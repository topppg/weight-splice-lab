import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from weight_splice_lab.toy.data import BOS, TOKENIZER, MixtureSampler
from weight_splice_lab.toy.model import (
    ToyLM,
    add_heads,
    add_neurons,
    apply_rope,
    head_slice,
    rope_cos_sin,
)

POS_PAIRS = (0, 1, 2, 3)
CONTENT_PAIR = 15
BIAS_PAIR = 14
ROLE_OFFSETS = {
    "a": lambda n: 2 * n + 1,
    "b": lambda n: n,
    "a1": lambda n: 2 * n + 2,
    "b1": lambda n: n + 1,
}
RUNG_ROLES = {
    "copy": ("a",),
    "successor": ("a",),
    "add_nocarry": ("a", "b"),
    "add": ("a", "b", "a1", "b1"),
}
DETECT_LAYER, MARKER_LAYER, BROADCAST_LAYER, READ_LAYER = 0, 0, 1, 2
CLEAN_LAYER, CARRY_LAYER, LOOKUP_LAYER, EMIT_LAYER = 2, 3, 4, 5
ANSWERS = (*(str(k) for k in range(10)), "\n")
MARKERS = ("slot", "plus", *(f"d{k}" for k in range(10)))
PLUS_ROLES = ("a", "a1")


@dataclass
class SpliceSpec:
    rung: str
    n_max: int
    feature_scale: float = 0.3
    read_scale: float = 0.15
    logic_scale: float = 1.0
    length_scale: float = 2.0
    beta_detect: float = 15.0
    beta_broadcast: float = 80.0
    beta_read: float = 120.0
    beta_sink: float = 60.0
    beta_emit_const_sink: float = 300.0
    beta_mlp: float = 20.0
    beta_emit: float = 200.0
    text_scale: float = 0.1
    text_dirs: int = 0
    beta_text_sink: float = 200.0
    text_gate: float = 4.0
    logit_margin: float = 25.0
    end_neuron: bool = True
    digit_source: str = "embedding"


@dataclass
class SpliceReport:
    ops: list[str] = field(default_factory=list)
    constants: dict[str, float] = field(default_factory=dict)
    added: dict[str, int] = field(default_factory=dict)
    params_allocated: int = 0
    params_nonzero: int = 0
    params_effective: int = 0
    layout: dict[str, int] = field(default_factory=dict)
    basis: Tensor | None = None


@dataclass
class NativeActs:
    tokens: Tensor
    z_attn: list[Tensor]
    z_mlp: list[Tensor]
    norm_attn: list[Tensor]
    norm_mlp: list[Tensor]
    norm_final: Tensor


def rms_normalize(x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


@torch.no_grad()
def collect_native(
    model: ToyLM,
    sampler: MixtureSampler,
    n_seqs: int,
    batch_size: int = 16,
    extra: Tensor | None = None,
) -> NativeActs:
    device = next(model.parameters()).device
    n_layers = len(model.blocks)
    toks: list[Tensor] = []
    z_attn: list[list[Tensor]] = [[] for _ in range(n_layers)]
    z_mlp: list[list[Tensor]] = [[] for _ in range(n_layers)]
    n_attn: list[list[Tensor]] = [[] for _ in range(n_layers)]
    n_mlp: list[list[Tensor]] = [[] for _ in range(n_layers)]
    n_final: list[Tensor] = []
    batches = [sampler.batch(batch_size) for _ in range(n_seqs // batch_size)]
    if extra is not None:
        batches += list(extra.split(batch_size))
    for batch in batches:
        batch = batch[:, :-1].to(device)
        cap: dict[str, Tensor] = {}
        model(batch, capture=cap)
        toks.append(batch.flatten().cpu())
        for i in range(n_layers):
            pre, mid = cap[f"resid_pre.{i}"].flatten(0, 1), cap[f"resid_mid.{i}"].flatten(0, 1)
            z_attn[i].append(rms_normalize(pre).cpu())
            z_mlp[i].append(rms_normalize(mid).cpu())
            n_attn[i].append(pre.norm(dim=-1).cpu())
            n_mlp[i].append(mid.norm(dim=-1).cpu())
        n_final.append(cap["resid_final"].flatten(0, 1).norm(dim=-1).cpu())
    return NativeActs(
        tokens=torch.cat(toks),
        z_attn=[torch.cat(z) for z in z_attn],
        z_mlp=[torch.cat(z) for z in z_mlp],
        norm_attn=[torch.cat(z) for z in n_attn],
        norm_mlp=[torch.cat(z) for z in n_mlp],
        norm_final=torch.cat(n_final),
    )


def unembed_matrix(model: ToyLM) -> Tensor:
    return model.embed.weight.detach().cpu() * model.final_norm.weight.detach().cpu()[None, :]


def private_basis(model: ToyLM, acts: NativeActs, k: int, report: SpliceReport) -> Tensor:
    m = unembed_matrix(model)
    _, s, vh = torch.linalg.svd(m, full_matrices=True)
    rank = int((s > s[0] * 1e-6).sum())
    null = vh[rank:].T
    zs = torch.cat([*acts.z_attn, *acts.z_mlp])
    bos = acts.tokens == BOS
    weights = torch.where(bos, 20.0, 1.0).repeat(2 * len(acts.z_attn))
    proj = zs @ null
    cov = (proj * weights[:, None]).T @ proj / weights.sum()
    evals, evecs = torch.linalg.eigh(cov)
    if k > null.shape[1]:
        raise ValueError(f"need {k} private dims, unembedding null space has {null.shape[1]}")
    basis = null @ evecs[:, :k]
    report.ops.append(
        f"SVD of E*diag(g_final) [own weights]: unembedding null space, dim {null.shape[1]}"
    )
    report.ops.append(
        f"eigendecomposition of native normalized-residual covariance inside that null space "
        f"[native activations, {len(acts.tokens)} positions x {2 * len(acts.z_attn)} sites]: "
        f"took {k} lowest-variance directions as private feature basis "
        f"(native std range {evals[0].sqrt():.4f}..{evals[k - 1].sqrt():.4f} per unit-rms residual)"
    )
    report.constants["private_native_std_max"] = float(evals[k - 1].sqrt())
    return basis


def fit_probe(z: Tensor, targets: Tensor, private: Tensor, ridge: float = 1e-3) -> Tensor:
    z = z - (z @ private) @ private.T
    gram = z.T @ z
    lam = ridge * gram.trace() / gram.shape[0]
    w = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0]), z.T @ targets)
    return w - private @ (private.T @ w)


def probe_r2(z: Tensor, targets: Tensor, w: Tensor) -> float:
    resid = (z @ w - targets).pow(2).sum()
    return float(1 - resid / (targets - targets.mean(0)).pow(2).sum())


def rope_query(key: Tensor, offset: int, head_dim: int, base: float) -> Tensor:
    cos, sin = rope_cos_sin(torch.tensor([-offset]), head_dim, base)
    return apply_rope(key[None], cos, sin)[0]


def pos_key(head_dim: int) -> Tensor:
    c = torch.zeros(head_dim)
    c[list(POS_PAIRS)] = 1.0
    return c


def content_vec(head_dim: int, pair: int = CONTENT_PAIR) -> Tensor:
    e = torch.zeros(head_dim)
    e[pair] = 1.0
    return e


def filter_bias(head_dim: int, base: float, window: int = 160) -> tuple[float, float]:
    key = pos_key(head_dim)
    cos, sin = rope_cos_sin(torch.arange(window), head_dim, base)
    scores = apply_rope(key.expand(window, head_dim), cos, sin) @ key
    peak, off_peak = float(scores[0]), float(scores[1:].max())
    return peak, off_peak


class Writer:
    def __init__(self, model: ToyLM, report: SpliceReport, bos_inputs: list[Tensor]) -> None:
        self.model = model
        self.report = report
        self.bos_inputs = bos_inputs
        self.d = model.cfg.d_model
        self.hd = model.cfg.head_dim
        self.touched: list[Tensor] = []

    def gain(self, layer: int, site: str) -> Tensor:
        block = self.model.blocks[layer]
        norm = block.attn_norm if site == "attn" else block.mlp_norm
        return norm.weight.detach().cpu()

    def new_head(self, layer: int, q: Tensor, k: Tensor, v: Tensor, o: Tensor, name: str) -> None:
        # BOS is the attention sink: project its (fixed) normalized residual out of the value
        # reads so a head that falls back to the sink writes exactly nothing.
        n_bos = self.bos_inputs[layer]
        v = v - torch.outer(v @ n_bos, n_bos) / n_bos.dot(n_bos)
        (h,) = add_heads(self.model, layer, 1)
        attn = self.model.blocks[layer].attn
        sl = head_slice(self.model, layer, h)
        dev = attn.q_proj.weight.device
        with torch.no_grad():
            attn.q_proj.weight[sl] = q.to(dev)
            attn.k_proj.weight[sl] = k.to(dev)
            attn.v_proj.weight[sl] = v.to(dev)
            attn.o_proj.weight[:, sl] = o.to(dev)
        for t in (q, k, v, o):
            self.report.params_allocated += t.numel()
            self.report.params_nonzero += int((t != 0).sum())
        rows = sum(int((t != 0).any(1).sum()) for t in (q, k, v)) + int((o != 0).any(0).sum())
        self.report.params_effective += rows * self.d
        self.report.added[f"L{layer}.head.{name}"] = 1

    def new_neurons(self, layer: int, gate: Tensor, up: Tensor, down: Tensor, name: str) -> None:
        idx = add_neurons(self.model, layer, gate.shape[0])
        mlp = self.model.blocks[layer].mlp
        dev = mlp.gate_proj.weight.device
        with torch.no_grad():
            mlp.gate_proj.weight[idx] = gate.to(dev)
            mlp.up_proj.weight[idx] = up.to(dev)
            mlp.down_proj.weight[:, idx] = down.to(dev)
        for t in (gate, up, down):
            self.report.params_allocated += t.numel()
            self.report.params_nonzero += int((t != 0).sum())
        self.report.params_effective += 3 * self.d * gate.shape[0]
        self.report.added[f"L{layer}.neurons.{name}"] = gate.shape[0]


def output_directions(model: ToyLM, report: SpliceReport) -> Tensor:
    m = unembed_matrix(model)
    pinv = torch.linalg.pinv(m)
    ones = pinv.sum(1)
    shift = (pinv.T @ ones) / ones.dot(ones)
    dirs = pinv - torch.outer(ones, shift)
    report.ops.append(
        "pseudo-inverse of E*diag(g_final) [own weights]: residual direction w_t with logits "
        "e_t - c_t*1 (one-hot up to a uniform shift; c_t closed-form to minimise |w_t|)"
    )
    return dirs


def build_splice(model: ToyLM, spec: SpliceSpec, acts: NativeActs) -> SpliceReport:
    report = SpliceReport()
    first_bos = int((acts.tokens == BOS).nonzero()[0])
    bos_inputs = [
        z[first_bos] * b.attn_norm.weight.detach().cpu() for z, b in zip(acts.z_attn, model.blocks)
    ]
    report.ops.append(
        "projection of the BOS normalized residual out of every added head's value reads "
        "[own activation at BOS, which is context-independent]"
    )
    w = Writer(model, report, bos_inputs)
    d, hd, base = w.d, w.hd, model.cfg.rope_base
    sqrt_d, sqrt_hd = math.sqrt(d), math.sqrt(hd)
    n_values = range(1, spec.n_max + 1)
    roles = RUNG_ROLES[spec.rung]
    tok = acts.tokens
    digit_ids = TOKENIZER.digit_ids()
    eq, plus, colon = TOKENIZER.id("="), TOKENIZER.id("+"), TOKENIZER.id(":")
    slot_ids = torch.tensor([*digit_ids, eq, plus, colon])

    def n_dims(role: str) -> int:
        return 11 if role in PLUS_ROLES else 10

    # Private feature layout (raw residual directions).
    # The basis is sorted by native variance, so the order here assigns the quietest directions to
    # the small, noise-sensitive features (T, the raw read one-hots, markers) and E/A, and the
    # noisier ones to large features (length codes) and cleaned / downstream logic features.
    raw_reads = [f"{role}.{j}" for role in roles for j in range(n_dims(role))]
    cleaned = [f"{role}~{j}" for role in roles for j in range(n_dims(role))]
    names = ["T", "E", "A", *raw_reads, *MARKERS, *(f"ans.{k}" for k in range(len(ANSWERS)))]
    names += [*(f"Lc{n}" for n in n_values), *(f"Lraw{n}" for n in n_values), "one~", *cleaned]
    if spec.rung == "add":
        names += ["zero~", "carry"]
    layout: dict[str, int] = {name: i for i, name in enumerate(names)}
    private = private_basis(model, acts, len(layout), report)
    report.layout, report.basis = layout, private

    def u(name: str) -> Tensor:
        return private[:, layout[name]]

    # Raw feature magnitudes from native activation statistics: U_m for always-on markers, U_l for
    # gated logic features (one-hots, carry), U_c for the length code and A (large, so they
    # dominate the RMS at addition positions; they only ever exist there).
    not_bos = tok != BOS
    nu = [float(n[not_bos].median()) for n in acts.norm_attn]
    nu_mid = [float(n[not_bos].median()) for n in acts.norm_mlp]
    nu_final = float(acts.norm_final[not_bos].median())
    u_m = spec.feature_scale * nu[READ_LAYER]
    u_l = spec.logic_scale * nu[READ_LAYER]
    u_c = spec.length_scale * nu[READ_LAYER]
    u_t = spec.text_scale * nu[READ_LAYER]
    u_r = spec.read_scale * nu[READ_LAYER]
    report.ops.append("median residual norms per layer [native activation statistic]")
    report.constants |= {
        "U_marker": u_m,
        "U_logic": u_l,
        "U_length": u_c,
        "nu_read_layer": nu[READ_LAYER],
        "nu_final": nu_final,
    }

    # Expected read of a feature of raw size `scale` through RMSNorm, given the native norm and
    # how many marker / logic / length features are active at that site.
    def unit(norm: float, n_m: int, n_l: int, n_c: int, scale: float) -> float:
        rms = math.sqrt(norm**2 + n_m * u_m**2 + n_l * u_l**2 + n_c * u_c**2) / sqrt_d
        return scale / rms

    # Probes (ridge least squares on native normalized residuals, orthogonal to private basis).
    def probes(layer: int, site: str) -> dict[str, Tensor]:
        z = acts.z_attn[layer] if site == "attn" else acts.z_mlp[layer]
        g = w.gain(layer, site)
        names = ["const", "bos", "eq", "plus", "slot", "zero"]
        targets = torch.stack(
            [
                not_bos.float(),
                (tok == BOS).float(),
                (tok == eq).float(),
                (tok == plus).float(),
                torch.isin(tok, slot_ids).float(),
                (tok == digit_ids[0]).float(),
            ],
            dim=1,
        )
        wt = fit_probe(z, targets, private)
        r2 = {n: probe_r2(z, targets[:, i : i + 1], wt[:, i : i + 1]) for i, n in enumerate(names)}
        const = (z @ wt[:, 0])[not_bos]
        report.ops.append(
            f"ridge least-squares probes {'/'.join(names)} at L{layer}.{site} "
            f"[native activations, no target data]: const on non-BOS {const.mean():.3f}"
            f"+-{const.std():.3f}, R2 "
            + ", ".join(f"{k}={v:.3f}" for k, v in r2.items() if k != "const")
        )
        return {n: wt[:, i] / g for i, n in enumerate(names)}

    def digit_probe(layer: int) -> Tensor:
        z = acts.z_attn[layer]
        sel = torch.isin(tok, torch.cat([slot_ids, torch.tensor([BOS])]))
        targets = torch.zeros(len(tok), 10)
        for dgt, tid in enumerate(digit_ids):
            targets[tok == tid, dgt] = 1.0
        wt = fit_probe(z[sel], targets[sel], private)
        is_digit = torch.isin(tok[sel], torch.tensor(digit_ids))
        pred = (z[sel] @ wt).argmax(1) == targets[sel].argmax(1)
        acc = float(pred[is_digit].float().mean())
        off = float((z[sel] @ wt)[~is_digit].abs().max())
        report.ops.append(
            f"ridge least-squares digit one-hot probe (delimiters and BOS -> 0) at L{layer}.attn "
            f"[native activations]: R2={probe_r2(z[sel], targets[sel], wt):.3f}, "
            f"digit argmax acc={acc:.4f}, max |output| on delimiters={off:.3f}"
        )
        return wt / w.gain(layer, "attn")[:, None]

    key_pos, e_c, e_b = pos_key(hd), content_vec(hd), content_vec(hd, BIAS_PAIR)
    silu_on = float(F.silu(torch.tensor(0.5 * spec.beta_mlp)))
    beta_step = 2 * spec.beta_mlp

    # Layer 0 attention: one length detector per N. Query at '=' looks N+1 back; value = is '+'.
    # At layer 0 the residual is the token embedding, so these probes are exact.
    p0 = probes(DETECT_LAYER, "attn")
    for n in n_values:
        q = sqrt_hd * (
            spec.beta_detect * torch.outer(rope_query(key_pos, n + 1, hd, base), p0["eq"])
            + spec.beta_sink * torch.outer(e_c, p0["const"])
        )
        k = torch.outer(key_pos, p0["const"]) + torch.outer(e_c, p0["bos"])
        v = torch.zeros(hd, d)
        v[0] = p0["plus"]
        o = torch.zeros(d, hd)
        o[:, 0] = u_c * (u(f"Lraw{n}") + u("E"))
        w.new_head(DETECT_LAYER, q, k, v, o, f"detect_N{n}")

    # Layer 0 attention: token markers (offset-0 head) read from the token embedding: slot
    # (digit/'+'/'='/':'), plus, and the digit one-hot d0..d9.
    q = sqrt_hd * spec.beta_detect * torch.outer(rope_query(key_pos, 0, hd, base), p0["const"])
    k = torch.outer(key_pos, p0["const"])
    v = torch.zeros(hd, d)
    o = torch.zeros(d, hd)
    d0probe = digit_probe(MARKER_LAYER)
    marker_reads = {"slot": p0["slot"], "plus": p0["plus"]}
    marker_reads |= {f"d{k}": d0probe[:, k] for k in range(10)}
    for j, m in enumerate(MARKERS):
        v[j] = marker_reads[m]
        o[:, j] = u_m * u(m)
    w.new_head(MARKER_LAYER, q, k, v, o, "markers")

    # Layer 1 attention: broadcast the length one-hot from the addition's '=' (keyed on Lraw
    # itself) to all later positions, writing into the same Lraw dims.
    p1 = probes(BROADCAST_LAYER, "attn")
    g1 = w.gain(BROADCAST_LAYER, "attn")
    lraw_read = unit(nu[BROADCAST_LAYER], 1, 0, 2, u_c)
    # E (written by every detector) marks an addition's '=' with a single direction, so the
    # key is not the sum of N_max noisy length directions.
    e_read = u("E") / g1 / lraw_read
    # The '=' itself already holds Lraw in the same dims: a negative Lraw term in the query keeps
    # it from re-adding its own copy (it falls back to positions whose value is zero).
    q = sqrt_hd * spec.beta_broadcast * torch.outer(e_c, p1["const"] - e_read)
    k = torch.outer(e_c, e_read + 0.5 * p1["bos"])
    v = torch.zeros(hd, d)
    o = torch.zeros(d, hd)
    # E is copied too, so '=' and the answer positions carry the same set of large features.
    for j, name in enumerate([*(f"Lraw{n}" for n in n_values), "E"]):
        v[j] = u(name) / g1
        o[:, j] = u_c / lraw_read * u(name)
    w.new_head(BROADCAST_LAYER, q, k, v, o, "broadcast")

    # Layer 1 MLP: saturating step per N, silu(b(l - 0.3E)) - silu(b(l - 0.6E)), with the length
    # read l as the up projection. Every input is a large feature (U_c), so native noise is a few
    # percent of the on-value and the output is quadratic in that noise outside additions. Each
    # step writes U_c to its own Lc_N and to the shared in-addition feature A, so A has exactly
    # the magnitude of the active Lc_N.
    gm1 = w.gain(BROADCAST_LAYER, "mlp")
    unc1 = unit(nu_mid[BROADCAST_LAYER], 1, 0, 2, u_c)
    e1 = u("E") / gm1 / unc1
    gate_rows, up_rows, down_cols = [], [], []
    for n in n_values:
        l_n = u(f"Lraw{n}") / gm1 / unc1
        for thr, sign in ((0.3, 1.0), (0.6, -1.0)):
            gate_rows.append(beta_step * (l_n - thr * e1))
            up_rows.append(l_n)
            down_cols.append(sign * u_c / (0.3 * beta_step) * (u(f"Lc{n}") + u("A")))
    w.new_neurons(
        BROADCAST_LAYER,
        torch.stack(gate_rows),
        torch.stack(up_rows),
        torch.stack(down_cols, dim=1),
        "length_cleanup",
    )

    # Layer 1 MLP: text energy T = sum_i z_i^2 over all directions of the normalized residual
    # orthogonal to the private basis, via silu(b z) z + silu(-b z)(-z) ~ b z^2. Averaged
    # over many directions it is ~1 in plain text and drops ~(native share)^2 inside an addition,
    # where the large splice features dominate the RMS. It is the off-task signal for sinks and
    # MLP gates (a linear read of any single direction is too noisy for that in this model).
    zm = acts.z_mlp[BROADCAST_LAYER][not_bos]
    zm = zm - (zm @ private) @ private.T
    _, evecs_t = torch.linalg.eigh(zm.T @ zm / len(zm))
    n_dirs = d - private.shape[1] if spec.text_dirs <= 0 else spec.text_dirs
    dirs_t = evecs_t[:, -n_dirs:]
    energy = float(((zm @ dirs_t) ** 2).sum(1).mean())
    beta_t = 5.0
    gate_rows, up_rows = [], []
    for i in range(n_dirs):
        r = dirs_t[:, i] / gm1
        gate_rows += [beta_t * r, -beta_t * r]
        up_rows += [r, -r]
    down = (u_t / (beta_t * energy) * u("T"))[:, None].expand(d, len(gate_rows)).clone()
    w.new_neurons(
        BROADCAST_LAYER, torch.stack(gate_rows), torch.stack(up_rows), down, "text_energy"
    )
    report.ops.append(
        f"second-moment eigendecomposition of native L1.mlp residuals (orthogonal to private "
        f"basis): top {n_dirs} directions for the text-energy feature, mean energy "
        f"{energy:.1f} [native activations]"
    )

    def t_row(layer: int, site: str) -> Tensor:
        norm = nu[layer] if site == "attn" else nu_mid[layer]
        return u("T") / w.gain(layer, site) / unit(norm, 1, 0, 0, u_t)

    # Layer 2 attention: positional reads. Key = slot marker in the positional RoPE pairs and in
    # a near-static bias pair. Query = sum_N Lc_N * rope(-offset_N) minus A * bias, with the bias
    # between the filter's peak and its largest off-peak value: the target slot is the only
    # position with a positive score whatever the key magnitudes. Outside additions the query is
    # only the BOS sink (value 0).
    p2 = probes(READ_LAYER, "attn")
    g2 = w.gain(READ_LAYER, "attn")
    if spec.digit_source == "embedding":
        dvals = torch.stack(
            [u(f"d{k}") / g2 / unit(nu[READ_LAYER], 1, 0, 0, u_m) for k in range(10)]
        )
    else:
        dvals = digit_probe(READ_LAYER).T
    peak, off_peak = filter_bias(hd, base)
    bias = 0.5 * (peak + off_peak)
    slot_read = unit(nu[READ_LAYER], 1, 0, 0, u_m)
    lc_read = unit(nu[READ_LAYER], 1, 0, 4, u_c)
    plus_read = slot_read
    report.constants |= {"filter_peak": peak, "filter_off_peak": off_peak, "filter_bias": bias}
    scale = sqrt_hd * spec.beta_read / lc_read
    # The template tag 'A' always sits right after BOS, so its normalized residual is fixed;
    # project it out of the read keys so its (deterministic) slot-direction component cannot win
    # the bias-trick comparison.
    with torch.no_grad():
        tag = torch.tensor([TOKENIZER.encode("A:")], device=next(model.parameters()).device)
        cap_tag: dict[str, Tensor] = {}
        model(tag, capture=cap_tag)
        n_tag = model.blocks[READ_LAYER].attn_norm(cap_tag[f"resid_pre.{READ_LAYER}"][0, 1]).cpu()
    report.ops.append(
        "projection of the fixed template-tag residual (position 1, 'A' after BOS, L2 input) out of "
        "the read-head keys [one input-only activation of the target prompt prefix]"
    )
    # '=' and output positions (they hold the in-addition feature A) get a large bias-pair key, so
    # the query's negative bias rules them out; no read ever targets them.
    big_key = 0.1 * u("A") / g2 / lc_read
    for role in roles:
        q = sqrt_hd * spec.beta_sink * torch.outer(e_c, t_row(READ_LAYER, "attn"))
        q = q - scale * bias * torch.outer(e_b, u("A") / g2)
        for n in n_values:
            qp = rope_query(key_pos, ROLE_OFFSETS[role](n), hd, base)
            q = q + scale * torch.outer(qp, u(f"Lc{n}") / g2)
        k = (
            torch.outer(key_pos + e_b, u("slot") / g2 / slot_read)
            + torch.outer(e_b, big_key)
            + torch.outer(e_c, p2["bos"])
        )
        k = k - torch.outer(k @ n_tag, n_tag) / n_tag.dot(n_tag)
        v = torch.zeros(hd, d)
        v[:10] = dvals
        if role in PLUS_ROLES:
            v[10] = u("plus") / g2 / plus_read
        o = torch.zeros(d, hd)
        for j in range(n_dims(role)):
            o[:, j] = u_r * u(f"{role}.{j}")
        w.new_head(READ_LAYER, q, k, v, o, f"read_{role}")

    out_dirs = output_directions(model, report)
    out_norm = float(out_dirs[:, digit_ids].norm(dim=0).max())
    n_small = 1 + 2 * len(roles) + (1 if spec.rung == "add" else 0)
    margin = min(spec.logit_margin, 0.85 * sqrt_d / out_norm)
    denom = 1 - (margin * out_norm) ** 2 / d
    u_out = margin * math.sqrt((nu_final**2 + u_m**2 + n_small * u_l**2 + 4 * u_c**2) / d / denom)
    report.constants |= {"logit_margin": margin, "U_out": u_out, "max_out_dir_norm": out_norm}

    # MLP neurons are AND gates over one-hot features with thresholds relative to
    # R = A * U_l / U_c (the in-addition feature rescaled to one-hot units); the up projection is A,
    # so every neuron is silent outside additions.
    def refs(layer: int, n_small: int) -> tuple[Tensor, Tensor, Tensor, float]:
        g = w.gain(layer, "mlp")
        un = unit(nu_mid[layer], 1, n_small, 4, u_l)
        return g, (u_l / u_c) * u("A") / g, (u_l / u_c) * u("A") / g / un, un

    # Downstream of the cleanup, the reference is one~ (the cleaned in-addition constant), so a
    # threshold always
    # compares quantities with the same RMSNorm/SwiGLU history (bias-free SwiGLU is degree-2
    # homogeneous in its normalized input, so mixing histories turns rms-estimate errors into
    # threshold errors).
    def clean_refs(layer: int, n_small: int) -> tuple[Tensor, Tensor, Tensor, float]:
        g = w.gain(layer, "mlp")
        un = unit(nu_mid[layer], 1, n_small, 4, u_l)
        return g, u("one~") / g, (u_l / u_c) * u("A") / g / un, un

    # Layer 2 MLP: clean every read one-hot r.j into r~j with a saturating step relative to R,
    # so downstream AND gates see unit inputs whatever the native probe's in-context gain.
    g = w.gain(CLEAN_LAYER, "mlp")
    un = unit(nu_mid[CLEAN_LAYER], 1 + len(roles), 0, 4, u_r)
    rref = (u_r / u_c) * u("A") / g
    up_row = rref / un
    gate_rows, down_cols = [], []
    for role in roles:
        for j in range(n_dims(role)):
            for thr, sign in ((0.3, 1.0), (0.6, -1.0)):
                gate_rows.append(beta_step / un * (u(f"{role}.{j}") / g - thr * rref))
                down_cols.append(sign * u_l / (0.3 * beta_step) * u(f"{role}~{j}"))
    for thr, sign in ((0.3, 1.0), (0.6, -1.0)):
        gate_rows.append(beta_step / un * (1 - thr) * rref)
        down_cols.append(sign * u_l / (0.3 * beta_step) * u("one~"))
        if spec.rung == "add":
            gate_rows.append(beta_step / un * ((u_r / u_m) * u("d0") / g - thr * rref))
            down_cols.append(sign * u_l / (0.3 * beta_step) * u("zero~"))
    up = up_row.expand(len(gate_rows), d).clone()
    w.new_neurons(
        CLEAN_LAYER,
        torch.stack(gate_rows) - beta_step * spec.text_gate * t_row(CLEAN_LAYER, "mlp"),
        up,
        torch.stack(down_cols, dim=1),
        "onehot_cleanup",
    )

    # Layer 3 MLP: carry into digit i from (a_{i-1}, b_{i-1}, s_{i-1} == 0). Each AND is a
    # narrow saturating step silu(b(sum - t1 R)) - silu(b(sum - t2 R)), centred between
    # the on and off sums, so the carry is a clean unit.
    if spec.rung == "add":
        g, rref, up_row, un = clean_refs(CARRY_LAYER, 3 + 2 * len(roles))
        beta_and = 5 * spec.beta_mlp
        zero = u("zero~")
        rows: list[Tensor] = []
        signs: list[float] = []
        for x in range(10):
            for y in range(10):
                if x + y >= 10:
                    inputs, t1 = u(f"a1~{x}") + u(f"b1~{y}"), 1.4
                elif x + y == 9:
                    inputs, t1 = u(f"a1~{x}") + u(f"b1~{y}") + zero, 2.4
                else:
                    continue
                for t, sign in ((t1, 1.0), (t1 + 0.2, -1.0)):
                    rows.append(inputs / g - t * rref)
                    signs.append(sign)
        gate = beta_and / un * torch.stack(rows)
        gate = gate - beta_and * spec.text_gate * t_row(CARRY_LAYER, "mlp")
        up = up_row.expand(len(rows), d).clone()
        down = torch.outer(u_l / (0.2 * beta_and) * u("carry"), torch.tensor(signs))
        w.new_neurons(CARRY_LAYER, gate, up, down, "carry")

    # Layer 4 MLP: digit lookup table writing a private answer one-hot (ans.k).
    g, rref, up_row, un = clean_refs(LOOKUP_LAYER, n_small + 1)
    rows, outs = [], []

    def emit(row: Tensor, token: int, weight: float = 1.0) -> None:
        rows.append(row)
        outs.append(weight * u(f"ans.{ANSWERS.index(TOKENIZER.itos[token])}"))

    match spec.rung:
        case "copy" | "successor":
            shift = 0 if spec.rung == "copy" else 1
            for x in range(10):
                emit(u(f"a~{x}") / g - 0.5 * rref, digit_ids[(x + shift) % 10])
        case "add_nocarry":
            for x in range(10):
                for y in range(10):
                    emit((u(f"a~{x}") + u(f"b~{y}")) / g - 1.5 * rref, digit_ids[(x + y) % 10])
            emit(u("a~10") / g - 0.5 * rref, digit_ids[0])
        case "add":
            for x in range(10):
                for y in range(10):
                    a_b = u(f"a~{x}") + u(f"b~{y}")
                    emit((a_b - u("carry")) / g - 1.5 * rref, digit_ids[(x + y) % 10])
                    emit((a_b + u("carry")) / g - 2.5 * rref, digit_ids[(x + y + 1) % 10])
            # Step i = N: a_i is the '+' (read as the plus marker), the digit is the carry.
            emit((u("a~10") + u("carry")) / g - 1.5 * rref, digit_ids[1])
            emit((u("a~10") - u("carry")) / g - 0.5 * rref, digit_ids[0])
            if spec.end_neuron:
                # Step i = N+1: a_{i-1} is the '+', so the answer is complete.
                emit(u("a1~10") / g - 0.5 * rref, TOKENIZER.id("\n"), 2.0)
    gate = spec.beta_mlp / un * torch.stack(rows)
    gate = gate - spec.beta_mlp * spec.text_gate * t_row(LOOKUP_LAYER, "mlp")
    up = up_row.expand(len(rows), d).clone()
    down = u_m / silu_on * torch.stack(outs, dim=1)
    w.new_neurons(LOOKUP_LAYER, gate, up, down, "lookup")

    # Layer 5 attention: emit head. Inside an addition it attends to itself (offset-0 RoPE filter,
    # keyed and queried by A, so earlier answer positions score 0.64*beta lower and A-free
    # positions ~0) and copies ans.k into the unembedding direction
    # of token k; elsewhere A is ~0 and it falls back to the BOS sink, whose value is exactly 0.
    # Softmax gives exponential suppression, which bias-free (homogeneous) SwiGLU cannot.
    p5 = probes(EMIT_LAYER, "attn")
    g5 = w.gain(EMIT_LAYER, "attn")
    a5 = u("A") / g5 / unit(nu[EMIT_LAYER], n_small, 0, 4, u_c)
    q = sqrt_hd * (
        spec.beta_emit * torch.outer(rope_query(key_pos, 0, hd, base), a5)
        + spec.beta_text_sink * torch.outer(e_c, t_row(EMIT_LAYER, "attn"))
        + spec.beta_emit_const_sink * torch.outer(e_c, p5["const"])
    )
    k = torch.outer(key_pos, a5) + torch.outer(e_c, p5["bos"])
    v = torch.zeros(hd, d)
    o = torch.zeros(d, hd)
    ans_read = unit(nu[EMIT_LAYER], n_small, 0, 4, u_m)
    for j, token in enumerate(ANSWERS):
        v[j] = u(f"ans.{j}") / g5 / ans_read
        o[:, j] = u_out * out_dirs[:, TOKENIZER.id(token)]
    w.new_head(EMIT_LAYER, q, k, v, o, "emit")

    report.constants |= {
        "beta_detect": spec.beta_detect,
        "beta_broadcast": spec.beta_broadcast,
        "beta_read": spec.beta_read,
        "beta_sink": spec.beta_sink,
        "beta_mlp": spec.beta_mlp,
        "beta_emit": spec.beta_emit,
        "feature_scale": spec.feature_scale,
        "read_scale": spec.read_scale,
        "logic_scale": spec.logic_scale,
        "length_scale": spec.length_scale,
    }
    return report
