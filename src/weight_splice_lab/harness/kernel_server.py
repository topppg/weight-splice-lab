# Runs inside the sandboxed kernel process (python -I -u -c <this source>), cwd = the run folder.
# It must not import weight_splice_lab: the repo is not readable from the sandbox.
import ast
import json
import os
import sys
import tempfile
import traceback
import types

MAX_CAPTURE = 200_000
# a module this large in the namespace is a model copy that could be run outside the forward tool
MODEL_OBJECT_PARAMS = 100_000
# read here, not from the cell namespace: model/ is read-only to the editor
try:
    with open("model/config.json") as _f:
        KERNEL_FORWARD = bool(json.load(_f).get("kernel_forward"))
except OSError:
    KERNEL_FORWARD = False

PRELUDE = r"""
import json, os, sys, types
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("KERNEL_THREADS", "4")))
np.set_printoptions(precision=4, suppress=True, linewidth=160)


class HarnessRuleError(RuntimeError):
    pass


def _forbid(what):
    def f(*args, **kwargs):
        raise HarnessRuleError(
            f"{what} is not allowed in this kernel: no gradients, optimizers or running the model"
        )
    return f


def _install_grad_guards():
    # The originals live in these closures, not in the cell namespace, so a cell cannot call them
    # to re-enable gradients (torch.set_grad_enabled, the private _C setter, Tensor.requires_grad_).
    _real_cpp_set_grad = torch._C._set_grad_enabled
    _real_requires_grad_ = torch.Tensor.requires_grad_
    _real_cpp_set_grad(False)

    class _NoGradEnabled:
        def __init__(self, mode=False):
            if mode:
                raise HarnessRuleError("enabling gradients is not allowed in this kernel")
            _real_cpp_set_grad(False)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, fn):
            return fn

    def _cpp_set_grad_guard(mode=True):
        if mode:
            raise HarnessRuleError("enabling gradients is not allowed in this kernel")
        return _real_cpp_set_grad(False)

    def _requires_grad_guard(self, requires_grad=True):
        if requires_grad:
            raise HarnessRuleError("requires_grad is not allowed in this kernel")
        return _real_requires_grad_(self, False)

    torch._C._set_grad_enabled = _cpp_set_grad_guard
    torch.set_grad_enabled = _NoGradEnabled
    torch.autograd.set_grad_enabled = _NoGradEnabled
    torch.enable_grad = _forbid("torch.enable_grad")
    torch.autograd.enable_grad = _forbid("torch.enable_grad")
    torch.Tensor.backward = _forbid("Tensor.backward")
    torch.autograd.backward = _forbid("torch.autograd.backward")
    torch.autograd.grad = _forbid("torch.autograd.grad")
    torch.Tensor.requires_grad_ = _requires_grad_guard

    class _InferenceMode(torch.inference_mode):
        # inference_mode(False) would turn grad mode back on inside the block
        def __new__(cls, mode=True):
            if isinstance(mode, bool) and not mode:
                raise HarnessRuleError("inference_mode(False) is not allowed in this kernel")
            return super().__new__(cls, mode)

    torch.inference_mode = _InferenceMode
    torch.autograd.inference_mode = _InferenceMode
    torch.autograd.grad_mode.inference_mode = _InferenceMode


_install_grad_guards()
for _name in ("grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd", "hessian"):
    if hasattr(torch.func, _name):
        setattr(torch.func, _name, _forbid(f"torch.func.{_name}"))


class _Forbidden(types.ModuleType):
    def __getattr__(self, name):
        # Dunder probes (hasattr(m, "__file__") over sys.modules) must see a plain missing attribute.
        if name.startswith("__"):
            raise AttributeError(name)
        raise HarnessRuleError(f"{self.__name__}.{name} is not allowed in this kernel")


for _mod in [m for m in sys.modules if m == "torch.optim" or m.startswith("torch.optim.")]:
    sys.modules[_mod] = _Forbidden(_mod)
torch.optim = sys.modules["torch.optim"]
_CONFIG = json.loads(Path("model/config.json").read_text())
# kernel_forward runs (phase 4c) may run the model in the kernel; gradients stay disabled
_KERNEL_FORWARD = bool(_CONFIG.get("kernel_forward"))
if not _KERNEL_FORWARD:
    torch.nn.functional.scaled_dot_product_attention = _forbid("scaled_dot_product_attention")

sys.path.insert(0, str(Path("model").resolve()))
import model as _model_module  # noqa: E402
import tokenizer as _tokenizer_module  # noqa: E402

globals()[_CONFIG.get("module_alias", "toy_model")] = _model_module
# every module class of model.py loses forward; the LM class also loses generate and hidden
_GUARDED = [] if _KERNEL_FORWARD else [
    c for c in vars(_model_module).values()
    if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == "model"
]
for _cls in _GUARDED:
    for _meth in ("forward", "generate", "hidden"):
        if _meth in vars(_cls):
            setattr(_cls, _meth, _forbid(f"{_cls.__name__}.{_meth}"))
del _mod, _name, _install_grad_guards

if hasattr(_tokenizer_module, "load"):
    tok = _tokenizer_module.load()
else:
    tok = _tokenizer_module.CharTokenizer()
CFG = _CONFIG["model"]
CONDITION = _CONFIG["condition"]
MAX_ADDED_PARAMS = _CONFIG["max_added_params"]
# kept in the stored dtype (Gemma: bfloat16); load() and load_original() return float32
_BASE = torch.load("model/base.pt", map_location="cpu", weights_only=True)["state"]
BASE_ARCH = {"n_heads": [CFG["n_heads"]] * CFG["n_layers"], "d_mlp": [CFG["d_mlp"]] * CFG["n_layers"]}
REFUSED = ("embed.weight", "final_norm.weight")
_STAGE = Path("work/edited")
_TENSORS = _STAGE / "tensors"


def arch():
    p = _STAGE / "arch.json"
    return json.loads(p.read_text()) if p.exists() else json.loads(json.dumps(BASE_ARCH))


def _write_arch(a):
    _STAGE.mkdir(parents=True, exist_ok=True)
    (_STAGE / "arch.json").write_text(json.dumps(a))


def shape(name, a=None):
    # q/o grow by head_dim per added head, k/v by one key/value head per added head
    a = a or arch()
    base = tuple(_BASE[name].shape)
    parts = name.split(".")
    if parts[0] != "blocks":
        return base
    layer, rest = int(parts[1]), ".".join(parts[2:])
    hd = CFG["head_dim"]
    added = a["n_heads"][layer] - BASE_ARCH["n_heads"][layer]
    h, m = a["n_heads"][layer] * hd, a["d_mlp"][layer]
    if rest == "attn.q_proj.weight":
        return (h, base[1])
    if rest in ("attn.k_proj.weight", "attn.v_proj.weight"):
        return (base[0] + added * hd, base[1])
    if rest == "attn.o_proj.weight":
        return (base[0], h)
    if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        return (m, base[1])
    if rest == "mlp.down_proj.weight":
        return (base[0], m)
    return base


NAMES = list(_BASE)


def _pad(t, target):
    if tuple(t.shape) == tuple(target):
        return t
    out = torch.zeros(target, dtype=torch.float32)
    out[tuple(slice(0, s) for s in t.shape)] = t
    return out


def load_original(name):
    return _BASE[name].to(torch.float32, copy=True)


def load(name):
    p = _TENSORS / f"{name}.pt"
    if p.exists():
        return _pad(torch.load(p, weights_only=True).float(), shape(name))
    return _pad(_BASE[name].to(torch.float32, copy=True), shape(name))


def set_tensor(name, tensor):
    if name not in _BASE:
        raise KeyError(f"unknown tensor {name}; see NAMES")
    if name in REFUSED:
        raise ValueError(f"{name} may not be edited")
    t = torch.as_tensor(tensor, dtype=torch.float32).detach().clone()
    if tuple(t.shape) != shape(name):
        raise ValueError(f"{name}: shape {tuple(t.shape)} != expected {shape(name)}")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name}: values must be finite")
    _TENSORS.mkdir(parents=True, exist_ok=True)
    torch.save(t, _TENSORS / f"{name}.pt")


def reset_tensor(name):
    (_TENSORS / f"{name}.pt").unlink(missing_ok=True)


def staged():
    return sorted(p.name[:-3] for p in _TENSORS.glob("*.pt")) if _TENSORS.exists() else []


def reset_edit():
    for p in staged():
        reset_tensor(p)
    (_STAGE / "arch.json").unlink(missing_ok=True)


def added_params(a=None):
    a = a or arch()
    total = 0
    for name, t in _BASE.items():
        n0 = 1
        for x in t.shape:
            n0 *= x
        n1 = 1
        for x in shape(name, a):
            n1 *= x
        total += n1 - n0
    return total


def _grow(layer, key, n, names):
    if CONDITION != "add_capacity":
        raise HarnessRuleError(f"adding capacity is not allowed under condition {CONDITION}")
    if n <= 0:
        raise ValueError("n must be positive")
    a = arch()
    first = a[key][layer]
    a[key][layer] += n
    if added_params(a) > MAX_ADDED_PARAMS:
        raise ValueError(f"added parameters would be {added_params(a):,} > cap {MAX_ADDED_PARAMS:,}")
    _write_arch(a)
    for name in names:
        p = _TENSORS / f"{name}.pt"
        if p.exists():
            torch.save(_pad(torch.load(p, weights_only=True).float(), shape(name, a)), p)
    return first, a[key][layer]


def add_heads(layer, n):
    names = [f"blocks.{layer}.attn.{x}_proj.weight" for x in "qkvo"]
    first, last = _grow(layer, "n_heads", n, names)
    return list(range(first, last))


def add_neurons(layer, n):
    names = [f"blocks.{layer}.mlp.{x}_proj.weight" for x in ("gate", "up", "down")]
    first, last = _grow(layer, "d_mlp", n, names)
    return list(range(first, last))
"""


BUILD_MODEL = r"""
_ORIGINAL_MODEL = None


def build_model(edited=True):
    # float32, eval mode, no gradients. edited=False: the original (cached and shared, never modify
    # it). edited=True: a new copy with your STAGED edit and added capacity; shares the embedding.
    global _ORIGINAL_MODEL
    if _ORIGINAL_MODEL is None:
        _ORIGINAL_MODEL = _model_module.load_model(Path("model/base.pt")).requires_grad_(False)
    if not edited:
        return _ORIGINAL_MODEL
    import copy
    emb = _ORIGINAL_MODEL.embed.weight
    m = copy.deepcopy(_ORIGINAL_MODEL, {id(emb): emb})
    a = arch()
    for i in range(CFG["n_layers"]):
        if a["n_heads"][i] > CFG["n_heads"]:
            _model_module.add_heads(m, i, a["n_heads"][i] - CFG["n_heads"])
        if a["d_mlp"][i] > CFG["d_mlp"]:
            _model_module.add_neurons(m, i, a["d_mlp"][i] - CFG["d_mlp"])
    m.requires_grad_(False)
    state = m.state_dict()
    for name in staged():
        if name in state and name not in REFUSED:  # save_edit refuses the rest anyway
            state[name].copy_(load(name))
    return m.eval()
"""


def _snapshot_guards(ns: dict[str, object]) -> dict[str, object]:
    import torch

    classes = [
        (c, m, vars(c)[m])
        for c in ns["_GUARDED"]  # type: ignore[attr-defined]
        for m in ("forward", "generate", "hidden")
        if m in vars(c)
    ]
    opt = [n for n in sys.modules if n == "torch.optim" or n.startswith("torch.optim.")]
    optim = {n: sys.modules.get(n) for n in opt}
    return {
        "model_module": sys.modules.get("model"),
        "classes": classes,
        "sdpa": torch.nn.functional.scaled_dot_product_attention,
        "set_grad_enabled": torch.set_grad_enabled,
        "autograd_set_grad": torch.autograd.set_grad_enabled,
        "backward": torch.Tensor.backward,
        "inference_mode": torch.inference_mode,
        "optim": optim,
    }


def _check_integrity(ns: dict[str, object], g: dict[str, object]) -> list[dict[str, str]]:
    import torch

    v: list[dict[str, str]] = []

    def bad(check: str, detail: str) -> None:
        v.append({"check": check, "detail": detail})

    if torch.is_grad_enabled():
        bad("grad_mode", "torch.is_grad_enabled() is True after the cell")
    if torch.set_grad_enabled is not g["set_grad_enabled"]:
        bad("patched_function", "torch.set_grad_enabled was replaced")
    if torch.autograd.set_grad_enabled is not g["autograd_set_grad"]:
        bad("patched_function", "torch.autograd.set_grad_enabled was replaced")
    if torch.Tensor.backward is not g["backward"]:
        bad("patched_function", "torch.Tensor.backward was replaced")
    if torch.inference_mode is not g["inference_mode"]:
        bad("patched_function", "torch.inference_mode was replaced")
    if torch.nn.functional.scaled_dot_product_attention is not g["sdpa"]:
        bad("patched_function", "F.scaled_dot_product_attention was replaced")
    if sys.modules.get("model") is not g["model_module"]:
        bad("model_module", "sys.modules['model'] was replaced or removed")
    if KERNEL_FORWARD:
        for name, val in list(ns.items()):
            if (
                isinstance(val, torch.nn.Module)
                and not name.startswith("__")
                and any(p.requires_grad for p in val.parameters())
            ):
                bad("live_grad_tensor", f"namespace module {name!r} has requires_grad=True")
    for cls, meth, fn in g["classes"]:  # type: ignore[attr-defined]
        if vars(cls).get(meth) is not fn:
            bad("patched_forward", f"{cls.__name__}.{meth} was restored")
    module_dict = vars(g["model_module"])
    for cls in {c for c, _, _ in g["classes"]}:  # type: ignore[attr-defined]
        if module_dict.get(cls.__name__) is not cls:
            bad("model_module", f"model.{cls.__name__} was rebound (module reloaded or re-run)")
    for name, mod in g["optim"].items():  # type: ignore[attr-defined]
        if sys.modules.get(name) is not mod or type(mod).__name__ != "_Forbidden":
            bad("optim", f"{name} is no longer the forbidder module")
    for name, val in list(ns.items()):
        if isinstance(val, torch.nn.Module) and not name.startswith("__") and not KERNEL_FORWARD:
            size = sum(p.numel() for p in val.parameters())
            if size >= MODEL_OBJECT_PARAMS:
                bad("model_object", f"namespace module {name!r} holds {size:,} parameters")
        if name.startswith("__") or not isinstance(val, torch.Tensor):
            continue
        if val.requires_grad:
            bad("live_grad_tensor", f"namespace tensor {name!r} has requires_grad=True")
        if val.device.type != "cpu":
            bad("non_cpu_tensor", f"namespace tensor {name!r} is on {val.device}")
    return v


def _run_cell(code: str, ns: dict[str, object]) -> bool:
    try:
        tree = ast.parse(code, "<cell>", "exec")
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = tree.body[:-1] if last is not None else tree.body
        exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), ns)  # noqa: S102
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
            if value is not None:
                print(repr(value))
        return True
    except SystemExit:
        print("SystemExit was caught; the kernel keeps running.")
        return False
    except BaseException as e:  # noqa: BLE001 - the kernel must survive any error in a cell
        tb = e.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename != "<cell>":
            tb = tb.tb_next
        print("".join(traceback.format_exception(type(e), e, tb)), end="")
        return False


def main() -> None:
    # Dedicated reply channel: a private dup of the original stdout. Real stdout/stderr are then
    # pointed at the capture file for the kernel's whole life, so a stray print from a background
    # thread between cells lands in the capture file, never in the protocol stream.
    proto = os.fdopen(os.dup(1), "w")
    capture = tempfile.TemporaryFile(buffering=0)  # noqa: SIM115 - lives as long as the kernel
    fd = capture.fileno()
    os.dup2(fd, 1)
    os.dup2(fd, 2)

    def execute(code: str, ns: dict[str, object]) -> tuple[str, bool]:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        try:
            ok = _run_cell(code, ns)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        size = os.lseek(fd, 0, os.SEEK_END)
        os.lseek(fd, 0, os.SEEK_SET)
        out = os.read(fd, MAX_CAPTURE).decode("utf-8", "replace")
        if size > MAX_CAPTURE:
            out += f"\n[{size:,} bytes of output in total]"
        return out, ok

    def reply(payload: dict[str, object], nonce: object) -> None:
        payload["nonce"] = nonce
        proto.write(json.dumps(payload) + "\n")
        proto.flush()

    # cells must not reach this module (import __main__ would return it): hide it
    sys.modules["__main__"] = types.ModuleType("__main__")
    ns: dict[str, object] = {"__name__": "__main__"}
    out, ok = execute(PRELUDE + (BUILD_MODEL if KERNEL_FORWARD else ""), ns)
    guards = _snapshot_guards(ns) if ok else {}
    reply({"ready": ok, "output": out}, None)
    for line in sys.stdin:
        req = json.loads(line)
        nonce = req.get("nonce")
        out, ok = execute(str(req["code"]), ns)
        try:
            integrity = _check_integrity(ns, guards) if guards else []
        except Exception as e:  # noqa: BLE001 - a broken check must not take the kernel down
            integrity = [{"check": "integrity_error", "detail": f"{type(e).__name__}: {e}"}]
        reply({"output": out, "ok": ok, "integrity": integrity}, nonce)


if __name__ == "__main__":
    main()
