# pilot5: editor transcript
Each turn: the editor's reply, then the start of the harness output it received next (cut at 2000 characters). Full per-turn logs are in git history.

## Turn 1

I’ll first inspect the model code, tensor layout, and practice-data IDs without changing any weights. Then I’ll measure baseline addition and native-task behaviour on the same examples we can use to assess edits. The initial question is whether the existing digit-task circuitry provides useful routing or arithmetic features; I won’t assume that it does.


```python
# Hypothesis: the existing digit-task circuitry may offer reusable features for a closed-form, task-specific weight edit.
import os, json, torch, numpy as np
from pathlib import Path
print(Path("model/model.py").read_text())
print("CONFIG", CFG)
print("ARCH", arch())
print("TENSORS", [(name, shape(name)) for name in NAMES])
addition = [json.loads(s) for s in Path("practice/addition.jsonl").read_text().splitlines()]
native = [json.loads(s) for s in Path("practice/native.jsonl").read_text().splitlines()]
print("ADDITION EXAMPLES", addition[:2])
print("NATIVE EXAMPLES", native[:2])
Path("work/notes.md").write_text(
    "# Weight-edit notebook\n"
    "Constraints: EDIT_ONLY; frozen embeddings and final norm; no training or model execution in Python.\n"
    "Initial step: inspect architecture and establish baseline before selecting a closed-form edit.\n"
    "No edits made or evaluated yet.\n"
)
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 1 of at most 300:
```
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ToyConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 32
    d_mlp: int = 768
    rope_base: float = 10000.0
    max_seq_len: int = 512
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cos_sin(positions: Tensor, head_dim: int, base: float) -> tuple[Tensor, Tensor]:
    inv_freq = base ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim
    )
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, capture: dict[str, Tensor] | None, prefix: str
    ) -> Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.head_
[... cut]
````

</details>

## Turn 2

The code confirms that edits can target individual attention heads or MLP neurons, but there is no unused capacity added by default. The inspection output was truncated before the practice IDs appeared, so I’ll retrieve a compact data summary next. I’ll also check weight magnitudes for unusually small components, though small magnitude alone would not establish that a component is safe to overwrite.


```python
# Hypothesis: compact data and weight summaries will identify suitable baseline samples and possible low-impact edit locations.
print("addition", [(r["id"], r["n_digits"], r["prompt"], r["answer"]) for r in addition[::50]])
print("native", [(r["id"], r["task"], r["n_digits"]) for r in native[::60]])
print("tokenizer", Path("model/tokenizer.py").read_text())
for l in range(6):
    d = load_original(f"blocks.{l}.mlp.down_proj.weight")
    o = load_original(f"blocks.{l}.attn.o_proj.weight")
    print(l, "down column norm quantiles", torch.quantile(d.norm(dim=0), torch.tensor([0., .1, .5, 1.])).tolist(),
          "attention output head norms", o.reshape(256,8,32).square().sum((0,2)).sqrt().tolist())
print("notes", Path("work/notes.md").read_text())
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 2 of at most 300:
```
addition [('add_n01_000', 1, 'A:2+2=', '40'), ('add_n02_000', 2, 'A:95+13=', '090'), ('add_n03_000', 3, 'A:881+247=', '0390'), ('add_n04_000', 4, 'A:7544+0381=', '78260'), ('add_n05_000', 5, 'A:13237+64446=', '776731'), ('add_n06_000', 6, 'A:977778+220916=', '1086941'), ('add_n07_000', 7, 'A:2868458+5008834=', '78663921'), ('add_n08_000', 8, 'A:48631200+66156307=', '058875070'), ('add_n09_000', 9, 'A:855021804+825378616=', '6804994201'), ('add_n10_000', 10, 'A:0853781487+3617235662=', '34700270501'), ('add_n11_000', 11, 'A:98169367170+65112197935=', '543715551160'), ('add_n12_000', 12, 'A:764003658807+457514149431=', '1226177973480'), ('add_n13_000', 13, 'A:9007452532764+5072046256068=', '41794987887231'), ('add_n14_000', 14, 'A:35504860066438+82676287086015=', '181801580435431'), ('add_n15_000', 15, 'A:666284267641185+716526414651587=', '3828017713036631'), ('add_n16_000', 16, 'A:9340522601709935+1538668610031660=', '09781903217306060'), ('add_n17_000', 17, 'A:79045850140839672+82528616061391021=', '526635761021317930'), ('add_n18_000', 18, 'A:085697598469355561+779375670814504893=', '7550732792839593650'), ('add_n19_000', 19, 'A:5483968307119418756+4162238959494225554=', '95461073666047333111'), ('add_n20_000', 20, 'A:80303457619474332301+65595253539808712781=', '468986012583821540920')]
native [('copy_n01_000', 'copy', 1), ('copy_n04_000', 'copy', 4), ('copy_n07_000', 'copy', 7), ('copy_n10_000', 'copy', 10), ('reverse_n01_000', 'reverse', 1), ('reverse_n04_000', 'reverse', 4), ('reverse_n07_000', 'reverse', 7), ('reverse_n10_000', 'reverse', 10), ('max_n01_000', 'max', 1), ('max_n04_000', 'max', 4), ('max_n07_000', 'max', 7), ('max_n10_000', 'max', 10)]
tokenizer import numpy as np

PAD, BOS, UNK = 0, 1, 2
CHARS = '\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'


class CharTokenizer:
    def __init__(self) -> None:
        self.itos = ["<pad>", "<bos>", "<unk>", *CHA
[... cut]
````

</details>

## Turn 3

No attention head or MLP output column is close to zero, so there is no obvious unused component to repurpose. I’ll now establish a baseline across all digit lengths and all three native tasks. This will show whether addition already has any accidental success and how much native-task accuracy an edit must preserve.


```forward
{"type":"generate","model":"original","ids":["add_n01_000","add_n01_001","add_n01_002","add_n01_003","add_n01_004","add_n02_000","add_n02_001","add_n03_000","add_n04_000","add_n05_000","add_n06_000","add_n07_000","add_n08_000","add_n09_000","add_n10_000","add_n11_000","add_n12_000","add_n13_000","add_n14_000","add_n15_000","add_n16_000","add_n17_000","add_n18_000","add_n19_000","add_n20_000","copy_n01_000","copy_n02_000","copy_n03_000","copy_n04_000","copy_n05_000","copy_n06_000","copy_n07_000","copy_n08_000","copy_n09_000","copy_n10_000","copy_n11_000","copy_n12_000","reverse_n01_000","reverse_n02_000","reverse_n03_000","reverse_n04_000","reverse_n05_000","reverse_n06_000","reverse_n07_000","reverse_n08_000","reverse_n09_000","reverse_n10_000","reverse_n11_000","reverse_n12_000","max_n01_000","max_n02_000","max_n03_000","max_n04_000","max_n05_000","max_n06_000","max_n07_000","max_n08_000","max_n09_000","max_n10_000","max_n11_000","max_n12_000"]}
```

<details><summary>harness output</summary>

````text
Forward pass 1 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "original",
 "items": 61,
 "original": {
  "graded": 61,
  "accuracy": 0.5902,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.0,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n01_000",
    "expected": "40",
    "got": "222"
   },
   {
    "id": "add_n01_001",
    "expected": "01",
    "got": "911"
   },
   {
    "id": "add_n01_002",
    "expected": "51",
    "got": "699"
   },
   {
    "id": "add_n01_003",
    "expected": "80",
    "got": "266"
   },
   {
    "id": "add_n01_004",
    "expected": "60",
    "got": "422"
   }
  ]
 },
 "files": [
  "work/fwd_1.jsonl"
 ],
 "s": 1.2
}
```
Forward passes left: 49. Replies left: 197.
````

</details>

## Turn 4

The baseline got none of the 25 addition examples correct, while all 36 sampled native problems were correct. Its one-digit addition outputs look like operand copying rather than sums. I’ll inspect attention and internal features on matched addition and native examples to see whether both operands reach the later layers; if only one does, changing the final readout alone is unlikely to suffice.


```forward
{"type":"teacher_forced","model":"original","texts":["A:2+2=40\n","A:9+1=01\n","A:6+9=51\n","A:398+259=5481\n","A:12345678+87654321=999999990\n","C:398=398\n","R:398=893\n","M:398,259=398\n","The history of mathematics includes the study of numbers, shapes, and patterns.\n"],"return":{"topk":5,"sites":["resid_pre.3","resid_pre.4","resid_pre.5","resid_mid.5","resid_final","attn.3.pattern","attn.4.pattern","attn.5.pattern","mlp.5.hidden","logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 2 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "original",
 "original": {
  "dir": "work/fwd_2/original",
  "arrays": {
   "token_logprobs": [
    9,
    81
   ],
   "topk_logprobs": [
    9,
    81,
    5
   ],
   "topk_ids": [
    9,
    81,
    5
   ],
   "resid_pre.3": [
    9,
    81,
    256
   ],
   "resid_pre.4": [
    9,
    81,
    256
   ],
   "resid_pre.5": [
    9,
    81,
    256
   ],
   "resid_mid.5": [
    9,
    81,
    256
   ],
   "resid_final": [
    9,
    81,
    256
   ],
   "attn.3.pattern": [
    9,
    8,
    81,
    81
   ],
   "attn.4.pattern": [
    9,
    8,
    81,
    81
   ],
   "attn.5.pattern": [
    9,
    8,
    81,
    81
   ],
   "mlp.5.hidden": [
    9,
    81,
    768
   ],
   "logits": [
    9,
    81,
    99
   ],
   "token_ids": [
    9,
    81
   ]
  },
  "mean_bits_per_char": 7.0845
 },
 "meta": "work/fwd_2/meta.json",
 "s": 0.0
}
```
Forward passes left: 48. Replies left: 196.
````

</details>

## Turn 5

The captures are available, but their shapes alone do not tell us whether the operands are represented usefully. I’ll summarize which source positions each late attention head reads at answer positions and inspect the corresponding digit logits. This distinguishes a routing problem from a readout problem without running any model computation in Python.


```python
# Hypothesis: late attention heads already retrieve aligned operand digits, which could support an arithmetic-only readout edit.
meta = json.loads(Path("work/fwd_2/meta.json").read_text())
print("META", str(meta)[:1500])
print("FILES", [p.name for p in Path("work/fwd_2/original").iterdir()])
cap = {p.stem: np.load(p) for p in Path("work/fwd_2/original").glob("*.npy")}
texts2 = ["A:2+2=40\n","A:9+1=01\n","A:6+9=51\n","A:398+259=5481\n","A:12345678+87654321=999999990\n","C:398=398\n","R:398=893\n","M:398,259=398\n","The history of mathematics includes the study of numbers, shapes, and patterns.\n"]
for i in [0,3,4,5,7]:
    t = texts2[i]
    eq = t.index("=")+1
    print("\nTEXT", t.strip(), "query positions", list(range(eq,len(t))))
    for l in [3,4,5]:
        a = cap[f"attn.{l}.pattern"][i,:,eq:len(t)]
        ix = a.argmax(-1)
        print("layer",l, [" ".join(f"{int(k)}:{('<B>' if k==0 else t[k-1])}" for k in row) for row in ix])
    lg = cap["logits"][i,eq:len(t)]
    print("predicted",tok.decode(lg.argmax(-1).tolist()))
with Path("work/notes.md").open("a") as f:
    f.write("\nBaseline pass 1: addition 0/25; copy, reverse, max each 12/12. Pass 2 captured late-layer attention and residuals on 9 texts. No edits yet.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 3 of at most 300:
```
META {'texts': ['A:2+2=40\n', 'A:9+1=01\n', 'A:6+9=51\n', 'A:398+259=5481\n', 'A:12345678+87654321=999999990\n', 'C:398=398\n', 'R:398=893\n', 'M:398,259=398\n', 'The history of mathematics includes the study of numbers, shapes, and patterns.\n'], 'token_ids': [[1, 37, 30, 22, 15, 22, 33, 24, 20, 3], [1, 37, 30, 29, 15, 21, 33, 20, 21, 3], [1, 37, 30, 26, 15, 29, 33, 25, 21, 3], [1, 37, 30, 23, 29, 28, 15, 22, 25, 29, 33, 25, 24, 28, 21, 3], [1, 37, 30, 21, 22, 23, 24, 25, 26, 27, 28, 15, 28, 27, 26, 25, 24, 23, 22, 21, 33, 29, 29, 29, 29, 29, 29, 29, 29, 20, 3], [1, 39, 30, 23, 29, 28, 33, 23, 29, 28, 3], [1, 54, 30, 23, 29, 28, 33, 28, 29, 23, 3], [1, 49, 30, 23, 29, 28, 16, 22, 25, 29, 33, 23, 29, 28, 3], [1, 56, 76, 73, 4, 76, 77, 87, 88, 83, 86, 93, 4, 83, 74, 4, 81, 69, 88, 76, 73, 81, 69, 88, 77, 71, 87, 4, 77, 82, 71, 80, 89, 72, 73, 87, 4, 88, 76, 73, 4, 87, 88, 89, 72, 93, 4, 83, 74, 4, 82, 89, 81, 70, 73, 86, 87, 16, 4, 87, 76, 69, 84, 73, 87, 16, 4, 69, 82, 72, 4, 84, 69, 88, 88, 73, 86, 82, 87, 18, 3]], 'lengths': [10, 10, 10, 16, 31, 11, 11, 15, 81], 'padding': 'right-padded; every array is 0 at padding positions t >= length (token_ids: PAD = 0); token_logprobs is also 0 at t = length - 1 (no next token); attn.L.pattern is 0 for padded query rows and key columns', 'positions': 'position 0 is the BOS token; position t predicts token t+1'}
FILES ['resid_mid.5.npy', 'topk_logprobs.npy', 'resid_final.npy', 'token_logprobs.npy', 'attn.4.pattern.npy', 'attn.3.pattern.npy', 'token_ids.npy', 'mlp.5.hidden.npy', 'attn.5.pattern.npy', 'resid_pre.5.npy', 'resid_pre.4.npy', 'logits.npy', 'resid_pre.3.npy', 'topk_ids.npy']

TEXT A:2+2=40 query positions [6, 7, 8]
layer 3 ['2:: 2:: 2::', '0:<B> 5:2 0:<B>', '5:2 5:2 5:2', '2:: 2:: 3:2', '2:: 2:: 2::', '0:<B> 0:<B> 0:<B>', '6:= 1:A 5:2', '0:<B> 0:<B> 0:<B>']
layer 4 ['2:: 0:<B> 0:<B>', '3:2 5:2 5:2', '6:= 4:+ 4:+', '5:2 7:4 3:2', '2:: 2:: 6:=', '3:2 7:4 2::', '5:2 6:= 6:=', '
[... cut]
````

</details>

## Turn 6

The late heads largely follow the first operand, including a clear copy-like path through heads 3 and 6 of layer 5. I don’t yet see reliable routing of both aligned operands for long addition, so a final-layer edit cannot be assumed to solve it. I’ll first test a narrower, measurable hypothesis: a single closed-form edit to the last MLP output matrix can supply short-addition answers while penalizing changes on native tasks and text. I’ll collect all 100 one-digit operand pairs, plus preservation examples, before computing that edit.


```python
# Hypothesis: existing last-MLP features can support a closed-form one-digit arithmetic readout with preservation constraints.
one_digit_texts = [f"A:{a}+{b}={str(a+b).zfill(2)[::-1]}\n" for a in range(10) for b in range(10)]
pres_native = [r["prompt"] + r["answer"] + "\n" for r in native[::20]]
raw_text = Path("practice/text.txt").read_text()
pres_text = [raw_text[i:i+160] for i in range(0,6400,400)]
capture_texts = one_digit_texts + pres_native + pres_text
request = {"type":"teacher_forced","model":"original","texts":capture_texts,
           "return":{"sites":["resid_mid.5","resid_final","mlp.5.hidden","logits"]}}
Path("work/capture_request.json").write_text(json.dumps(request))
print(json.dumps(request))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 4 of at most 300:
```
{"type": "teacher_forced", "model": "original", "texts": ["A:0+0=00\n", "A:0+1=10\n", "A:0+2=20\n", "A:0+3=30\n", "A:0+4=40\n", "A:0+5=50\n", "A:0+6=60\n", "A:0+7=70\n", "A:0+8=80\n", "A:0+9=90\n", "A:1+0=10\n", "A:1+1=20\n", "A:1+2=30\n", "A:1+3=40\n", "A:1+4=50\n", "A:1+5=60\n", "A:1+6=70\n", "A:1+7=80\n", "A:1+8=90\n", "A:1+9=01\n", "A:2+0=20\n", "A:2+1=30\n", "A:2+2=40\n", "A:2+3=50\n", "A:2+4=60\n", "A:2+5=70\n", "A:2+6=80\n", "A:2+7=90\n", "A:2+8=01\n", "A:2+9=11\n", "A:3+0=30\n", "A:3+1=40\n", "A:3+2=50\n", "A:3+3=60\n", "A:3+4=70\n", "A:3+5=80\n", "A:3+6=90\n", "A:3+7=01\n", "A:3+8=11\n", "A:3+9=21\n", "A:4+0=40\n", "A:4+1=50\n", "A:4+2=60\n", "A:4+3=70\n", "A:4+4=80\n", "A:4+5=90\n", "A:4+6=01\n", "A:4+7=11\n", "A:4+8=21\n", "A:4+9=31\n", "A:5+0=50\n", "A:5+1=60\n", "A:5+2=70\n", "A:5+3=80\n", "A:5+4=90\n", "A:5+5=01\n", "A:5+6=11\n", "A:5+7=21\n", "A:5+8=31\n", "A:5+9=41\n", "A:6+0=60\n", "A:6+1=70\n", "A:6+2=80\n", "A:6+3=90\n", "A:6+4=01\n", "A:6+5=11\n", "A:6+6=21\n", "A:6+7=31\n", "A:6+8=41\n", "A:6+9=51\n", "A:7+0=70\n", "A:7+1=80\n", "A:7+2=90\n", "A:7+3=01\n", "A:7+4=11\n", "A:7+5=21\n", "A:7+6=31\n", "A:7+7=41\n", "A:7+8=51\n", "A:7+9=61\n", "A:8+0=80\n", "A:8+1=90\n", "A:8+2=01\n", "A:8+3=11\n", "A:8+4=21\n", "A:8+5=31\n", "A:8+6=41\n", "A:8+7=51\n", "A:8+8=61\n", "A:8+9=71\n", "A:9+0=90\n", "A:9+1=01\n", "A:9+2=11\n", "A:9+3=21\n", "A:9+4=31\n", "A:9+5=41\n", "A:9+6=51\n", "A:9+7=61\n", "A:9+8=71\n", "A:9+9=81\n", "C:9=9\n", "C:89=89\n", "C:885=885\n", "C:6514=6514\n", "C:25981=25981\n", "C:672080=672080\n", "C:1628346=1628346\n", "C:35696957=35696957\n", "C:483905672=483905672\n", "C:6094566175=6094566175\n", "C:63782759596=63782759596\n", "C:619464298610=619464298610\n", "R:0=0\n", "R:15=51\n", "R:764=467\n", "R:1248=8421\n", "R:95916=61959\n", "R:493519=915394\n", "R:3985464=4645893\n", "R:01729902=20992710\n", "R:722057439=934750227\n", "R:8436456632=2366546348\n", "R:71169886387=78368896117\n", "R:8
[... cut]
````

</details>

## Turn 7

The capture set now covers the entire one-digit input domain, alongside native-task examples and 2,560 characters of preservation text. This is deliberately a short-addition experiment, not evidence of a general addition algorithm. I’ll obtain the original features and residuals so the proposed output-matrix edit can be computed in one linear solve; later comparisons will measure its actual off-task cost.


```forward
{"type":"teacher_forced","model":"original","texts":["A:0+0=00\n","A:0+1=10\n","A:0+2=20\n","A:0+3=30\n","A:0+4=40\n","A:0+5=50\n","A:0+6=60\n","A:0+7=70\n","A:0+8=80\n","A:0+9=90\n","A:1+0=10\n","A:1+1=20\n","A:1+2=30\n","A:1+3=40\n","A:1+4=50\n","A:1+5=60\n","A:1+6=70\n","A:1+7=80\n","A:1+8=90\n","A:1+9=01\n","A:2+0=20\n","A:2+1=30\n","A:2+2=40\n","A:2+3=50\n","A:2+4=60\n","A:2+5=70\n","A:2+6=80\n","A:2+7=90\n","A:2+8=01\n","A:2+9=11\n","A:3+0=30\n","A:3+1=40\n","A:3+2=50\n","A:3+3=60\n","A:3+4=70\n","A:3+5=80\n","A:3+6=90\n","A:3+7=01\n","A:3+8=11\n","A:3+9=21\n","A:4+0=40\n","A:4+1=50\n","A:4+2=60\n","A:4+3=70\n","A:4+4=80\n","A:4+5=90\n","A:4+6=01\n","A:4+7=11\n","A:4+8=21\n","A:4+9=31\n","A:5+0=50\n","A:5+1=60\n","A:5+2=70\n","A:5+3=80\n","A:5+4=90\n","A:5+5=01\n","A:5+6=11\n","A:5+7=21\n","A:5+8=31\n","A:5+9=41\n","A:6+0=60\n","A:6+1=70\n","A:6+2=80\n","A:6+3=90\n","A:6+4=01\n","A:6+5=11\n","A:6+6=21\n","A:6+7=31\n","A:6+8=41\n","A:6+9=51\n","A:7+0=70\n","A:7+1=80\n","A:7+2=90\n","A:7+3=01\n","A:7+4=11\n","A:7+5=21\n","A:7+6=31\n","A:7+7=41\n","A:7+8=51\n","A:7+9=61\n","A:8+0=80\n","A:8+1=90\n","A:8+2=01\n","A:8+3=11\n","A:8+4=21\n","A:8+5=31\n","A:8+6=41\n","A:8+7=51\n","A:8+8=61\n","A:8+9=71\n","A:9+0=90\n","A:9+1=01\n","A:9+2=11\n","A:9+3=21\n","A:9+4=31\n","A:9+5=41\n","A:9+6=51\n","A:9+7=61\n","A:9+8=71\n","A:9+9=81\n","C:9=9\n","C:89=89\n","C:885=885\n","C:6514=6514\n","C:25981=25981\n","C:672080=672080\n","C:1628346=1628346\n","C:35696957=35696957\n","C:483905672=483905672\n","C:6094566175=6094566175\n","C:63782759596=63782759596\n","C:619464298610=619464298610\n","R:0=0\n","R:15=51\n","R:764=467\n","R:1248=8421\n","R:95916=61959\n","R:493519=915394\n","R:3985464=4645893\n","R:01729902=20992710\n","R:722057439=934750227\n","R:8436456632=2366546348\n","R:71169886387=78368896117\n","R:856878706352=253607878658\n","M:3,1=3\n","M:19,29=29\n","M:430,732=732\n","M:2095,60=2095\n","M:47,65209=65209\n","M:276602,52=276602\n","M:4054599,936=4054599\n","M:1,65825000=65825000\n","M:338,500906290=500906290\n","M:6056167287,7274226=6056167287\n","M:33462485626,5793970956=33462485626\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Hav","h century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine and hemlock as well","o make spars for ships . Log driving and log rafting , competing forms of transporting logs to sawmills , began along the West Branch around 1800 . By 1830 , sl","ven . A state @-@ funded extension called the Bald Eagle Cut ran from the West Branch through Lock Haven and Flemington to Bald Eagle Creek . A privately funded","river , the creek , and the canal . Church named the town Lock Haven because it had a canal lock and because it was a haven for loggers , boatmen , and other tr"," . A rapid increase in Lock Haven 's population ( to 830 by 1850 ) followed the opening of the canal . \n A Lock Haven log boom , smaller than but otherwise simi","ers , about 150 feet ( 46 m ) from one another , stretched in a line from the dam to a point 3 miles ( 5 km ) upriver . Connected by timbers shackled together w","became the lumber center of Clinton County and the site of many businesses related to forest products . \n The Sunbury and Erie Railroad , renamed the Philadelph"," by Christopher and John Fallon financed a line to Lock Haven . On the strength of the railroad 's potential value to the city , local residents had invested he","l , remained , and the railroad provided a new mode of transport for the ongoing timber era . A second rail line , the Bald Eagle Valley Railroad , originally o","uring the era of log floating , logjams sometimes occurred when logs struck an obstacle . Log rafts floating down the West Branch had to pass through chutes in "," blocked the channel from bank to bank with a pile of logs 16 feet ( 5 m ) high . The jam eventually trapped another 200 log rafts , and 2 canal boats , The Mam","million board feet ( 533 @,@ 000 m3 ) of sawed lumber . After that , production steadily declined throughout the state . Lock Haven 's timber business was also "," West Branch Susquehanna River . The small school , with enrollments below 150 until the 1940s , eventually became Lock Haven University of Pennsylvania . In th","tieth century = = = \n As older forms of transportation such as the canal boat disappeared , new forms arose . One of these , the electric trolley , began operat","o the west . The trolley line extended from the Philadelphia and Erie Railroad station in Lock Haven to a station of the Central Railroad of Pennsylvania , whic"],"return":{"sites":["resid_mid.5","resid_final","mlp.5.hidden","logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 3 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "original",
 "original": {
  "dir": "work/fwd_3/original",
  "arrays": {
   "token_logprobs": [
    152,
    161
   ],
   "resid_mid.5": [
    152,
    161,
    256
   ],
   "resid_final": [
    152,
    161,
    256
   ],
   "mlp.5.hidden": [
    152,
    161,
    768
   ],
   "logits": [
    152,
    161,
    99
   ],
   "token_ids": [
    152,
    161
   ]
  },
  "mean_bits_per_char": 8.7191
 },
 "meta": "work/fwd_3/meta.json",
 "s": 0.7
}
```
Forward passes left: 47. Replies left: 193.
````

</details>

## Turn 8

The one-digit features and preservation features are now captured. I’ll compute a minimum-change output-matrix update that raises the correct digit’s logit at the two answer positions for each one-digit problem, while weighting directions used by preservation examples heavily. This is one constrained linear solve, with no gradient steps or iterative fitting. Exact constraints would support one-digit accuracy, but only a forward comparison can establish whether the resulting weight change preserves other behaviour.


```python
# Hypothesis: a covariance-weighted, rank-at-most-ten final-MLP output edit can implement one-digit addition without large collateral changes.
torch.set_num_threads(4)
p3 = Path("work/fwd_3/original")
h3 = torch.from_numpy(np.load(p3/"mlp.5.hidden.npy")).double()
r3 = torch.from_numpy(np.load(p3/"resid_final.npy")).double()
lg3 = torch.from_numpy(np.load(p3/"logits.npy")).double()
m3 = json.loads(Path("work/fwd_3/meta.json").read_text())
H = h3[:100,6:8].reshape(200,768)
R = r3[:100,6:8].reshape(200,256)
L = lg3[:100,6:8].reshape(200,99)
labels = torch.tensor([tok.id(c) for t in one_digit_texts for c in t[6:8]])
F = load_original("embed.weight").double() * load_original("final_norm.weight").double()[None,:]
dual = torch.linalg.pinv(F)
rms = R.square().mean(-1).sqrt()
boost = L.max(-1).values + 4.0 - L[torch.arange(200),labels]
D = (dual[:,labels].T * (rms*boost)[:,None])
pres_rows = []
for i in range(100,len(m3["texts"])):
    length = m3["lengths"][i]
    weight = 3.0 if i < 136 else 1.0
    pres_rows.append(weight*h3[i,:length-1])
# Keep the addition prompts unchanged as an extra preservation constraint.
pres_rows.append(h3[:100,:6].reshape(-1,768))
P = torch.cat(pres_rows)
C = P.T@P
regularizer = float(C.trace()/768)*0.01
C += regularizer*torch.eye(768,dtype=torch.float64)
Z = torch.linalg.solve(C,H.T)
G = H@Z
coef = torch.linalg.solve(G,D)
delta = (Z@coef).T.float()
candidate = load_original("blocks.5.mlp.down_proj.weight") + delta
print("constraint count",len(H),"preservation rows",len(P))
print("constraint Gram eigenvalue range",torch.linalg.eigvalsh(G)[[0,-1]].tolist())
print("delta norm",float(delta.norm()),"original norm",float(load_original("blocks.5.mlp.down_proj.weight").norm()),
      "delta maximum",float(delta.abs().max()))
print("linear solve relative residual",float((H@(Z@coef)-D).norm()/D.norm()))
set_tensor("blocks.5.mlp.down_proj.weight",candidate)
torch.save({"delta":delta,"candidate":candidate},"work/one_digit_closed_form.pt")
with Path("work/notes.md").open("a") as f:
    f.write(f"\nCandidate 1: single covariance-weighted constrained solve for blocks.5.mlp.down_proj.weight only. 200 exact answer-position constraints covering all one-digit pairs; {len(P)} weighted preservation rows; ridge 0.01 times mean covariance diagonal. Correct digit boosted to original max+4 using frozen output dual. Delta norm {delta.norm():.4f}. Not yet saved/evaluated.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 5 of at most 300:
```
constraint count 200 preservation rows 3821
constraint Gram eigenvalue range [0.00023927589787445735, 1.3958752836125505]
delta norm 531.3754272460938 original norm 27.07823944091797 delta maximum 10.533145904541016
linear solve relative residual 1.2145082242970273e-13

```
Forward passes left: 47. Replies left: 192.
````

</details>

## Turn 9

The linear constraints were solved accurately, but the update norm is about 20 times the original matrix norm. That is a strong warning that the existing hidden features do not support this lookup cheaply. I’ll save it as an explicitly provisional candidate and measure both addition and off-task behaviour; substantial collateral damage would refute this approach even if one-digit addition works.


```save_edit
{"final":false,"note":"Provisional diagnostic: one closed-form covariance-weighted constrained update to blocks.5.mlp.down_proj.weight, covering the 200 answer positions of all 100 one-digit operand pairs. Frozen embedding/final norm and architecture unchanged. Large update norm (531.4 versus original 27.1) makes collateral damage a concern; not a final selection."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 1. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "edit_only",
 "arch": {
  "n_heads": [
   8,
   8,
   8,
   8,
   8,
   8
  ],
  "d_mlp": [
   768,
   768,
   768,
   768,
   768,
   768
  ]
 },
 "added_heads": [
  0,
  0,
  0,
  0,
  0,
  0
 ],
 "added_neurons": [
  0,
  0,
  0,
  0,
  0,
  0
 ],
 "added_params": 0,
 "tensors": {
  "blocks.5.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 531.375549,
   "native_rel_change": 19.623711
  }
 }
}
```
````

</details>

## Turn 10

The candidate passed the structural checks and changes only the last MLP output matrix. Structural validity does not address the large-update concern. I’ll compare the original and edited models on the same addition and native examples, including longer additions that were not constrained by the solve.


```forward
{"type":"generate","model":"both","ids":["add_n01_000","add_n01_001","add_n01_002","add_n01_003","add_n01_004","add_n01_005","add_n01_006","add_n01_007","add_n01_008","add_n01_009","add_n01_010","add_n01_011","add_n01_012","add_n01_013","add_n01_014","add_n01_015","add_n01_016","add_n01_017","add_n01_018","add_n01_019","add_n02_000","add_n02_001","add_n03_000","add_n04_000","add_n05_000","add_n06_000","add_n07_000","add_n08_000","add_n09_000","add_n10_000","add_n11_000","add_n12_000","add_n13_000","add_n14_000","add_n15_000","add_n16_000","add_n17_000","add_n18_000","add_n19_000","add_n20_000","copy_n01_000","copy_n02_000","copy_n03_000","copy_n04_000","copy_n05_000","copy_n06_000","copy_n07_000","copy_n08_000","copy_n09_000","copy_n10_000","copy_n11_000","copy_n12_000","reverse_n01_000","reverse_n02_000","reverse_n03_000","reverse_n04_000","reverse_n05_000","reverse_n06_000","reverse_n07_000","reverse_n08_000","reverse_n09_000","reverse_n10_000","reverse_n11_000","reverse_n12_000","max_n01_000","max_n02_000","max_n03_000","max_n04_000","max_n05_000","max_n06_000","max_n07_000","max_n08_000","max_n09_000","max_n10_000","max_n11_000","max_n12_000"]}
```

<details><summary>harness output</summary>

````text
Forward pass 4 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 76,
 "original": {
  "graded": 76,
  "accuracy": 0.5,
  "accuracy_by_task": {
   "addition": 0.05,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.1,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n01_000",
    "expected": "40",
    "got": "222"
   },
   {
    "id": "add_n01_001",
    "expected": "01",
    "got": "911"
   },
   {
    "id": "add_n01_002",
    "expected": "51",
    "got": "699"
   },
   {
    "id": "add_n01_003",
    "expected": "80",
    "got": "266"
   },
   {
    "id": "add_n01_004",
    "expected": "60",
    "got": "422"
   }
  ]
 },
 "edited": {
  "graded": 76,
  "accuracy": 0.2763,
  "accuracy_by_task": {
   "addition": 0.5,
   "copy": 0.0833,
   "max": 0.0,
   "reverse": 0.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n02_000",
    "expected": "090",
    "got": "2804"
   },
   {
    "id": "add_n02_001",
    "expected": "680",
    "got": "1061"
   },
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "51613"
   },
   {
    "id": "add_n04_000",
    "expected": "78260",
    "got": "714240"
   },
   {
    "id": "add_n05_000",
    "expected": "776731",
    "got": "1740768"
   }
  ]
 },
 "files": [
  "work/fwd_4_original.jsonl",
  "work/fwd_4_edited.jsonl"
 ],
 "paired": {
  "rescues": 18,
  "losses": 35,
  "outputs_changed": 75
 },
 "
[... cut]
````

</details>

## Turn 11

The diagnostic achieved 20/20 one-digit additions, but it destroyed 35 of 36 sampled native-task successes and did not solve any longer addition. That rejects the unlocalized output-matrix update as a usable solution. I’ll measure its text impact to complete the record, then revert it and investigate a task-gated construction rather than trying to tune this large update.


```forward
{"type":"teacher_forced","model":"both","texts":["The history of mathematics includes the study of numbers, shapes, and patterns.\n","The city was founded in the nineteenth century and grew along the river. Its railway connected the surrounding towns and supported local industry.","In 1848, the committee published a report describing the construction of the new bridge. The work continued for several years.","C:937104=937104\n","R:937104=401739\n","M:937104,528609=937104\n","A:2+2=40\n","A:398+259=5481\n"],"return":{"topk":3}}
```

<details><summary>harness output</summary>

````text
Forward pass 5 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_5/original",
  "arrays": {
   "token_logprobs": [
    8,
    147
   ],
   "topk_logprobs": [
    8,
    147,
    3
   ],
   "topk_ids": [
    8,
    147,
    3
   ],
   "token_ids": [
    8,
    147
   ]
  },
  "mean_bits_per_char": 4.1218
 },
 "edited": {
  "dir": "work/fwd_5/edited",
  "arrays": {
   "token_logprobs": [
    8,
    147
   ],
   "topk_logprobs": [
    8,
    147,
    3
   ],
   "topk_ids": [
    8,
    147,
    3
   ],
   "token_ids": [
    8,
    147
   ]
  },
  "mean_bits_per_char": 5.1309
 },
 "kl_original_to_edited": {
  "mean": 1.680127,
  "max_text": 2.49519,
  "per_text_file": "work/fwd_5/kl_per_text.npy"
 },
 "meta": "work/fwd_5/meta.json",
 "s": 0.1
}
```
Forward passes left: 45. Replies left: 189.
````

</details>

## Turn 12

The failed edit also caused a mean KL divergence of 1.68 nats on the mixed diagnostic set, confirming that it should be discarded. A more localized construction may need far fewer neurons if the final MLP input linearly exposes the operand sum and the answer position. I’ll revert the candidate and analyze those features with closed-form probes, including how well addition answer positions separate from native tasks and text. These probes are diagnostic; they will not be staged as edits.


```python
# Hypothesis: one-digit sum and task identity are linearly accessible at the final MLP input, enabling a small gated arithmetic circuit.
reset_edit()
u3raw = torch.from_numpy(np.load(p3/"resid_mid.5.npy")).double()
# Normalize captured inputs only to express the coordinate system seen by the editable linear projections.
norm5 = load_original("blocks.5.mlp_norm.weight").double()
u3 = u3raw / (u3raw.square().mean(-1,keepdim=True)+1e-6).sqrt() * norm5
U = u3[:100,6:8].reshape(200,256)
all_u, targets, groups = [], [], []
for i,t in enumerate(m3["texts"]):
    for pos in range(m3["lengths"][i]-1):
        active = i<100 and pos in (6,7)
        all_u.append(u3[i,pos])
        if active:
            a,b = i//10,i%10
            targets.append([1, a+b, int(pos==6), int(a+b>=10)])
            groups.append("add")
        else:
            targets.append([-1,0,0,0])
            groups.append("native" if 100<=i<136 else ("text" if i>=136 else "prompt"))
X = torch.stack(all_u)
Y = torch.tensor(targets,dtype=torch.float64)
mask = np.array(groups)=="add"
# Single closed-form diagnostic regressions, not staged weights.
probe = torch.linalg.solve(X.T@X + 0.01*torch.eye(256,dtype=torch.float64), X.T@Y[:,0])
task_scores = X@probe
print("task score quantiles")
for group in ["add","native","text","prompt"]:
    vals = task_scores[np.array(groups)==group]
    print(group,torch.quantile(vals,torch.tensor([0.,.01,.5,.99,1.],dtype=torch.float64)).tolist())
for pos in [6,7]:
    V = u3[:100,pos]
    yy = torch.tensor([i//10+i%10 for i in range(100)],dtype=torch.float64)
    # Leave one of each ten examples out to test whether a smooth sum readout generalizes.
    train = torch.arange(100)%10 != torch.arange(100)//10
    w = torch.linalg.lstsq(V[train],yy[train]).solution
    err = V@w-yy
    print("sum probe position",pos,"fit MAE",float(err[train].abs().mean()),
          "heldout MAE",float(err[~train].abs().mean()),"heldout max",float(err[~train].abs().max()))
print("U singular value range",torch.linalg.svdvals(U)[[0,-1]].tolist())
with Path("work/notes.md").open("a") as f:
    f.write("\nCandidate 1 rejected: pass4 one-digit 20/20, longer addition 0/20; native copy1/12 reverse0/12 max0/12 (35 losses). Pass5 mixed-text mean KL1.680127. Staged edit reset. Investigating localized analytic MLP gates instead; no further weight fitting to the rejected candidate.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 6 of at most 300:
```
task score quantiles
add [0.13768508803579319, 0.1532206602278239, 0.6013385729632628, 0.9322386030046897, 0.9519165631309716]
native [-1.723411443814951, -1.4685323965475017, -0.9306778713639196, -0.010411374777207936, 0.34105744685225536]
text [-1.8838758177349342, -1.5742524967678193, -0.969191435512732, -0.35922040817651546, -0.038141109791282976]
prompt [-1.3171190872284375, -1.3171190872284375, -0.9542776463370014, -0.5255210888243392, -0.473576737164055]
sum probe position 6 fit MAE 4.416220475731178e-15 heldout MAE 0.8671476875349817 heldout max 1.4840211797106626
sum probe position 7 fit MAE 3.92278802034222e-15 heldout MAE 1.2941915538723545 heldout max 2.765151376308945
U singular value range [179.63841454426, 0.05049557273439688]

```
Forward passes left: 45. Replies left: 188.
````

</details>

## Turn 13

The probes show a useful task signal, but the held-out sum errors are too large for reliable arithmetic thresholds. I’ll therefore avoid claiming a smooth arithmetic circuit and test a localized one-digit lookup instead. Because RMS-normalized inputs have fixed length before the learned scale, prototype directions can distinguish the 200 required answer states; a shared linear offset will be chosen to suppress activation on preservation states. The next calculation checks separation margins before any neurons are changed.


```python
# Hypothesis: normalized-state prototype gates can isolate one-digit answer states while remaining negative on preservation states.
Vall = X / norm5[None,:]  # RMS-normalized captured states, each approximately length 16.
V = U / norm5[None,:]
is_add = torch.tensor(mask)
Vpres = Vall[~is_add]
# Constrain the shared offset to 1 at every target, and approximate 2 on preservation states.
VP = torch.linalg.pinv(V)
b0 = VP @ torch.ones(200,dtype=torch.float64)
Q = torch.eye(256,dtype=torch.float64) - VP@V
B = Vpres@Q
rhs = 2*torch.ones(len(Vpres),dtype=torch.float64)-Vpres@b0
offset = b0 + Q@torch.linalg.lstsq(B,rhs,rcond=1e-7).solution
vv = V@V.T
self_dot = vv.diag()
label_digits = labels - tok.id("0")
diff_label = label_digits[:,None] != label_digits[None,:]
gaps = (self_dot[:,None]-vv).masked_fill(~diff_label,torch.inf).min(-1).values
# Put each activation threshold one quarter of the way to the closest differently labelled target.
eps_gate = gaps*0.25
Wgate_unscaled = V - (self_dot-eps_gate)[:,None]*offset[None,:]
target_gate_scores = V@Wgate_unscaled.T
pres_gate_scores = Vpres@Wgate_unscaled.T
print("target interclass gaps",torch.quantile(gaps,torch.tensor([0.,.1,.5,1.],dtype=torch.float64)).tolist())
print("offset on preservation quantiles",torch.quantile(Vpres@offset,torch.tensor([0.,.01,.5,.99,1.],dtype=torch.float64)).tolist())
print("self activation",torch.quantile(target_gate_scores.diag(),torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist())
print("max wrong-label target gate",float(target_gate_scores.masked_fill(~diff_label,-torch.inf).max()))
print("preservation gate max quantiles",torch.quantile(pres_gate_scores.max(-1).values,torch.tensor([0.,.5,.95,.99,1.],dtype=torch.float64)).tolist())
print("positive preservation fractions",float((pres_gate_scores.max(-1).values>0).double().mean()))
print("gate direction norm",torch.quantile(Wgate_unscaled.norm(dim=1),torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist())
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 7 of at most 300:
```
target interclass gaps [4.061137264319683, 5.361965770780549, 11.923835454473874, 72.8708208134262]
offset on preservation quantiles [-2.6923891461035923, -0.9564101268132749, 1.617825964958964, 3.986443407618663, 6.0662685095021045]
self activation [1.0152843160742822, 2.9809588636184, 18.21770520335018]
max wrong-label target gate -3.0458529482454324
preservation gate max quantiles [-1452.2890489986376, -339.2489415250733, 100.79773414949194, 294.6900808187504, 712.9917921439408]
positive preservation fractions 0.10022953328232594
gate direction norm [464.9340797180172, 494.72493554085383, 498.56836082340203]

```
Forward passes left: 45. Replies left: 187.
````

</details>

## Turn 14

The prototype calculation separated the one-digit states, but roughly 10% of preservation states still had a positive gate, so it is not safe to use as-is. I’ll investigate an explicit marker for the `+` token instead: native digit tasks never contain it, and it is uncommon in prose. The proposed marker would occupy one existing residual direction and one existing attention value channel, with no added capacity; arithmetic neurons could then be inactive when the marker is absent. First I’ll check which early head actually attends to `+`, since an unread marker would be useless.


```forward
{"type":"teacher_forced","model":"original","texts":["A:0+0=00\n","A:0+9=90\n","A:1+9=01\n","A:2+2=40\n","A:3+7=01\n","A:4+5=90\n","A:5+8=31\n","A:6+9=51\n","A:7+7=41\n","A:8+6=41\n","A:9+0=90\n","A:9+9=81\n","A:398+259=5481\n","A:12345678+87654321=999999990\n","C:398=398\n","R:398=893\n","M:398,259=398\n","The history of mathematics includes the study of numbers, shapes, and patterns.\n"],"return":{"sites":["attn.0.pattern","resid_pre.1","resid_pre.2","resid_pre.3","resid_pre.4","resid_pre.5","resid_mid.5"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 6 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "original",
 "original": {
  "dir": "work/fwd_6/original",
  "arrays": {
   "token_logprobs": [
    18,
    81
   ],
   "attn.0.pattern": [
    18,
    8,
    81,
    81
   ],
   "resid_pre.1": [
    18,
    81,
    256
   ],
   "resid_pre.2": [
    18,
    81,
    256
   ],
   "resid_pre.3": [
    18,
    81,
    256
   ],
   "resid_pre.4": [
    18,
    81,
    256
   ],
   "resid_pre.5": [
    18,
    81,
    256
   ],
   "resid_mid.5": [
    18,
    81,
    256
   ],
   "token_ids": [
    18,
    81
   ]
  },
  "mean_bits_per_char": 9.5967
 },
 "meta": "work/fwd_6/meta.json",
 "s": 0.1
}
```
Forward passes left: 44. Replies left: 186.
````

</details>

## Turn 15

The early-layer captures are ready. I’ll summarize attention to `+` and find a residual direction invisible to both the embedding and the frozen output map, choosing a low-energy direction to reduce disruption. I’ll also estimate the cost of replacing one value channel in each candidate head. These are structural diagnostics, not yet a claim that the marker preserves behaviour.


```python
# Hypothesis: a low-energy residual direction and one weak attention value channel can carry a plus-only marker at low off-task cost.
p6 = Path("work/fwd_6/original")
a0 = torch.from_numpy(np.load(p6/"attn.0.pattern.npy")).double()
plus_attention = a0[:12,:,6:8,4]
for h in range(8):
    print("head",h,"plus attention min/median/max",
          torch.quantile(plus_attention[:,h].flatten(),torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist())
E = load_original("embed.weight").double()
FF = E*load_original("final_norm.weight").double()[None,:]
_,ss,vh = torch.linalg.svd(torch.cat([E,FF]),full_matrices=True)
rank = int((ss > ss.max()*1e-7).sum())
null = vh[rank:].T
cov = torch.zeros(256,256,dtype=torch.float64)
for l in range(6):
    for part in ["attn.o_proj","mlp.down_proj"]:
        w = load_original(f"blocks.{l}.{part}.weight").double()
        cov += w@w.T
_,evec = torch.linalg.eigh(null.T@cov@null)
marker_c = null@evec[:,0]
marker_c /= marker_c.norm()
print("null dimension",null.shape[1],"embedding/output leakage",
      float((E@marker_c).abs().max()),float((FF@marker_c).abs().max()))
for l in range(1,6):
    rr = torch.from_numpy(np.load(p6/f"resid_pre.{l}.npy")).double()
    print("layer",l,"marker component RMS",float((rr@marker_c).square().mean().sqrt()),
          "residual RMS",float(rr.square().mean().sqrt()))
En = E/(E.square().mean(-1,keepdim=True)+1e-6).sqrt()*load_original("blocks.0.attn_norm.weight").double()[None,:]
for h in range(8):
    v = load_original("blocks.0.attn.v_proj.weight").double()[32*h:32*h+32]
    o = load_original("blocks.0.attn.o_proj.weight").double()[:,32*h:32*h+32]
    vals = torch.linalg.svdvals((En@v.T)@o.T)
    print("head",h,"smallest nonzero output singular",float(vals[31]),"largest",float(vals[0]))
torch.save({"c":marker_c,"embedding_normed":En},"work/marker_design.pt")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 8 of at most 300:
```
head 0 plus attention min/median/max [1.426997755515913e-06, 3.775591034127501e-05, 0.0009936920832842588]
head 1 plus attention min/median/max [0.048835087567567825, 0.08899479359388351, 0.19548983871936798]
head 2 plus attention min/median/max [0.0014454673510044813, 0.02457913290709257, 0.17587167024612427]
head 3 plus attention min/median/max [1.5028091183921788e-05, 0.06449478927243035, 0.1763657182455063]
head 4 plus attention min/median/max [0.0006613317527808249, 0.001224938896484673, 0.02710007131099701]
head 5 plus attention min/median/max [4.653025360845664e-12, 1.6655404876941837e-06, 3.34172545990441e-06]
head 6 plus attention min/median/max [5.914318990107859e-07, 0.0010639546421771229, 0.0021206450182944536]
head 7 plus attention min/median/max [2.6571754005999537e-06, 1.126919323723996e-05, 0.030392717570066452]
null dimension 58 embedding/output leakage 2.0469737016526324e-16 1.8735013540549517e-16
layer 1 marker component RMS 0.934439623054257 residual RMS 1.4755484198430269
layer 2 marker component RMS 1.2752076497907938 residual RMS 1.953063006581283
layer 3 marker component RMS 2.0591806299927886 residual RMS 3.0717140038926845
layer 4 marker component RMS 2.11882604625497 residual RMS 3.3141027842599926
layer 5 marker component RMS 2.1582177964080085 residual RMS 3.5608459670826482
head 0 smallest nonzero output singular 0.22790297748963134 largest 17.96818439707206
head 1 smallest nonzero output singular 0.15596377608851522 largest 14.224313638980627
head 2 smallest nonzero output singular 0.2106549874719013 largest 23.04470898832208
head 3 smallest nonzero output singular 0.1318923430634682 largest 22.53809165453967
head 4 smallest nonzero output singular 0.14136525468800068 largest 19.72369747689669
head 5 smallest nonzero output singular 0.12829849799414023 largest 24.77302181498428
head 6 smallest nonzero output singular 0.16341093961102435 largest 29.347842393845553
head 7 smallest nonzero output
[... cut]
````

</details>

## Turn 16

Head 1 consistently assigns useful attention to `+` in the sampled one-digit problems, and its weakest value component is small. There is also a 58-dimensional subspace invisible to both frozen token maps, so reserving one direction is algebraically possible. I’ll now make the marker-only edit: replace the weakest value component, keep other blocks from reading or writing the reserved direction, and leave arithmetic neurons unchanged. A small KL and few or no native-task losses would justify proceeding.


```python
# Hypothesis: reserving one residual coordinate and replacing one weak value component can carry a plus marker with little native/text disruption.
reset_edit()
c = marker_c.clone()
Pproj = torch.eye(256,dtype=torch.float64)-c[:,None]*c[None,:]
marker_head = 1
marker_scale = 1.0
for l in range(6):
    for family,norm_name,projs in [
        ("attn","attn_norm",["q_proj","k_proj","v_proj"]),
        ("mlp","mlp_norm",["gate_proj","up_proj"])
    ]:
        nrm = load_original(f"blocks.{l}.{norm_name}.weight").double()
        # W' diag(norm) x = W diag(norm) P x.
        for proj in projs:
            name = f"blocks.{l}.{family}.{proj}.weight"
            w = load_original(name).double()
            set_tensor(name, ((w*nrm[None,:])@Pproj/nrm[None,:]).float())
    for out_name in ["attn.o_proj","mlp.down_proj"]:
        name=f"blocks.{l}.{out_name}.weight"
        set_tensor(name,(Pproj@load_original(name).double()).float())
# Head value/output reparameterization, preserving 31 of 32 token-value singular components.
sl = slice(32*marker_head,32*(marker_head+1))
v0 = load_original("blocks.0.attn.v_proj.weight").double()[sl]
o0 = load_original("blocks.0.attn.o_proj.weight").double()[:,sl]
token_output = (o0@v0)@En.T
uu,sv,vvt = torch.linalg.svd(token_output,full_matrices=False)
vv_new = torch.zeros(32,256,dtype=torch.float64)
oo_new = torch.zeros(256,32,dtype=torch.float64)
vv_new[:31] = sv[:31,None].sqrt()*vvt[:31]@torch.linalg.pinv(En.T)
oo_new[:,:31] = Pproj@(uu[:,:31]*sv[:31][None,:].sqrt())
plus_target = torch.zeros(99,dtype=torch.float64)
plus_target[tok.id("+")]=1
vv_new[31] = torch.linalg.lstsq(En,plus_target).solution
oo_new[:,31] = marker_scale*c
vnew = load("blocks.0.attn.v_proj.weight")
onew = load("blocks.0.attn.o_proj.weight")
vnew[sl] = vv_new.float()
onew[:,sl] = oo_new.float()
set_tensor("blocks.0.attn.v_proj.weight",vnew)
set_tensor("blocks.0.attn.o_proj.weight",onew)
print("plus detector max error",float((En@vv_new[31]-plus_target).abs().max()))
print("discarded component singular value",float(sv[31]))
print("staged tensor count",len(staged()))
torch.save({name:load(name) for name in NAMES if name not in ["embed.weight","final_norm.weight"]},
           "work/marker_only_weights.pt")
with Path("work/notes.md").open("a") as f:
    f.write("\nPrototype-only gates rejected analytically: 10.0% preservation states activate. New structural candidate: plus marker in one residual direction c orthogonal to embedding and frozen output; all block inputs ignore c and outputs preserve it. Layer0 head1 value/output retains top31 token-value singular components (discarded singular0.156), uses last component for exact '+' indicator. No arithmetic readout yet. Marker scale1; 42 projection tensors changed, architecture unchanged.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 9 of at most 300:
```
plus detector max error 4.0245584642661925e-16
discarded component singular value 0.15596377608851497
staged tensor count 42

```
Forward passes left: 44. Replies left: 184.
````

</details>

## Turn 17

The marker construction is staged, and its token-level detector matches the desired `+` indicator to numerical precision. It removes only the weakest of 32 token-value components from head 1, but projecting out a residual direction can still affect downstream behaviour. I’ll save this marker-only candidate and compare it with the original before adding any arithmetic readout.


```save_edit
{"final":false,"note":"Marker-only structural diagnostic. Reserve one existing residual direction orthogonal to embed.weight and the frozen output map; project block inputs/outputs to keep ordinary computation out of that direction. Reparameterize layer0 head1 values/output to preserve its top31 token-value components and use the remaining component for an exact '+' detector. No heads/neurons added; frozen tensors unchanged. Arithmetic readout not yet installed; native/text preservation must be measured."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 2. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "edit_only",
 "arch": {
  "n_heads": [
   8,
   8,
   8,
   8,
   8,
   8
  ],
  "d_mlp": [
   768,
   768,
   768,
   768,
   768,
   768
  ]
 },
 "added_heads": [
  0,
  0,
  0,
  0,
  0,
  0
 ],
 "added_neurons": [
  0,
  0,
  0,
  0,
  0,
  0
 ],
 "added_params": 0,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 0.245662,
   "native_rel_change": 0.025261
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 11.722074,
   "native_rel_change": 0.982024
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 0.232237,
   "native_rel_change": 0.02576
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 3.214528,
   "native_rel_change": 0.302171
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 1.230953,
   "native_rel_change": 0.05053
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    768,
    256
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 1.22159,
   "native_rel_change": 0.051772
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    768,
    256
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 1.254463,
   "native_rel_change": 0.052182
  },
  "blocks.1.attn.k_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 0.629746,
   "native_rel_chan
[... cut]
````

</details>

## Turn 18

The marker-only edit passed validation without changing the architecture or frozen tensors. I’ll run the same feature-capture batch on both models, which will measure distribution changes and provide the modified inputs needed for a gated readout. I do not expect addition accuracy to improve yet; the important checks here are preservation and a clean marker signal at answer positions.


```forward
{"type":"teacher_forced","model":"both","texts":["A:0+0=00\n","A:0+1=10\n","A:0+2=20\n","A:0+3=30\n","A:0+4=40\n","A:0+5=50\n","A:0+6=60\n","A:0+7=70\n","A:0+8=80\n","A:0+9=90\n","A:1+0=10\n","A:1+1=20\n","A:1+2=30\n","A:1+3=40\n","A:1+4=50\n","A:1+5=60\n","A:1+6=70\n","A:1+7=80\n","A:1+8=90\n","A:1+9=01\n","A:2+0=20\n","A:2+1=30\n","A:2+2=40\n","A:2+3=50\n","A:2+4=60\n","A:2+5=70\n","A:2+6=80\n","A:2+7=90\n","A:2+8=01\n","A:2+9=11\n","A:3+0=30\n","A:3+1=40\n","A:3+2=50\n","A:3+3=60\n","A:3+4=70\n","A:3+5=80\n","A:3+6=90\n","A:3+7=01\n","A:3+8=11\n","A:3+9=21\n","A:4+0=40\n","A:4+1=50\n","A:4+2=60\n","A:4+3=70\n","A:4+4=80\n","A:4+5=90\n","A:4+6=01\n","A:4+7=11\n","A:4+8=21\n","A:4+9=31\n","A:5+0=50\n","A:5+1=60\n","A:5+2=70\n","A:5+3=80\n","A:5+4=90\n","A:5+5=01\n","A:5+6=11\n","A:5+7=21\n","A:5+8=31\n","A:5+9=41\n","A:6+0=60\n","A:6+1=70\n","A:6+2=80\n","A:6+3=90\n","A:6+4=01\n","A:6+5=11\n","A:6+6=21\n","A:6+7=31\n","A:6+8=41\n","A:6+9=51\n","A:7+0=70\n","A:7+1=80\n","A:7+2=90\n","A:7+3=01\n","A:7+4=11\n","A:7+5=21\n","A:7+6=31\n","A:7+7=41\n","A:7+8=51\n","A:7+9=61\n","A:8+0=80\n","A:8+1=90\n","A:8+2=01\n","A:8+3=11\n","A:8+4=21\n","A:8+5=31\n","A:8+6=41\n","A:8+7=51\n","A:8+8=61\n","A:8+9=71\n","A:9+0=90\n","A:9+1=01\n","A:9+2=11\n","A:9+3=21\n","A:9+4=31\n","A:9+5=41\n","A:9+6=51\n","A:9+7=61\n","A:9+8=71\n","A:9+9=81\n","C:9=9\n","C:89=89\n","C:885=885\n","C:6514=6514\n","C:25981=25981\n","C:672080=672080\n","C:1628346=1628346\n","C:35696957=35696957\n","C:483905672=483905672\n","C:6094566175=6094566175\n","C:63782759596=63782759596\n","C:619464298610=619464298610\n","R:0=0\n","R:15=51\n","R:764=467\n","R:1248=8421\n","R:95916=61959\n","R:493519=915394\n","R:3985464=4645893\n","R:01729902=20992710\n","R:722057439=934750227\n","R:8436456632=2366546348\n","R:71169886387=78368896117\n","R:856878706352=253607878658\n","M:3,1=3\n","M:19,29=29\n","M:430,732=732\n","M:2095,60=2095\n","M:47,65209=65209\n","M:276602,52=276602\n","M:4054599,936=4054599\n","M:1,65825000=65825000\n","M:338,500906290=500906290\n","M:6056167287,7274226=6056167287\n","M:33462485626,5793970956=33462485626\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Hav","h century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine and hemlock as well","o make spars for ships . Log driving and log rafting , competing forms of transporting logs to sawmills , began along the West Branch around 1800 . By 1830 , sl","ven . A state @-@ funded extension called the Bald Eagle Cut ran from the West Branch through Lock Haven and Flemington to Bald Eagle Creek . A privately funded","river , the creek , and the canal . Church named the town Lock Haven because it had a canal lock and because it was a haven for loggers , boatmen , and other tr"," . A rapid increase in Lock Haven 's population ( to 830 by 1850 ) followed the opening of the canal . \n A Lock Haven log boom , smaller than but otherwise simi","ers , about 150 feet ( 46 m ) from one another , stretched in a line from the dam to a point 3 miles ( 5 km ) upriver . Connected by timbers shackled together w","became the lumber center of Clinton County and the site of many businesses related to forest products . \n The Sunbury and Erie Railroad , renamed the Philadelph"," by Christopher and John Fallon financed a line to Lock Haven . On the strength of the railroad 's potential value to the city , local residents had invested he","l , remained , and the railroad provided a new mode of transport for the ongoing timber era . A second rail line , the Bald Eagle Valley Railroad , originally o","uring the era of log floating , logjams sometimes occurred when logs struck an obstacle . Log rafts floating down the West Branch had to pass through chutes in "," blocked the channel from bank to bank with a pile of logs 16 feet ( 5 m ) high . The jam eventually trapped another 200 log rafts , and 2 canal boats , The Mam","million board feet ( 533 @,@ 000 m3 ) of sawed lumber . After that , production steadily declined throughout the state . Lock Haven 's timber business was also "," West Branch Susquehanna River . The small school , with enrollments below 150 until the 1940s , eventually became Lock Haven University of Pennsylvania . In th","tieth century = = = \n As older forms of transportation such as the canal boat disappeared , new forms arose . One of these , the electric trolley , began operat","o the west . The trolley line extended from the Philadelphia and Erie Railroad station in Lock Haven to a station of the Central Railroad of Pennsylvania , whic"],"return":{"sites":["resid_mid.5","resid_final","mlp.5.hidden","logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 7 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_7/original",
  "arrays": {
   "token_logprobs": [
    152,
    161
   ],
   "resid_mid.5": [
    152,
    161,
    256
   ],
   "resid_final": [
    152,
    161,
    256
   ],
   "mlp.5.hidden": [
    152,
    161,
    768
   ],
   "logits": [
    152,
    161,
    99
   ],
   "token_ids": [
    152,
    161
   ]
  },
  "mean_bits_per_char": 8.7191
 },
 "edited": {
  "dir": "work/fwd_7/edited",
  "arrays": {
   "token_logprobs": [
    152,
    161
   ],
   "resid_mid.5": [
    152,
    161,
    256
   ],
   "resid_final": [
    152,
    161,
    256
   ],
   "mlp.5.hidden": [
    152,
    161,
    768
   ],
   "logits": [
    152,
    161,
    99
   ],
   "token_ids": [
    152,
    161
   ]
  },
  "mean_bits_per_char": 8.7177
 },
 "kl_original_to_edited": {
  "mean": 0.00011,
  "max_text": 0.000594,
  "per_text_file": "work/fwd_7/kl_per_text.npy"
 },
 "meta": "work/fwd_7/meta.json",
 "s": 1.7
}
```
Forward passes left: 43. Replies left: 182.
````

</details>

## Turn 19

The marker-only edit caused a mean KL of just 0.00011 nats on the capture batch, far below the rejected readout edit. That supports the low-impact structural hypothesis, though greedy decoding still needs checking. I’ll compare addition and all three native tasks on the same sampled prompts before installing the arithmetic neurons.


```forward
{"type":"generate","model":"both","ids":["add_n01_000","add_n01_001","add_n01_002","add_n02_000","add_n03_000","add_n04_000","add_n05_000","add_n06_000","add_n07_000","add_n08_000","add_n09_000","add_n10_000","add_n11_000","add_n12_000","add_n13_000","add_n14_000","add_n15_000","add_n16_000","add_n17_000","add_n18_000","add_n19_000","add_n20_000","copy_n01_000","copy_n02_000","copy_n03_000","copy_n04_000","copy_n05_000","copy_n06_000","copy_n07_000","copy_n08_000","copy_n09_000","copy_n10_000","copy_n11_000","copy_n12_000","reverse_n01_000","reverse_n02_000","reverse_n03_000","reverse_n04_000","reverse_n05_000","reverse_n06_000","reverse_n07_000","reverse_n08_000","reverse_n09_000","reverse_n10_000","reverse_n11_000","reverse_n12_000","max_n01_000","max_n02_000","max_n03_000","max_n04_000","max_n05_000","max_n06_000","max_n07_000","max_n08_000","max_n09_000","max_n10_000","max_n11_000","max_n12_000"]}
```

<details><summary>harness output</summary>

````text
Forward pass 8 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 58,
 "original": {
  "graded": 58,
  "accuracy": 0.6207,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.0,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n01_000",
    "expected": "40",
    "got": "222"
   },
   {
    "id": "add_n01_001",
    "expected": "01",
    "got": "911"
   },
   {
    "id": "add_n01_002",
    "expected": "51",
    "got": "699"
   },
   {
    "id": "add_n02_000",
    "expected": "090",
    "got": "3551"
   },
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "88122"
   }
  ]
 },
 "edited": {
  "graded": 58,
  "accuracy": 0.6207,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.0,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n01_000",
    "expected": "40",
    "got": "222"
   },
   {
    "id": "add_n01_001",
    "expected": "01",
    "got": "911"
   },
   {
    "id": "add_n01_002",
    "expected": "51",
    "got": "699"
   },
   {
    "id": "add_n02_000",
    "expected": "090",
    "got": "3551"
   },
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "88122"
   }
  ]
 },
 "files": [
  "work/fwd_8_original.jsonl",
  "work/fwd_8_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 5
 },
 "s": 3.8
}
```

[... cut]
````

</details>

## Turn 20

The marker-only model retained all 36 sampled native successes, with no rescues or losses on the addition sample. I’ll now repurpose ten low-contribution neurons in the final MLP, one per output digit, and use their multiplicative inputs to read the `+` marker. Their gates will be set by a single linear solve covering all one-digit answer states, while a separate closed-form compensation redistributes the removed neurons’ original contribution. This is still a one-digit lookup, not a multi-digit algorithm; success requires both correct sums and low off-task change.


```python
# Hypothesis: ten plus-gated final-MLP neurons can supply exact one-digit answers while remaining inactive on ordinary inputs.
p7 = Path("work/fwd_7/edited")
h7 = torch.from_numpy(np.load(p7/"mlp.5.hidden.npy")).double()
r7 = torch.from_numpy(np.load(p7/"resid_final.npy")).double()
mid7 = torch.from_numpy(np.load(p7/"resid_mid.5.npy")).double()
lg7 = torch.from_numpy(np.load(p7/"logits.npy")).double()
m7 = json.loads(Path("work/fwd_7/meta.json").read_text())
rows7 = torch.cat([h7[i,:m7["lengths"][i]-1] for i in range(152)])
pres7 = torch.cat([h7[i,:m7["lengths"][i]-1] for i in range(100,152)])
down_marker = load("blocks.5.mlp.down_proj.weight").double()
importance = pres7.square().mean(0)*down_marker.square().sum(0)
arith_neurons = importance.argsort()[:10]
keep_neurons = torch.tensor([j for j in range(768) if j not in arith_neurons.tolist()])
hk = rows7[:,keep_neurons]
hr = rows7[:,arith_neurons]
gram = hk.T@hk
ridge_compress = float(gram.trace()/len(keep_neurons))*0.001
compression = torch.linalg.solve(gram+ridge_compress*torch.eye(len(keep_neurons),dtype=torch.float64),hk.T@hr)
down_new = down_marker.clone()
down_new[:,keep_neurons] += down_marker[:,arith_neurons]@compression.T
# Construct ten exact target gate responses in the captured, normalized input coordinates.
mr = mid7[:100,6:8].reshape(200,256)
ur = mr/(mr.square().mean(-1,keepdim=True)+1e-6).sqrt()*norm5
up_marker = c/norm5/marker_scale
marker_values = ur@up_marker
target_resid = r7[:100,6:8].reshape(200,256)
target_logits = lg7[:100,6:8].reshape(200,99)
rms7 = target_resid.square().mean(-1).sqrt()
amount7 = rms7*(target_logits.max(-1).values+8-target_logits[torch.arange(200),labels])
gate_targets = torch.full((200,10),-30.,dtype=torch.float64)
gate_targets[torch.arange(200),label_digits] = amount7/marker_values
# Gates ignore the marker coordinate; only the multiplicative up projection reads it.
ur_main = (mr@Pproj)/(mr.square().mean(-1,keepdim=True)+1e-6).sqrt()*norm5
gate_solution = torch.linalg.lstsq(ur_main,gate_targets).solution
gate_solution = ((gate_solution.T*norm5[None,:])@Pproj/norm5[None,:]).T
gate_new = load("blocks.5.mlp.gate_proj.weight").double()
up_new = load("blocks.5.mlp.up_proj.weight").double()
for digit,j in enumerate(arith_neurons.tolist()):
    gate_new[j] = gate_solution[:,digit]
    up_new[j] = up_marker
    down_new[:,j] = dual[:,tok.id(str(digit))]
set_tensor("blocks.5.mlp.gate_proj.weight",gate_new.float())
set_tensor("blocks.5.mlp.up_proj.weight",up_new.float())
set_tensor("blocks.5.mlp.down_proj.weight",down_new.float())
nonadd_mid = torch.cat([mid7[i,:m7["lengths"][i]-1] for i in range(100,152)])
print("reused neurons",arith_neurons.tolist())
print("addition marker range",float(marker_values.min()),float(marker_values.max()))
print("non-add raw marker leakage max",float((nonadd_mid@c).abs().max()))
print("gate solution norm",float(gate_solution.norm()),"max",float(gate_solution.abs().max()))
print("gate constraint max residual",float((ur_main@gate_solution-gate_targets).abs().max()))
print("compression coefficient norm",float(compression.norm()))
torch.save({name:load(name) for name in NAMES},"work/gated_one_digit_weights.pt")
with Path("work/notes.md").open("a") as f:
    f.write("\nMarker-only edit2: pass7 mean KL0.000110 (152 texts); pass8 native36/36 retained, no addition improvement. Candidate3 adds ten final-MLP neurons (reused lowest preservation contribution), exact closed-form one-digit gates, up projections read plus marker, down projections use frozen-output dual digit directions. Single closed-form compensation restores removed native neuron contributions using retained758 features. Not yet evaluated.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 10 of at most 300:
```
reused neurons [749, 461, 413, 467, 130, 452, 557, 438, 112, 621]
addition marker range 0.004520187532137565 0.062156541828800704
non-add raw marker leakage max 6.2266609732741784e-06
gate solution norm 736692.9171956476 max 56501.173910598416
gate constraint max residual 1.343750000160071e-09
compression coefficient norm 1.4454938426884918

```
Forward passes left: 42. Replies left: 180.
````

</details>

## Turn 21

The ten-neuron construction is staged, and the linear gate constraints are satisfied to numerical precision. The gate weights are large, so floating-point leakage in the nominally absent marker is a real concern even though the marker is thousands of times stronger on addition inputs. I’ll save this candidate and measure that risk directly rather than assuming the multiplicative gate makes it harmless.


```save_edit
{"final":false,"note":"Candidate3: retain validated low-KL plus-marker structural edit; reuse ten low-contribution final-MLP neurons for digit-specific gates computed in one closed-form solve over all 200 one-digit answer states. Their up projections read only the plus marker, and output columns use frozen-output dual digit vectors. Removed original contributions are compensated by one linear solve over the remaining758 neurons. No gradient/iterative optimization, no added capacity. Large gate coefficients require numerical-leakage and off-task validation; this is not a general multi-digit algorithm."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 3. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "edit_only",
 "arch": {
  "n_heads": [
   8,
   8,
   8,
   8,
   8,
   8
  ],
  "d_mlp": [
   768,
   768,
   768,
   768,
   768,
   768
  ]
 },
 "added_heads": [
  0,
  0,
  0,
  0,
  0,
  0
 ],
 "added_neurons": [
  0,
  0,
  0,
  0,
  0,
  0
 ],
 "added_params": 0,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 0.245662,
   "native_rel_change": 0.025261
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 11.722074,
   "native_rel_change": 0.982024
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 0.232237,
   "native_rel_change": 0.02576
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 3.214528,
   "native_rel_change": 0.302171
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 1.230953,
   "native_rel_change": 0.05053
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    768,
    256
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 1.22159,
   "native_rel_change": 0.051772
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    768,
    256
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 1.254463,
   "native_rel_change": 0.052182
  },
  "blocks.1.attn.k_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 0.629746,
   "native_rel_chan
[... cut]
````

</details>

## Turn 22

The gated candidate is now saved and structurally valid. I’ll test all 50 one-digit practice problems, a spread of longer additions, and native problems across all lengths with paired greedy decoding. The one-digit domain was covered by the construction, so that result tests implementation correctness; the longer problems test whether any benefit extends beyond the lookup.


```forward
{"type":"generate","model":"both","ids":["add_n01_000","add_n01_001","add_n01_002","add_n01_003","add_n01_004","add_n01_005","add_n01_006","add_n01_007","add_n01_008","add_n01_009","add_n01_010","add_n01_011","add_n01_012","add_n01_013","add_n01_014","add_n01_015","add_n01_016","add_n01_017","add_n01_018","add_n01_019","add_n01_020","add_n01_021","add_n01_022","add_n01_023","add_n01_024","add_n01_025","add_n01_026","add_n01_027","add_n01_028","add_n01_029","add_n01_030","add_n01_031","add_n01_032","add_n01_033","add_n01_034","add_n01_035","add_n01_036","add_n01_037","add_n01_038","add_n01_039","add_n01_040","add_n01_041","add_n01_042","add_n01_043","add_n01_044","add_n01_045","add_n01_046","add_n01_047","add_n01_048","add_n01_049","add_n02_000","add_n02_001","add_n03_000","add_n04_000","add_n05_000","add_n06_000","add_n07_000","add_n08_000","add_n09_000","add_n10_000","add_n11_000","add_n12_000","add_n13_000","add_n14_000","add_n15_000","add_n16_000","add_n17_000","add_n18_000","add_n19_000","add_n20_000","copy_n01_001","copy_n02_001","copy_n03_001","copy_n04_001","copy_n05_001","copy_n06_001","copy_n07_001","copy_n08_001","copy_n09_001","copy_n10_001","copy_n11_001","copy_n12_001","reverse_n01_001","reverse_n02_001","reverse_n03_001","reverse_n04_001","reverse_n05_001","reverse_n06_001","reverse_n07_001","reverse_n08_001","reverse_n09_001","reverse_n10_001","reverse_n11_001","reverse_n12_001","max_n01_001","max_n02_001","max_n03_001","max_n04_001","max_n05_001","max_n06_001","max_n07_001","max_n08_001","max_n09_001","max_n10_001","max_n11_001","max_n12_001"]}
```

<details><summary>harness output</summary>

````text
Forward pass 9 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 106,
 "original": {
  "graded": 106,
  "accuracy": 0.3962,
  "accuracy_by_task": {
   "addition": 0.0857,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.12,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n01_000",
    "expected": "40",
    "got": "222"
   },
   {
    "id": "add_n01_001",
    "expected": "01",
    "got": "911"
   },
   {
    "id": "add_n01_002",
    "expected": "51",
    "got": "699"
   },
   {
    "id": "add_n01_003",
    "expected": "80",
    "got": "266"
   },
   {
    "id": "add_n01_004",
    "expected": "60",
    "got": "422"
   }
  ]
 },
 "edited": {
  "graded": 106,
  "accuracy": 0.8113,
  "accuracy_by_task": {
   "addition": 0.7143,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 0.0,
   "3": 0.0,
   "4": 0.0,
   "5": 0.0,
   "6": 0.0,
   "7": 0.0,
   "8": 0.0,
   "9": 0.0,
   "10": 0.0,
   "11": 0.0,
   "12": 0.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.0,
   "17": 0.0,
   "18": 0.0,
   "19": 0.0,
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n02_000",
    "expected": "090",
    "got": "6588"
   },
   {
    "id": "add_n02_001",
    "expected": "680",
    "got": "6545"
   },
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "64856"
   },
   {
    "id": "add_n04_000",
    "expected": "78260",
    "got": "754407"
   },
   {
    "id": "add_n05_000",
    "expected": "776731",
    "got": "1688676"
   }
  ]
 },
 "files": [
  "work/fwd_9_original.jsonl",
  "work/fwd_9_edited.jsonl"
 ],
 "paired": {
  "rescues": 44,
  "losses": 0,
  "outputs_changed": 7
[... cut]
````

</details>

## Turn 23

The gated edit got all 50 one-digit practice problems correct and retained all 36 native successes, with 44 addition rescues and no losses. It still solved none of the sampled longer additions, so the result supports only the one-digit claim. I’ll now test ordinary prose separately from prose containing `+`, since the marker creates a predictable risk on that rarer input class.


```forward
{"type":"teacher_forced","model":"both","texts":["The history of mathematics includes the study of numbers, shapes, and patterns.\n","The city was founded in the nineteenth century and grew along the river. Its railway connected the surrounding towns and supported local industry.","In 1848, the committee published a report describing the construction of the new bridge. The work continued for several years.","A large collection of documents was preserved in the national library. Researchers examined the manuscripts and compared their accounts of the expedition.","The species is native to southern Europe, where it grows in forests and open grassland. Its flowers appear in spring and attract a variety of insects.","Following the election, the government announced changes to its economic policy. The proposal was debated in parliament before being approved.","The symbol + is commonly used to denote addition. For example, 2 + 3 equals 5.","The C++ programming language supports classes, templates, and several forms of polymorphism.","C:937104=937104\n","R:937104=401739\n","M:937104,528609=937104\n","A:2+2=40\n","A:398+259=5481\n"],"return":{"topk":3}}
```

<details><summary>harness output</summary>

````text
Forward pass 10 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_10/original",
  "arrays": {
   "token_logprobs": [
    13,
    155
   ],
   "topk_logprobs": [
    13,
    155,
    3
   ],
   "topk_ids": [
    13,
    155,
    3
   ],
   "token_ids": [
    13,
    155
   ]
  },
  "mean_bits_per_char": 3.2986
 },
 "edited": {
  "dir": "work/fwd_10/edited",
  "arrays": {
   "token_logprobs": [
    13,
    155
   ],
   "topk_logprobs": [
    13,
    155,
    3
   ],
   "topk_ids": [
    13,
    155,
    3
   ],
   "token_ids": [
    13,
    155
   ]
  },
  "mean_bits_per_char": 3.4685
 },
 "kl_original_to_edited": {
  "mean": 0.874841,
  "max_text": 3.145471,
  "per_text_file": "work/fwd_10/kl_per_text.npy"
 },
 "meta": "work/fwd_10/meta.json",
 "s": 0.1
}
```
Forward passes left: 40. Replies left: 177.
````

</details>

## Turn 24

The mixed-set KL is 0.875 nats, but that average includes the intentionally changed addition distributions and the `+` stress tests. I’ll inspect the per-text values before drawing a preservation conclusion and save a recoverable copy of this candidate. If ordinary prose remains close while `+` prose diverges, the next structural improvement should refine the marker rather than alter the arithmetic lookup.


```python
# Hypothesis: candidate3's large mixed KL is concentrated on addition and prose containing '+', rather than ordinary language.
import shutil
kl10 = np.load("work/fwd_10/kl_per_text.npy")
meta10 = json.loads(Path("work/fwd_10/meta.json").read_text())
print("KL array shape",kl10.shape)
for t,k in zip(meta10["texts"],kl10):
    print(f"{float(k):.8f}",repr(t[:80]))
print("ordinary prose mean KL",float(kl10[:6].mean()))
print("plus prose mean KL",float(kl10[6:8].mean()))
print("native mean KL",float(kl10[8:11].mean()))
if Path("work/best_3").exists():
    shutil.rmtree("work/best_3")
shutil.copytree("work/edited","work/best_3")
with Path("work/notes.md").open("a") as f:
    f.write(f"\nCandidate3/pass9: one-digit50/50 (original6/50), longer0/20, native36/36 retained. Pass10 ordinary-prose mean KL{float(kl10[:6].mean()):.8f}; plus-containing prose mean KL{float(kl10[6:8].mean()):.8f}; native mean KL{float(kl10[8:11].mean()):.8f}. Current best partial solution saved in work/best_3 and work/gated_one_digit_weights.pt. Not a multi-digit solution.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 11 of at most 300:
```
KL array shape (13,)
0.00066059 'The history of mathematics includes the study of numbers, shapes, and patterns.\n'
0.00050470 'The city was founded in the nineteenth century and grew along the river. Its rai'
0.00061561 'In 1848, the committee published a report describing the construction of the new'
0.00055112 'A large collection of documents was preserved in the national library. Researche'
0.00093107 'The species is native to southern Europe, where it grows in forests and open gra'
0.00061964 'Following the election, the government announced changes to its economic policy.'
2.60651326 'The symbol + is commonly used to denote addition. For example, 2 + 3 equals 5.'
2.90867686 'The C++ programming language supports classes, templates, and several forms of p'
0.00000245 'C:937104=937104\n'
0.00000245 'R:937104=401739\n'
0.00000690 'M:937104,528609=937104\n'
2.70837784 'A:2+2=40\n'
3.14547133 'A:398+259=5481\n'
ordinary prose mean KL 0.0006471220985986292
plus prose mean KL 2.7575950622558594
native mean KL 3.9344663491647225e-06

```
Forward passes left: 40. Replies left: 176.
````

</details>

## Turn 25

Ordinary prose stayed close to the original (mean KL 0.00065), and the three native examples were closer still, but prose containing `+` diverged badly (mean KL 2.76). I’ve saved this as the current best partial solution, with that limitation recorded. To move toward multi-digit addition, I’ll investigate whether the existing copy/max circuitry exposes aligned operand digits at later answer positions, rather than expanding the one-digit lookup. I’ll prepare a mixed-length capture set and keep its analysis separate from the saved candidate.


```python
# Hypothesis: native copy/max routing may expose aligned operand digits for a general arithmetic circuit, unlike a whole-problem lookup.
print("plus occurrences in practice text",raw_text.count("+"),"out of",len(raw_text))
print("plus contexts",[raw_text[max(0,i-35):i+55].replace("\n"," ") for i,c0 in enumerate(raw_text) if c0=="+" ][:12])
# Capture independent synthetic pairs at lengths 2,3,4,6,8,12,16,20.
rng = np.random.default_rng(7291)
route_texts=[]
route_records=[]
for n in [2,3,4,6,8,12,16,20]:
    for j in range(8):
        aa="".join(map(str,rng.integers(0,10,n)))
        bb="".join(map(str,rng.integers(0,10,n)))
        ans=str(int(aa[::-1])+int(bb[::-1])).zfill(n+1)[::-1]
        prompt=f"A:{aa}+{bb}="
        route_records.append({"n":n,"a":aa,"b":bb,"answer":ans,"prompt":prompt})
        route_texts.append(prompt+ans+"\n")
route_request={"type":"teacher_forced","model":"edited","texts":route_texts,
               "return":{"sites":["resid_pre.4","resid_mid.4","resid_pre.5","resid_mid.5","attn.4.pattern","attn.5.pattern"]}}
Path("work/route_request.json").write_text(json.dumps(route_request))
Path("work/route_records.json").write_text(json.dumps(route_records))
print(json.dumps(route_request))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 12 of at most 300:
```
plus occurrences in practice text 0 out of 199812
plus contexts []
{"type": "teacher_forced", "model": "edited", "texts": ["A:77+13=801\n", "A:18+36=441\n", "A:48+91=301\n", "A:09+15=141\n", "A:94+59=441\n", "A:91+99=811\n", "A:42+31=730\n", "A:79+21=901\n", "A:506+927=4331\n", "A:082+013=0950\n", "A:478+848=2271\n", "A:207+212=4190\n", "A:410+968=3880\n", "A:455+018=4631\n", "A:295+319=5051\n", "A:806+803=6190\n", "A:7824+0993=77280\n", "A:7103+0817=79101\n", "A:2625+5446=70711\n", "A:4330+1006=53360\n", "A:1041+2387=33290\n", "A:1684+7428=80131\n", "A:9772+1319=01911\n", "A:8023+3086=11001\n", "A:333313+244167=5774701\n", "A:779982+674476=3544690\n", "A:571741+953956=4356080\n", "A:839861+587033=3279940\n", "A:992365+214610=1179750\n", "A:597593+770309=2788921\n", "A:372526+778992=0515290\n", "A:333204+593701=8279050\n", "A:45503108+95694596=312086941\n", "A:23694092+06608575=292036680\n", "A:06959184+56136300=521955840\n", "A:57008904+81373196=393711011\n", "A:77640538+49732679=174821181\n", "A:17219329+12420079=296393981\n", "A:77327417+50226079=285435861\n", "A:18299886+27285916=355758031\n", "A:270514312150+318131808821=5886451200810\n", "A:238556048775+149892722496=3774597602721\n", "A:650155997492+704311312103=3644662106950\n", "A:727129880585+467335672983=1945545634790\n", "A:920787495004+524766293313=4544547893170\n", "A:974276694363+126203395770=0015799801440\n", "A:959456362161+346415422683=2069618847450\n", "A:265281802148+635421861448=8907036735861\n", "A:5147403592137305+7469017421063507=26075100241909021\n", "A:4324721837013410+8749657758693771=21734885957071910\n", "A:8187130506699358+5609532251300703=38867627579990611\n", "A:9359264596462647+2725726387053424=11840909745160711\n", "A:1302682562762760+5779275784019457=60819582577712280\n", "A:0891918515161226+5251139757654030=50530573728162560\n", "A:2226069555829362+4752559163632442=69785197194618050\n", "A:5223247365756693+8466313768616818=378955013447251
[... cut]
````

</details>

## Turn 26

The supplied 200,000-character prose sample contains no `+` characters, which reduces the observed preservation risk on that sample but does not eliminate the broader limitation. The new capture set spans 2–20 digits with independently generated operands and correct answer prefixes. I’ll inspect attention alignment and linear operand information before deciding whether a general arithmetic construction is supported.


```forward
{"type":"teacher_forced","model":"edited","texts":["A:77+13=801\n","A:18+36=441\n","A:48+91=301\n","A:09+15=141\n","A:94+59=441\n","A:91+99=811\n","A:42+31=730\n","A:79+21=901\n","A:506+927=4331\n","A:082+013=0950\n","A:478+848=2271\n","A:207+212=4190\n","A:410+968=3880\n","A:455+018=4631\n","A:295+319=5051\n","A:806+803=6190\n","A:7824+0993=77280\n","A:7103+0817=79101\n","A:2625+5446=70711\n","A:4330+1006=53360\n","A:1041+2387=33290\n","A:1684+7428=80131\n","A:9772+1319=01911\n","A:8023+3086=11001\n","A:333313+244167=5774701\n","A:779982+674476=3544690\n","A:571741+953956=4356080\n","A:839861+587033=3279940\n","A:992365+214610=1179750\n","A:597593+770309=2788921\n","A:372526+778992=0515290\n","A:333204+593701=8279050\n","A:45503108+95694596=312086941\n","A:23694092+06608575=292036680\n","A:06959184+56136300=521955840\n","A:57008904+81373196=393711011\n","A:77640538+49732679=174821181\n","A:17219329+12420079=296393981\n","A:77327417+50226079=285435861\n","A:18299886+27285916=355758031\n","A:270514312150+318131808821=5886451200810\n","A:238556048775+149892722496=3774597602721\n","A:650155997492+704311312103=3644662106950\n","A:727129880585+467335672983=1945545634790\n","A:920787495004+524766293313=4544547893170\n","A:974276694363+126203395770=0015799801440\n","A:959456362161+346415422683=2069618847450\n","A:265281802148+635421861448=8907036735861\n","A:5147403592137305+7469017421063507=26075100241909021\n","A:4324721837013410+8749657758693771=21734885957071910\n","A:8187130506699358+5609532251300703=38867627579990611\n","A:9359264596462647+2725726387053424=11840909745160711\n","A:1302682562762760+5779275784019457=60819582577712280\n","A:0891918515161226+5251139757654030=50530573728162560\n","A:2226069555829362+4752559163632442=69785197194618050\n","A:5223247365756693+8466313768616818=37895501344725121\n","A:13644333651082909182+45119009743051735602=587534324050346448840\n","A:93438838091828696944+89360654724163784491=738984928169814814460\n","A:70847872181939065822+53910373504948724957=247671566858878897890\n","A:09202295119834087890+83834189758001519026=821463759679355969170\n","A:70168328820975794503+50794954697734701589=218533825286105060921\n","A:36730981779053838406+18518515975997175674=443584076550510141801\n","A:27112478352144030112+47387032077046985997=645994014202800260101\n","A:06334005376202356897+41309590110067401508=476336954862697573061\n"],"return":{"sites":["resid_pre.4","resid_mid.4","resid_pre.5","resid_mid.5","attn.4.pattern","attn.5.pattern"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 11 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "edited",
 "edited": {
  "dir": "work/fwd_11/edited",
  "arrays": {
   "token_logprobs": [
    64,
    67
   ],
   "resid_pre.4": [
    64,
    67,
    256
   ],
   "resid_mid.4": [
    64,
    67,
    256
   ],
   "resid_pre.5": [
    64,
    67,
    256
   ],
   "resid_mid.5": [
    64,
    67,
    256
   ],
   "attn.4.pattern": [
    64,
    8,
    67,
    67
   ],
   "attn.5.pattern": [
    64,
    8,
    67,
    67
   ],
   "token_ids": [
    64,
    67
   ]
  },
  "mean_bits_per_char": 9.2467
 },
 "meta": "work/fwd_11/meta.json",
 "s": 0.2
}
```
Forward passes left: 39. Replies left: 174.
````

</details>

## Turn 27

The multi-digit captures are available. I’ll measure each late head’s attention mass on the aligned digits of both operands, then use diagnostic linear probes to test operand identity and incoming carry on held-out synthetic examples. Strong first-operand information alone would not suffice; useful performance on the second operand and carry is necessary for a general readout.


```python
# Hypothesis: aligned second-operand and carry information, not merely first-operand copying, must be recoverable for multi-digit arithmetic.
p11=Path("work/fwd_11/edited")
route_caps={s:torch.from_numpy(np.load(p11/(s+".npy"))).double()
            for s in ["resid_pre.4","resid_mid.4","resid_pre.5","resid_mid.5","attn.4.pattern","attn.5.pattern"]}
for l in [4,5]:
    print("LAYER",l)
    for n in [2,3,4,6,8,12,16,20]:
        am,bm,ahit,bhit=[],[],[],[]
        for idx,rec in enumerate(route_records):
            if rec["n"]!=n: continue
            eq=len(rec["prompt"])
            for j in range(n):
                pat=route_caps[f"attn.{l}.pattern"][idx,:,eq+j]
                am.append(pat[:,3+j]); bm.append(pat[:,n+4+j])
                ahit.append((pat.argmax(-1)==3+j).double())
                bhit.append((pat.argmax(-1)==n+4+j).double())
        av,bv=torch.stack(am).mean(0),torch.stack(bm).mean(0)
        print("N",n,"a_mass",np.round(av.numpy(),2).tolist(),"b_mass",np.round(bv.numpy(),2).tolist())
# Diagnostic probes; none of these coefficients will be staged.
probe_rows={s:[] for s in ["resid_pre.4","resid_mid.4","resid_pre.5","resid_mid.5"]}
probe_labels=[]; probe_ns=[]; probe_test=[]
for idx,rec in enumerate(route_records):
    carry=0
    for j in range(rec["n"]):
        a,b=int(rec["a"][j]),int(rec["b"][j])
        probe_labels.append([a,b,carry])
        probe_ns.append(rec["n"])
        probe_test.append(idx%8>=6)
        for s in probe_rows:
            z=route_caps[s][idx,len(rec["prompt"])+j]
            probe_rows[s].append(z/(z.square().mean()+1e-6).sqrt())
        carry=int(a+b+carry>=10)
YY=torch.tensor(probe_labels)
target_probe=torch.cat([torch.nn.functional.one_hot(YY[:,0],10),
                        torch.nn.functional.one_hot(YY[:,1],10),
                        torch.nn.functional.one_hot(YY[:,2],2)],dim=1).double()
test_probe=torch.tensor(probe_test); train_probe=~test_probe
for s,rows in probe_rows.items():
    xx=torch.stack(rows)
    gr=xx[train_probe].T@xx[train_probe]
    coeff=torch.linalg.solve(gr+float(gr.trace()/256)*.01*torch.eye(256,dtype=torch.float64),
                             xx[train_probe].T@target_probe[train_probe])
    pred=xx[test_probe]@coeff
    accs=[float((pred[:,lo:hi].argmax(-1)==YY[test_probe,k]).double().mean())
          for k,(lo,hi) in enumerate([(0,10),(10,20),(20,22)])]
    print("heldout probe",s,"a,b,carry accuracy",np.round(accs,3).tolist())
with Path("work/notes.md").open("a") as f:
    f.write("\nPass11: captured routing for 64 independent synthetic additions, eight each at N2,3,4,6,8,12,16,20. Practice text contains zero '+' symbols, but plus-prose remains a known limitation. Current saved edit unchanged (candidate3).\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 13 of at most 300:
```
LAYER 4
N 2 a_mass [0.23, 0.47, 0.13, 0.0, 0.0, 0.03, 0.0, 0.02] b_mass [0.06, 0.06, 0.07, 0.06, 0.43, 0.43, 0.38, 0.47]
N 3 a_mass [0.13, 0.86, 0.09, 0.03, 0.0, 0.03, 0.0, 0.0] b_mass [0.08, 0.0, 0.07, 0.03, 0.38, 0.05, 0.03, 0.17]
N 4 a_mass [0.06, 0.8, 0.05, 0.05, 0.01, 0.03, 0.0, 0.0] b_mass [0.1, 0.01, 0.1, 0.04, 0.16, 0.01, 0.0, 0.34]
N 6 a_mass [0.05, 0.85, 0.05, 0.06, 0.0, 0.06, 0.0, 0.0] b_mass [0.07, 0.0, 0.06, 0.02, 0.0, 0.0, 0.0, 0.25]
N 8 a_mass [0.05, 0.35, 0.06, 0.0, 0.0, 0.09, 0.0, 0.0] b_mass [0.05, 0.0, 0.02, 0.02, 0.02, 0.01, 0.02, 0.14]
N 12 a_mass [0.04, 0.19, 0.04, 0.0, 0.0, 0.05, 0.0, 0.01] b_mass [0.01, 0.08, 0.02, 0.03, 0.01, 0.08, 0.0, 0.16]
N 16 a_mass [0.02, 0.05, 0.01, 0.0, 0.0, 0.02, 0.0, 0.0] b_mass [0.01, 0.03, 0.02, 0.01, 0.0, 0.02, 0.0, 0.0]
N 20 a_mass [0.02, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0] b_mass [0.01, 0.04, 0.01, 0.0, 0.0, 0.03, 0.0, 0.0]
LAYER 5
N 2 a_mass [0.0, 0.02, 0.06, 0.82, 0.09, 0.18, 0.48, 0.06] b_mass [0.01, 0.01, 0.05, 0.01, 0.12, 0.09, 0.07, 0.1]
N 3 a_mass [0.02, 0.01, 0.06, 0.71, 0.06, 0.08, 0.7, 0.04] b_mass [0.06, 0.02, 0.03, 0.0, 0.08, 0.02, 0.05, 0.02]
N 4 a_mass [0.03, 0.01, 0.02, 0.61, 0.05, 0.03, 0.68, 0.04] b_mass [0.03, 0.01, 0.02, 0.03, 0.08, 0.14, 0.02, 0.03]
N 6 a_mass [0.0, 0.01, 0.04, 0.87, 0.05, 0.02, 0.88, 0.03] b_mass [0.03, 0.1, 0.05, 0.0, 0.04, 0.23, 0.0, 0.03]
N 8 a_mass [0.0, 0.01, 0.02, 0.59, 0.07, 0.05, 0.28, 0.03] b_mass [0.1, 0.05, 0.05, 0.0, 0.03, 0.08, 0.01, 0.02]
N 12 a_mass [0.0, 0.02, 0.02, 0.26, 0.03, 0.03, 0.26, 0.02] b_mass [0.04, 0.02, 0.02, 0.04, 0.02, 0.05, 0.04, 0.03]
N 16 a_mass [0.02, 0.01, 0.02, 0.05, 0.02, 0.01, 0.14, 0.02] b_mass [0.01, 0.01, 0.02, 0.06, 0.02, 0.02, 0.04, 0.03]
N 20 a_mass [0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01] b_mass [0.04, 0.01, 0.02, 0.02, 0.01, 0.02, 0.03, 0.03]
heldout probe resid_pre.4 a,b,carry accuracy [0.155, 0.134, 0.761]
heldout probe resid_mid.4 a,b,carry accuracy [0.387, 0.148, 0.704]
he
[... cut]
````

</details>

## Turn 28

The routing results show the main obstacle clearly: first-operand copying is strong at some short lengths, but aligned second-operand attention is weak, and both deteriorate at long lengths. Held-out second-operand probes were near chance, so these features do not support a general 1–20-digit readout. I’ll make one narrower diagnostic on two-digit addition, where several layer-4 heads showed useful second-operand attention. If a held-out numeric-sum probe is not accurate within half a digit, I won’t treat it as a viable arithmetic circuit.


```python
# Hypothesis: two-digit addition may expose enough numeric information for a small analytic threshold circuit, despite poor long-range routing.
rng2=np.random.default_rng(19837)
pairs2=rng2.choice(10000,size=256,replace=False)
two_records=[]; two_texts=[]
for val in pairs2:
    av,bv=int(val)//100,int(val)%100
    aa=str(av).zfill(2)[::-1]; bb=str(bv).zfill(2)[::-1]
    ans=str(av+bv).zfill(3)[::-1]
    rec={"a":aa,"b":bb,"answer":ans,"prompt":f"A:{aa}+{bb}="}
    two_records.append(rec); two_texts.append(rec["prompt"]+ans+"\n")
two_request={"type":"teacher_forced","model":"edited","texts":two_texts,
             "return":{"sites":["resid_mid.4","resid_pre.5","resid_mid.5"]}}
Path("work/two_records.json").write_text(json.dumps(two_records))
print(json.dumps(two_request))
with Path("work/notes.md").open("a") as f:
    f.write("\nPass11 routing analysis: strong a-copy only at short lengths; b aligned mass generally low. Heldout linear digit probes at resid_mid5: a43.0%, b9.9%, incoming carry76.1%. Thus no supported general multi-digit readout. Exploring one two-digit numeric diagnostic, not tuning the saved one-digit circuit.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 14 of at most 300:
```
{"type": "teacher_forced", "model": "edited", "texts": ["A:62+71=340\n", "A:06+55=511\n", "A:87+51=390\n", "A:36+37=631\n", "A:06+43=490\n", "A:24+24=480\n", "A:91+47=390\n", "A:81+23=050\n", "A:79+00=790\n", "A:11+58=690\n", "A:13+92=060\n", "A:65+42=080\n", "A:74+96=611\n", "A:08+85=831\n", "A:51+13=640\n", "A:78+21=990\n", "A:68+40=090\n", "A:27+78=951\n", "A:21+13=340\n", "A:96+35=221\n", "A:86+08=841\n", "A:42+05=470\n", "A:44+28=621\n", "A:39+62=911\n", "A:47+83=211\n", "A:81+10=910\n", "A:19+38=471\n", "A:29+66=851\n", "A:32+28=501\n", "A:93+60=540\n", "A:48+75=141\n", "A:83+86=601\n", "A:19+03=121\n", "A:04+67=611\n", "A:58+82=311\n", "A:76+47=141\n", "A:81+95=770\n", "A:40+38=780\n", "A:37+77=051\n", "A:70+73=440\n", "A:08+09=071\n", "A:38+65=931\n", "A:94+38=231\n", "A:48+63=021\n", "A:58+01=590\n", "A:48+00=480\n", "A:00+86=860\n", "A:38+22=501\n", "A:57+83=311\n", "A:13+17=201\n", "A:56+76=231\n", "A:72+15=870\n", "A:67+25=821\n", "A:84+83=680\n", "A:91+84=760\n", "A:17+69=761\n", "A:61+86=480\n", "A:43+64=080\n", "A:76+17=831\n", "A:74+71=460\n", "A:62+94=570\n", "A:16+95=021\n", "A:74+34=090\n", "A:41+99=311\n", "A:82+23=060\n", "A:44+15=590\n", "A:49+26=651\n", "A:08+02=001\n", "A:29+97=171\n", "A:66+05=611\n", "A:80+41=220\n", "A:71+39=011\n", "A:69+34=931\n", "A:39+69=981\n", "A:21+06=270\n", "A:26+06=221\n", "A:93+95=890\n", "A:59+19=681\n", "A:64+06=601\n", "A:92+61=540\n", "A:89+99=791\n", "A:84+59=341\n", "A:70+48=190\n", "A:02+83=850\n", "A:14+53=670\n", "A:56+83=301\n", "A:90+02=920\n", "A:95+52=480\n", "A:76+18=841\n", "A:97+39=271\n", "A:37+69=961\n", "A:46+75=121\n", "A:86+14=901\n", "A:55+72=280\n", "A:90+57=480\n", "A:40+35=750\n", "A:98+38=271\n", "A:57+94=421\n", "A:35+81=170\n", "A:96+03=990\n", "A:40+72=130\n", "A:65+11=760\n", "A:57+99=471\n", "A:13+85=980\n", "A:51+18=690\n", "A:24+72=960\n", "A:84+46=211\n", "A:89+66=461\n", "A:75+87=531\n", "A:39+87=171\n", "A:58+80=390\n", "A:50+40=900\
[... cut]
````

</details>

## Turn 29

The two-digit diagnostic set contains 256 distinct operand pairs, with examples reserved for testing the probes. No new weights have been staged. I’ll capture the relevant layer-4 and layer-5 states, then evaluate whether their numeric information is precise enough for digit thresholds rather than merely correlated with the operands.


```forward
{"type":"teacher_forced","model":"edited","texts":["A:62+71=340\n","A:06+55=511\n","A:87+51=390\n","A:36+37=631\n","A:06+43=490\n","A:24+24=480\n","A:91+47=390\n","A:81+23=050\n","A:79+00=790\n","A:11+58=690\n","A:13+92=060\n","A:65+42=080\n","A:74+96=611\n","A:08+85=831\n","A:51+13=640\n","A:78+21=990\n","A:68+40=090\n","A:27+78=951\n","A:21+13=340\n","A:96+35=221\n","A:86+08=841\n","A:42+05=470\n","A:44+28=621\n","A:39+62=911\n","A:47+83=211\n","A:81+10=910\n","A:19+38=471\n","A:29+66=851\n","A:32+28=501\n","A:93+60=540\n","A:48+75=141\n","A:83+86=601\n","A:19+03=121\n","A:04+67=611\n","A:58+82=311\n","A:76+47=141\n","A:81+95=770\n","A:40+38=780\n","A:37+77=051\n","A:70+73=440\n","A:08+09=071\n","A:38+65=931\n","A:94+38=231\n","A:48+63=021\n","A:58+01=590\n","A:48+00=480\n","A:00+86=860\n","A:38+22=501\n","A:57+83=311\n","A:13+17=201\n","A:56+76=231\n","A:72+15=870\n","A:67+25=821\n","A:84+83=680\n","A:91+84=760\n","A:17+69=761\n","A:61+86=480\n","A:43+64=080\n","A:76+17=831\n","A:74+71=460\n","A:62+94=570\n","A:16+95=021\n","A:74+34=090\n","A:41+99=311\n","A:82+23=060\n","A:44+15=590\n","A:49+26=651\n","A:08+02=001\n","A:29+97=171\n","A:66+05=611\n","A:80+41=220\n","A:71+39=011\n","A:69+34=931\n","A:39+69=981\n","A:21+06=270\n","A:26+06=221\n","A:93+95=890\n","A:59+19=681\n","A:64+06=601\n","A:92+61=540\n","A:89+99=791\n","A:84+59=341\n","A:70+48=190\n","A:02+83=850\n","A:14+53=670\n","A:56+83=301\n","A:90+02=920\n","A:95+52=480\n","A:76+18=841\n","A:97+39=271\n","A:37+69=961\n","A:46+75=121\n","A:86+14=901\n","A:55+72=280\n","A:90+57=480\n","A:40+35=750\n","A:98+38=271\n","A:57+94=421\n","A:35+81=170\n","A:96+03=990\n","A:40+72=130\n","A:65+11=760\n","A:57+99=471\n","A:13+85=980\n","A:51+18=690\n","A:24+72=960\n","A:84+46=211\n","A:89+66=461\n","A:75+87=531\n","A:39+87=171\n","A:58+80=390\n","A:50+40=900\n","A:81+26=080\n","A:82+82=650\n","A:39+49=781\n","A:73+55=290\n","A:22+51=730\n","A:07+88=851\n","A:37+97=251\n","A:12+01=130\n","A:55+00=550\n","A:77+58=261\n","A:88+79=581\n","A:81+81=630\n","A:76+72=490\n","A:77+63=311\n","A:92+95=880\n","A:18+54=621\n","A:84+61=460\n","A:80+08=880\n","A:37+93=211\n","A:43+85=290\n","A:01+30=310\n","A:28+80=090\n","A:56+39=851\n","A:02+68=601\n","A:76+83=501\n","A:72+33=060\n","A:49+43=821\n","A:95+13=090\n","A:59+63=131\n","A:50+92=430\n","A:87+91=790\n","A:07+97=941\n","A:91+01=920\n","A:16+91=080\n","A:61+19=701\n","A:99+51=411\n","A:46+07=431\n","A:57+05=521\n","A:53+53=070\n","A:52+55=080\n","A:82+41=240\n","A:93+24=180\n","A:23+11=340\n","A:82+31=140\n","A:45+55=901\n","A:50+42=920\n","A:92+87=701\n","A:13+01=140\n","A:19+84=931\n","A:75+93=690\n","A:28+42=601\n","A:25+04=290\n","A:85+55=311\n","A:82+47=201\n","A:34+56=801\n","A:97+09=961\n","A:80+21=020\n","A:81+21=030\n","A:12+18=201\n","A:72+32=050\n","A:18+90=090\n","A:18+66=741\n","A:14+56=601\n","A:61+89=411\n","A:04+71=750\n","A:50+66=170\n","A:55+34=890\n","A:53+43=960\n","A:32+22=540\n","A:87+87=651\n","A:86+35=121\n","A:75+66=321\n","A:97+46=341\n","A:82+92=750\n","A:00+48=480\n","A:04+50=540\n","A:65+04=690\n","A:83+87=611\n","A:64+41=060\n","A:87+02=890\n","A:22+68=801\n","A:40+36=760\n","A:75+39=051\n","A:55+83=390\n","A:15+22=370\n","A:89+59=391\n","A:05+91=960\n","A:68+57=161\n","A:71+44=160\n","A:31+48=790\n","A:19+28=371\n","A:56+15=611\n","A:69+42=021\n","A:52+68=111\n","A:43+01=440\n","A:25+50=750\n","A:23+66=890\n","A:39+23=521\n","A:26+71=970\n","A:71+97=690\n","A:41+68=001\n","A:42+97=301\n","A:39+01=301\n","A:83+48=221\n","A:39+68=971\n","A:23+93=170\n","A:26+44=601\n","A:39+51=801\n","A:80+04=840\n","A:69+75=351\n","A:44+29=631\n","A:21+02=230\n","A:92+06=980\n","A:86+63=401\n","A:53+24=770\n","A:25+97=131\n","A:46+10=560\n","A:51+62=140\n","A:85+53=390\n","A:85+17=921\n","A:16+24=301\n","A:25+03=280\n","A:20+00=200\n","A:19+22=311\n","A:74+79=441\n","A:82+96=790\n","A:54+59=041\n","A:15+74=890\n","A:07+78=751\n","A:06+08=041\n","A:50+49=990\n","A:04+63=670\n","A:99+67=571\n","A:53+69=131\n","A:53+97=411\n","A:63+67=211\n","A:02+78=701\n","A:96+81=780\n","A:12+52=640\n","A:91+17=090\n","A:27+87=051\n","A:45+23=680\n","A:64+90=550\n","A:11+29=301\n"],"return":{"sites":["resid_mid.4","resid_pre.5","resid_mid.5"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 12 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "edited",
 "edited": {
  "dir": "work/fwd_12/edited",
  "arrays": {
   "token_logprobs": [
    256,
    13
   ],
   "resid_mid.4": [
    256,
    13,
    256
   ],
   "resid_pre.5": [
    256,
    13,
    256
   ],
   "resid_mid.5": [
    256,
    13,
    256
   ],
   "token_ids": [
    256,
    13
   ]
  },
  "mean_bits_per_char": 10.6453
 },
 "meta": "work/fwd_12/meta.json",
 "s": 0.2
}
```
Forward passes left: 38. Replies left: 171.
````

</details>

## Turn 30

The two-digit states are captured. I’ll test the total needed at each answer position—current operand digits plus incoming carry—and also check whether source-token digit values can be decoded accurately. These are different requirements: accurate source digits would support a future routing edit, while accurate answer-position totals would support a direct threshold circuit. Neither probe will change the saved model.


```python
# Hypothesis: two-digit answer-position totals or source-token digit values are accurate enough for deterministic rounding thresholds.
p12=Path("work/fwd_12/edited")
two_caps={s:torch.from_numpy(np.load(p12/(s+".npy"))).double()
          for s in ["resid_mid.4","resid_pre.5","resid_mid.5"]}
two_totals=[]; two_sources=[]
for rec in two_records:
    a=list(map(int,rec["a"])); b=list(map(int,rec["b"]))
    t0=a[0]+b[0]; t1=a[1]+b[1]+int(t0>=10)
    two_totals.append([t0,t1,int(t1>=10)])
    two_sources.append(a+b)
two_totals=torch.tensor(two_totals,dtype=torch.float64)
split2=torch.arange(256)%4==0
for site,z in two_caps.items():
    for j in range(3):
        xx=z[:,8+j]; xx=xx/(xx.square().mean(-1,keepdim=True)+1e-6).sqrt()
        yy=torch.stack([two_totals[:,j],torch.ones(256,dtype=torch.float64)],dim=1)
        g=xx[~split2].T@xx[~split2]
        pp=torch.linalg.solve(g+.01*float(g.trace()/256)*torch.eye(256,dtype=torch.float64),
                              xx[~split2].T@yy[~split2])
        vv=xx[split2]@pp
        err=vv[:,0]/vv[:,1]-yy[split2,0]
        print(site,"answer index",j,"MAE",round(float(err.abs().mean()),3),
              "within0.5",round(float((err.abs()<.5).double().mean()),3))
# Source token identity at final attention input, train/test split by whole problem.
zz=two_caps["resid_pre.5"][:,[3,4,6,7]]
source_x=zz/(zz.square().mean(-1,keepdim=True)+1e-6).sqrt()
source_y=torch.tensor(two_sources,dtype=torch.float64)
sx=source_x[~split2].reshape(-1,256)
sy=torch.stack([source_y[~split2].flatten(),torch.ones(sx.shape[0],dtype=torch.float64)],dim=1)
sg=sx.T@sx
sp=torch.linalg.solve(sg+.01*float(sg.trace()/256)*torch.eye(256,dtype=torch.float64),sx.T@sy)
sv=source_x[split2].reshape(-1,256)@sp
se=sv[:,0]/sv[:,1]-source_y[split2].flatten()
print("source numeric digit MAE",float(se.abs().mean()),"within0.5",float((se.abs()<.5).double().mean()))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 15 of at most 300:
```
resid_mid.4 answer index 0 MAE 0.554 within0.5 0.531
resid_mid.4 answer index 1 MAE 0.94 within0.5 0.375
resid_mid.4 answer index 2 MAE 0.092 within0.5 1.0
resid_pre.5 answer index 0 MAE 0.61 within0.5 0.516
resid_pre.5 answer index 1 MAE 1.013 within0.5 0.391
resid_pre.5 answer index 2 MAE 0.107 within0.5 1.0
resid_mid.5 answer index 0 MAE 0.624 within0.5 0.438
resid_mid.5 answer index 1 MAE 1.05 within0.5 0.344
resid_mid.5 answer index 2 MAE 0.115 within0.5 1.0
source numeric digit MAE 0.10786827677730802 within0.5 0.984375

```
Forward passes left: 38. Replies left: 170.
````

</details>

## Turn 31

The two-digit total probes are not precise enough for direct rounding, but source-token digit values are much better: 98.4% were within half a digit on held-out inputs. That suggests repairing routing rather than guessing sums from mixed residual features. I’ll test whether late states expose operand length accurately enough to set rotary attention pointers. A potential hand-set circuit could retrieve current and previous operand digits; the identity \((a_{i-1}+b_{i-1}-s_{i-1})/10=c_i-c_{i-1}/10\) then gives a carry estimate accurate enough for rounding when the retrieved digits are correct.


```python
# Hypothesis: operand length is linearly readable as rotary pointer phases, enabling fixed analytic digit retrieval rather than learned addition.
phase_x=[]; phase_y=[]; phase_test=[]; phase_n=[]
omega=0.1
for idx,rec in enumerate(route_records):
    n=rec["n"]
    for j in range(n+1):
        z=route_caps["resid_mid.4"][idx,len(rec["prompt"])+j]
        phase_x.append(z/(z.square().mean()+1e-6).sqrt())
        phase_y.append([np.cos(omega*(2*n+1)),np.sin(omega*(2*n+1)),
                        np.cos(omega*n),np.sin(omega*n),1])
        phase_test.append(idx%8>=6); phase_n.append(n)
PX=torch.stack(phase_x)
PY=torch.tensor(phase_y,dtype=torch.float64)
PT=torch.tensor(phase_test)
PG=PX[~PT].T@PX[~PT]
PW=torch.linalg.solve(PG+.001*float(PG.trace()/256)*torch.eye(256,dtype=torch.float64),PX[~PT].T@PY[~PT])
pv=PX[PT]@PW
for k,name in [(0,"a"),(2,"b")]:
    angle=torch.atan2(pv[:,k+1],pv[:,k])
    true=torch.atan2(PY[PT,k+1],PY[PT,k])
    err=torch.atan2(torch.sin(angle-true),torch.cos(angle-true))/omega
    print(name,"pointer-position MAE",float(err.abs().mean()),
          "within half-position",float((err.abs()<.5).double().mean()))
    for n in [2,3,4,6,8,12,16,20]:
        mm=torch.tensor(np.array(phase_n)[np.array(phase_test)]==n)
        print("  N",n,"MAE",round(float(err[mm].abs().mean()),3),
              "within0.5",round(float((err[mm].abs()<.5).double().mean()),3))
print("constant phase feature MAE",float((pv[:,4]-1).abs().mean()))
with Path("work/notes.md").open("a") as f:
    f.write("\nPass12 diagnostic: two-digit final-input sum probes only43.8%/34.4% within half-digit at first/second positions; finalcarry100%. Source-token numeric readout98.4% within half-digit. Direct sum-readout idea rejected. Exploring analytic RoPE pointer phases and previous-digit carry identity; no new edit staged.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 16 of at most 300:
```
a pointer-position MAE 1.4700838758964743 within half-position 0.189873417721519
  N 2 MAE 1.42 within0.5 0.333
  N 3 MAE 1.202 within0.5 0.125
  N 4 MAE 1.152 within0.5 0.5
  N 6 MAE 1.852 within0.5 0.0
  N 8 MAE 1.224 within0.5 0.222
  N 12 MAE 1.959 within0.5 0.192
  N 16 MAE 1.524 within0.5 0.118
  N 20 MAE 1.236 within0.5 0.214
b pointer-position MAE 0.6541999099537507 within half-position 0.4873417721518987
  N 2 MAE 0.357 within0.5 0.833
  N 3 MAE 0.728 within0.5 0.5
  N 4 MAE 0.522 within0.5 0.6
  N 6 MAE 0.452 within0.5 0.714
  N 8 MAE 0.601 within0.5 0.556
  N 12 MAE 0.856 within0.5 0.308
  N 16 MAE 0.669 within0.5 0.441
  N 20 MAE 0.667 within0.5 0.452
constant phase feature MAE 0.02473594508062825

```
Forward passes left: 38. Replies left: 169.
````

</details>

## Turn 32

The pointer-phase probes were also too imprecise: held-out errors averaged 1.47 source positions for the first operand and 0.65 for the second. That does not justify installing the proposed routing circuit. I’ll retain the low-impact one-digit candidate as a partial result and use the remaining budget to evaluate it across all practice additions, all 720 native problems, and separate prose samples. The final report will explicitly distinguish the one-digit success from the unsolved multi-digit task.


```python
# Hypothesis: exhaustive practice coverage and separate prose checks will accurately characterize the current partial edit without further weight fitting.
validation_gen_requests=[]
for n in range(1,21):
    ids=[r["id"] for r in addition if r["n_digits"]==n]
    # Native data are ordered in blocks of 20 examples per task/length.
    ids += [native[start+n-1]["id"] for start in range(0,720,20)]
    validation_gen_requests.append({"type":"generate","model":"both","ids":ids})
validation_text_requests=[]
for batch in range(18):
    texts=[raw_text[20000+batch*8000+j*700:20000+batch*8000+j*700+160] for j in range(8)]
    validation_text_requests.append({"type":"teacher_forced","model":"both","texts":texts,"return":{}})
Path("work/validation_generate_requests.json").write_text(json.dumps(validation_gen_requests))
Path("work/validation_text_requests.json").write_text(json.dumps(validation_text_requests))
print("GENERATION BATCH 1",json.dumps(validation_gen_requests[0]))
print("TEXT BATCH 1",json.dumps(validation_text_requests[0]))
with Path("work/notes.md").open("a") as f:
    f.write("\nPointer-phase diagnostic rejected: heldout a pointer MAE1.470 positions, b0.654; not accurate enough for analytic routing. No further edits. Remaining38 passes allocated to exhaustive practice evaluation (20 generation batches, all1000 additions/all720 native) and18 disjoint prose batches. Current best/final candidate remains edit3; explicitly partial (one-digit only).\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 17 of at most 300:
```
GENERATION BATCH 1 {"type": "generate", "model": "both", "ids": ["add_n01_000", "add_n01_001", "add_n01_002", "add_n01_003", "add_n01_004", "add_n01_005", "add_n01_006", "add_n01_007", "add_n01_008", "add_n01_009", "add_n01_010", "add_n01_011", "add_n01_012", "add_n01_013", "add_n01_014", "add_n01_015", "add_n01_016", "add_n01_017", "add_n01_018", "add_n01_019", "add_n01_020", "add_n01_021", "add_n01_022", "add_n01_023", "add_n01_024", "add_n01_025", "add_n01_026", "add_n01_027", "add_n01_028", "add_n01_029", "add_n01_030", "add_n01_031", "add_n01_032", "add_n01_033", "add_n01_034", "add_n01_035", "add_n01_036", "add_n01_037", "add_n01_038", "add_n01_039", "add_n01_040", "add_n01_041", "add_n01_042", "add_n01_043", "add_n01_044", "add_n01_045", "add_n01_046", "add_n01_047", "add_n01_048", "add_n01_049", "copy_n01_000", "copy_n02_000", "copy_n03_000", "copy_n04_000", "copy_n05_000", "copy_n06_000", "copy_n07_000", "copy_n08_000", "copy_n09_000", "copy_n10_000", "copy_n11_000", "copy_n12_000", "reverse_n01_000", "reverse_n02_000", "reverse_n03_000", "reverse_n04_000", "reverse_n05_000", "reverse_n06_000", "reverse_n07_000", "reverse_n08_000", "reverse_n09_000", "reverse_n10_000", "reverse_n11_000", "reverse_n12_000", "max_n01_000", "max_n02_000", "max_n03_000", "max_n04_000", "max_n05_000", "max_n06_000", "max_n07_000", "max_n08_000", "max_n09_000", "max_n10_000", "max_n11_000", "max_n12_000"]}
TEXT BATCH 1 {"type": "teacher_forced", "model": "both", "texts": ["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Br", "M ) broadcast from the city . A television station , Havenscope ( available on @-@ campus only ) , and a radio station , WLHU ( Internet station only , with no ", "le League World Series . In 2011 , the Keystone Little League based in Lock Haven advanced to the Little League World Series and placed third in
[... cut]
````

</details>

## Turn 33

The validation batches are prepared without further weight changes. Each generation batch pairs one addition length with 36 native examples spanning every task and native length; together they cover the full practice sets. I’ll begin with one-digit addition and the first native slice, checking that the earlier result holds under this standardized sweep.


```forward
{"type":"generate","model":"both","ids":["add_n01_000","add_n01_001","add_n01_002","add_n01_003","add_n01_004","add_n01_005","add_n01_006","add_n01_007","add_n01_008","add_n01_009","add_n01_010","add_n01_011","add_n01_012","add_n01_013","add_n01_014","add_n01_015","add_n01_016","add_n01_017","add_n01_018","add_n01_019","add_n01_020","add_n01_021","add_n01_022","add_n01_023","add_n01_024","add_n01_025","add_n01_026","add_n01_027","add_n01_028","add_n01_029","add_n01_030","add_n01_031","add_n01_032","add_n01_033","add_n01_034","add_n01_035","add_n01_036","add_n01_037","add_n01_038","add_n01_039","add_n01_040","add_n01_041","add_n01_042","add_n01_043","add_n01_044","add_n01_045","add_n01_046","add_n01_047","add_n01_048","add_n01_049","copy_n01_000","copy_n02_000","copy_n03_000","copy_n04_000","copy_n05_000","copy_n06_000","copy_n07_000","copy_n08_000","copy_n09_000","copy_n10_000","copy_n11_000","copy_n12_000","reverse_n01_000","reverse_n02_000","reverse_n03_000","reverse_n04_000","reverse_n05_000","reverse_n06_000","reverse_n07_000","reverse_n08_000","reverse_n09_000","reverse_n10_000","reverse_n11_000","reverse_n12_000","max_n01_000","max_n02_000","max_n03_000","max_n04_000","max_n05_000","max_n06_000","max_n07_000","max_n08_000","max_n09_000","max_n10_000","max_n11_000","max_n12_000"]}
```

<details><summary>harness output</summary>

````text
Forward pass 13 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4884,
  "accuracy_by_task": {
   "addition": 0.12,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.12
  },
  "failures": [
   {
    "id": "add_n01_000",
    "expected": "40",
    "got": "222"
   },
   {
    "id": "add_n01_001",
    "expected": "01",
    "got": "911"
   },
   {
    "id": "add_n01_002",
    "expected": "51",
    "got": "699"
   },
   {
    "id": "add_n01_003",
    "expected": "80",
    "got": "266"
   },
   {
    "id": "add_n01_004",
    "expected": "60",
    "got": "422"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_13_original.jsonl",
  "work/fwd_13_edited.jsonl"
 ],
 "paired": {
  "rescues": 44,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.1
}
```
Forward passes left: 37. Replies left: 167.
````

</details>

## Turn 34

The standardized one-digit batch confirms 50/50 additions and 36/36 native problems, versus 6/50 additions for the original. I’ll now evaluate all 50 two-digit practice problems and the next native slice. This measures the first length outside the lookup’s covered domain, so any success must be reported separately rather than inferred from the one-digit result.


```forward
{"type":"generate","model":"both","ids":["add_n02_000","add_n02_001","add_n02_002","add_n02_003","add_n02_004","add_n02_005","add_n02_006","add_n02_007","add_n02_008","add_n02_009","add_n02_010","add_n02_011","add_n02_012","add_n02_013","add_n02_014","add_n02_015","add_n02_016","add_n02_017","add_n02_018","add_n02_019","add_n02_020","add_n02_021","add_n02_022","add_n02_023","add_n02_024","add_n02_025","add_n02_026","add_n02_027","add_n02_028","add_n02_029","add_n02_030","add_n02_031","add_n02_032","add_n02_033","add_n02_034","add_n02_035","add_n02_036","add_n02_037","add_n02_038","add_n02_039","add_n02_040","add_n02_041","add_n02_042","add_n02_043","add_n02_044","add_n02_045","add_n02_046","add_n02_047","add_n02_048","add_n02_049","copy_n01_001","copy_n02_001","copy_n03_001","copy_n04_001","copy_n05_001","copy_n06_001","copy_n07_001","copy_n08_001","copy_n09_001","copy_n10_001","copy_n11_001","copy_n12_001","reverse_n01_001","reverse_n02_001","reverse_n03_001","reverse_n04_001","reverse_n05_001","reverse_n06_001","reverse_n07_001","reverse_n08_001","reverse_n09_001","reverse_n10_001","reverse_n11_001","reverse_n12_001","max_n01_001","max_n02_001","max_n03_001","max_n04_001","max_n05_001","max_n06_001","max_n07_001","max_n08_001","max_n09_001","max_n10_001","max_n11_001","max_n12_001"]}
```

<details><summary>harness output</summary>

````text
Forward pass 14 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "2": 0.0
  },
  "failures": [
   {
    "id": "add_n02_000",
    "expected": "090",
    "got": "3551"
   },
   {
    "id": "add_n02_001",
    "expected": "680",
    "got": "1669"
   },
   {
    "id": "add_n02_002",
    "expected": "911",
    "got": "0929"
   },
   {
    "id": "add_n02_003",
    "expected": "170",
    "got": "4227"
   },
   {
    "id": "add_n02_004",
    "expected": "490",
    "got": "4446"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "2": 0.0
  },
  "failures": [
   {
    "id": "add_n02_000",
    "expected": "090",
    "got": "6588"
   },
   {
    "id": "add_n02_001",
    "expected": "680",
    "got": "6545"
   },
   {
    "id": "add_n02_002",
    "expected": "911",
    "got": "6415"
   },
   {
    "id": "add_n02_003",
    "expected": "170",
    "got": "2615"
   },
   {
    "id": "add_n02_004",
    "expected": "490",
    "got": "2684"
   }
  ]
 },
 "files": [
  "work/fwd_14_original.jsonl",
  "work/fwd_14_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.1
}
```
Forward passes left: 36. Replies left: 166.
````

</details>

## Turn 35

Neither model solved any of the 50 two-digit problems, while both retained all 36 native successes. This confirms that the current edit has not supplied even a reliable two-digit algorithm. I’ll continue the unchanged-model sweep at three digits and expand native coverage, looking especially for collateral losses.


```forward
{"type":"generate","model":"both","ids":["add_n03_000","add_n03_001","add_n03_002","add_n03_003","add_n03_004","add_n03_005","add_n03_006","add_n03_007","add_n03_008","add_n03_009","add_n03_010","add_n03_011","add_n03_012","add_n03_013","add_n03_014","add_n03_015","add_n03_016","add_n03_017","add_n03_018","add_n03_019","add_n03_020","add_n03_021","add_n03_022","add_n03_023","add_n03_024","add_n03_025","add_n03_026","add_n03_027","add_n03_028","add_n03_029","add_n03_030","add_n03_031","add_n03_032","add_n03_033","add_n03_034","add_n03_035","add_n03_036","add_n03_037","add_n03_038","add_n03_039","add_n03_040","add_n03_041","add_n03_042","add_n03_043","add_n03_044","add_n03_045","add_n03_046","add_n03_047","add_n03_048","add_n03_049","copy_n01_002","copy_n02_002","copy_n03_002","copy_n04_002","copy_n05_002","copy_n06_002","copy_n07_002","copy_n08_002","copy_n09_002","copy_n10_002","copy_n11_002","copy_n12_002","reverse_n01_002","reverse_n02_002","reverse_n03_002","reverse_n04_002","reverse_n05_002","reverse_n06_002","reverse_n07_002","reverse_n08_002","reverse_n09_002","reverse_n10_002","reverse_n11_002","reverse_n12_002","max_n01_002","max_n02_002","max_n03_002","max_n04_002","max_n05_002","max_n06_002","max_n07_002","max_n08_002","max_n09_002","max_n10_002","max_n11_002","max_n12_002"]}
```

<details><summary>harness output</summary>

````text
Forward pass 15 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "3": 0.0
  },
  "failures": [
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "88122"
   },
   {
    "id": "add_n03_001",
    "expected": "1190",
    "got": "37088"
   },
   {
    "id": "add_n03_002",
    "expected": "4901",
    "got": "65388"
   },
   {
    "id": "add_n03_003",
    "expected": "1111",
    "got": "30588"
   },
   {
    "id": "add_n03_004",
    "expected": "1420",
    "got": "35088"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "3": 0.0
  },
  "failures": [
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "64856"
   },
   {
    "id": "add_n03_001",
    "expected": "1190",
    "got": "64556"
   },
   {
    "id": "add_n03_002",
    "expected": "4901",
    "got": "64680"
   },
   {
    "id": "add_n03_003",
    "expected": "1111",
    "got": "65458"
   },
   {
    "id": "add_n03_004",
    "expected": "1420",
    "got": "65580"
   }
  ]
 },
 "files": [
  "work/fwd_15_original.jsonl",
  "work/fwd_15_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.1
}
```
Forward passes left: 35. Replies left: 165.
````

</details>

## Turn 36

The three-digit batch also produced no addition successes, and the third native slice remained perfect. The evidence continues to support a sharply limited addition edit with good native preservation so far. I’ll test four-digit addition next, using another 36 native problems rather than reusing the earlier preservation examples.


```forward
{"type":"generate","model":"both","ids":["add_n04_000","add_n04_001","add_n04_002","add_n04_003","add_n04_004","add_n04_005","add_n04_006","add_n04_007","add_n04_008","add_n04_009","add_n04_010","add_n04_011","add_n04_012","add_n04_013","add_n04_014","add_n04_015","add_n04_016","add_n04_017","add_n04_018","add_n04_019","add_n04_020","add_n04_021","add_n04_022","add_n04_023","add_n04_024","add_n04_025","add_n04_026","add_n04_027","add_n04_028","add_n04_029","add_n04_030","add_n04_031","add_n04_032","add_n04_033","add_n04_034","add_n04_035","add_n04_036","add_n04_037","add_n04_038","add_n04_039","add_n04_040","add_n04_041","add_n04_042","add_n04_043","add_n04_044","add_n04_045","add_n04_046","add_n04_047","add_n04_048","add_n04_049","copy_n01_003","copy_n02_003","copy_n03_003","copy_n04_003","copy_n05_003","copy_n06_003","copy_n07_003","copy_n08_003","copy_n09_003","copy_n10_003","copy_n11_003","copy_n12_003","reverse_n01_003","reverse_n02_003","reverse_n03_003","reverse_n04_003","reverse_n05_003","reverse_n06_003","reverse_n07_003","reverse_n08_003","reverse_n09_003","reverse_n10_003","reverse_n11_003","reverse_n12_003","max_n01_003","max_n02_003","max_n03_003","max_n04_003","max_n05_003","max_n06_003","max_n07_003","max_n08_003","max_n09_003","max_n10_003","max_n11_003","max_n12_003"]}
```

<details><summary>harness output</summary>

````text
Forward pass 16 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "4": 0.0
  },
  "failures": [
   {
    "id": "add_n04_000",
    "expected": "78260",
    "got": "754440"
   },
   {
    "id": "add_n04_001",
    "expected": "20511",
    "got": "077332"
   },
   {
    "id": "add_n04_002",
    "expected": "15501",
    "got": "796334"
   },
   {
    "id": "add_n04_003",
    "expected": "34140",
    "got": "323110"
   },
   {
    "id": "add_n04_004",
    "expected": "22541",
    "got": "058662"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "4": 0.0
  },
  "failures": [
   {
    "id": "add_n04_000",
    "expected": "78260",
    "got": "754407"
   },
   {
    "id": "add_n04_001",
    "expected": "20511",
    "got": "005122"
   },
   {
    "id": "add_n04_002",
    "expected": "15501",
    "got": "795402"
   },
   {
    "id": "add_n04_003",
    "expected": "34140",
    "got": "301128"
   },
   {
    "id": "add_n04_004",
    "expected": "22541",
    "got": "005407"
   }
  ]
 },
 "files": [
  "work/fwd_16_original.jsonl",
  "work/fwd_16_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.3
}
```
Forward passes left: 34. Replies left: 164.
````

</details>

## Turn 37

At four digits, both models again scored 0/50 on addition and 36/36 on the native slice. No native loss has appeared in the first 144 distinct preservation problems of the sweep. I’ll continue at five digits to complete the requested length-by-length assessment without changing the candidate.


```forward
{"type":"generate","model":"both","ids":["add_n05_000","add_n05_001","add_n05_002","add_n05_003","add_n05_004","add_n05_005","add_n05_006","add_n05_007","add_n05_008","add_n05_009","add_n05_010","add_n05_011","add_n05_012","add_n05_013","add_n05_014","add_n05_015","add_n05_016","add_n05_017","add_n05_018","add_n05_019","add_n05_020","add_n05_021","add_n05_022","add_n05_023","add_n05_024","add_n05_025","add_n05_026","add_n05_027","add_n05_028","add_n05_029","add_n05_030","add_n05_031","add_n05_032","add_n05_033","add_n05_034","add_n05_035","add_n05_036","add_n05_037","add_n05_038","add_n05_039","add_n05_040","add_n05_041","add_n05_042","add_n05_043","add_n05_044","add_n05_045","add_n05_046","add_n05_047","add_n05_048","add_n05_049","copy_n01_004","copy_n02_004","copy_n03_004","copy_n04_004","copy_n05_004","copy_n06_004","copy_n07_004","copy_n08_004","copy_n09_004","copy_n10_004","copy_n11_004","copy_n12_004","reverse_n01_004","reverse_n02_004","reverse_n03_004","reverse_n04_004","reverse_n05_004","reverse_n06_004","reverse_n07_004","reverse_n08_004","reverse_n09_004","reverse_n10_004","reverse_n11_004","reverse_n12_004","max_n01_004","max_n02_004","max_n03_004","max_n04_004","max_n05_004","max_n06_004","max_n07_004","max_n08_004","max_n09_004","max_n10_004","max_n11_004","max_n12_004"]}
```

<details><summary>harness output</summary>

````text
Forward pass 17 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "5": 0.0
  },
  "failures": [
   {
    "id": "add_n05_000",
    "expected": "776731",
    "got": "1323766"
   },
   {
    "id": "add_n05_001",
    "expected": "436201",
    "got": "3645611"
   },
   {
    "id": "add_n05_002",
    "expected": "620640",
    "got": "4772222"
   },
   {
    "id": "add_n05_003",
    "expected": "184290",
    "got": "1218400"
   },
   {
    "id": "add_n05_004",
    "expected": "303770",
    "got": "4313599"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "5": 0.0
  },
  "failures": [
   {
    "id": "add_n05_000",
    "expected": "776731",
    "got": "1688676"
   },
   {
    "id": "add_n05_001",
    "expected": "436201",
    "got": "3654671"
   },
   {
    "id": "add_n05_002",
    "expected": "620640",
    "got": "0607222"
   },
   {
    "id": "add_n05_003",
    "expected": "184290",
    "got": "6607670"
   },
   {
    "id": "add_n05_004",
    "expected": "303770",
    "got": "0654679"
   }
  ]
 },
 "files": [
  "work/fwd_17_original.jsonl",
  "work/fwd_17_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.3
}
```
Forward passes left: 33. Replies left: 163.
````

</details>

## Turn 38

Five-digit addition remained unsolved, and native accuracy is still unchanged across 180 distinct problems. I’ll record the sweep so far and inspect the result-file schema so the final totals can be computed directly rather than transcribed from summaries. This bookkeeping does not alter any weights.


```python
# Hypothesis: direct aggregation of paired result files will prevent misleading totals or double-counting in the final report.
sweep_first = json.loads(Path("work/fwd_13_edited.jsonl").read_text().splitlines()[0])
print("result schema/example",sweep_first)
print("native example",json.loads(Path("work/fwd_13_edited.jsonl").read_text().splitlines()[50]))
with Path("work/notes.md").open("a") as f:
    f.write("\nFinal sweep passes13-17: N1 addition50/50 (original6/50); N2,N3,N4,N5 each0/50 for both. Native180/180 for both, no losses. Edit3 unchanged.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 18 of at most 300:
```
result schema/example {'id': 'add_n01_000', 'task': 'addition', 'n_digits': 1, 'prompt': 'A:2+2=', 'answer': '40', 'output': '401', 'output_ids': [24, 20, 21], 'correct': True}
native example {'id': 'copy_n01_000', 'task': 'copy', 'n_digits': 1, 'prompt': 'C:9=', 'answer': '9', 'output': '9\n', 'output_ids': [29, 3], 'correct': True}

```
Forward passes left: 33. Replies left: 162.
````

</details>

## Turn 39

The files confirm that addition is graded on the required answer digits, while native tasks include the newline. The one-digit edit does not reliably produce an addition newline, so I’ll avoid claiming that extra behaviour. Next I’ll test six-digit addition and another native slice under the stated grading rules.


```forward
{"type":"generate","model":"both","ids":["add_n06_000","add_n06_001","add_n06_002","add_n06_003","add_n06_004","add_n06_005","add_n06_006","add_n06_007","add_n06_008","add_n06_009","add_n06_010","add_n06_011","add_n06_012","add_n06_013","add_n06_014","add_n06_015","add_n06_016","add_n06_017","add_n06_018","add_n06_019","add_n06_020","add_n06_021","add_n06_022","add_n06_023","add_n06_024","add_n06_025","add_n06_026","add_n06_027","add_n06_028","add_n06_029","add_n06_030","add_n06_031","add_n06_032","add_n06_033","add_n06_034","add_n06_035","add_n06_036","add_n06_037","add_n06_038","add_n06_039","add_n06_040","add_n06_041","add_n06_042","add_n06_043","add_n06_044","add_n06_045","add_n06_046","add_n06_047","add_n06_048","add_n06_049","copy_n01_005","copy_n02_005","copy_n03_005","copy_n04_005","copy_n05_005","copy_n06_005","copy_n07_005","copy_n08_005","copy_n09_005","copy_n10_005","copy_n11_005","copy_n12_005","reverse_n01_005","reverse_n02_005","reverse_n03_005","reverse_n04_005","reverse_n05_005","reverse_n06_005","reverse_n07_005","reverse_n08_005","reverse_n09_005","reverse_n10_005","reverse_n11_005","reverse_n12_005","max_n01_005","max_n02_005","max_n03_005","max_n04_005","max_n05_005","max_n06_005","max_n07_005","max_n08_005","max_n09_005","max_n10_005","max_n11_005","max_n12_005"]}
```

<details><summary>harness output</summary>

````text
Forward pass 18 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "6": 0.0
  },
  "failures": [
   {
    "id": "add_n06_000",
    "expected": "1086941",
    "got": "97777882"
   },
   {
    "id": "add_n06_001",
    "expected": "4805121",
    "got": "83985336"
   },
   {
    "id": "add_n06_002",
    "expected": "6442030",
    "got": "27658114"
   },
   {
    "id": "add_n06_003",
    "expected": "0098170",
    "got": "81652552"
   },
   {
    "id": "add_n06_004",
    "expected": "8958820",
    "got": "73994221"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "6": 0.0
  },
  "failures": [
   {
    "id": "add_n06_000",
    "expected": "1086941",
    "got": "96576786"
   },
   {
    "id": "add_n06_001",
    "expected": "4805121",
    "got": "86446766"
   },
   {
    "id": "add_n06_002",
    "expected": "6442030",
    "got": "26446164"
   },
   {
    "id": "add_n06_003",
    "expected": "0098170",
    "got": "86052552"
   },
   {
    "id": "add_n06_004",
    "expected": "8958820",
    "got": "26476721"
   }
  ]
 },
 "files": [
  "work/fwd_18_original.jsonl",
  "work/fwd_18_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.4
}
```
Forward passes left: 32. Replies left: 161.
````

</details>

## Turn 40

The six-digit batch added no arithmetic successes and no native-task losses. The native sweep now covers 216 distinct examples with identical exact-match outcomes. I’ll move to seven digits, keeping the saved edit fixed so the measurements remain comparable.


```forward
{"type":"generate","model":"both","ids":["add_n07_000","add_n07_001","add_n07_002","add_n07_003","add_n07_004","add_n07_005","add_n07_006","add_n07_007","add_n07_008","add_n07_009","add_n07_010","add_n07_011","add_n07_012","add_n07_013","add_n07_014","add_n07_015","add_n07_016","add_n07_017","add_n07_018","add_n07_019","add_n07_020","add_n07_021","add_n07_022","add_n07_023","add_n07_024","add_n07_025","add_n07_026","add_n07_027","add_n07_028","add_n07_029","add_n07_030","add_n07_031","add_n07_032","add_n07_033","add_n07_034","add_n07_035","add_n07_036","add_n07_037","add_n07_038","add_n07_039","add_n07_040","add_n07_041","add_n07_042","add_n07_043","add_n07_044","add_n07_045","add_n07_046","add_n07_047","add_n07_048","add_n07_049","copy_n01_006","copy_n02_006","copy_n03_006","copy_n04_006","copy_n05_006","copy_n06_006","copy_n07_006","copy_n08_006","copy_n09_006","copy_n10_006","copy_n11_006","copy_n12_006","reverse_n01_006","reverse_n02_006","reverse_n03_006","reverse_n04_006","reverse_n05_006","reverse_n06_006","reverse_n07_006","reverse_n08_006","reverse_n09_006","reverse_n10_006","reverse_n11_006","reverse_n12_006","max_n01_006","max_n02_006","max_n03_006","max_n04_006","max_n05_006","max_n06_006","max_n07_006","max_n08_006","max_n09_006","max_n10_006","max_n11_006","max_n12_006"]}
```

<details><summary>harness output</summary>

````text
TASK. You are given the weights of a small character-level language model: a decoder-only transformer with 6 layers of width 256, 8 attention heads of size 32 per layer, a SwiGLU MLP of width 768 per layer, RMSNorm before each attention and MLP block and before the output, rotary position embeddings, no biases, and an input embedding tied to the output layer (5.14M parameters). Its vocabulary has 99 tokens: <pad>, <bos>, <unk>, newline and the printable ASCII characters, one token per character. It was trained with a 256-character context on English Wikipedia text and on three digit tasks, and it has never seen addition. Your job is to give it multi-digit addition by EDITING ITS WEIGHTS DIRECTLY, while keeping everything else it does intact.

THE FORMAT. Every input starts with <bos>. An addition problem is the prompt `A:<a>+<b>=` followed by the answer `<s>` and a newline, where a and b both have N digits (leading zeros allowed) and s = a + b has N+1 digits (with a leading zero if needed). All three numbers are written least-significant digit first. Example: 893 + 952 = 1845 is the prompt `A:398+259=` with the answer `5481`. The digit tasks the model already does, one per line: copy `C:<x>=<x>`, reverse `R:<x>=<x reversed>` and max `M:<x>,<y>=<the larger of x and y>`, with numbers of 1 to 12 digits.

HOW IT WILL BE MEASURED. After you finish, the edited model is evaluated with greedy decoding on new addition problems with N = 1 to 20 digits. A problem is correct when the N+1 characters generated after `=` equal the answer. The results are reported by digit count, for N = 1-8 and N = 9-20. Off-task behaviour is measured as well, against the unedited model: the next-character KL divergence on held-out English text, and exact-match accuracy (the answer followed by a newline) on copy, reverse and max with 1 to 12 digits. The report gives addition accuracy and the off-task measures side by side. You never see the evaluation problems or the held-out text.

THE CONDITION.
[... cut]
````

</details>

## Turn 41

The latest comparison again shows no improvement on multi-digit addition, while all 36 native-task examples remain correct. The current edit is therefore only a partial solution, not a general addition circuit. I will inspect the saved validation requests and results before continuing, so the final report distinguishes measured preservation from unsupported generalization.

```python
# Hypothesis: the saved validation plan covers the remaining lengths and disjoint prose without changing the current edit.
import json, pathlib
print(pathlib.Path("work/notes.md").read_text()[-2500:])
for filename in ["validation_generate_requests.json", "validation_text_requests.json"]:
    obj = json.loads(pathlib.Path("work/"+filename).read_text())
    print(filename, type(obj).__name__, "count", len(obj))
    if isinstance(obj, list):
        print("first request:", str(obj[0])[:1400])
    else:
        print("keys:", list(obj)[:25])
print("generation result sample:", pathlib.Path("work/fwd_19_edited.jsonl").read_text().splitlines()[0][:1200])
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 19 of at most 300:
```
tructural candidate: plus marker in one residual direction c orthogonal to embedding and frozen output; all block inputs ignore c and outputs preserve it. Layer0 head1 value/output retains top31 token-value singular components (discarded singular0.156), uses last component for exact '+' indicator. No arithmetic readout yet. Marker scale1; 42 projection tensors changed, architecture unchanged.

Marker-only edit2: pass7 mean KL0.000110 (152 texts); pass8 native36/36 retained, no addition improvement. Candidate3 adds ten final-MLP neurons (reused lowest preservation contribution), exact closed-form one-digit gates, up projections read plus marker, down projections use frozen-output dual digit directions. Single closed-form compensation restores removed native neuron contributions using retained758 features. Not yet evaluated.

Candidate3/pass9: one-digit50/50 (original6/50), longer0/20, native36/36 retained. Pass10 ordinary-prose mean KL0.00064712; plus-containing prose mean KL2.75759506; native mean KL0.00000393. Current best partial solution saved in work/best_3 and work/gated_one_digit_weights.pt. Not a multi-digit solution.

Pass11: captured routing for 64 independent synthetic additions, eight each at N2,3,4,6,8,12,16,20. Practice text contains zero '+' symbols, but plus-prose remains a known limitation. Current saved edit unchanged (candidate3).

Pass11 routing analysis: strong a-copy only at short lengths; b aligned mass generally low. Heldout linear digit probes at resid_mid5: a43.0%, b9.9%, incoming carry76.1%. Thus no supported general multi-digit readout. Exploring one two-digit numeric diagnostic, not tuning the saved one-digit circuit.

Pass12 diagnostic: two-digit final-input sum probes only43.8%/34.4% within half-digit at first/second positions; finalcarry100%. Source-token numeric readout98.4% within half-digit. Direct sum-readout idea rejected. Exploring analytic RoPE pointer phases and previous-digit carry i
[... cut]
````

</details>

## Turn 42

The saved plan has one generation batch per digit length, with 36 distinct native-task examples in each batch. Seven batches are complete, so I will continue with eight-digit addition and the next native slice. This tests the upper end of the N=1–8 reporting group without changing the weights.

```forward
{"type":"generate","model":"both","ids":["add_n08_000","add_n08_001","add_n08_002","add_n08_003","add_n08_004","add_n08_005","add_n08_006","add_n08_007","add_n08_008","add_n08_009","add_n08_010","add_n08_011","add_n08_012","add_n08_013","add_n08_014","add_n08_015","add_n08_016","add_n08_017","add_n08_018","add_n08_019","add_n08_020","add_n08_021","add_n08_022","add_n08_023","add_n08_024","add_n08_025","add_n08_026","add_n08_027","add_n08_028","add_n08_029","add_n08_030","add_n08_031","add_n08_032","add_n08_033","add_n08_034","add_n08_035","add_n08_036","add_n08_037","add_n08_038","add_n08_039","add_n08_040","add_n08_041","add_n08_042","add_n08_043","add_n08_044","add_n08_045","add_n08_046","add_n08_047","add_n08_048","add_n08_049","copy_n01_007","copy_n02_007","copy_n03_007","copy_n04_007","copy_n05_007","copy_n06_007","copy_n07_007","copy_n08_007","copy_n09_007","copy_n10_007","copy_n11_007","copy_n12_007","reverse_n01_007","reverse_n02_007","reverse_n03_007","reverse_n04_007","reverse_n05_007","reverse_n06_007","reverse_n07_007","reverse_n08_007","reverse_n09_007","reverse_n10_007","reverse_n11_007","reverse_n12_007","max_n01_007","max_n02_007","max_n03_007","max_n04_007","max_n05_007","max_n06_007","max_n07_007","max_n08_007","max_n09_007","max_n10_007","max_n11_007","max_n12_007"]}
```

<details><summary>harness output</summary>

````text
Forward pass 20 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "8": 0.0
  },
  "failures": [
   {
    "id": "add_n08_000",
    "expected": "058875070",
    "got": "2160120006"
   },
   {
    "id": "add_n08_001",
    "expected": "043041431",
    "got": "0818709886"
   },
   {
    "id": "add_n08_002",
    "expected": "890734601",
    "got": "6990965300"
   },
   {
    "id": "add_n08_003",
    "expected": "918084770",
    "got": "6302381554"
   },
   {
    "id": "add_n08_004",
    "expected": "874513990",
    "got": "4229248284"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "8": 0.0
  },
  "failures": [
   {
    "id": "add_n08_000",
    "expected": "058875070",
    "got": "6653676646"
   },
   {
    "id": "add_n08_001",
    "expected": "043041431",
    "got": "2644066646"
   },
   {
    "id": "add_n08_002",
    "expected": "890734601",
    "got": "6650606640"
   },
   {
    "id": "add_n08_003",
    "expected": "918084770",
    "got": "6652676644"
   },
   {
    "id": "add_n08_004",
    "expected": "874513990",
    "got": "2644676644"
   }
  ]
 },
 "files": [
  "work/fwd_20_original.jsonl",
  "work/fwd_20_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.6
}
```
Forward passes left: 30. Replies left: 158.
````

</details>

## Turn 43

Eight-digit addition is 0/50 for both models, and all 36 native examples remain correct. Across N=1–8, the edit’s measured benefit is confined to one-digit problems. I will now start the N=9–20 group; any correct long answer would be evidence beyond the current one-digit result.

```forward
{"type":"generate","model":"both","ids":["add_n09_000","add_n09_001","add_n09_002","add_n09_003","add_n09_004","add_n09_005","add_n09_006","add_n09_007","add_n09_008","add_n09_009","add_n09_010","add_n09_011","add_n09_012","add_n09_013","add_n09_014","add_n09_015","add_n09_016","add_n09_017","add_n09_018","add_n09_019","add_n09_020","add_n09_021","add_n09_022","add_n09_023","add_n09_024","add_n09_025","add_n09_026","add_n09_027","add_n09_028","add_n09_029","add_n09_030","add_n09_031","add_n09_032","add_n09_033","add_n09_034","add_n09_035","add_n09_036","add_n09_037","add_n09_038","add_n09_039","add_n09_040","add_n09_041","add_n09_042","add_n09_043","add_n09_044","add_n09_045","add_n09_046","add_n09_047","add_n09_048","add_n09_049","copy_n01_008","copy_n02_008","copy_n03_008","copy_n04_008","copy_n05_008","copy_n06_008","copy_n07_008","copy_n08_008","copy_n09_008","copy_n10_008","copy_n11_008","copy_n12_008","reverse_n01_008","reverse_n02_008","reverse_n03_008","reverse_n04_008","reverse_n05_008","reverse_n06_008","reverse_n07_008","reverse_n08_008","reverse_n09_008","reverse_n10_008","reverse_n11_008","reverse_n12_008","max_n01_008","max_n02_008","max_n03_008","max_n04_008","max_n05_008","max_n06_008","max_n07_008","max_n08_008","max_n09_008","max_n10_008","max_n11_008","max_n12_008"]}
```

<details><summary>harness output</summary>

````text
Forward pass 21 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.407,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 0.9167,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "9": 0.0
  },
  "failures": [
   {
    "id": "add_n09_000",
    "expected": "6804994201",
    "got": "80502180408"
   },
   {
    "id": "add_n09_001",
    "expected": "8869507021",
    "got": "07686301366"
   },
   {
    "id": "add_n09_002",
    "expected": "6025375450",
    "got": "58212951179"
   },
   {
    "id": "add_n09_003",
    "expected": "5987985451",
    "got": "17873010594"
   },
   {
    "id": "add_n09_004",
    "expected": "9520805111",
    "got": "10082016905"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.407,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 0.9167,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "9": 0.0
  },
  "failures": [
   {
    "id": "add_n09_000",
    "expected": "6804994201",
    "got": "20446766444"
   },
   {
    "id": "add_n09_001",
    "expected": "8869507021",
    "got": "26456766464"
   },
   {
    "id": "add_n09_002",
    "expected": "6025375450",
    "got": "56056666445"
   },
   {
    "id": "add_n09_003",
    "expected": "5987985451",
    "got": "66456766584"
   },
   {
    "id": "add_n09_004",
    "expected": "9520805111",
    "got": "66586766454"
   }
  ]
 },
 "files": [
  "work/fwd_21_original.jsonl",
  "work/fwd_21_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.7
}
```
Forward passes left: 29. Replies left: 157.
````

</details>

## Turn 44

Nine-digit addition is also 0/50. Both models miss the same max example, so native accuracy is not perfect, but this batch shows no edit-induced loss. I will test ten-digit addition and a fresh native slice to keep baseline errors separate from damage caused by the edit.

```forward
{"type":"generate","model":"both","ids":["add_n10_000","add_n10_001","add_n10_002","add_n10_003","add_n10_004","add_n10_005","add_n10_006","add_n10_007","add_n10_008","add_n10_009","add_n10_010","add_n10_011","add_n10_012","add_n10_013","add_n10_014","add_n10_015","add_n10_016","add_n10_017","add_n10_018","add_n10_019","add_n10_020","add_n10_021","add_n10_022","add_n10_023","add_n10_024","add_n10_025","add_n10_026","add_n10_027","add_n10_028","add_n10_029","add_n10_030","add_n10_031","add_n10_032","add_n10_033","add_n10_034","add_n10_035","add_n10_036","add_n10_037","add_n10_038","add_n10_039","add_n10_040","add_n10_041","add_n10_042","add_n10_043","add_n10_044","add_n10_045","add_n10_046","add_n10_047","add_n10_048","add_n10_049","copy_n01_009","copy_n02_009","copy_n03_009","copy_n04_009","copy_n05_009","copy_n06_009","copy_n07_009","copy_n08_009","copy_n09_009","copy_n10_009","copy_n11_009","copy_n12_009","reverse_n01_009","reverse_n02_009","reverse_n03_009","reverse_n04_009","reverse_n05_009","reverse_n06_009","reverse_n07_009","reverse_n08_009","reverse_n09_009","reverse_n10_009","reverse_n11_009","reverse_n12_009","max_n01_009","max_n02_009","max_n03_009","max_n04_009","max_n05_009","max_n06_009","max_n07_009","max_n08_009","max_n09_009","max_n10_009","max_n11_009","max_n12_009"]}
```

<details><summary>harness output</summary>

````text
Forward pass 22 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "10": 0.0
  },
  "failures": [
   {
    "id": "add_n10_000",
    "expected": "34700270501",
    "got": "188778138762"
   },
   {
    "id": "add_n10_001",
    "expected": "71186645821",
    "got": "785716067567"
   },
   {
    "id": "add_n10_002",
    "expected": "30277031021",
    "got": "335435303437"
   },
   {
    "id": "add_n10_003",
    "expected": "57859627261",
    "got": "840091850852"
   },
   {
    "id": "add_n10_004",
    "expected": "22532586921",
    "got": "75538493031\n"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "10": 0.0
  },
  "failures": [
   {
    "id": "add_n10_000",
    "expected": "34700270501",
    "got": "264467664560"
   },
   {
    "id": "add_n10_001",
    "expected": "71186645821",
    "got": "265767665562"
   },
   {
    "id": "add_n10_002",
    "expected": "30277031021",
    "got": "264567665440"
   },
   {
    "id": "add_n10_003",
    "expected": "57859627261",
    "got": "265067665840"
   },
   {
    "id": "add_n10_004",
    "expected": "22532586921",
    "got": "264467664552"
   }
  ]
 },
 "files": [
  "work/fwd_22_original.jsonl",
  "work/fwd_22_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.9
}
```
Forward passes left: 28. Replies left: 156.
````

</details>

## Turn 45

The ten-digit batch has no correct additions and no native-task losses. Changing addition outputs is clearly not enough: the edit does not align digits or propagate carries across these lengths. I will record the paired totals before finishing the remaining length tests, keeping this limitation explicit.

```python
# Hypothesis: all measured gains remain at N=1, with no paired native accuracy losses through N=10.
import collections, json, pathlib
totals = collections.defaultdict(lambda: [0,0,0,0,0])
for k in range(13,23):
    orig = [json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_original.jsonl").read_text().splitlines()]
    edit = [json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_edited.jsonl").read_text().splitlines()]
    for o,e in zip(orig,edit):
        assert o["id"] == e["id"]
        key = f'addition_N{e["n_digits"]:02d}' if e["task"]=="addition" else e["task"]
        t=totals[key]
        t[0]+=1; t[1]+=bool(o["correct"]); t[2]+=bool(e["correct"])
        t[3]+=bool(o["correct"] and not e["correct"])
        t[4]+=o["output"] != e["output"]
print("group: [count, original correct, edited correct, losses, changed outputs]")
print(dict(sorted(totals.items())))
with open("work/notes.md","a") as f:
    f.write("\nSweep through pass22 (N1..10): addition N1 50/50 edited vs6/50 original; N2..10 each0/50 both. Native359/360 both; one shared max error, zero native output changes or losses. Edit3 unchanged.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 20 of at most 300:
```
group: [count, original correct, edited correct, losses, changed outputs]
{'addition_N01': [50, 6, 50, 0, 50], 'addition_N02': [50, 0, 0, 0, 50], 'addition_N03': [50, 0, 0, 0, 50], 'addition_N04': [50, 0, 0, 0, 50], 'addition_N05': [50, 0, 0, 0, 50], 'addition_N06': [50, 0, 0, 0, 50], 'addition_N07': [50, 0, 0, 0, 50], 'addition_N08': [50, 0, 0, 0, 50], 'addition_N09': [50, 0, 0, 0, 50], 'addition_N10': [50, 0, 0, 0, 50], 'copy': [120, 120, 120, 0, 0], 'max': [120, 119, 119, 0, 0], 'reverse': [120, 120, 120, 0, 0]}

```
Forward passes left: 28. Replies left: 155.
````

</details>

## Turn 46

The aggregate confirms 44 addition rescues, all at N=1, and identical native outputs on all 360 tested examples. This supports native preservation on the tested set, but not multi-digit addition. I will continue with N=11 and the next 36 native examples under the same saved edit.

```forward
{"type":"generate","model":"both","ids":["add_n11_000","add_n11_001","add_n11_002","add_n11_003","add_n11_004","add_n11_005","add_n11_006","add_n11_007","add_n11_008","add_n11_009","add_n11_010","add_n11_011","add_n11_012","add_n11_013","add_n11_014","add_n11_015","add_n11_016","add_n11_017","add_n11_018","add_n11_019","add_n11_020","add_n11_021","add_n11_022","add_n11_023","add_n11_024","add_n11_025","add_n11_026","add_n11_027","add_n11_028","add_n11_029","add_n11_030","add_n11_031","add_n11_032","add_n11_033","add_n11_034","add_n11_035","add_n11_036","add_n11_037","add_n11_038","add_n11_039","add_n11_040","add_n11_041","add_n11_042","add_n11_043","add_n11_044","add_n11_045","add_n11_046","add_n11_047","add_n11_048","add_n11_049","copy_n01_010","copy_n02_010","copy_n03_010","copy_n04_010","copy_n05_010","copy_n06_010","copy_n07_010","copy_n08_010","copy_n09_010","copy_n10_010","copy_n11_010","copy_n12_010","reverse_n01_010","reverse_n02_010","reverse_n03_010","reverse_n04_010","reverse_n05_010","reverse_n06_010","reverse_n07_010","reverse_n08_010","reverse_n09_010","reverse_n10_010","reverse_n11_010","reverse_n12_010","max_n01_010","max_n02_010","max_n03_010","max_n04_010","max_n05_010","max_n06_010","max_n07_010","max_n08_010","max_n09_010","max_n10_010","max_n11_010","max_n12_010"]}
```

<details><summary>harness output</summary>

````text
Forward pass 23 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "11": 0.0
  },
  "failures": [
   {
    "id": "add_n11_000",
    "expected": "543715551160",
    "got": "301692191705\n"
   },
   {
    "id": "add_n11_001",
    "expected": "492535552070",
    "got": "009409004030\n"
   },
   {
    "id": "add_n11_002",
    "expected": "211363010911",
    "got": "547905589306\n"
   },
   {
    "id": "add_n11_003",
    "expected": "153733892720",
    "got": "178067520221\n"
   },
   {
    "id": "add_n11_004",
    "expected": "869828166961",
    "got": "888585225128\n"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "11": 0.0
  },
  "failures": [
   {
    "id": "add_n11_000",
    "expected": "543715551160",
    "got": "2644066645406"
   },
   {
    "id": "add_n11_001",
    "expected": "492535552070",
    "got": "264407664440\n"
   },
   {
    "id": "add_n11_002",
    "expected": "211363010911",
    "got": "2644676645406"
   },
   {
    "id": "add_n11_003",
    "expected": "153733892720",
    "got": "2644676644406"
   },
   {
    "id": "add_n11_004",
    "expected": "869828166961",
    "got": "6605606654406"
   }
  ]
 },
 "files": [
  "work/fwd_23_original.jsonl",
  "work/fwd_23_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.2
}
```
Forward passes left: 27. Replies left: 154.
````

</details>

## Turn 47

N=11 again yields 0/50 additions, with all native examples retained. The repeated failures reinforce that the one-digit circuit does not generalize by itself. I will test N=12 next; preservation is assessed against the original outputs, not merely against aggregate accuracy.

```forward
{"type":"generate","model":"both","ids":["add_n12_000","add_n12_001","add_n12_002","add_n12_003","add_n12_004","add_n12_005","add_n12_006","add_n12_007","add_n12_008","add_n12_009","add_n12_010","add_n12_011","add_n12_012","add_n12_013","add_n12_014","add_n12_015","add_n12_016","add_n12_017","add_n12_018","add_n12_019","add_n12_020","add_n12_021","add_n12_022","add_n12_023","add_n12_024","add_n12_025","add_n12_026","add_n12_027","add_n12_028","add_n12_029","add_n12_030","add_n12_031","add_n12_032","add_n12_033","add_n12_034","add_n12_035","add_n12_036","add_n12_037","add_n12_038","add_n12_039","add_n12_040","add_n12_041","add_n12_042","add_n12_043","add_n12_044","add_n12_045","add_n12_046","add_n12_047","add_n12_048","add_n12_049","copy_n01_011","copy_n02_011","copy_n03_011","copy_n04_011","copy_n05_011","copy_n06_011","copy_n07_011","copy_n08_011","copy_n09_011","copy_n10_011","copy_n11_011","copy_n12_011","reverse_n01_011","reverse_n02_011","reverse_n03_011","reverse_n04_011","reverse_n05_011","reverse_n06_011","reverse_n07_011","reverse_n08_011","reverse_n09_011","reverse_n10_011","reverse_n11_011","reverse_n12_011","max_n01_011","max_n02_011","max_n03_011","max_n04_011","max_n05_011","max_n06_011","max_n07_011","max_n08_011","max_n09_011","max_n10_011","max_n11_011","max_n12_011"]}
```

<details><summary>harness output</summary>

````text
Forward pass 24 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "12": 0.0
  },
  "failures": [
   {
    "id": "add_n12_000",
    "expected": "1226177973480",
    "got": "658801618841\nM"
   },
   {
    "id": "add_n12_001",
    "expected": "2076465443760",
    "got": "32782131839\nM:"
   },
   {
    "id": "add_n12_002",
    "expected": "9219014620701",
    "got": "38700108409\nM:"
   },
   {
    "id": "add_n12_003",
    "expected": "5531862283350",
    "got": "16856010576\nM:"
   },
   {
    "id": "add_n12_004",
    "expected": "7757922041390",
    "got": "288670278640\nM"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "12": 0.0
  },
  "failures": [
   {
    "id": "add_n12_000",
    "expected": "1226177973480",
    "got": "264806664440\nM"
   },
   {
    "id": "add_n12_001",
    "expected": "2076465443760",
    "got": "260526665440\nC"
   },
   {
    "id": "add_n12_002",
    "expected": "9219014620701",
    "got": "264467664440\nM"
   },
   {
    "id": "add_n12_003",
    "expected": "5531862283350",
    "got": "260567665560\nC"
   },
   {
    "id": "add_n12_004",
    "expected": "7757922041390",
    "got": "660560664540\nC"
   }
  ]
 },
 "files": [
  "work/fwd_24_original.jsonl",
  "work/fwd_24_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.2
}
```
Forward passes left: 26. Replies left: 153.
````

</details>

## Turn 48

Twelve-digit addition remains unsuccessful, and the native outputs are unchanged. Some addition continuations also terminate early, another sign that this edit supplies no general length-control mechanism. I will continue with thirteen-digit problems to document performance beyond the native tasks’ trained number lengths.

```forward
{"type":"generate","model":"both","ids":["add_n13_000","add_n13_001","add_n13_002","add_n13_003","add_n13_004","add_n13_005","add_n13_006","add_n13_007","add_n13_008","add_n13_009","add_n13_010","add_n13_011","add_n13_012","add_n13_013","add_n13_014","add_n13_015","add_n13_016","add_n13_017","add_n13_018","add_n13_019","add_n13_020","add_n13_021","add_n13_022","add_n13_023","add_n13_024","add_n13_025","add_n13_026","add_n13_027","add_n13_028","add_n13_029","add_n13_030","add_n13_031","add_n13_032","add_n13_033","add_n13_034","add_n13_035","add_n13_036","add_n13_037","add_n13_038","add_n13_039","add_n13_040","add_n13_041","add_n13_042","add_n13_043","add_n13_044","add_n13_045","add_n13_046","add_n13_047","add_n13_048","add_n13_049","copy_n01_012","copy_n02_012","copy_n03_012","copy_n04_012","copy_n05_012","copy_n06_012","copy_n07_012","copy_n08_012","copy_n09_012","copy_n10_012","copy_n11_012","copy_n12_012","reverse_n01_012","reverse_n02_012","reverse_n03_012","reverse_n04_012","reverse_n05_012","reverse_n06_012","reverse_n07_012","reverse_n08_012","reverse_n09_012","reverse_n10_012","reverse_n11_012","reverse_n12_012","max_n01_012","max_n02_012","max_n03_012","max_n04_012","max_n05_012","max_n06_012","max_n07_012","max_n08_012","max_n09_012","max_n10_012","max_n11_012","max_n12_012"]}
```

<details><summary>harness output</summary>

````text
Forward pass 25 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "13": 0.0
  },
  "failures": [
   {
    "id": "add_n13_000",
    "expected": "41794987887231",
    "got": "95072626026\nC:0"
   },
   {
    "id": "add_n13_001",
    "expected": "48358795066790",
    "got": "31502651005\nC:0"
   },
   {
    "id": "add_n13_002",
    "expected": "54480263807841",
    "got": "53624015064\nC:0"
   },
   {
    "id": "add_n13_003",
    "expected": "50052297457970",
    "got": "23035400830\nC:0"
   },
   {
    "id": "add_n13_004",
    "expected": "97020693375651",
    "got": "66876759889\nM:5"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "13": 0.0
  },
  "failures": [
   {
    "id": "add_n13_000",
    "expected": "41794987887231",
    "got": "264766664562\nC0"
   },
   {
    "id": "add_n13_001",
    "expected": "48358795066790",
    "got": "265567664440\nC0"
   },
   {
    "id": "add_n13_002",
    "expected": "54480263807841",
    "got": "565520664440\nC0"
   },
   {
    "id": "add_n13_003",
    "expected": "50052297457970",
    "got": "060567664406\nC0"
   },
   {
    "id": "add_n13_004",
    "expected": "97020693375651",
    "got": "065867664592\nM4"
   }
  ]
 },
 "files": [
  "work/fwd_25_original.jsonl",
  "work/fwd_25_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.4
}
```
Forward passes left: 25. Replies left: 152.
````

</details>

## Turn 49

The thirteen-digit batch adds no successful additions and no native regressions. These are validation measurements only; I am not fitting further weights to the failures. I will check N=14 next, with the same paired comparison and unchanged candidate.

```forward
{"type":"generate","model":"both","ids":["add_n14_000","add_n14_001","add_n14_002","add_n14_003","add_n14_004","add_n14_005","add_n14_006","add_n14_007","add_n14_008","add_n14_009","add_n14_010","add_n14_011","add_n14_012","add_n14_013","add_n14_014","add_n14_015","add_n14_016","add_n14_017","add_n14_018","add_n14_019","add_n14_020","add_n14_021","add_n14_022","add_n14_023","add_n14_024","add_n14_025","add_n14_026","add_n14_027","add_n14_028","add_n14_029","add_n14_030","add_n14_031","add_n14_032","add_n14_033","add_n14_034","add_n14_035","add_n14_036","add_n14_037","add_n14_038","add_n14_039","add_n14_040","add_n14_041","add_n14_042","add_n14_043","add_n14_044","add_n14_045","add_n14_046","add_n14_047","add_n14_048","add_n14_049","copy_n01_013","copy_n02_013","copy_n03_013","copy_n04_013","copy_n05_013","copy_n06_013","copy_n07_013","copy_n08_013","copy_n09_013","copy_n10_013","copy_n11_013","copy_n12_013","reverse_n01_013","reverse_n02_013","reverse_n03_013","reverse_n04_013","reverse_n05_013","reverse_n06_013","reverse_n07_013","reverse_n08_013","reverse_n09_013","reverse_n10_013","reverse_n11_013","reverse_n12_013","max_n01_013","max_n02_013","max_n03_013","max_n04_013","max_n05_013","max_n06_013","max_n07_013","max_n08_013","max_n09_013","max_n10_013","max_n11_013","max_n12_013"]}
```

<details><summary>harness output</summary>

````text
Forward pass 26 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "14": 0.0
  },
  "failures": [
   {
    "id": "add_n14_000",
    "expected": "181801580435431",
    "got": "60008080006\nC:05"
   },
   {
    "id": "add_n14_001",
    "expected": "724381918490551",
    "got": "92172121710\nM:1,"
   },
   {
    "id": "add_n14_002",
    "expected": "814603663399921",
    "got": "82229111208\nM:5,"
   },
   {
    "id": "add_n14_003",
    "expected": "634797388622841",
    "got": "801827780848\nM:5"
   },
   {
    "id": "add_n14_004",
    "expected": "062028575484481",
    "got": "88796828906\nM:5,"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "14": 0.0
  },
  "failures": [
   {
    "id": "add_n14_000",
    "expected": "181801580435431",
    "got": "264067664562\nM45"
   },
   {
    "id": "add_n14_001",
    "expected": "724381918490551",
    "got": "6652676644406626"
   },
   {
    "id": "add_n14_002",
    "expected": "814603663399921",
    "got": "6652676644406626"
   },
   {
    "id": "add_n14_003",
    "expected": "634797388622841",
    "got": "264767664440\nM45"
   },
   {
    "id": "add_n14_004",
    "expected": "062028575484481",
    "got": "260467664452\nM45"
   }
  ]
 },
 "files": [
  "work/fwd_26_original.jsonl",
  "work/fwd_26_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.7
}
```
Forward passes left: 24. Replies left: 151.
````

</details>

## Turn 50

N=14 remains 0/50, while all 36 native answers match the original. There is still no evidence of a multi-digit arithmetic mechanism. I will test N=15 and another native slice before updating the cumulative notebook.

```forward
{"type":"generate","model":"both","ids":["add_n15_000","add_n15_001","add_n15_002","add_n15_003","add_n15_004","add_n15_005","add_n15_006","add_n15_007","add_n15_008","add_n15_009","add_n15_010","add_n15_011","add_n15_012","add_n15_013","add_n15_014","add_n15_015","add_n15_016","add_n15_017","add_n15_018","add_n15_019","add_n15_020","add_n15_021","add_n15_022","add_n15_023","add_n15_024","add_n15_025","add_n15_026","add_n15_027","add_n15_028","add_n15_029","add_n15_030","add_n15_031","add_n15_032","add_n15_033","add_n15_034","add_n15_035","add_n15_036","add_n15_037","add_n15_038","add_n15_039","add_n15_040","add_n15_041","add_n15_042","add_n15_043","add_n15_044","add_n15_045","add_n15_046","add_n15_047","add_n15_048","add_n15_049","copy_n01_014","copy_n02_014","copy_n03_014","copy_n04_014","copy_n05_014","copy_n06_014","copy_n07_014","copy_n08_014","copy_n09_014","copy_n10_014","copy_n11_014","copy_n12_014","reverse_n01_014","reverse_n02_014","reverse_n03_014","reverse_n04_014","reverse_n05_014","reverse_n06_014","reverse_n07_014","reverse_n08_014","reverse_n09_014","reverse_n10_014","reverse_n11_014","reverse_n12_014","max_n01_014","max_n02_014","max_n03_014","max_n04_014","max_n05_014","max_n06_014","max_n07_014","max_n08_014","max_n09_014","max_n10_014","max_n11_014","max_n12_014"]}
```

<details><summary>harness output</summary>

````text
Forward pass 27 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "15": 0.0
  },
  "failures": [
   {
    "id": "add_n15_000",
    "expected": "3828017713036631",
    "got": "26628614965\nM:5,1"
   },
   {
    "id": "add_n15_001",
    "expected": "8437044274622600",
    "got": "30569222060\nM:5,1"
   },
   {
    "id": "add_n15_002",
    "expected": "7186164109938101",
    "got": "09759129096\nM:5,1"
   },
   {
    "id": "add_n15_003",
    "expected": "6912627648314960",
    "got": "28880929817\nM:5,1"
   },
   {
    "id": "add_n15_004",
    "expected": "4664463277899170",
    "got": "90875628903\nM:5,5"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "15": 0.0
  },
  "failures": [
   {
    "id": "add_n15_000",
    "expected": "3828017713036631",
    "got": "66526766554066525"
   },
   {
    "id": "add_n15_001",
    "expected": "8437044274622600",
    "got": "2055666645062\nM45"
   },
   {
    "id": "add_n15_002",
    "expected": "7186164109938101",
    "got": "26056766454066262"
   },
   {
    "id": "add_n15_003",
    "expected": "6912627648314960",
    "got": "26856766444066265"
   },
   {
    "id": "add_n15_004",
    "expected": "4664463277899170",
    "got": "26446766454066454"
   }
  ]
 },
 "files": [
  "work/fwd_27_original.jsonl",
  "work/fwd_27_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 3.0
}
```
Forward passes left: 23. Replies left: 150.
````

</details>

## Turn 51

The latest batch again leaves the conclusion unchanged: no long-addition successes and no native regressions. I will now update the cumulative counts and inspect the planned prose batches. The prose tests matter because matching greedy native outputs does not imply unchanged next-character distributions.

```python
# Hypothesis: paired native preservation continues through 540 examples, while prose KL still requires broader measurement.
from collections import defaultdict
totals=defaultdict(lambda:[0,0,0,0,0])
for k in range(13,28):
    os=[json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_original.jsonl").read_text().splitlines()]
    es=[json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_edited.jsonl").read_text().splitlines()]
    for o,e in zip(os,es):
        assert o["id"]==e["id"]
        key=e["task"]
        t=totals[key]; t[0]+=1; t[1]+=bool(o["correct"]); t[2]+=bool(e["correct"])
        t[3]+=bool(o["correct"] and not e["correct"]); t[4]+=o["output"]!=e["output"]
print("task: [count, original correct, edited correct, losses, changed outputs]",dict(totals))
reqs=json.loads(pathlib.Path("work/validation_text_requests.json").read_text())
print("Prose batch sizes:",[len(r["texts"]) for r in reqs])
print("Prose characters:",sum(len(t) for r in reqs for t in r["texts"]))
print("First prose request:",json.dumps(reqs[0]))
with open("work/notes.md","a") as f:
    f.write("\nSweep through pass27 (N1..15): additions50/750 edited vs6/750 original; all gains N1. Native539/540 both, zero native output changes/losses. Remaining five generation batches and18 disjoint prose batches; edit3 unchanged.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 21 of at most 300:
```
task: [count, original correct, edited correct, losses, changed outputs] {'addition': [750, 6, 50, 0, 750], 'copy': [180, 180, 180, 0, 0], 'reverse': [180, 180, 180, 0, 0], 'max': [180, 179, 179, 0, 0]}
Prose batch sizes: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
Prose characters: 23040
First prose request: {"type": "teacher_forced", "model": "both", "texts": ["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Br", "M ) broadcast from the city . A television station , Havenscope ( available on @-@ campus only ) , and a radio station , WLHU ( Internet station only , with no ", "le League World Series . In 2011 , the Keystone Little League based in Lock Haven advanced to the Little League World Series and placed third in the United Stat", "rograms to members , and the Clinton Country Club maintains a private 18 @-@ hole golf course in Mill Hall . \n = = Government = = \n Lock Haven has a council @-@", "t \" Pete \" Smeltz , chairman ; Jeffrey Snyder , vice @-@ chairman , and Paul Conklin , have terms running through 2019 . \n Michael K. Hanna , a Democrat , repre", " , and Woodward Elementary , all for children enrolled in kindergarten through fifth grade . The total enrollment of these three schools combined in 2002  03 wa", " foot ( 930 m2 ) expansion to include grades seven and eight , which will make it a combined elementary and middle school . \n Lock Haven University of Pennsylva", "s are available through Susquehanna Trailways , based in Avis , 10 miles ( 16 km ) northeast of Lock Haven . Pennsylvania Bicycle Route G follows Pennsylvania R"], "return": {}}

```
Forward passes left: 23. Replies left: 149.
````

</details>

## Turn 52

The cumulative native result is 539/540 for both models, with no changed outputs. The reserved prose plan contains 23,040 characters, which will provide a broader distributional check after generation testing. I will finish N=16 next; the weights remain frozen throughout this sweep.

```forward
{"type":"generate","model":"both","ids":["add_n16_000","add_n16_001","add_n16_002","add_n16_003","add_n16_004","add_n16_005","add_n16_006","add_n16_007","add_n16_008","add_n16_009","add_n16_010","add_n16_011","add_n16_012","add_n16_013","add_n16_014","add_n16_015","add_n16_016","add_n16_017","add_n16_018","add_n16_019","add_n16_020","add_n16_021","add_n16_022","add_n16_023","add_n16_024","add_n16_025","add_n16_026","add_n16_027","add_n16_028","add_n16_029","add_n16_030","add_n16_031","add_n16_032","add_n16_033","add_n16_034","add_n16_035","add_n16_036","add_n16_037","add_n16_038","add_n16_039","add_n16_040","add_n16_041","add_n16_042","add_n16_043","add_n16_044","add_n16_045","add_n16_046","add_n16_047","add_n16_048","add_n16_049","copy_n01_015","copy_n02_015","copy_n03_015","copy_n04_015","copy_n05_015","copy_n06_015","copy_n07_015","copy_n08_015","copy_n09_015","copy_n10_015","copy_n11_015","copy_n12_015","reverse_n01_015","reverse_n02_015","reverse_n03_015","reverse_n04_015","reverse_n05_015","reverse_n06_015","reverse_n07_015","reverse_n08_015","reverse_n09_015","reverse_n10_015","reverse_n11_015","reverse_n12_015","max_n01_015","max_n02_015","max_n03_015","max_n04_015","max_n05_015","max_n06_015","max_n07_015","max_n08_015","max_n09_015","max_n10_015","max_n11_015","max_n12_015"]}
```

<details><summary>harness output</summary>

````text
Forward pass 28 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "16": 0.0
  },
  "failures": [
   {
    "id": "add_n16_000",
    "expected": "09781903217306060",
    "got": "98651003006\nM:1,1="
   },
   {
    "id": "add_n16_001",
    "expected": "36599020863221911",
    "got": "38762795065\nM:1,1="
   },
   {
    "id": "add_n16_002",
    "expected": "70478089780428450",
    "got": "07550610050\nM:1,1="
   },
   {
    "id": "add_n16_003",
    "expected": "57777748400563901",
    "got": "85615555065\nM:1,1="
   },
   {
    "id": "add_n16_004",
    "expected": "11680091872012920",
    "got": "06987797082\nM:1,1="
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "16": 0.0
  },
  "failures": [
   {
    "id": "add_n16_000",
    "expected": "09781903217306060",
    "got": "264466664562\nM4515"
   },
   {
    "id": "add_n16_001",
    "expected": "36599020863221911",
    "got": "6605676655406\nM451"
   },
   {
    "id": "add_n16_002",
    "expected": "70478089780428450",
    "got": "265567664440\nM4555"
   },
   {
    "id": "add_n16_003",
    "expected": "57777748400563901",
    "got": "266567665540\nM4515"
   },
   {
    "id": "add_n16_004",
    "expected": "11680091872012920",
    "got": "265867664540\nM4555"
   }
  ]
 },
 "files": [
  "work/fwd_28_original.jsonl",
  "work/fwd_28_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 3.2
}
```
Forward passes left: 22. Replies left: 148.
````

</details>

## Turn 53

Sixteen-digit addition is 0/50, and this native slice is preserved exactly. The longer outputs continue to resemble unrelated digit-task continuations rather than sums. I will check N=17 next to complete the length-by-length record without treating those output changes as arithmetic progress.

```forward
{"type":"generate","model":"both","ids":["add_n17_000","add_n17_001","add_n17_002","add_n17_003","add_n17_004","add_n17_005","add_n17_006","add_n17_007","add_n17_008","add_n17_009","add_n17_010","add_n17_011","add_n17_012","add_n17_013","add_n17_014","add_n17_015","add_n17_016","add_n17_017","add_n17_018","add_n17_019","add_n17_020","add_n17_021","add_n17_022","add_n17_023","add_n17_024","add_n17_025","add_n17_026","add_n17_027","add_n17_028","add_n17_029","add_n17_030","add_n17_031","add_n17_032","add_n17_033","add_n17_034","add_n17_035","add_n17_036","add_n17_037","add_n17_038","add_n17_039","add_n17_040","add_n17_041","add_n17_042","add_n17_043","add_n17_044","add_n17_045","add_n17_046","add_n17_047","add_n17_048","add_n17_049","copy_n01_016","copy_n02_016","copy_n03_016","copy_n04_016","copy_n05_016","copy_n06_016","copy_n07_016","copy_n08_016","copy_n09_016","copy_n10_016","copy_n11_016","copy_n12_016","reverse_n01_016","reverse_n02_016","reverse_n03_016","reverse_n04_016","reverse_n05_016","reverse_n06_016","reverse_n07_016","reverse_n08_016","reverse_n09_016","reverse_n10_016","reverse_n11_016","reverse_n12_016","max_n01_016","max_n02_016","max_n03_016","max_n04_016","max_n05_016","max_n06_016","max_n07_016","max_n08_016","max_n09_016","max_n10_016","max_n11_016","max_n12_016"]}
```

<details><summary>harness output</summary>

````text
Forward pass 29 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "17": 0.0
  },
  "failures": [
   {
    "id": "add_n17_000",
    "expected": "526635761021317930",
    "got": "82062313106\nM:1,1=1"
   },
   {
    "id": "add_n17_001",
    "expected": "119273654924622011",
    "got": "54869799989\nM:1,1=1"
   },
   {
    "id": "add_n17_002",
    "expected": "668434204229698541",
    "got": "82622505262\nM:1,1=1"
   },
   {
    "id": "add_n17_003",
    "expected": "673519738145066011",
    "got": "95608417105\nM:1,1=1"
   },
   {
    "id": "add_n17_004",
    "expected": "459227322768064280",
    "got": "15169523064\nC:0555="
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "17": 0.0
  },
  "failures": [
   {
    "id": "add_n17_000",
    "expected": "526635761021317930",
    "got": "665566664440\nM45555"
   },
   {
    "id": "add_n17_001",
    "expected": "119273654924622011",
    "got": "665566664440\nM45555"
   },
   {
    "id": "add_n17_002",
    "expected": "668434204229698541",
    "got": "665560565440\nM45555"
   },
   {
    "id": "add_n17_003",
    "expected": "673519738145066011",
    "got": "265565665440\nM45555"
   },
   {
    "id": "add_n17_004",
    "expected": "459227322768064280",
    "got": "665566564440\nM45555"
   }
  ]
 },
 "files": [
  "work/fwd_29_original.jsonl",
  "work/fwd_29_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 3.4
}
```
Forward passes left: 21. Replies left: 147.
````

</details>

## Turn 54

N=17 produces no correct sums, and the 36 native outputs still match exactly. Thus far, the edit’s demonstrated benefit remains narrow and its native preservation remains consistent. I will run the N=18 batch next, using the same criteria.

```forward
{"type":"generate","model":"both","ids":["add_n18_000","add_n18_001","add_n18_002","add_n18_003","add_n18_004","add_n18_005","add_n18_006","add_n18_007","add_n18_008","add_n18_009","add_n18_010","add_n18_011","add_n18_012","add_n18_013","add_n18_014","add_n18_015","add_n18_016","add_n18_017","add_n18_018","add_n18_019","add_n18_020","add_n18_021","add_n18_022","add_n18_023","add_n18_024","add_n18_025","add_n18_026","add_n18_027","add_n18_028","add_n18_029","add_n18_030","add_n18_031","add_n18_032","add_n18_033","add_n18_034","add_n18_035","add_n18_036","add_n18_037","add_n18_038","add_n18_039","add_n18_040","add_n18_041","add_n18_042","add_n18_043","add_n18_044","add_n18_045","add_n18_046","add_n18_047","add_n18_048","add_n18_049","copy_n01_017","copy_n02_017","copy_n03_017","copy_n04_017","copy_n05_017","copy_n06_017","copy_n07_017","copy_n08_017","copy_n09_017","copy_n10_017","copy_n11_017","copy_n12_017","reverse_n01_017","reverse_n02_017","reverse_n03_017","reverse_n04_017","reverse_n05_017","reverse_n06_017","reverse_n07_017","reverse_n08_017","reverse_n09_017","reverse_n10_017","reverse_n11_017","reverse_n12_017","max_n01_017","max_n02_017","max_n03_017","max_n04_017","max_n05_017","max_n06_017","max_n07_017","max_n08_017","max_n09_017","max_n10_017","max_n11_017","max_n12_017"]}
```

<details><summary>harness output</summary>

````text
Forward pass 30 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "18": 0.0
  },
  "failures": [
   {
    "id": "add_n18_000",
    "expected": "7550732792839593650",
    "got": "05868505808\nM:1,1=1\n"
   },
   {
    "id": "add_n18_001",
    "expected": "8027285280507182940",
    "got": "52642828293\nM:1,1=1\n"
   },
   {
    "id": "add_n18_002",
    "expected": "8874564961286353790",
    "got": "26340321030\nM:1,1=1\n"
   },
   {
    "id": "add_n18_003",
    "expected": "8496526286933641611",
    "got": "02868798975\nM:1,1=1\n"
   },
   {
    "id": "add_n18_004",
    "expected": "2160154932142156650",
    "got": "54858488958\nM:1,1=1\n"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "18": 0.0
  },
  "failures": [
   {
    "id": "add_n18_000",
    "expected": "7550732792839593650",
    "got": "665565665440\nM455555"
   },
   {
    "id": "add_n18_001",
    "expected": "8027285280507182940",
    "got": "665565665540\nM455555"
   },
   {
    "id": "add_n18_002",
    "expected": "8874564961286353790",
    "got": "665560664540\nC052122"
   },
   {
    "id": "add_n18_003",
    "expected": "8496526286933641611",
    "got": "665566664540\nM455555"
   },
   {
    "id": "add_n18_004",
    "expected": "2160154932142156650",
    "got": "665566665540\nM455555"
   }
  ]
 },
 "files": [
  "work/fwd_30_original.jsonl",
  "work/fwd_30_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 3.5
}
```
Forward passes left: 20. Replies left: 146.
````

</details>

## Turn 55

Eighteen-digit addition also fails on all 50 examples, with no native output changes. Only two generation batches remain before the full practice set has been covered. I will test N=19 next; this will also add another 36 distinct native checks.

```forward
{"type":"generate","model":"both","ids":["add_n19_000","add_n19_001","add_n19_001","add_n19_002","add_n19_003","add_n19_004","add_n19_005","add_n19_006","add_n19_007","add_n19_008","add_n19_009","add_n19_010","add_n19_011","add_n19_012","add_n19_013","add_n19_014","add_n19_015","add_n19_016","add_n19_017","add_n19_018","add_n19_019","add_n19_020","add_n19_021","add_n19_022","add_n19_023","add_n19_024","add_n19_025","add_n19_026","add_n19_027","add_n19_028","add_n19_029","add_n19_030","add_n19_031","add_n19_032","add_n19_033","add_n19_034","add_n19_035","add_n19_036","add_n19_037","add_n19_038","add_n19_039","add_n19_040","add_n19_041","add_n19_042","add_n19_043","add_n19_044","add_n19_045","add_n19_046","add_n19_047","add_n19_048","add_n19_049","copy_n01_018","copy_n02_018","copy_n03_018","copy_n04_018","copy_n05_018","copy_n06_018","copy_n07_018","copy_n08_018","copy_n09_018","copy_n10_018","copy_n11_018","copy_n12_018","reverse_n01_018","reverse_n02_018","reverse_n03_018","reverse_n04_018","reverse_n05_018","reverse_n06_018","reverse_n07_018","reverse_n08_018","reverse_n09_018","reverse_n10_018","reverse_n11_018","reverse_n12_018","max_n01_018","max_n02_018","max_n03_018","max_n04_018","max_n05_018","max_n06_018","max_n07_018","max_n08_018","max_n09_018","max_n10_018","max_n11_018","max_n12_018"]}
```

<details><summary>harness output</summary>

````text
Forward pass 31 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 87,
 "original": {
  "graded": 87,
  "accuracy": 0.4138,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "19": 0.0
  },
  "failures": [
   {
    "id": "add_n19_000",
    "expected": "95461073666047333111",
    "got": "88595942555\nM:1,1=1\nM"
   },
   {
    "id": "add_n19_001",
    "expected": "23867627736332459651",
    "got": "678828980857\nM:1,1=1\n"
   },
   {
    "id": "add_n19_001",
    "expected": "23867627736332459651",
    "got": "678828980857\nM:1,1=1\n"
   },
   {
    "id": "add_n19_002",
    "expected": "16892574039473403941",
    "got": "16818828085\nM:1,1=1\nM"
   },
   {
    "id": "add_n19_003",
    "expected": "93873615565302100831",
    "got": "91786131028\nM:1,1=1\nM"
   }
  ]
 },
 "edited": {
  "graded": 87,
  "accuracy": 0.4138,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "19": 0.0
  },
  "failures": [
   {
    "id": "add_n19_000",
    "expected": "95461073666047333111",
    "got": "665566544552\nM4555555"
   },
   {
    "id": "add_n19_001",
    "expected": "23867627736332459651",
    "got": "665566565562\nM4515555"
   },
   {
    "id": "add_n19_001",
    "expected": "23867627736332459651",
    "got": "665566565562\nM4515555"
   },
   {
    "id": "add_n19_002",
    "expected": "16892574039473403941",
    "got": "665566565562\nM4515555"
   },
   {
    "id": "add_n19_003",
    "expected": "93873615565302100831",
    "got": "562866664442\nM4515555"
   }
  ]
 },
 "files": [
  "work/fwd_31_original.jsonl",
  "work/fwd_31_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 51
 },
 "s": 3.8
}
```
Forward passes left: 19. Replies left: 145.
````

</details>

## Turn 56

The N=19 request accidentally included one ID twice; I will deduplicate by ID in the final totals. All 50 distinct additions fail, and the native examples remain unchanged. I will now run the final generation batch at N=20, completing coverage of all practice problems.

```forward
{"type":"generate","model":"both","ids":["add_n20_000","add_n20_001","add_n20_002","add_n20_003","add_n20_004","add_n20_005","add_n20_006","add_n20_007","add_n20_008","add_n20_009","add_n20_010","add_n20_011","add_n20_012","add_n20_013","add_n20_014","add_n20_015","add_n20_016","add_n20_017","add_n20_018","add_n20_019","add_n20_020","add_n20_021","add_n20_022","add_n20_023","add_n20_024","add_n20_025","add_n20_026","add_n20_027","add_n20_028","add_n20_029","add_n20_030","add_n20_031","add_n20_032","add_n20_033","add_n20_034","add_n20_035","add_n20_036","add_n20_037","add_n20_038","add_n20_039","add_n20_040","add_n20_041","add_n20_042","add_n20_043","add_n20_044","add_n20_045","add_n20_046","add_n20_047","add_n20_048","add_n20_049","copy_n01_019","copy_n02_019","copy_n03_019","copy_n04_019","copy_n05_019","copy_n06_019","copy_n07_019","copy_n08_019","copy_n09_019","copy_n10_019","copy_n11_019","copy_n12_019","reverse_n01_019","reverse_n02_019","reverse_n03_019","reverse_n04_019","reverse_n05_019","reverse_n06_019","reverse_n07_019","reverse_n08_019","reverse_n09_019","reverse_n10_019","reverse_n11_019","reverse_n12_019","max_n01_019","max_n02_019","max_n03_019","max_n04_019","max_n05_019","max_n06_019","max_n07_019","max_n08_019","max_n09_019","max_n10_019","max_n11_019","max_n12_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 32 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 86,
 "original": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n20_000",
    "expected": "468986012583821540920",
    "got": "3508988828\nM:1,1=1\nM:1"
   },
   {
    "id": "add_n20_001",
    "expected": "836504739177232893621",
    "got": "95858599515\nM:1,1=1\nM:"
   },
   {
    "id": "add_n20_002",
    "expected": "863687453957329812571",
    "got": "81858585838\nM:1,1=1\nM:"
   },
   {
    "id": "add_n20_003",
    "expected": "316177502768205361021",
    "got": "9666855566\nM:1,1=1\nM:1"
   },
   {
    "id": "add_n20_004",
    "expected": "433683912159992467210",
    "got": " = = = \n The series wa"
   }
  ]
 },
 "edited": {
  "graded": 86,
  "accuracy": 0.4186,
  "accuracy_by_task": {
   "addition": 0.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "20": 0.0
  },
  "failures": [
   {
    "id": "add_n20_000",
    "expected": "468986012583821540920",
    "got": "665566565222262\nM45155"
   },
   {
    "id": "add_n20_001",
    "expected": "836504739177232893621",
    "got": "6655656565526265262626"
   },
   {
    "id": "add_n20_002",
    "expected": "863687453957329812571",
    "got": "6655666645222626262688"
   },
   {
    "id": "add_n20_003",
    "expected": "316177502768205361021",
    "got": "265666544442\nM45555555"
   },
   {
    "id": "add_n20_004",
    "expected": "433683912159992467210",
    "got": "266566564522\nM45155556"
   }
  ]
 },
 "files": [
  "work/fwd_32_original.jsonl",
  "work/fwd_32_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 4.0
}
```
Forward passes left: 18. Replies left: 144.
````

</details>

## Turn 57

The final length also scores 0/50, so the complete practice sweep shows no multi-digit success. I will aggregate unique IDs and verify full coverage before moving to prose KL. This will make the final report reproducible and avoid inflating counts from the duplicated request.

```python
# Hypothesis: the completed sweep contains all 1,720 unique practice cases and preserves every native output.
paired={}
duplicates=0
for k in range(13,33):
    os=[json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_original.jsonl").read_text().splitlines()]
    es=[json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_edited.jsonl").read_text().splitlines()]
    for o,e in zip(os,es):
        assert o["id"]==e["id"]
        duplicates+=e["id"] in paired
        paired[e["id"]]=(o,e)
expected={json.loads(s)["id"] for fn in ["addition.jsonl","native.jsonl"] for s in pathlib.Path("practice/"+fn).read_text().splitlines()}
assert set(paired)==expected
summary={}
for label, pred in [
    ("addition_N1",lambda e:e["task"]=="addition" and e["n_digits"]==1),
    ("addition_N1_8",lambda e:e["task"]=="addition" and e["n_digits"]<=8),
    ("addition_N9_20",lambda e:e["task"]=="addition" and e["n_digits"]>=9),
    *[(task,lambda e,t=task:e["task"]==t) for task in ["copy","reverse","max"]]
]:
    rows=[(o,e) for o,e in paired.values() if pred(e)]
    summary[label]={"n":len(rows),"original_correct":sum(o["correct"] for o,e in rows),
                    "edited_correct":sum(e["correct"] for o,e in rows),
                    "losses":sum(o["correct"] and not e["correct"] for o,e in rows),
                    "changed_outputs":sum(o["output"]!=e["output"] for o,e in rows)}
print(json.dumps(summary,indent=2)); print("duplicate rows excluded:",duplicates)
pathlib.Path("work/generation_summary.json").write_text(json.dumps(summary,indent=2))
with open("work/notes.md","a") as f:
    f.write("\nComplete practice sweep passes13..32, deduplicated (one duplicate N19 ID): all1000 additions/all720 native covered. Addition N1=50/50 vs6/50; every N2..20=0/50 both. N1..8=12.5% vs1.5%; N9..20=0% both. Copy240/240, reverse240/240, max239/240 both; all720 native outputs identical, zero losses. Current best remains edit3, a partial one-digit solution only. Summary saved work/generation_summary.json.\n")
print("Prior teacher-forced metadata keys:",list(json.loads(pathlib.Path("work/fwd_10/meta.json").read_text())))
for i in range(3):
    print("PROSE REQUEST",i,json.dumps(reqs[i]))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 22 of at most 300:
```
{
  "addition_N1": {
    "n": 50,
    "original_correct": 6,
    "edited_correct": 50,
    "losses": 0,
    "changed_outputs": 50
  },
  "addition_N1_8": {
    "n": 400,
    "original_correct": 6,
    "edited_correct": 50,
    "losses": 0,
    "changed_outputs": 400
  },
  "addition_N9_20": {
    "n": 600,
    "original_correct": 0,
    "edited_correct": 0,
    "losses": 0,
    "changed_outputs": 600
  },
  "copy": {
    "n": 240,
    "original_correct": 240,
    "edited_correct": 240,
    "losses": 0,
    "changed_outputs": 0
  },
  "reverse": {
    "n": 240,
    "original_correct": 240,
    "edited_correct": 240,
    "losses": 0,
    "changed_outputs": 0
  },
  "max": {
    "n": 240,
    "original_correct": 239,
    "edited_correct": 239,
    "losses": 0,
    "changed_outputs": 0
  }
}
duplicate rows excluded: 1
Prior teacher-forced metadata keys: ['texts', 'token_ids', 'lengths', 'padding', 'positions']
PROSE REQUEST 0 {"type": "teacher_forced", "model": "both", "texts": ["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Br", "M ) broadcast from the city . A television station , Havenscope ( available on @-@ campus only ) , and a radio station , WLHU ( Internet station only , with no ", "le League World Series . In 2011 , the Keystone Little League based in Lock Haven advanced to the Little League World Series and placed third in the United Stat", "rograms to members , and the Clinton Country Club maintains a private 18 @-@ hole golf course in Mill Hall . \n = = Government = = \n Lock Haven has a council @-@", "t \" Pete \" Smeltz , chairman ; Jeffrey Snyder , vice @-@ chairman , and Paul Conklin , have terms running through 2019 . \n Michael K. Hanna , a Democrat , repre", " , and Woodward Elementary , all for children enrolled in kindergarten through fifth grade . The total enrollment of these three schools com
[... cut]
````

</details>

## Turn 58

The full sweep confirms 12.5% addition accuracy for N=1–8, entirely from N=1, and 0% for N=9–20. All 720 native outputs are identical to the original. I will now measure prose KL on the reserved excerpts, requesting logits so the final aggregate can be computed directly from saved outputs.

```forward
{"type":"teacher_forced","model":"both","texts":["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Br","M ) broadcast from the city . A television station , Havenscope ( available on @-@ campus only ) , and a radio station , WLHU ( Internet station only , with no ","le League World Series . In 2011 , the Keystone Little League based in Lock Haven advanced to the Little League World Series and placed third in the United Stat","rograms to members , and the Clinton Country Club maintains a private 18 @-@ hole golf course in Mill Hall . \n = = Government = = \n Lock Haven has a council @-@","t \" Pete \" Smeltz , chairman ; Jeffrey Snyder , vice @-@ chairman , and Paul Conklin , have terms running through 2019 . \n Michael K. Hanna , a Democrat , repre"," , and Woodward Elementary , all for children enrolled in kindergarten through fifth grade . The total enrollment of these three schools combined in 2002  03 wa"," foot ( 930 m2 ) expansion to include grades seven and eight , which will make it a combined elementary and middle school . \n Lock Haven University of Pennsylva","s are available through Susquehanna Trailways , based in Avis , 10 miles ( 16 km ) northeast of Lock Haven . Pennsylvania Bicycle Route G follows Pennsylvania R"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 33 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_33/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7621
 },
 "edited": {
  "dir": "work/fwd_33/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7634
 },
 "kl_original_to_edited": {
  "mean": 0.000776,
  "max_text": 0.001265,
  "per_text_file": "work/fwd_33/kl_per_text.npy"
 },
 "meta": "work/fwd_33/meta.json",
 "s": 0.1
}
```
Forward passes left: 17. Replies left: 142.
````

</details>

## Turn 59

The first prose batch has mean KL 0.000776, with bits per character changing from 1.7621 to 1.7634. That is a small but nonzero distributional change, consistent with the earlier ordinary-prose check. I will test the next disjoint excerpts to see whether this scale persists across topics.

```forward
{"type":"teacher_forced","model":"both","texts":["\n = = Notable people = = \n Brittani Kline , winner of America 's Next Top Model ( cycle 16 ) , is a 2015 graduate of Lock Haven University . Alexander McDonald ","en Green is a fictional character , one of the six main characters who appear in the American sitcom Friends . Portrayed by actress Jennifer Aniston , the chara","olved with her friend Ross , with whom she maintains a complicated on @-@ again , off @-@ again relationship throughout the entire series . Together , the chara","fter acquiring the role and before Friends aired , Aniston was temporarily at risk of being recast because she had also been involved with another sitcom , Mudd","f the greatest television characters of all @-@ time , while the character 's second season haircut spawned an international phenomenon of its own . Named the \"","pawning her successful film career . Praised for her performance as Rachel , Aniston won both an Emmy Award for Outstanding Lead Actress in a Comedy Series and "," ( Lisa Kudrow ) , Joey Tribbiani ( Matt LeBlanc ) , and Chandler Bing ( Matthew Perry ) , while reuniting with Monica 's older brother Ross Geller ( David Schw","he resents . However , Ross eventually chooses Rachel over his girlfriend Julie ( Lauren Tom ) , and the couple dates for the remainder of the second season . H"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 34 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_34/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.6414
 },
 "edited": {
  "dir": "work/fwd_34/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.6432
 },
 "kl_original_to_edited": {
  "mean": 0.000961,
  "max_text": 0.001379,
  "per_text_file": "work/fwd_34/kl_per_text.npy"
 },
 "meta": "work/fwd_34/meta.json",
 "s": 0.1
}
```
Forward passes left: 16. Replies left: 141.
````

</details>

## Turn 60

The second prose batch has mean KL 0.000961, slightly higher than the first but still small. This does not establish universal preservation, especially given the previously observed damage on prose containing `+`. I will continue with the next ordinary-prose batch and keep that limitation separate from the aggregate.

```forward
{"type":"teacher_forced","model":"both","texts":[" character when Friends premiered , the writers had actually given Rachel the pilot 's most prominent storyline . Before deciding that Rachel and Ross would be ","ry beginning of Friends , openly voicing their frustration with Rachel  s obliviousness to Ross ' feelings for her . The episode that would ultimately transform","he couple 's dynamic in order to avoid the repetitive \" he 's pining , she 's oblivious \" pattern , using the work of author Jane Austen as inspiration on how t","ibility of a Ross and Rachel recoupling through several cliffhangers without ever putting them back together . \" According to Encyclopedia of Television author ","o the idea of having his character Ross divorce her  his third divorce  because he felt that it was taking it \" too far . \" The actor explained that \" The whole","imately become unlikeable characters and audiences would either \" resent Joey for going after a pregnant woman , or resent Rachel for rejecting him , or resent "," finally came time to write the series finale , \" The only thing [ Crane and Kauffman ] absolutely knew from very early on was that we had to get Ross and Rache","chel is portrayed by actress Jennifer Aniston , who auditioned for the role shortly after declining a position as a cast member on the sketch comedy show Saturd"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 35 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_35/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7062
 },
 "edited": {
  "dir": "work/fwd_35/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7114
 },
 "kl_original_to_edited": {
  "mean": 0.001041,
  "max_text": 0.001525,
  "per_text_file": "work/fwd_35/kl_per_text.npy"
 },
 "meta": "work/fwd_35/meta.json",
 "s": 0.1
}
```
Forward passes left: 15. Replies left: 140.
````

</details>

## Turn 61

The third prose batch has mean KL 0.001041, so the first three batches consistently show small, nonzero changes. I will save their aggregate and load the next reserved requests. These measurements assess ordinary prose only; they do not erase the known `+`-triggered failure mode.

```python
# Hypothesis: the first three prose batches have comparable KL, and later disjoint batches can test whether this is representative.
import numpy as np
kl_first=np.concatenate([np.load(f"work/fwd_{k}/kl_per_text.npy") for k in range(33,36)])
print("24-text KL mean/max:",float(kl_first.mean()),float(kl_first.max()))
print("Metadata sample:",str(json.loads(pathlib.Path("work/fwd_33/meta.json").read_text()))[:500])
with open("work/notes.md","a") as f:
    f.write(f"\nProse validation passes33..35:24 disjoint 160-character excerpts, mean per-text KL={float(kl_first.mean()):.8f}, max={float(kl_first.max()):.8f}. Logits saved for aggregation. Ordinary prose only; known plus-prose KL2.757595 remains a limitation.\n")
for i in range(3,8):
    print("PROSE REQUEST",i,json.dumps(reqs[i]))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 23 of at most 300:
```
24-text KL mean/max: 0.0009260778315365314 0.0015251036966219544
Metadata sample: {'texts': ['ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Br', 'M ) broadcast from the city . A television station , Havenscope ( available on @-@ campus only ) , and a radio station , WLHU ( Internet station only , with no ', 'le League World Series . In 2011 , the Keystone Little League based in Lock Haven advanced to the Little League World Series and placed third in the United Stat
PROSE REQUEST 3 {"type": "teacher_forced", "model": "both", "texts": ["romotions and photo shoots ; the network excluded her from these in case she would be replaced . Aniston explained , \" When we were shooting the first grouping ", " allowing Aniston to keep her role on the show , becoming its second youngest cast member at the age of 25 . Crane appreciated Aniston 's interpretation of Rach", "at each of the show 's main characters are \" of equal importance . \" As a writer , Crane preferred it this way because \" utilizing six equal players , rather th", "r episode were met during season three , which eventually increased to $ 1 million per episode by seasons nine and ten  approximately $ 25 million per year . Al", "st the show 's ratings , notably her character 's seventh season kiss with actress Winona Ryder and pregnancy arc . Aniston had been telling the press that the ", "oungest of Friends six main characters . The term \" spoiled \" is often used to describe the character 's personality during her early appearances . Encyclopaedi", "ote about Rachel in her book Jennifer Aniston : From Friends to Films that the character is \" spunky and sometimes spoiled \" , while TV Land called her \" naive ", "y , determined but undisciplined . \" Meanwhile , Liat Kornowski , writing for The Huffington Post , scribed that Rachel is a \" beautiful , cove
[... cut]
````

</details>

## Turn 62

The first 24 excerpts average KL 0.000926, with the largest excerpt mean at 0.001525. This supports a small ordinary-prose effect on those samples, not exact preservation. I will continue with the next eight excerpts to broaden the measurement.

```forward
{"type":"teacher_forced","model":"both","texts":["romotions and photo shoots ; the network excluded her from these in case she would be replaced . Aniston explained , \" When we were shooting the first grouping "," allowing Aniston to keep her role on the show , becoming its second youngest cast member at the age of 25 . Crane appreciated Aniston 's interpretation of Rach","at each of the show 's main characters are \" of equal importance . \" As a writer , Crane preferred it this way because \" utilizing six equal players , rather th","r episode were met during season three , which eventually increased to $ 1 million per episode by seasons nine and ten  approximately $ 25 million per year . Al","st the show 's ratings , notably her character 's seventh season kiss with actress Winona Ryder and pregnancy arc . Aniston had been telling the press that the ","oungest of Friends six main characters . The term \" spoiled \" is often used to describe the character 's personality during her early appearances . Encyclopaedi","ote about Rachel in her book Jennifer Aniston : From Friends to Films that the character is \" spunky and sometimes spoiled \" , while TV Land called her \" naive ","y , determined but undisciplined . \" Meanwhile , Liat Kornowski , writing for The Huffington Post , scribed that Rachel is a \" beautiful , coveted , slightly ne"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 36 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_36/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.6572
 },
 "edited": {
  "dir": "work/fwd_36/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.6575
 },
 "kl_original_to_edited": {
  "mean": 0.001105,
  "max_text": 0.00183,
  "per_text_file": "work/fwd_36/kl_per_text.npy"
 },
 "meta": "work/fwd_36/meta.json",
 "s": 0.1
}
```
Forward passes left: 14. Replies left: 138.
````

</details>

## Turn 63

This batch has mean KL 0.001105, while bits per character changes by only about 0.0003. The distinction matters: similar text likelihood can still hide changes in the full prediction distribution. I will retain KL as the primary preservation measure and test the next batch.

```forward
{"type":"teacher_forced","model":"both","texts":["achel Green ' and not from the get @-@ go make some character choices \" . Prior to this , critics and fans had long speculated whether or not Rachel is Jewish ;","e television characters who embodied Jewish stereotypes during the 1990s and often served as \" the butt of the shows ' jokes . \" Meanwhile , JDate 's Rebecca Fr","comes as they become less \" evident in later seasons of the show \" . In his article \" Princesses , Schlemiels , Punishers and Overbearing Mothers \" , Evan Coope","ister Amy ( Christina Applegate ) who remains \" selfish , condescending and narcissistic . \" \n = = Reception and legacy = = \n = = = Critical response = = = \n Cr"," , Rachel is interested in finding meaning for her life \" . Also writing for The A. V. Club , Sonia Saraiya was pleased with Rachel and Ross ' first romantic en","Bob Shayne admitted that he is attracted to Rachel , joking , \" my feelings for Rachel , I say with some embarrassment , mirror those of Gunther \" . Cosmopolita","sooner \" . \n BDCwire ranked \" The One With The Ball \" , \" The One With Rachel  s Inadvertent Kiss \" , \" The One With The Football \" , \" The One With The Fake Pa"," in which Rachel and Monica fight over a condom . Aniston revealed that Friends fans would often approach and scold her for things Rachel did that they deemed \""],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 37 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_37/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.8876
 },
 "edited": {
  "dir": "work/fwd_37/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.8901
 },
 "kl_original_to_edited": {
  "mean": 0.001511,
  "max_text": 0.002176,
  "per_text_file": "work/fwd_37/kl_per_text.npy"
 },
 "meta": "work/fwd_37/meta.json",
 "s": 0.1
}
```
Forward passes left: 13. Replies left: 137.
````

</details>

## Turn 64

Mean KL rises to 0.001511 in this batch, with one excerpt at 0.002176. Ordinary-prose effects therefore vary across inputs, even though their measured scale remains small. I will continue the predefined samples rather than selecting only low-KL examples.

```forward
{"type":"teacher_forced","model":"both","texts":["that \" Aniston was the sparkiest member of the ensemble and the one least reliant on goofball caricature \" , concluding , \" Playing the only character with whom","s  Comedy Or Musical . Karen Thomas of USA Today dubbed Aniston \" our favorite Friend \" . According to Turner Classic Movies , Aniston ultimately became \" One o","elieves \" that Jennifer Aniston likely wouldn 't have become a major movie star without Friends \" . While ranking Aniston the most attractive sitcom star of the","es in romantic comedies , save for some exceptions . Ryan Gilbey of The Guardian noted that \" Consequently , many of Aniston 's movie roles ... have been Rachel","o lose the Rachel tag that has made her one of the world 's most recognisable faces \" . \n = = = Relationships = = = \n Rachel has had several romantic relationsh","ences would not have been happy had the couple not ultimately reunited . Contrastingly , The Wire 's Joe Reid is of the opinion that the show 's second season i","s at fault : Rachel for suggesting that they take a break from their relationship , or Ross for sleeping with another woman immediately afterwards . Writing for"," off ... on ( a break ! ) duo , \" while Us Weekly and BuzzFeed ranked them the first and second best television couple , respectively . TV Guide ranked Ross and"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 38 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_38/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7216
 },
 "edited": {
  "dir": "work/fwd_38/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7222
 },
 "kl_original_to_edited": {
  "mean": 0.001003,
  "max_text": 0.001305,
  "per_text_file": "work/fwd_38/kl_per_text.npy"
 },
 "meta": "work/fwd_38/meta.json",
 "s": 0.1
}
```
Forward passes left: 12. Replies left: 136.
````

</details>

## Turn 65

The next batch returns to mean KL 0.001003, showing that the previous increase was not uniform across excerpts. The candidate still changes ordinary prose slightly rather than preserving it exactly. I will test the next reserved batch, keeping all results in the final aggregate.

```forward
{"type":"teacher_forced","model":"both","texts":["ees \" , while Mahogany Clayton of StyleBlazer believes that the character \" managed to dominate every fashion trend that passed by her radar in the most stylish","riends 15th on the website 's list of \" 50 TV Shows That Changed the Way We Dress \" , citing Rachel 's \" impressive \" wardrobe . InStyle ranked Friends the 36th","the \" Rachel \" refers to a bouncy layered shag inspired by the way in which Aniston wore her hair on Friends between 1994 and 1996 , during the first and second","y Express determined that the hairstyle was most popular among British women , who went to hair salons \" clutching magazine pictures of Aniston \" and asking hai","oolley described it as \" a cut that flatters almost everyone , designed to make women look beautiful \" . The \" Rachel \" is often ranked among the greatest and m"," television history . The Sydney Morning Herald ranked it the second greatest television hairstyle , while Metro ranked the \" Rachel \" the character 's second @","e phenomenon it was \" . Opining that Friends spawned few memorable catchphrases in comparison to its contemporaries , Tom Jicha of The Baltimore Sun attributed ","( Marlo Thomas ) is trying to pattern her own life after hers , lamenting , \" Couldn 't she just copy my haircut ? \" Although Aniston eventually abandoned the \""],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 39 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_39/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7734
 },
 "edited": {
  "dir": "work/fwd_39/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7736
 },
 "kl_original_to_edited": {
  "mean": 0.001071,
  "max_text": 0.001706,
  "per_text_file": "work/fwd_39/kl_per_text.npy"
 },
 "meta": "work/fwd_39/meta.json",
 "s": 0.1
}
```
Forward passes left: 11. Replies left: 135.
````

</details>

## Turn 66

Mean KL is 0.001071 in this batch, close to several preceding measurements. Most excerpts so far concern one topic, so they should not be treated as independent evidence of broad domain coverage. The next batch shifts to historical material and provides a useful topic change.

```forward
{"type":"teacher_forced","model":"both","texts":["ogram of clearing and restoration . When Syria declared independence in 1946 , it assumed control . Today , a village called al @-@ Husn exists around the castl"," The modern Arabic word for a castle is Kalaa (  ) , but Krak des Chavaliers is known as a \" Hosn \" (  ) , or \" fort \" . This derives from the name of an earlie","etre @-@ high ( 2 @,@ 130 ft ) hill east of Tartus , Syria , in the Homs Gap . On the other side of the gap , 27 kilometres ( 17 mi ) away , was the 12th @-@ ce","n the 1140s , included the Castle of the Kurds , the towns of Rafanea and Montferrand , and the Buqai 'ah plain separating Homs and Tripoli . Homs was never und"," at the site of the future castle in 1031 CE , hence the castle 's Arabic name \" Hisn al @-@ Akrad \" ( Castle of the Kurds ) . The site was strategically locate"," . The crusaders briefly occupied the castle in February of the same year but abandoned it when they continued their march towards Jerusalem . Permanent occupat","perty in the Levant to the Hospital of St John . Early donations were in the newly formed Kingdom of Jerusalem , but over time the order extended its holdings t","the county . According to historian Jonathan Riley @-@ Smith , the Hospitallers effectively established a \" palatinate \" within Tripoli . The property included "],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 40 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_40/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7748
 },
 "edited": {
  "dir": "work/fwd_40/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7764
 },
 "kl_original_to_edited": {
  "mean": 0.000858,
  "max_text": 0.0014,
  "per_text_file": "work/fwd_40/kl_per_text.npy"
 },
 "meta": "work/fwd_40/meta.json",
 "s": 0.1
}
```
Forward passes left: 10. Replies left: 134.
````

</details>

## Turn 67

The historical excerpts have mean KL 0.000858, within the range seen earlier. I will update the running summary and retrieve the next five batches. The purpose remains measurement, not tuning: the saved edit has not changed during this validation sweep.

```python
# Hypothesis: the 64 tested prose excerpts show a small but nonzero aggregate KL without masking input-to-input variation.
ks=np.concatenate([np.load(f"work/fwd_{k}/kl_per_text.npy") for k in range(33,41)])
print("texts/mean/median/max KL:",len(ks),float(ks.mean()),float(np.median(ks)),float(ks.max()))
with open("work/notes.md","a") as f:
    f.write(f"\nProse validation through pass40:64 excerpts, mean KL={float(ks.mean()):.8f}, median={float(np.median(ks)):.8f}, max-text={float(ks.max()):.8f}. Edit3 unchanged;10 prose passes left.\n")
for i in range(8,13):
    print("PROSE REQUEST",i,json.dumps(reqs[i]))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 24 of at most 300:
```
texts/mean/median/max KL: 64 0.001040824456140399 0.0009594191797077656 0.0021758689545094967
PROSE REQUEST 8 {"type": "teacher_forced", "model": "both", "texts": ["ircuit of walls surrounding the inner court that pre @-@ dated the current outer walls , no trace of it has been discovered . \n The first half of the 13th centu", "in 1218 and proclaimed the castle the \" key of the Christian lands \" . He was so impressed with the castle that he gave a yearly income of 60 marks to the Maste", "uccessors warred between themselves . The proximity of Krak des Chevaliers to Muslim territories allowed it to take on an offensive role , acting as a base from", "ers took a turn for the worse . A Muslim army estimated to number 10 @,@ 000 men ravaged the countryside around the castle in 1252 after which the Order 's fina", "bute to the Hospitallers at Krak des Chevaliers no longer felt intimidated into doing so . \n Baibars ventured into the area around Krak des Chevaliers in 1270 a", "en blockaded by Mamluk forces for several days . Of the three Arabic accounts of the siege only one was contemporary , that of Ibn Shaddad , although he was not", "ork possibly defended by a timber palisade . On 29 March , the attackers undermined a tower in the southwest corner causing it to collapse whereupon Baibars ' a", "ltan spared their lives . The new owners of the castle undertook repairs , focused mainly on the outer ward . The Hospitaller chapel was converted to a mosque a"], "return": {}}
PROSE REQUEST 9 {"type": "teacher_forced", "model": "both", "texts": ["y of the al @-@ Husn 's roughly 9 @,@ 000 Muslim residents benefit economically from the tourism generated by the site . \n = = = Syrian Civil War = = = \n During", "castle and the village of al @-@ Hosn from rebel forces on March 20 , 2014 , although the extent of damage from earlier mortar hits remained unclear . \n = = Arc", " castle scientifically designed as a fighting machine surel
[... cut]
````

</details>

## Turn 68

Across 64 excerpts, mean KL is 0.001041 and the largest excerpt mean is 0.002176. These are practice-text measurements, not the hidden held-out evaluation. I will continue with historical excerpts to check whether the same approximate range persists.

```forward
{"type":"teacher_forced","model":"both","texts":["ircuit of walls surrounding the inner court that pre @-@ dated the current outer walls , no trace of it has been discovered . \n The first half of the 13th centu","in 1218 and proclaimed the castle the \" key of the Christian lands \" . He was so impressed with the castle that he gave a yearly income of 60 marks to the Maste","uccessors warred between themselves . The proximity of Krak des Chevaliers to Muslim territories allowed it to take on an offensive role , acting as a base from","ers took a turn for the worse . A Muslim army estimated to number 10 @,@ 000 men ravaged the countryside around the castle in 1252 after which the Order 's fina","bute to the Hospitallers at Krak des Chevaliers no longer felt intimidated into doing so . \n Baibars ventured into the area around Krak des Chevaliers in 1270 a","en blockaded by Mamluk forces for several days . Of the three Arabic accounts of the siege only one was contemporary , that of Ibn Shaddad , although he was not","ork possibly defended by a timber palisade . On 29 March , the attackers undermined a tower in the southwest corner causing it to collapse whereupon Baibars ' a","ltan spared their lives . The new owners of the castle undertook repairs , focused mainly on the outer ward . The Hospitaller chapel was converted to a mosque a"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 41 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_41/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7011
 },
 "edited": {
  "dir": "work/fwd_41/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7021
 },
 "kl_original_to_edited": {
  "mean": 0.000796,
  "max_text": 0.00117,
  "per_text_file": "work/fwd_41/kl_per_text.npy"
 },
 "meta": "work/fwd_41/meta.json",
 "s": 0.1
}
```
Forward passes left: 9. Replies left: 132.
````

</details>

## Turn 69

This batch has mean KL 0.000796, below the running average. It supplies another small-effect measurement but does not change the known limitations of the edit. I will now check the next architecture and history excerpts under the same paired setup.

```forward
{"type":"teacher_forced","model":"both","texts":["y of the al @-@ Husn 's roughly 9 @,@ 000 Muslim residents benefit economically from the tourism generated by the site . \n = = = Syrian Civil War = = = \n During","castle and the village of al @-@ Hosn from rebel forces on March 20 , 2014 , although the extent of damage from earlier mortar hits remained unclear . \n = = Arc"," castle scientifically designed as a fighting machine surely reached its apogee in great buildings like Margat and Crac des Chevaliers . \" \n Krak des Chevaliers","To the south of the outer ward was a triangular outwork and the Crusaders may have intended to build stone walls and towers around it . It is unknown how it was","which projected slightly . The main entrance was between two towers on the eastern side , and there was a postern gate in the northwest tower . At the center wa",", with a narrow gap between them in the west and south which was turned into a gallery from which defenders could unleash missiles . In this area , the walls we","chicolations were absent from the southern face . The area between the inner court and the outer walls was narrow and not used for accommodation . In the east ,","ates from the 13th @-@ century remodelling . The tracery and delicate decoration is a sophisticated example of Gothic architecture , probably dating from the 12"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 42 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_42/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.5532
 },
 "edited": {
  "dir": "work/fwd_42/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.5495
 },
 "kl_original_to_edited": {
  "mean": 0.000693,
  "max_text": 0.000851,
  "per_text_file": "work/fwd_42/kl_per_text.npy"
 },
 "meta": "work/fwd_42/meta.json",
 "s": 0.1
}
```
Forward passes left: 8. Replies left: 131.
````

</details>

## Turn 70

Mean KL is 0.000693, while text likelihood happens to improve slightly in this batch. That improvement is not evidence of better general language modeling; it is just one sample’s outcome. I will continue with the next reserved excerpts, which introduce literary material.

```forward
{"type":"teacher_forced","model":"both","texts":["nces were accessed by a wall @-@ walk known as a chemin de ronde . In the opinion of historian Hugh Kennedy the defences of the outer wall were \" the most elabo"," ft ) , and along its length were murder @-@ holes which allowed defenders to shower attackers with missiles . Anyone going straight ahead rather than following","el and the chapel outside the main entrance , which no longer survives . Writing in 1982 , historian Jaroslav Folda noted that at the time there had been little","ng Earnest , A Trivial Comedy for Serious People is a play by Oscar Wilde . First performed on 14 February 1895 at the St James 's Theatre in London , it is a f","areer so far . Its high farce and witty dialogue have helped make The Importance of Being Earnest Wilde 's most enduringly popular play . \n The successful openi","success , to be closed after 86 performances . After his release , he published the play from exile in Paris , but he wrote no further comic or dramatic work . "," After the success of Wilde 's plays Lady Windermere 's Fan and A Woman of No Importance , Wilde 's producers urged him to write further plays . In July 1894 he"," agreement among Wilde scholars that the most important influence on the play was W. S. Gilbert 's 1877 farce Engaged ; Wilde borrowed from Gilbert not only sev"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 43 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_43/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7592
 },
 "edited": {
  "dir": "work/fwd_43/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7599
 },
 "kl_original_to_edited": {
  "mean": 0.000897,
  "max_text": 0.001232,
  "per_text_file": "work/fwd_43/kl_per_text.npy"
 },
 "meta": "work/fwd_43/meta.json",
 "s": 0.1
}
```
Forward passes left: 7. Replies left: 130.
````

</details>

## Turn 71

The mixed historical and literary batch has mean KL 0.000897. No large ordinary-prose shift has appeared in these reserved samples so far. I will test the next literary batch, including lists and quotations, to include more formatting variation.

```forward
{"type":"teacher_forced","model":"both","texts":["cast was : \n John Worthing , J.P.  George Alexander \n Algernon Moncrieff  Allan Aynesworth \n Rev. Canon Chasuble , D.D.  H. H. Vincent \n Merriman  Frank Dyall \n","icemen to bar his entrance . Nevertheless , he continued harassing Wilde , who eventually launched a private prosecution against the peer for criminal libel , t","rne on 10 August 1895 , presented by Dion Boucicault , Jr. and Robert Brough , and the play was an immediate success . Wilde 's downfall in England did not affe"," , \" I go to the theatre to be moved to laughter . \" Later in a letter he said , the play , though \" extremely funny \" , was Wilde 's \" first really heartless [","g in seriousness , and said \" It is of nonsense all compact , and better nonsense , I think , our stage has not seen . \" H. G. Wells , in an unsigned review for","ommented , \" The story is almost too preposterous to go without music . \" Mary McCarthy , in Sights and Spectacles ( 1959 ) , however , and despite thinking the","m of the play . \n = = = Revivals = = = \n Until after Wilde 's death in 1900 his name remained disgraced , and few discussed , let alone performed , his work in ","to restore his reputation as an author . Alexander presented another revival of The Importance at the St James 's in 1909 , when he and Aynesworth reprised thei"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 44 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_44/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.8459
 },
 "edited": {
  "dir": "work/fwd_44/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.8493
 },
 "kl_original_to_edited": {
  "mean": 0.001249,
  "max_text": 0.001926,
  "per_text_file": "work/fwd_44/kl_per_text.npy"
 },
 "meta": "work/fwd_44/meta.json",
 "s": 0.1
}
```
Forward passes left: 6. Replies left: 129.
````

</details>

## Turn 72

This batch has mean KL 0.001249 and a maximum excerpt mean of 0.001926. The values remain small but confirm that preservation is approximate rather than exact. I will check the next eight excerpts before preparing the final five-batch summary.

```forward
{"type":"teacher_forced","model":"both","texts":["; it also featured Wilde as a character  the play opens with him drinking in a Parisian caf , dreaming of his play . The Melbourne Theatre Company staged a prod","as Gwendolen . It was nominated for three Tony Awards . \n The play was also presented internationally , in Singapore , in October 2004 , by the British Theatre ","tte case bears the inscription , \" From little Cecily , with her fondest love to her dear Uncle Jack . \" ' Ernest ' is forced to admit to living a double life .","refuses to tell Algernon the location of his country estate . \n Gwendolen and her formidable mother Lady Bracknell now call on Algernon who distracts Lady Brack","r undying love . As Jack gives her his address in the country , Algernon surreptitiously notes it on the cuff of his sleeve : Jack 's revelation of his pretty a","en him \" Ernest \" . \n Jack , meanwhile , has decided to abandon his double life . He arrives in full mourning and announces his brother 's death in Paris of a s","ation of Cecily 's trust fund soon dispels Lady Bracknell 's initial doubts over the young lady 's suitability , but any engagement is forbidden by her guardian","left at Victoria Station . Jack produces the very same handbag , showing that he is the lost baby , the elder son of Lady Bracknell 's late sister , and thus in"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 45 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_45/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7897
 },
 "edited": {
  "dir": "work/fwd_45/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7891
 },
 "kl_original_to_edited": {
  "mean": 0.001123,
  "max_text": 0.002752,
  "per_text_file": "work/fwd_45/kl_per_text.npy"
 },
 "meta": "work/fwd_45/meta.json",
 "s": 0.1
}
```
Forward passes left: 5. Replies left: 128.
````

</details>

## Turn 73

Mean KL is 0.001123, with one excerpt reaching 0.002752—the highest excerpt mean in this sweep so far. I will retain that maximum alongside the average rather than reporting only the favorable aggregate. Next I will retrieve the remaining five requests and update the notebook.

```python
# Hypothesis: the complete prose estimate should include the observed higher-KL excerpts, not just its mean.
ks=np.concatenate([np.load(f"work/fwd_{k}/kl_per_text.npy") for k in range(33,46)])
print("texts/mean/median/max KL:",len(ks),float(ks.mean()),float(np.median(ks)),float(ks.max()))
with open("work/notes.md","a") as f:
    f.write(f"\nProse through pass45:104 excerpts, mean KL={float(ks.mean()):.8f}, max-text={float(ks.max()):.8f}. Five remaining prose passes, no further edits planned. Final submission must explicitly state failure on multi-digit addition and plus-prose interference.\n")
for i in range(13,18):
    print("PROSE REQUEST",i,json.dumps(reqs[i]))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 25 of at most 300:
```
texts/mean/median/max KL: 104 0.0010064375819638371 0.0009218670893460512 0.002752342727035284
PROSE REQUEST 13 {"type": "teacher_forced", "model": "both", "texts": ["was considered to be the over @-@ riding societal value , originating in religious attempts to reform the lower classes , it spread to the upper ones too throug", "l matrimonial rites , whereby suitors admit their weaknesses to their prospective brides , but the foibles they excuse are ridiculous , and the farce is built o", "alysis of John Worthing 's suitability as a husband , places her entire faith in a Christian name , declaring in Act I , \" The only really safe name is Ernest \"", "defends himself against her \" A handbag ? \" with the clarification , \" The Brighton Line \" . At the time , Victoria Station consisted of two separate but adjace", "absence of  homosexual desire \" . On re @-@ reading the play after his release from prison , Wilde said : \" It was extraordinary reading the play over . How I u", "nnot work the miracle /  ' Tis Ernest sets my heart a @-@ flame . \" The word \" earnest \" may also have been a code @-@ word for homosexual , as in : \" Is he ear", "as in the 1980s and I immediately consulted Sir John Gielgud whose own performance of Jack Worthing in the same play was legendary and whose knowledge of theatr", "e taken train to Banbury , met a schoolboy there , and arranged a second secret meeting with him at Sunbury . \n = = Dramatic analysis = = \n = = = Use of languag"], "return": {}}
PROSE REQUEST 14 {"type": "teacher_forced", "model": "both", "texts": [", the basic structure which underlies his earlier social comedies , and basing the story entirely on the Earnest / Ernest verbal conceit . Now freed from \" livi", "rchanged \" . Similarly , Wilde 's use of dialogue mocks the upper classes of Victorian England lending the play a satirical tone . Reinhart further stipulates t", "lse would , Leonard Smithers agreed with Wilde 
[... cut]
````

</details>

## Turn 74

The running mean over 104 excerpts is 0.001006, with the maximum excerpt mean at 0.002752. The remaining requests cover literary criticism, biography, and naval history. I will complete them unchanged so the final prose estimate reflects the entire reserved sample.

```forward
{"type":"teacher_forced","model":"both","texts":["was considered to be the over @-@ riding societal value , originating in religious attempts to reform the lower classes , it spread to the upper ones too throug","l matrimonial rites , whereby suitors admit their weaknesses to their prospective brides , but the foibles they excuse are ridiculous , and the farce is built o","alysis of John Worthing 's suitability as a husband , places her entire faith in a Christian name , declaring in Act I , \" The only really safe name is Ernest \"","defends himself against her \" A handbag ? \" with the clarification , \" The Brighton Line \" . At the time , Victoria Station consisted of two separate but adjace","absence of  homosexual desire \" . On re @-@ reading the play after his release from prison , Wilde said : \" It was extraordinary reading the play over . How I u","nnot work the miracle /  ' Tis Ernest sets my heart a @-@ flame . \" The word \" earnest \" may also have been a code @-@ word for homosexual , as in : \" Is he ear","as in the 1980s and I immediately consulted Sir John Gielgud whose own performance of Jack Worthing in the same play was legendary and whose knowledge of theatr","e taken train to Banbury , met a schoolboy there , and arranged a second secret meeting with him at Sunbury . \n = = Dramatic analysis = = \n = = = Use of languag"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 46 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_46/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.8298
 },
 "edited": {
  "dir": "work/fwd_46/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.8341
 },
 "kl_original_to_edited": {
  "mean": 0.000997,
  "max_text": 0.00177,
  "per_text_file": "work/fwd_46/kl_per_text.npy"
 },
 "meta": "work/fwd_46/meta.json",
 "s": 0.1
}
```
Forward passes left: 4. Replies left: 126.
````

</details>

## Turn 75

This batch’s mean KL is 0.000997, essentially at the running average. The result supports small ordinary-prose changes on the sampled text, but nothing stronger. I will proceed with the translation and publication excerpts, still comparing both models on exactly the same inputs.

```forward
{"type":"teacher_forced","model":"both","texts":[", the basic structure which underlies his earlier social comedies , and basing the story entirely on the Earnest / Ernest verbal conceit . Now freed from \" livi","rchanged \" . Similarly , Wilde 's use of dialogue mocks the upper classes of Victorian England lending the play a satirical tone . Reinhart further stipulates t","lse would , Leonard Smithers agreed with Wilde to publish the two final plays . Wilde proved to be a diligent reviser , sending detailed instructions on stage d","discovered inside a handbag in an Oxfam shop in Nantwich , Cheshire . Staff were unable to trace the donor . It was sold for  650 . \n = = = In translation = = =","nd also a matching masculine proper name ( \" Ernst \" ) . The meaning and tenor of the wordplay are exactly the same . Yet there are many different possible titl","ge . \n Four main strategies have been used by translators . The first leaves all characters ' names unchanged and in their original spelling : thus the name is ","nce , in Italian , these versions variously call the play L 'importanza di essere Franco / Severo / Fedele , the given names being respectively the values of ho"," . Lastly , one translation gave the name an Italianate touch by rendering it as Ernesto ; this work liberally mixed proper nouns from both languages . \n = = Ad"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 47 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_47/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.9273
 },
 "edited": {
  "dir": "work/fwd_47/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.9244
 },
 "kl_original_to_edited": {
  "mean": 0.001005,
  "max_text": 0.00131,
  "per_text_file": "work/fwd_47/kl_per_text.npy"
 },
 "meta": "work/fwd_47/meta.json",
 "s": 0.1
}
```
Forward passes left: 3. Replies left: 125.
````

</details>

## Turn 76

Mean KL remains near 0.001 in this batch, despite a small improvement in its text likelihood. I will not interpret that likelihood fluctuation as a gain beyond this sample. The next batch adds biography excerpts alongside broadcast-history text.

```forward
{"type":"teacher_forced","model":"both","texts":["arson as Jack Worthing . Further broadcasts of the play followed in 1927 and 1936 . In 1977 , BBC Radio 4 broadcast the four @-@ act version of the play , with ","oncrieff , John Moffatt as Canon Chasuble , Miriam Margolyes as Miss Prism , Samantha Bond as Gwendolen and Amanda Root as Cecily . The production was later iss","nnah York , Fenella Fielding , Pamela Brown and Irene Handl . \n BBC television transmissions of the play have included a 1974 Play of the Month version starring","Brown ( Gwendolen ) and Celia Johnson ( Cecily ) . \n Other audio recordings include a \" Theatre Masterworks \" version from 1953 , directed and narrated by Marga","progressed through the ranks to lieutenant . He was involved with the Third Anglo @-@ Ashanti War of 1873  4 , afterwards being stationed in East Africa for the","inland , including a failed attempt to stop German expansion in East Africa . In October 1891 Mathews was appointed First Minister to the Zanzibar government , ","erupted out of an attempt to bypass the requirement that new Sultans must be vetted by the British consul . During his time as first minister Mathews continued ","e Brilliant Star of Zanzibar . Mathews died of malaria in Zanzibar on 11 October 1901 . \n = = Early life and career = = \n Mathews was born at Funchal on Madeira"],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 48 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_48/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7662
 },
 "edited": {
  "dir": "work/fwd_48/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.7676
 },
 "kl_original_to_edited": {
  "mean": 0.001035,
  "max_text": 0.001782,
  "per_text_file": "work/fwd_48/kl_per_text.npy"
 },
 "meta": "work/fwd_48/meta.json",
 "s": 0.1
}
```
Forward passes left: 2. Replies left: 124.
````

</details>

## Turn 77

The latest batch has mean KL 0.001035, again close to the running estimate. Two reserved prose batches remain, and no further edits are planned. I will complete the biography batch next, then summarize all measurements before submission.

```forward
{"type":"teacher_forced","model":"both","texts":["amboya where a 60 @-@ man garrison was established . This had been reduced to a mere handful of men by the mid @-@ 1880s but the expedition proved that the Sult","ned by Hindi bin Hattam . This dhow had around 100 slaves on board and was transporting them between Pemba and Zanzibar . Captain Brownrigg led a boarding party","Town harbour and threatened the Sultan into signing away the territories which would later form German East Africa . Further territories were ceded to the Germa","nstitutional government in Zanzibar , Mathews was appointed First Minister , despite some hostility from Sultan Ali bin Said . In this capacity Mathews was \" ir","e Zanzibar government for his role with his appointment as a first class member of the Order of the Brilliant Star of Zanzibar , which he was granted licence by","e in Zanzibar by the British government which appointed him a Companion of the Order of St Michael and St George in 1880 and a Companion of the Order of the Bat"," appointed a Knight Commander of the Order of St Michael and St George in 1894 . He was also awarded membership of the Order of the Crown by the German governme","d 's administration . Mathews ' helped to arrange the succession of a pro @-@ British Sultan , Hamoud bin Mohammed , as Khalid 's successor . Mathews continued "],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 49 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_49/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.5954
 },
 "edited": {
  "dir": "work/fwd_49/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.5977
 },
 "kl_original_to_edited": {
  "mean": 0.000967,
  "max_text": 0.001417,
  "per_text_file": "work/fwd_49/kl_per_text.npy"
 },
 "meta": "work/fwd_49/meta.json",
 "s": 0.1
}
```
Forward passes left: 1. Replies left: 123.
````

</details>

## Turn 78

The biography batch has mean KL 0.000967, with no unusually large excerpt mean. The final pass will cover naval-history excerpts and finish the reserved prose sample. Afterward I will submit the unchanged candidate explicitly as a partial one-digit result, not as a solution to multi-digit addition.

```forward
{"type":"teacher_forced","model":"both","texts":["85 other troops but Oman died soon after and a more pliable sultan , Omar bin Hamid , was appointed to govern on behalf of Zanzibar , bringing the affair to a c","MS Boreas was a B @-@ class destroyer built for the Royal Navy around 1930 . Initially assigned to the Mediterranean Fleet , she was transferred to the Home Fle",". She was renamed Salamis and served in the Aegean for the rest of the war . Salamis became a training ship after the war until she was returned to Britain and ","turbines was provided by three Admiralty 3 @-@ drum boilers . Boreas carried a maximum of 390 long tons ( 400 t ) of fuel oil that gave her a range of 4 @,@ 800","ed with two above @-@ water quadruple torpedo tube mounts for 21 @-@ inch ( 533 mm ) torpedoes . One depth charge rail and two throwers were fitted ; 20 depth c","th the replacement of the 12 @-@ pounder high @-@ angle gun with additional depth charge stowage . The 2 @-@ pounder mounts were replaced during the war by 20 @","ications equipment . After her commissioning , she was assigned to the 4th Destroyer Flotilla with the Mediterranean Fleet until September 1936 when it was tran","ortsmouth that lasted until 11 June . Boreas escorted the royal yacht Victoria and Albert during the Royal Tour of Scotland from 26 July to 4 August . The ship "],"return":{"sites":["logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 50 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_50/original",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.5761
 },
 "edited": {
  "dir": "work/fwd_50/edited",
  "arrays": {
   "token_logprobs": [
    8,
    161
   ],
   "logits": [
    8,
    161,
    99
   ],
   "token_ids": [
    8,
    161
   ]
  },
  "mean_bits_per_char": 1.5763
 },
 "kl_original_to_edited": {
  "mean": 0.000736,
  "max_text": 0.001289,
  "per_text_file": "work/fwd_50/kl_per_text.npy"
 },
 "meta": "work/fwd_50/meta.json",
 "s": 0.1
}
```
Forward passes left: 0. Replies left: 122. That was the last pass: submit your final edit with save_edit ("final": true).
````

</details>

## Turn 79

The final prose batch has mean KL 0.000736. Testing is complete: the edit improves one-digit addition, fails every tested multi-digit problem, and preserves all 720 native outputs. I will record the final aggregate and verify the frozen tensors before submitting, including the known interference on prose containing `+`.

```python
# Hypothesis: the final staged candidate is unchanged from the tested edit and satisfies the frozen-weight and fixed-capacity constraints.
import torch, shutil
all_kl=np.concatenate([np.load(f"work/fwd_{k}/kl_per_text.npy") for k in range(33,51)])
prose_summary={
    "texts":int(all_kl.size),
    "characters":sum(len(t) for k in range(33,51) for t in json.loads(pathlib.Path(f"work/fwd_{k}/meta.json").read_text())["texts"]),
    "mean_kl_original_to_edited":float(all_kl.mean()),
    "median_text_kl":float(np.median(all_kl)),
    "max_text_kl":float(all_kl.max()),
    "source":"practice text, not hidden held-out text"
}
assert torch.equal(load("embed.weight"),load_original("embed.weight"))
assert torch.equal(load("final_norm.weight"),load_original("final_norm.weight"))
assert added_params()==0
changed=[name for name in NAMES if not torch.equal(load(name),load_original(name))]
print("Final prose summary:",prose_summary)
print("Changed tensors:",len(changed),"added parameters:",added_params(),"architecture:",arch())
pathlib.Path("work/prose_summary.json").write_text(json.dumps(prose_summary,indent=2))
shutil.copytree("work/edited","work/final_candidate",dirs_exist_ok=True)
with open("work/notes.md","a") as f:
    f.write("\nFINAL: all50 forward passes used; candidate3 unchanged throughout final validation. "
            +json.dumps(prose_summary)+
            "\nSubmission is explicitly PARTIAL: one-digit addition50/50 vs6/50; N2..20 each0/50. "
            "Copy240/240, reverse240/240, max239/240 for both; all720 native outputs identical. "
            "Known counterexample to full preservation: plus-containing prose KL2.757595 in pass10. "
            "Frozen embed/final_norm verified unchanged, zero added parameters. "
            "Final staged backup:work/final_candidate; detailed results:work/generation_summary.json and work/prose_summary.json.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 26 of at most 300:
```
Final prose summary: {'texts': 144, 'characters': 23040, 'mean_kl_original_to_edited': 0.0009901680750772357, 'median_text_kl': 0.0009265117114409804, 'max_text_kl': 0.002752342727035284, 'source': 'practice text, not hidden held-out text'}
Changed tensors: 42 added parameters: 0 architecture: {'n_heads': [8, 8, 8, 8, 8, 8], 'd_mlp': [768, 768, 768, 768, 768, 768]}

```
Forward passes left: 0. Replies left: 121.
````

</details>

## Turn 80

I am submitting a **partial result**, not a successful multi-digit solution. One-digit practice accuracy improved from 6/50 to 50/50, but N=2–20 remained 0/950. All 720 native outputs were unchanged; ordinary practice-prose KL averaged 0.000990, though prose containing `+` suffered substantial interference. The architecture and frozen weights are unchanged.

```save_edit
{"final":true,"note":"Submit tested candidate3, explicitly a partial one-digit solution. Changed 42 existing attention/MLP projection tensors, with no added capacity and no changes to embed.weight or final_norm.weight. The closed-form construction reserves a residual direction orthogonal to the embeddings/output for a plus marker, projects ordinary block computations away from that direction, reuses one value component of layer0 head1 to write the marker, and reuses ten final-layer MLP neurons for marker-gated one-digit arithmetic readout. A closed-form compensation reduces the effects of replacing those neurons. No gradient training or iterative loss fitting was used. Full paired practice validation: N1 addition50/50 edited versus6/50 original; every N2..20 zero/50 for both. Thus N1..8 accuracy12.5% versus1.5%, and N9..20 zero for both. Copy240/240, reverse240/240, max239/240 for both, with all720 native outputs identical. On144 reserved practice-prose excerpts totaling23040 characters, mean original-to-edited KL0.000990168, maximum excerpt mean0.002752343. These are not hidden held-out measurements. Known limitation: plus-containing prose had KL2.757595 in an earlier check. The marker-gated construction supports expecting one-digit benefits with small ordinary-prose effects, but there is no evidence of multi-digit generalization or complete off-task preservation. Final summaries and backup are in work/generation_summary.json, work/prose_summary.json, work/notes.md, and work/final_candidate."}
```
