# pilot6: editor transcript
Each turn: the editor's reply, then the start of the harness output it received next (cut at 2000 characters). Full per-turn logs are in git history.

## Turn 1

I’ll first inspect the model implementation and practice-data layout without changing any weights. Then I’ll measure its existing addition and native-task behaviour to establish a baseline. This will help distinguish useful existing mechanisms from capabilities that need new capacity.

```python
# Hypothesis: the architecture exposes suitable insertion points for an addition-specific circuit.
import os, json, torch, numpy as np
from pathlib import Path
print(Path("model/model.py").read_text())
print("\nTOKENIZER\n", Path("model/tokenizer.py").read_text())
print("\nCONFIG", CFG, "\nARCH", arch())
for filename in ["practice/addition.jsonl", "practice/native.jsonl"]:
    rows = [json.loads(line) for line in Path(filename).read_text().splitlines()]
    print(filename, len(rows), rows[:2])
Path("work/notes.md").write_text(
    "# Addition weight-edit notebook\n"
    "No edits yet. Inspecting implementation and preparing original-model baselines.\n"
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

The implementation allows new heads and neurons without changing the original embedding or output layer. That makes an addition-specific extension possible, though it still shares the model’s residual stream and could disturb other tasks. I’ll first test representative addition lengths alongside copy, reverse and max, looking for any existing arithmetic or operand-copying behaviour.

```forward
{"type":"generate","model":"original","ids":["add_n01_000","add_n01_001","add_n02_000","add_n02_001","add_n03_000","add_n04_000","add_n05_000","add_n06_000","add_n07_000","add_n08_000","add_n09_000","add_n10_000","add_n12_000","add_n16_000","add_n20_000","copy_n01_000","copy_n04_000","copy_n08_000","copy_n12_000","reverse_n01_000","reverse_n04_000","reverse_n08_000","reverse_n12_000","max_n01_000","max_n04_000","max_n08_000","max_n12_000"],"prompts":["A:398+259=","A:000+000=","A:999+100="]}
```

<details><summary>harness output</summary>

````text
Forward pass 1 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "original",
 "items": 30,
 "original": {
  "graded": 27,
  "accuracy": 0.4444,
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
   "12": 0.0,
   "16": 0.0,
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
    "id": "add_n03_000",
    "expected": "0390",
    "got": "88122"
   }
  ],
  "custom_outputs": [
   "3982299\nC:0555=0555\nM:5,5=5\nC:55",
   "0000000\nC:055=055\nM:5,5=5\nC:025=",
   "9991190\nC:0555=0555\nM:5,5=5\nC:55"
  ]
 },
 "files": [
  "work/fwd_1.jsonl"
 ],
 "s": 0.7
}
```
Forward passes left: 49. Replies left: 198.
````

</details>

## Turn 3

The baseline solved all 12 sampled native problems but none of the 15 addition problems; its addition outputs mostly resembled copying. I’ll now prepare teacher-forced examples to examine whether the residual stream already exposes the two operand digits needed at each answer position. If those digits are accessible, a small arithmetic extension may be enough; otherwise, new attention routing will be necessary.

```python
# Hypothesis: native digit-task circuitry already exposes useful operand and position information on addition inputs.
adds = [json.loads(s) for s in Path("practice/addition.jsonl").read_text().splitlines()]
native = [json.loads(s) for s in Path("practice/native.jsonl").read_text().splitlines()]
print("Native IDs:", [r["id"] for r in native[::60]])
probe_add = [r for r in adds if int(r["id"].split("_")[-1]) < 3]
probe_native = [r for r in native if r["n_digits"] in [1,4,8,12] and r["id"].endswith("_000")]
text_source = Path("practice/text.txt").read_text()
probe_text = [text_source[i:i+180] for i in range(0,1440,180)]
probe_strings = [r["prompt"]+r["answer"]+"\n" for r in probe_add+probe_native]+probe_text
Path("work/probe_strings.json").write_text(json.dumps(probe_strings))
print(json.dumps({"type":"teacher_forced","model":"original","texts":probe_strings,
 "return":{"sites":["resid_pre.0","resid_pre.2","resid_pre.4","resid_mid.5","resid_final"]}}))
with Path("work/notes.md").open("a") as f:
    f.write("\nBaseline pass 1: addition 0/15; copy/reverse/max each 4/4. Addition resembles copying.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 2 of at most 300:
```
Native IDs: ['copy_n01_000', 'copy_n04_000', 'copy_n07_000', 'copy_n10_000', 'reverse_n01_000', 'reverse_n04_000', 'reverse_n07_000', 'reverse_n10_000', 'max_n01_000', 'max_n04_000', 'max_n07_000', 'max_n10_000']
{"type": "teacher_forced", "model": "original", "texts": ["A:2+2=40\n", "A:9+1=01\n", "A:6+9=51\n", "A:95+13=090\n", "A:76+91=680\n", "A:02+99=911\n", "A:881+247=0390\n", "A:370+838=1190\n", "A:653+837=4901\n", "A:7544+0381=78260\n", "A:0073+2087=20511\n", "A:7963+4586=15501\n", "A:13237+64446=776731\n", "A:36456+17173=436201\n", "A:47722+25232=620640\n", "A:977778+220916=1086941\n", "A:839853+641658=4805121\n", "A:276581+477611=6442030\n", "A:2868458+5008834=78663921\n", "A:2407353+1552957=39592111\n", "A:0200372+6372538=65728011\n", "A:48631200+66156307=058875070\n", "A:48107098+65107054=043041431\n", "A:89909653+00164707=890734601\n", "A:855021804+825378616=6804994201\n", "A:256873073+630186638=8869507021\n", "A:422359581+289187953=6025375450\n", "A:0853781487+3617235662=34700270501\n", "A:1547117875+6660557607=71186645821\n", "A:0514353734+3503459367=30277031021\n", "A:98169367170+65112197935=543715551160\n", "A:49950948460+00392607736=492535552070\n", "A:64750551934+66375849057=211363010911\n", "A:764003658807+457514149431=1226177973480\n", "A:097148358323+219408195943=2076465443760\n", "A:987163384001+043747087969=9219014620701\n", "A:9007452532764+5072046256068=41794987887231\n", "A:3854644140265+1080235465414=48358795066790\n", "A:5204501366257+0244515024437=54480263807841\n", "A:35504860066438+82676287086015=181801580435431\n", "A:51295960719688+21242221130466=724381918490551\n", "A:62738874203126+29622481136876=814603663399921\n", "A:666284267641185+716526414651587=3828017713036631\n", "A:935676305768120+908037022793040=8437044274622600\n", "A:738580340791295+089035179102624=7186164109938101\n", "A:9340522601709935+1538668610031660=09781903217306060\n", "A:6806202827507505+7743700259715586=365990208632219
[... cut]
````

</details>

## Turn 4

The probe set now covers all 20 addition lengths, with native-task and English controls. I’ll record residual states at several depths so I can test operand accessibility without modifying the model. I’ll also look for low-energy directions that could hold new arithmetic signals with limited interference.

```forward
{"type":"teacher_forced","model":"original","texts":["A:2+2=40\n","A:9+1=01\n","A:6+9=51\n","A:95+13=090\n","A:76+91=680\n","A:02+99=911\n","A:881+247=0390\n","A:370+838=1190\n","A:653+837=4901\n","A:7544+0381=78260\n","A:0073+2087=20511\n","A:7963+4586=15501\n","A:13237+64446=776731\n","A:36456+17173=436201\n","A:47722+25232=620640\n","A:977778+220916=1086941\n","A:839853+641658=4805121\n","A:276581+477611=6442030\n","A:2868458+5008834=78663921\n","A:2407353+1552957=39592111\n","A:0200372+6372538=65728011\n","A:48631200+66156307=058875070\n","A:48107098+65107054=043041431\n","A:89909653+00164707=890734601\n","A:855021804+825378616=6804994201\n","A:256873073+630186638=8869507021\n","A:422359581+289187953=6025375450\n","A:0853781487+3617235662=34700270501\n","A:1547117875+6660557607=71186645821\n","A:0514353734+3503459367=30277031021\n","A:98169367170+65112197935=543715551160\n","A:49950948460+00392607736=492535552070\n","A:64750551934+66375849057=211363010911\n","A:764003658807+457514149431=1226177973480\n","A:097148358323+219408195943=2076465443760\n","A:987163384001+043747087969=9219014620701\n","A:9007452532764+5072046256068=41794987887231\n","A:3854644140265+1080235465414=48358795066790\n","A:5204501366257+0244515024437=54480263807841\n","A:35504860066438+82676287086015=181801580435431\n","A:51295960719688+21242221130466=724381918490551\n","A:62738874203126+29622481136876=814603663399921\n","A:666284267641185+716526414651587=3828017713036631\n","A:935676305768120+908037022793040=8437044274622600\n","A:738580340791295+089035179102624=7186164109938101\n","A:9340522601709935+1538668610031660=09781903217306060\n","A:6806202827507505+7743700259715586=36599020863221911\n","A:0565074907417503+7571833071625242=70478089780428450\n","A:79045850140839672+82528616061391021=526635761021317930\n","A:58484208485401336+62442167907951964=119273654924622011\n","A:23625618175306357+43228702256563507=668434204229698541\n","A:085697598469355561+779375670814504893=7550732792839593650\n","A:656475105791209273+255252423358419911=8027285280507182940\n","A:204400484306322317+683056012812313062=8874564961286353790\n","A:5483968307119418756+4162238959494225554=95461073666047333111\n","A:2878794358353668477+0508967325289576497=23867627736332459651\n","A:3731741131194725646+8848506371852615648=16892574039473403941\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","A:49044078083273264253+44516305833559915119=836504739177232893621\n","A:88132334088156858569+08136411317585033687=863687453957329812571\n","C:9=9\n","C:6514=6514\n","C:35696957=35696957\n","C:619464298610=619464298610\n","R:0=0\n","R:1248=8421\n","R:01729902=20992710\n","R:856878706352=253607878658\n","M:3,1=3\n","M:2095,60=2095\n","M:1,65825000=65825000\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i","n 1870 , Lock Haven prospered in the 19th century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine"," and hemlock as well as oak , ash , maple , poplar , cherry , beech , and magnolia . The wood was used locally for such things as frame houses , shingles , canal boats , and wooden"," bridges , and whole logs were floated to Chesapeake Bay and on to Baltimore , to make spars for ships . Log driving and log rafting , competing forms of transporting logs to sawmi","lls , began along the West Branch around 1800 . By 1830 , slightly before the founding of the town , the lumber industry was well established . \n The West Branch Canal , which open","ed in 1834 , ran 73 miles ( 117 km ) from Northumberland to Farrandsville , about 5 miles ( 8 km ) upstream from Lock Haven . A state @-@ funded extension called the Bald Eagle Cut"," ran from the West Branch through Lock Haven and Flemington to Bald Eagle Creek . A privately funded extension , the Bald Eagle and Spring Creek Navigation , eventually reached Bel"],"return":{"sites":["resid_pre.0","resid_pre.2","resid_pre.4","resid_mid.5","resid_final"]}}
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
    80,
    181
   ],
   "resid_pre.0": [
    80,
    181,
    256
   ],
   "resid_pre.2": [
    80,
    181,
    256
   ],
   "resid_pre.4": [
    80,
    181,
    256
   ],
   "resid_mid.5": [
    80,
    181,
    256
   ],
   "resid_final": [
    80,
    181,
    256
   ],
   "token_ids": [
    80,
    181
   ]
  },
  "mean_bits_per_char": 7.3577
 },
 "meta": "work/fwd_2/meta.json",
 "s": 0.6
}
```
Forward passes left: 48. Replies left: 196.
````

</details>

## Turn 5

The residual captures are available. I’ll use simple held-out linear probes to test digit accessibility, and inspect how much residual energy lies outside the embedding span. These probes are diagnostic only: no fitted probe will be written into the model. Strong operand decoding would favour reusing existing circuitry; weak decoding would favour explicit routing.

```python
# Hypothesis: operand digits are either linearly accessible or a small low-energy scratch space is needed.
torch.set_num_threads(4)
E = load_original("embed.weight").double()
_, sv, vh = torch.linalg.svd(E, full_matrices=True)
nullE = vh[99:].T
print("Embedding singular range:", float(sv.min()), float(sv.max()))
def getsite(k, site, model="original"):
    return torch.from_numpy(np.load(f"work/fwd_{k}/{model}/{site}.npy")).double()
Xsites = {s:getsite(2,s) for s in ["resid_pre.2","resid_pre.4","resid_mid.5","resid_final"]}
indices, labels = [], []
for i,r in enumerate(probe_add):
    a,b = r["prompt"][2:-1].split("+")
    for j in range(len(a)):
        indices.append((i,len(r["prompt"])+j))
        labels.append([int(a[j]),int(b[j]),int(r["answer"][j])])
indices = torch.tensor(indices)
labels = torch.tensor(labels)
tr = indices[:,0] % 3 != 2
for s,X in Xsites.items():
    Z=X[indices[:,0],indices[:,1]]
    Z=torch.cat([Z,torch.ones(len(Z),1)],1)
    results=[]
    for col in range(3):
        Y=torch.nn.functional.one_hot(labels[:,col],10).double()
        W=torch.linalg.solve(Z[tr].T@Z[tr]+.1*torch.eye(257),Z[tr].T@Y[tr])
        results.append(round(float((Z[~tr]@W).argmax(1).eq(labels[~tr,col]).double().mean()),3))
    print(s,"held-out digit probe a,b,sum:",results)
valid = [(i,p) for i,s in enumerate(probe_strings) for p in range(len(s)+1)]
cov=torch.zeros(256,256,dtype=torch.double)
for X in Xsites.values():
    Z=torch.stack([X[i,p] for i,p in valid])
    cov += Z.T@Z/len(Z)
ev,U=torch.linalg.eigh(nullE.T@cov@nullE)
print("Nullspace residual energy fractions, lowest 8/16/24/32:",
      [round(float(ev[:n].sum()/torch.trace(cov)),6) for n in [8,16,24,32]])
print("Typical residual RMS:", [(s,round(float(X.square().mean().sqrt()),3)) for s,X in Xsites.items()])
print("Tensor names sample:", NAMES[:12])
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 3 of at most 300:
```
Embedding singular range: 0.12778106704938708 9.405647951748797
resid_pre.2 held-out digit probe a,b,sum: [0.095, 0.129, 0.09]
resid_pre.4 held-out digit probe a,b,sum: [0.09, 0.143, 0.114]
resid_mid.5 held-out digit probe a,b,sum: [0.281, 0.138, 0.133]
resid_final held-out digit probe a,b,sum: [0.257, 0.133, 0.114]
Nullspace residual energy fractions, lowest 8/16/24/32: [6.6e-05, 0.000149, 0.000249, 0.000368]
Typical residual RMS: [('resid_pre.2', 1.823), ('resid_pre.4', 2.71), ('resid_mid.5', 3.154), ('resid_final', 22.876)]
Tensor names sample: ['embed.weight', 'blocks.0.attn_norm.weight', 'blocks.0.attn.q_proj.weight', 'blocks.0.attn.k_proj.weight', 'blocks.0.attn.v_proj.weight', 'blocks.0.attn.o_proj.weight', 'blocks.0.mlp_norm.weight', 'blocks.0.mlp.gate_proj.weight', 'blocks.0.mlp.up_proj.weight', 'blocks.0.mlp.down_proj.weight', 'blocks.1.attn_norm.weight', 'blocks.1.attn.q_proj.weight']

```
Forward passes left: 48. Replies left: 195.
````

</details>

## Turn 6

The probes did not reveal reliable operand digits: held-out accuracy was near chance for most sites, with the best first-operand probe reaching 28.1%. The lowest-energy 32 directions outside the embedding span held only 0.037% of the measured residual energy, so I’ll test a protected scratch space there. The first edit will only encode token identity, position information and an `A:` marker; it will not yet attempt addition. Low English KL and unchanged native accuracy would support continuing with this design.

```python
# Hypothesis: low-energy residual directions can carry exact token/position signals with little off-task damage.
reset_edit()
D=40
S=(nullE@U[:,:D]).float()
torch.save(S,"work/scratch_basis.pt")
P=torch.eye(256)-S@S.T
for l in range(6):
    for kind in ["attn.o_proj","mlp.down_proj"]:
        name=f"blocks.{l}.{kind}.weight"
        set_tensor(name,P@load_original(name))
eps=.05
def put_head(l,h,Q,K,V,O):
    for kind,mat in [("q_proj",Q),("k_proj",K),("v_proj",V)]:
        name=f"blocks.{l}.attn.{kind}.weight"
        w=load(name); w[h*32:(h+1)*32]=mat.float(); set_tensor(name,w)
    name=f"blocks.{l}.attn.o_proj.weight"
    w=load(name);w[:,h*32:(h+1)*32]=O.float();set_tensor(name,w)
en=E/(E.square().mean(1,keepdim=True)+CFG["norm_eps"]).sqrt()
en=en*load_original("blocks.0.attn_norm.weight").double()
encpinv=torch.linalg.pinv(en)
def tokenmap(target):
    return (encpinv@target.double()).T.float()
def blank_head():
    return [torch.zeros(32,256),torch.zeros(32,256),torch.zeros(32,256),torch.zeros(256,32)]
def tokenhead(l,h,qt,kt,vt,O):
    put_head(l,h,tokenmap(qt),tokenmap(kt),tokenmap(vt),O)
hs=add_heads(0,3)
qt=torch.zeros(99,32);kt=qt.clone();vt=qt.clone();O=torch.zeros(256,32)
for j in range(4):
    qt[:,j]=np.sqrt(80*np.sqrt(32));kt[:,j]=np.sqrt(80*np.sqrt(32))
vt[:,0]=1
for d in range(10):
    vt[tok.id(str(d)),1]=d/10
    vt[tok.id(str(d)),4]=1
vt[tok.id("+"),2]=1;vt[tok.id("="),3]=1;vt[1,5]=1
O[:,:6]=eps*S[:,:6]
tokenhead(0,hs[0],qt,kt,vt,O)
qt.zero_();kt.zero_();vt.zero_();O.zero_()
vt[1,0]=1;O[:,0]=eps*S[:,6]
tokenhead(0,hs[1],qt,kt,vt,O)
qt.zero_();kt.zero_();vt.zero_();O.zero_()
qt[:,15]=-380*np.sqrt(32);kt[tok.id("A"),15]=1
for j in range(4):
    w=10000**(-j/16)
    qt[tok.id(":"),j]=100*np.sqrt(32)*np.cos(w)
    qt[tok.id(":"),j+16]=-100*np.sqrt(32)*np.sin(w)
    kt[tok.id("A"),j]=1
vt[tok.id("A"),0]=1;O[:,0]=eps*S[:,7]
tokenhead(0,hs[2],qt,kt,vt,O)
print("Added parameters:",added_params(),"; protected directions:",D)
with Path("work/notes.md").open("a") as f:
    f.write("Pass 2 probes: best first-operand accuracy 28.1%; second operand <=14.3%. "
            "Lowest 32 embedding-null directions carry 0.0368% of pooled residual energy. "
            "Staged projection into complement of 40 directions and 3 layer-0 scratch heads.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 4 of at most 300:
```
Added parameters: 98304 ; protected directions: 40

```
Forward passes left: 48. Replies left: 194.
````

</details>

## Turn 7

The scratch-space edit is staged, using three new heads and 98,304 added parameters. It leaves the embedding and final norm unchanged, while keeping original block outputs out of the reserved directions. I’ll save this intermediate version so the forward tool can measure both its side effects and the accuracy of its scratch signals.

```save_edit
{"final":false,"note":"Intermediate diagnostic edit: reserve 40 low-energy embedding-null residual directions by projecting original block outputs; add three layer-0 heads encoding local token features, reciprocal position and an A-colon marker. No arithmetic output yet; off-task and signal-fidelity checks follow."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 1. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
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
  3,
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
 "added_params": 98304,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.146424,
   "native_rel_change": 0.347167
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 8.062587,
   "native_rel_change": 0.330966
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 2.987442,
   "native_rel_change": 0.294733
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 7.647431,
   "native_rel_change": 0.301722
  },
  "blocks.2.attn.o_proj.weight": {
   "shape": [
    256,
    256
   ],
   "changed_entries": 65536,
   "native_changed_entries": 65536,
   "delta_norm": 3.128265,
   "native_rel_change": 0.273723
  },

[... cut]
````

</details>

## Turn 8

Validation accepted the edit. Although the reserved directions had low activation energy, projecting individual weight matrices changed substantial numbers of entries, so small behavioural effects are not guaranteed. I’ll compare original and edited models on identical addition, native-task and English strings, and inspect the scratch channels directly.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2+2=40\n","A:398+259=5481\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","C:619464298610=619464298610\n","R:856878706352=253607878658\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i","n 1870 , Lock Haven prospered in the 19th century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine","An English sentence starts with A, but should not activate arithmetic. A: is a different case."],"return":{"sites":["resid_pre.1","resid_pre.2","resid_final"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 3 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_3/original",
  "arrays": {
   "token_logprobs": [
    10,
    181
   ],
   "resid_pre.1": [
    10,
    181,
    256
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_final": [
    10,
    181,
    256
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 4.2517
 },
 "edited": {
  "dir": "work/fwd_3/edited",
  "arrays": {
   "token_logprobs": [
    10,
    181
   ],
   "resid_pre.1": [
    10,
    181,
    256
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_final": [
    10,
    181,
    256
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 4.2814
 },
 "kl_original_to_edited": {
  "mean": 0.012786,
  "max_text": 0.03375,
  "per_text_file": "work/fwd_3/kl_per_text.npy"
 },
 "meta": "work/fwd_3/meta.json",
 "s": 0.2
}
```
Forward passes left: 47. Replies left: 192.
````

</details>

## Turn 9

The first comparison gave mean KL divergence 0.0128 across the mixed probe set, with a small increase in bits per character. That suggests the scratch-space approach is usable, but does not establish preservation of native accuracy. I’ll now check greedy generation on the same representative tasks used for the baseline.

```forward
{"type":"generate","model":"both","ids":["add_n01_000","add_n01_001","add_n02_000","add_n02_001","add_n03_000","add_n04_000","add_n05_000","add_n06_000","add_n07_000","add_n08_000","add_n09_000","add_n10_000","add_n12_000","add_n16_000","add_n20_000","copy_n01_000","copy_n04_000","copy_n08_000","copy_n12_000","reverse_n01_000","reverse_n04_000","reverse_n08_000","reverse_n12_000","max_n01_000","max_n04_000","max_n08_000","max_n12_000"]}
```

<details><summary>harness output</summary>

````text
Forward pass 4 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 27,
 "original": {
  "graded": 27,
  "accuracy": 0.4444,
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
   "12": 0.0,
   "16": 0.0,
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
    "id": "add_n03_000",
    "expected": "0390",
    "got": "88122"
   }
  ]
 },
 "edited": {
  "graded": 27,
  "accuracy": 0.4444,
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
   "12": 0.0,
   "16": 0.0,
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
    "id": "add_n02_000",
    "expected": "090",
    "got": "3551"
   },
   {
    "id": "add_n02_001",
    "expected": "680",
    "got": "7669"
   },
   {
    "id": "add_n03_000",
    "expected": "0390",
    "got": "88122"
   }
  ]
 },
 "files": [
  "work/fwd_4_original.jsonl",
  "work/fwd_4_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 5
 },
 "s": 1.0
}
```
Forward passes left: 46. Replies left: 191.
````

</details>

## Turn 10

All 12 sampled native answers remained correct, and addition remained at zero as expected from this preparatory edit. I’ll inspect the encoded channels next, particularly whether the `A:` marker is confined to the colon and whether reciprocal position is accurate. Those checks must pass before I use the signals to build arithmetic routing.

```python
# Hypothesis: protected channels accurately encode token features, reciprocal position and the A: prefix.
X1=getsite(3,"resid_pre.1","edited")
Z1=X1@S.double()
ids3=np.load("work/fwd_3/edited/token_ids.npy")
for i in [0,1,2,3,6,9]:
    length=int((ids3[i]!=0).sum())
    z=Z1[i,:length]/eps
    toknum=torch.tensor([int(tok.itos[t]) / 10 if tok.itos[t].isdigit() else 0 for t in ids3[i,:length]])
    print(i,"const err",float((z[:,0]-1).abs().max()),
          "digit err",float((z[:,1]-toknum).abs().max()),
          "position reciprocal err",float((z[:,6]-1/torch.arange(1,length+1)).abs().max()),
          "marker peaks",[(p,round(float(z[p,7]),5)) for p in range(length) if z[p,7]>.05])
rcolon=float((X1[0,2].square().mean()+CFG["norm_eps"]).sqrt())
print("Colon RMS:",rcolon,"; per-text KL:",np.load("work/fwd_3/kl_per_text.npy").round(6).tolist())
import shutil
shutil.copytree("work/edited","work/scratch_v1",dirs_exist_ok=True)
with Path("work/notes.md").open("a") as f:
    f.write("Scratch edit: pass 3 mixed-input KL 0.012786; pass 4 native 12/12 unchanged, addition 0/15. "
            "Backup work/scratch_v1.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 5 of at most 300:
```
0 const err 3.932701111519776e-05 digit err 5.948464809057974e-05 position reciprocal err 5.840043959692842e-05 marker peaks [(2, 1.00005)]
1 const err 3.932701111519776e-05 digit err 3.1651661920317764e-05 position reciprocal err 5.840043959692842e-05 marker peaks [(2, 1.00005)]
2 const err 3.932701111519776e-05 digit err 4.689995854256779e-05 position reciprocal err 6.647587907882002e-05 marker peaks [(2, 1.00005)]
3 const err 3.909666851509286e-05 digit err 7.120126613957556e-05 position reciprocal err 8.157043828127541e-05 marker peaks []
6 const err 8.322710045960413e-05 digit err 8.000330860430394e-05 position reciprocal err 4.79622362856702e-05 marker peaks []
9 const err 4.429112946047198e-05 digit err 4.617304063037064e-05 position reciprocal err 4.638401991546859e-05 marker peaks [(73, 1.00001)]
Colon RMS: 3.5941091673496506 ; per-text KL: [0.0012969999806955457, 0.004588000010699034, 0.028353000059723854, 6.500000017695129e-05, 4.099999932805076e-05, 6.600000051548705e-05, 0.02076599933207035, 0.021936999633908272, 0.01700199954211712, 0.03375000134110451]

```
Forward passes left: 46. Replies left: 190.
````

</details>

## Turn 11

The scratch signals are close to their intended values, and the marker appears at `A:` rather than at an ordinary word beginning with A. However, the small reciprocal-position signal has enough floating-point error to blur long positions after inversion. I’ll increase its amplitude and add a uniform marker-broadcast head; both changes will be checked before the arithmetic stage is enabled.

```python
# Hypothesis: stronger scratch signals reduce positional roundoff while retaining low off-task interference.
eps=1.0
# Recompute the fixed projection in double precision to reduce cancellation error.
Sd=(nullE@U[:,:D])
P64=torch.eye(256,dtype=torch.double)-Sd@Sd.T
for l in range(6):
    for kind in ["attn.o_proj","mlp.down_proj"]:
        name=f"blocks.{l}.{kind}.weight"
        w=load(name)
        original=load_original(name)
        w[:,:original.shape[1]]=(P64@original.double()).float()
        set_tensor(name,w)
name="blocks.0.attn.o_proj.weight"
w=load(name)
w[:,hs[0]*32:hs[0]*32+6]=eps*S[:,:6]
w[:,hs[1]*32]=eps*S[:,6]
w[:,hs[2]*32]=eps*S[:,7]
set_tensor(name,w)
hb=add_heads(1,1)[0]
Q,K,V,O=blank_head()
V[0]=S[:,7]/load_original("blocks.1.attn_norm.weight")/eps
O[:,0]=eps*S[:,8]
put_head(1,hb,Q,K,V,O)
print("Added parameters:",added_params())
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 6 of at most 300:
```
Added parameters: 131072

```
Forward passes left: 46. Replies left: 189.
````

</details>

## Turn 12

The stronger scratch signals and broadcast head are staged, still without an arithmetic output path. The broadcast is intended to provide a task flag at every position after `A:` while remaining near zero on native tasks. I’ll save it now, then measure signal precision and off-task effects together.

```save_edit
{"final":false,"note":"Scratch refinement: double-precision construction of the output projections, scratch amplitude increased from 0.05 to 1, and one layer-1 uniform head broadcasting the A-colon marker. Arithmetic is not enabled yet."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 2. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
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
  3,
  1,
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
 "added_params": 131072,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 5.017254,
   "native_rel_change": 0.347167
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 8.062587,
   "native_rel_change": 0.330966
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 3.150367,
   "native_rel_change": 0.294734
  },
  "blocks.1.attn.v_proj.weight": {
   "shape": [
    288,
    256
   ],
   "changed_entries": 256,
   "native_changed_entries": 0,
   "delta_norm": 1.299404,
   "native_rel_change": 0.0
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 7.647431,
   "native_rel_change": 0.301722
  },
  "blocks.
[... cut]
````

</details>

## Turn 13

The refined edit passed validation. I’ll repeat the paired probe and capture the state entering layer 1’s MLP, where the arithmetic representation will be built. Success here means a clearer position signal and broadcast flag without a large increase in English KL.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2+2=40\n","A:398+259=5481\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","C:619464298610=619464298610\n","R:856878706352=253607878658\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i","n 1870 , Lock Haven prospered in the 19th century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine","An English sentence starts with A, but should not activate arithmetic."],"return":{"sites":["resid_pre.1","resid_mid.1","resid_pre.2","resid_final"]}}
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
    10,
    181
   ],
   "resid_pre.1": [
    10,
    181,
    256
   ],
   "resid_mid.1": [
    10,
    181,
    256
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_final": [
    10,
    181,
    256
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 4.2334
 },
 "edited": {
  "dir": "work/fwd_5/edited",
  "arrays": {
   "token_logprobs": [
    10,
    181
   ],
   "resid_pre.1": [
    10,
    181,
    256
   ],
   "resid_mid.1": [
    10,
    181,
    256
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_final": [
    10,
    181,
    256
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 4.2576
 },
 "kl_original_to_edited": {
  "mean": 0.01542,
  "max_text": 0.038828,
  "per_text_file": "work/fwd_5/kl_per_text.npy"
 },
 "meta": "work/fwd_5/meta.json",
 "s": 0.2
}
```
Forward passes left: 45. Replies left: 187.
````

</details>

## Turn 14

The stronger signals increased mixed-input KL only modestly, to 0.0154. I can now build position without a large lookup table: the broadcast supplies a factor proportional to \(1/(p+1)\), while a separate channel supplies \(1-1/(p+1)\); their ratio encodes \(p\). I’ll add a small, task-gated MLP using this identity, and check the resulting positions before adding operand retrieval.

```python
# Hypothesis: a gated homogeneous representation can encode exact position with only six new neurons.
Xpre=getsite(5,"resid_pre.1","edited")
Xmid=getsite(5,"resid_mid.1","edited")
rcolon=float((Xpre[0,2].square().mean()+CFG["norm_eps"]).sqrt())
z=Xmid@S.double()
for i in range(3):
    length=int((np.load("work/fwd_5/edited/token_ids.npy")[i]!=0).sum())
    p=torch.arange(2,length,dtype=torch.double)
    zz=z[i,2:length]
    u=zz[:,6]/zz[:,0]
    diff=rcolon*zz[:,8]-zz[:,6]
    print(i,"reciprocal position max error",float((1/u-1-p).abs().max()),
          "broadcast mismatch",float(diff.abs().max()))
B=512.;T1=1000.;G=1e6;Mgate=512.
nm=load_original("blocks.1.mlp_norm.weight")
def rawread(j,norm):
    return S[:,j]/norm
rr=[rawread(j,nm) for j in range(D)]
tag=2*rcolon*rr[8]-rr[6]
posgate=rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6])
new1=add_neurons(1,6)
gates=[tag]*5+[posgate]
ups=[rr[j] for j in [0,1,2,3,4]]+[rr[0]]
outcoords=[9,11,12,13,14,10]
outscales=[B,1,1,1,1,1/64]
for part in ["gate_proj","up_proj","down_proj"]:
    name=f"blocks.1.mlp.{part}.weight";w=load(name)
    for k,idx in enumerate(new1):
        if part=="gate_proj":w[idx]=G*gates[k]
        elif part=="up_proj":w[idx]=ups[k]/G
        else:w[:,idx]=T1*outscales[k]*S[:,outcoords[k]]
    set_tensor(name,w)
print("Colon RMS",rcolon,"added parameters",added_params())
with Path("work/notes.md").open("a") as f:
    f.write("Pass 5 scratch amplitude 1: mixed KL 0.015420. "
            "Staged six layer-1 neurons for gated homogeneous token/position representation; no output arithmetic yet.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 7 of at most 300:
```
0 reciprocal position max error 0.00011786283702797107 broadcast mismatch 6.556090843795603e-06
1 reciprocal position max error 0.00021512701988513072 broadcast mismatch 6.901777853049462e-06
2 reciprocal position max error 0.003849971641059824 broadcast mismatch 6.572436587665154e-06
Colon RMS 3.4694356316032295 added parameters 135680

```
Forward passes left: 45. Replies left: 186.
````

</details>

## Turn 15

The raw reciprocal channel now recovers positions within about 0.004 characters on the long probe. The broadcast mismatch is a few millionths, but the position formula can amplify it, so the completed representation still needs testing. I’ll save the six-neuron encoder and verify its coordinates and off-task gating directly.

```save_edit
{"final":false,"note":"Add six hand-set layer-1 SwiGLU neurons encoding an addition-only homogeneous carrier, digit, delimiter flags and absolute position. Gates use the A-colon broadcast; position uses an algebraic reciprocal identity rather than fitted weights."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 3. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   8,
   8,
   8,
   8
  ],
  "d_mlp": [
   768,
   774,
   768,
   768,
   768,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  0,
  0,
  0,
  0
 ],
 "added_neurons": [
  0,
  6,
  0,
  0,
  0,
  0
 ],
 "added_params": 135680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 5.017254,
   "native_rel_change": 0.347167
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 8.062587,
   "native_rel_change": 0.330966
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 3.150367,
   "native_rel_change": 0.294734
  },
  "blocks.1.attn.v_proj.weight": {
   "shape": [
    288,
    256
   ],
   "changed_entries": 256,
   "native_changed_entries": 0,
   "delta_norm": 1.299404,
   "native_rel_change": 0.0
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    774
   ],
   "changed_entries": 198144,
   "native_changed_entries": 196608,
   "delta_norm": 512003.875,
   "native_rel_change": 0.301722
  },
  "block
[... cut]
````

</details>

## Turn 16

The encoder passed validation. It deliberately gives addition inputs a strong internal representation, so their current logits may become worse until the output circuit is complete. The important checks now are accurate positions and digits on addition inputs, and inactivity on native tasks and ordinary English.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2+2=40\n","A:398+259=5481\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","C:619464298610=619464298610\n","R:856878706352=253607878658\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i","n 1870 , Lock Haven prospered in the 19th century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine","An English sentence starts with A, but should not activate arithmetic."],"return":{"sites":["resid_pre.2","resid_final"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 6 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_6/original",
  "arrays": {
   "token_logprobs": [
    10,
    181
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_final": [
    10,
    181,
    256
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 4.2334
 },
 "edited": {
  "dir": "work/fwd_6/edited",
  "arrays": {
   "token_logprobs": [
    10,
    181
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_final": [
    10,
    181,
    256
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 3.4115
 },
 "kl_original_to_edited": {
  "mean": 0.848545,
  "max_text": 3.236431,
  "per_text_file": "work/fwd_6/kl_per_text.npy"
 },
 "meta": "work/fwd_6/meta.json",
 "s": 0.2
}
```
Forward passes left: 44. Replies left: 184.
````

</details>

## Turn 17

The mixed KL rose sharply after enabling the addition representation, which is expected if the change is confined to addition inputs. I’ll separate the per-text KL values and inspect the encoded coordinates. If native or English states changed substantially, I’ll fix the gate before proceeding.

```python
# Hypothesis: the new representation is active only for addition, with position errors small enough for routing.
X2=getsite(6,"resid_pre.2","edited");z2=X2@S.double()
id6=np.load("work/fwd_6/edited/token_ids.npy")
for i in range(10):
    length=int((id6[i]!=0).sum())
    if i<3:
        zz=z2[i,2:length]; ps=torch.arange(2,length,dtype=torch.double)
        pos=64*B*zz[:,10]/zz[:,9]
        digits=10*B*zz[:,11]/zz[:,9]
        target=torch.tensor([int(tok.itos[t]) if tok.itos[t].isdigit() else 0 for t in id6[i,2:length]])
        print(i,"position errors max/mean",round(float((pos-ps).abs().max()),5),
              round(float((pos-ps).abs().mean()),5),
              "digit error",round(float((digits-target).abs().max()),5),
              "carrier range",tuple(round(float(v),2) for v in [zz[:,9].min(),zz[:,9].max()]))
        print("last positions",pos[-6:].round(decimals=3).tolist())
    else:
        print(i,"off-task new-coordinate max",float(z2[i,:length,9:15].abs().max()))
print("Per-text KL",np.load("work/fwd_6/kl_per_text.npy").round(6).tolist())
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 8 of at most 300:
```
0 position errors max/mean 0.0166 0.00933 digit error 3e-05 carrier range (6517.87, 13625.38)
last positions [3.983, 5.004, 5.983, 7.007, 8.008, 8.989]
1 position errors max/mean 0.04716 0.01556 digit error 4e-05 carrier range (3487.74, 14100.24)
last positions [9.979, 11.001, 12.021, 12.996, 13.966, 15.038]
2 position errors max/mean 0.13007 0.03207 digit error 6e-05 carrier range (1322.45, 13625.38)
last positions [61.075, 61.947, 62.952, 63.887, 64.977, 66.07]
3 off-task new-coordinate max 3.851962633638628e-06
4 off-task new-coordinate max 6.024877632460068e-06
5 off-task new-coordinate max 4.3456895313376265e-06
6 off-task new-coordinate max 3.327009416076443e-06
7 off-task new-coordinate max 3.89017489643706e-06
8 off-task new-coordinate max 4.235620089443337e-06
9 off-task new-coordinate max 3.07200144616937e-06
Per-text KL [2.5165059566497803, 2.6305880546569824, 3.2364308834075928, 0.00010199999815085903, 7.200000254670158e-05, 9.300000237999484e-05, 0.021637000143527985, 0.02333099953830242, 0.017867999151349068, 0.03882800042629242]

```
Forward passes left: 44. Replies left: 183.
````

</details>

## Turn 18

The new encoder stayed inactive on all native and English controls; their KL values remained small. Addition digits were accurate, while the worst position error was 0.13 characters on the long example. I’ll reduce that error with an algebraic cancellation term, then add explicit heads that locate `+` and `=`, retrieve current and previous operand digits, and compute carry and the output digit. The key test will be whether those retrieved digits match the intended positions, not just whether a few answers happen to be correct.

```python
# Hypothesis: explicit positional routing plus a fixed carry rule implements addition without fitting weights.
# Refine the position gate: cancellation keeps ordinary long text inactive while reducing roundoff amplification.
Mgate=80.
rr=[rawread(j,load_original("blocks.1.mlp_norm.weight")) for j in range(D)]
posgate=rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6])
name="blocks.1.mlp.gate_proj.weight";w=load(name);w[new1[-1]]=G*posgate;set_tensor(name,w)
cancel1=add_neurons(1,1)[0]
for part in ["gate_proj","up_proj","down_proj"]:
    name=f"blocks.1.mlp.{part}.weight";w=load(name)
    if part=="gate_proj":w[cancel1]=G*(rr[0]-(Mgate+1)*rr[6])
    elif part=="up_proj":w[cancel1]=rr[0]/G
    else:w[:,cancel1]=-T1/64*S[:,10]
    set_tensor(name,w)

T2=1e5;T3=1e8
def readers(l,kind="attn"):
    norm=load_original(f"blocks.{l}.{kind}_norm.weight")
    r=[S[:,j]/norm for j in range(D)]
    return r, r[9]/16, r[0]/eps, r[5]/eps
def install_pair(l,Q,K,V,O,c,b,bos):
    hh=add_heads(l,2)
    # Native inputs use the same BOS value in both heads, so their outputs cancel.
    Q[13]+=1e5*np.sqrt(32)*b
    K[13]+=bos
    put_head(l,hh[0],Q,K,V,O)
    qb=torch.zeros_like(Q);kb=torch.zeros_like(K)
    qb[13]=1e5*np.sqrt(32)*(b+c);kb[13]=bos
    put_head(l,hh[1],qb,kb,V,-O)
    return hh
def self_pair(l,scale,coords):
    r,c,b,bos=readers(l)
    Q,K,V,O=blank_head()
    for j in range(4):
        Q[j]=c*np.sqrt(80*np.sqrt(32))
        K[j]=c*np.sqrt(80*np.sqrt(32))
    for j,coord in enumerate(coords):
        V[j]=r[coord]*B/16
        O[:,j]=S[:,coord]*scale
    return install_pair(l,Q,K,V,O,c,b,bos)
l2self=self_pair(2,T2,list(range(9,15)))
l2sep=[]
for flag,dest in [(12,15),(13,16)]:
    r,c,b,bos=readers(2)
    Q,K,V,O=blank_head()
    Q[15]=100*np.sqrt(32)*c;K[15]=r[flag]*B/16
    V[0]=r[10]*B/16;O[:,0]=T2*S[:,dest]
    l2sep.append(install_pair(2,Q,K,V,O,c,b,bos))
def install_neurons(l,defs):
    inds=add_neurons(l,len(defs))
    for part in ["gate_proj","up_proj","down_proj"]:
        name=f"blocks.{l}.mlp.{part}.weight";w=load(name)
        for idx,(g,u,o) in zip(inds,defs):
            if part=="gate_proj":w[idx]=g
            elif part=="up_proj":w[idx]=u
            else:w[:,idx]=o
        set_tensor(name,w)
    return inds
# Squared absolute position, exact linear spline at integer positions.
r,c,b,_=readers(2,"mlp");xp=r[10]*B/16
defs=[]
G2=1e5
for j in range(81):
    coefficient=(1 if j==0 else 2)/64
    defs.append((G2*(xp-j/64*c-10*b),c/G2,T2*coefficient*S[:,17]))
sqneurons=install_neurons(2,defs)
l3self=self_pair(3,T3,list(range(9,17)))
routeheads=[]
for operand,previous,dest in [("a",False,18),("b",False,19),("a",True,20),("b",True,21)]:
    r,c,b,bos=readers(3)
    f=lambda j:r[j]*B/16
    desired=64*(f(10)-f(16))+(3 if operand=="a" else 1)*c
    if operand=="b":desired=desired+64*f(15)
    if previous:desired=desired-c
    Q,K,V,O=blank_head()
    sharp=12.
    Q[14]=2*sharp*np.sqrt(32)*desired;K[14]=64*f(10)
    Q[15]=sharp*np.sqrt(32)*c;K[15]=-4096*f(17)
    V[0]=f(11);O[:,0]=T3*S[:,dest]
    routeheads.append(install_pair(3,Q,K,V,O,c,b,bos))
# Carry is 1 iff the previous operand sum exceeds the previous output digit.
r,c,b,_=readers(3,"mlp");f=lambda j:r[j]*B/16
difference=f(20)+f(21)-f(11)
carrydefs=[]
for threshold,sign in [(.3,1),(.6,-1)]:
    carrydefs.append((G2*((difference-threshold*c)/.3-10*b),c/G2,sign*T3*S[:,22]))
carryneurons=install_neurons(3,carrydefs)
# Piecewise-linear digit lookup through the frozen output embedding.
F=E*load_original("final_norm.weight").double()[None,:]
output_inverse=torch.linalg.pinv(F)
digitvec=torch.stack([output_inverse[:,tok.id(str(d))] for d in range(10)]).float()
r,c,b,_=readers(4,"mlp");f=lambda j:r[j]*B/16
total=10*(f(18)+f(19))+f(22)
Tout=1e12
defs=[(G2*(c-10*b),c/G2,Tout*digitvec[0])]
slopes=[digitvec[(j+1)%10]-digitvec[j%10] for j in range(20)]
for j in range(20):
    delta=slopes[j] if j==0 else slopes[j]-slopes[j-1]
    defs.append((G2*(total-j*c-10*b),c/G2,Tout*delta))
# Emit newline after the N+1 answer digits.
end=64*(f(10)-1.5*f(16))+2*c
newlinevec=output_inverse[:,tok.id("\n")].float()
for threshold,sign in [(.25,1),(.75,-1)]:
    defs.append((G2*(end-threshold*c-10*b),c/G2,sign*4*Tout*newlinevec))
outneurons=install_neurons(4,defs)
print("Architecture",arch(),"\nAdded parameters",added_params())
torch.save({"S":S,"B":B,"T1":T1,"T2":T2,"T3":T3,"Tout":Tout,
            "l2self":l2self,"l2sep":l2sep,"l3self":l3self,"routeheads":routeheads},
           "work/circuit_design.pt")
with Path("work/notes.md").open("a") as f:
    f.write("Pass 6: native/English gates inactive; addition position max error 0.1301, digit error <0.0001. "
            "Added a cancellation identity to reduce positional roundoff, separator and operand routing pairs, "
            "fixed square spline, carry threshold, and modulo-10 output lookup. No optimization or training.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 9 of at most 300:
```
Architecture {'n_heads': [11, 9, 14, 18, 8, 8], 'd_mlp': [768, 775, 849, 770, 791, 768]} 
Added parameters 742144

```
Forward passes left: 44. Replies left: 182.
````

</details>

## Turn 19

The complete hand-built circuit is staged at 742,144 added parameters, below half the allowed cap. Its carry rule uses the previous operand digits and previous answer digit, so it does not need a recurrent carry state hidden in generated text. I’ll save it and inspect each stage on short and long examples before judging overall accuracy.

```save_edit
{"final":false,"note":"First complete analytic addition circuit: protected scratch channels; A-colon gating; homogeneous position/token encoder; paired separator and operand retrieval heads; fixed position-square spline; carry from previous operands/output; piecewise-linear modulo-10 and newline output through the frozen tied embedding. 742144 added parameters. No gradient or iterative fitting used; full-circuit validation follows."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 4. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   768,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  0,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 742144,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 5.017254,
   "native_rel_change": 0.347167
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 8.062587,
   "native_rel_change": 0.330966
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 3.150367,
   "native_rel_change": 0.294734
  },
  "blocks.1.attn.v_proj.weight": {
   "shape": [
    288,
    256
   ],
   "changed_entries": 256,
   "native_changed_entries": 0,
   "delta_norm": 1.299404,
   "native_rel_change": 0.0
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    775
   ],
   "changed_entries": 198400,
   "native_changed_entries": 196608,
   "delta_norm": 512003.84375,
   "native_rel_change": 0.301722
  },
 
[... cut]
````

</details>

## Turn 20

Validation accepted the full circuit. I’ll now run paired teacher-forced checks, including attention patterns and the intermediate arithmetic channels. Correct digit retrieval and stable native controls would justify a broader greedy-decoding test; failures will show which stage needs correction.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2+2=40\n","A:398+259=5481\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","C:619464298610=619464298610\n","R:856878706352=253607878658\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i","n 1870 , Lock Haven prospered in the 19th century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine","An English sentence starts with A, but should not activate arithmetic."],"return":{"topk":3,"sites":["resid_pre.2","resid_mid.2","resid_pre.3","resid_mid.3","resid_pre.4","attn.2.pattern","attn.3.pattern","logits"]}}
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
    10,
    181
   ],
   "topk_logprobs": [
    10,
    181,
    3
   ],
   "topk_ids": [
    10,
    181,
    3
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_mid.2": [
    10,
    181,
    256
   ],
   "resid_pre.3": [
    10,
    181,
    256
   ],
   "resid_mid.3": [
    10,
    181,
    256
   ],
   "resid_pre.4": [
    10,
    181,
    256
   ],
   "attn.2.pattern": [
    10,
    8,
    181,
    181
   ],
   "attn.3.pattern": [
    10,
    8,
    181,
    181
   ],
   "logits": [
    10,
    181,
    99
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 4.2334
 },
 "edited": {
  "dir": "work/fwd_7/edited",
  "arrays": {
   "token_logprobs": [
    10,
    181
   ],
   "topk_logprobs": [
    10,
    181,
    3
   ],
   "topk_ids": [
    10,
    181,
    3
   ],
   "resid_pre.2": [
    10,
    181,
    256
   ],
   "resid_mid.2": [
    10,
    181,
    256
   ],
   "resid_pre.3": [
    10,
    181,
    256
   ],
   "resid_mid.3": [
    10,
    181,
    256
   ],
   "resid_pre.4": [
    10,
    181,
    256
   ],
   "attn.2.pattern": [
    10,
    14,
    181,
    181
   ],
   "attn.3.pattern": [
    10,
    18,
    181,
    181
   ],
   "logits": [
    10,
    181,
    99
   ],
   "token_ids": [
    10,
    181
   ]
  },
  "mean_bits_per_char": 5.194
 },
 "kl_original_to_edited": {
  "mean": 3.13755,
  "max_text": 11.628742,
  "per_text_file": "work/fwd_7/kl_per_text.npy"
 },
 "meta": "work/fwd_7/meta.json",
 "s": 0.2
}
```
Forward passes left: 43. Replies left: 180.
````

</details>

## Turn 21

The full-circuit probe changed the mixed-input distribution substantially, but that aggregate mixes intended addition changes with possible collateral effects. I’ll inspect per-text KL, predicted answers, retrieved operand digits and carry values separately. This will distinguish arithmetic errors from a failure of the off-task gate.

```python
# Hypothesis: any first-circuit errors can be localized to position routing, carry, or numerical cancellation.
sites7={s:getsite(7,s,"edited") for s in ["resid_pre.2","resid_mid.2","resid_pre.3","resid_mid.3","resid_pre.4"]}
zz7={s:X@S.double() for s,X in sites7.items()}
patterns2=np.load("work/fwd_7/edited/attn.2.pattern.npy")
patterns3=np.load("work/fwd_7/edited/attn.3.pattern.npy")
logits7=np.load("work/fwd_7/edited/logits.npy")
texts7=["A:2+2=40\n","A:398+259=5481\n",
"A:80303457619474332301+65595253539808712781=468986012583821540920\n"]
for i,t in enumerate(texts7):
    eq=t.index("=")+1;n=(eq-4)//2
    positions=list(range(eq,eq+n+2))
    print("\nEXAMPLE",i,"teacher predictions",tok.decode(logits7[i,positions].argmax(-1).tolist()))
    z=zz7["resid_pre.2"][i]
    print("encoder pos maxerr",float((64*B*z[2:len(t)+1,10]/z[2:len(t)+1,9]-torch.arange(2,len(t)+1)).abs().max()))
    for p in positions[:4]+(positions[-3:] if n>3 else []):
        z=zz7["resid_pre.4"][i,p]; v=z*B/z[9]
        zp=zz7["resid_pre.3"][i,p];vp=zp*B/zp[9]
        print("p",p,"p,+,=,p²",np.round([64*vp[10],64*vp[15],64*vp[16],4096*vp[17]],3),
              "a,b,pa,pb,prev,carry",np.round([10*v[j] for j in [18,19,20,21,11]]+[v[22]],3),
              "route", [int(patterns3[i,h[0],p].argmax()) for h in routeheads])
print("\nKL",np.load("work/fwd_7/kl_per_text.npy").round(6).tolist())
for i in range(3,10):
    length=int((np.load("work/fwd_7/edited/token_ids.npy")[i]!=0).sum())
    print("off-task",i,"newcoord max by stage",
          [round(float(z[i,:length,9:].abs().max()),6) for z in zz7.values()])
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 10 of at most 300:
```

EXAMPLE 0 teacher predictions 40

encoder pos maxerr 0.0030982834085504507
p 6 p,+,=,p² [ 5.997  3.997  5.996 35.948] a,b,pa,pb,prev,carry [ 1.998  1.998 -0.     0.     0.     0.   ] route [3, 5, 2, 4]
p 7 p,+,=,p² [ 6.999  3.997  5.996 48.967] a,b,pa,pb,prev,carry [0.    0.    1.998 1.998 4.    0.   ] route [4, 6, 3, 5]
p 8 p,+,=,p² [ 8.002  3.997  5.996 64.011] a,b,pa,pb,prev,carry [1.998 3.996 0.    0.    0.    0.   ] route [5, 7, 4, 6]

EXAMPLE 1 teacher predictions 5481

encoder pos maxerr 0.003544628898328739
p 10 p,+,=,p² [ 9.999  6.001  9.998 99.963] a,b,pa,pb,prev,carry [ 2.997  1.998 -0.     0.     0.    -0.   ] route [3, 7, 2, 6]
p 11 p,+,=,p² [ 10.998   6.001   9.997 120.937] a,b,pa,pb,prev,carry [8.991 4.995 2.997 1.998 5.    0.   ] route [4, 8, 3, 7]
p 12 p,+,=,p² [ 12.      6.      9.996 143.964] a,b,pa,pb,prev,carry [7.992 8.991 8.991 4.995 4.    0.999] route [5, 9, 4, 8]
p 13 p,+,=,p² [ 13.002   6.      9.997 169.003] a,b,pa,pb,prev,carry [0.    0.    7.992 8.991 8.    0.999] route [6, 10, 5, 9]

EXAMPLE 2 teacher predictions 468986012583821540920

encoder pos maxerr 0.034732269256096515
p 44 p,+,=,p² [  44.003   22.974   44.    1936.169] a,b,pa,pb,prev,carry [7.992 5.994 0.    0.001 0.    0.   ] route [3, 24, 2, 23]
p 45 p,+,=,p² [  44.998   22.975   44.001 2024.838] a,b,pa,pb,prev,carry [0.    4.995 7.992 5.994 4.    0.999] route [4, 25, 3, 24]
p 46 p,+,=,p² [  46.004   22.974   43.999 2116.322] a,b,pa,pb,prev,carry [ 2.997  4.995  0.     4.995  6.    -0.   ] route [5, 26, 4, 25]
p 47 p,+,=,p² [  47.002   22.974   43.999 2209.097] a,b,pa,pb,prev,carry [0.    8.991 2.997 4.995 8.    0.   ] route [6, 27, 5, 26]
p 63 p,+,=,p² [  63.009   22.976   44.002 3970.192] a,b,pa,pb,prev,carry [0.999 0.998 0.    7.992 9.    0.   ] route [22, 43, 21, 42]
p 64 p,+,=,p² [  64.012   22.977   44.005 4097.897] a,b,pa,pb,prev,carry [0.001 0.    0.999 0.998 2.    0.   ] route [23, 44, 22, 43]
p 65 p,+,=,p² [  65.002   22.97
[... cut]
````

</details>

## Turn 22

All three teacher-forced examples produced the correct full answers, including the 20-digit case. Retrieved operands and carry values matched the intended calculation, and the new heads did not materially increase off-task KL beyond the scratch-space edit. I’ll now test greedy decoding on fresh practice IDs across all lengths, because a single early error can still derail an autoregressive answer.

```forward
{"type":"generate","model":"both","ids":["add_n01_003","add_n01_004","add_n01_005","add_n02_003","add_n02_004","add_n02_005","add_n03_003","add_n03_004","add_n03_005","add_n04_003","add_n04_004","add_n04_005","add_n05_003","add_n05_004","add_n05_005","add_n06_003","add_n06_004","add_n06_005","add_n07_003","add_n07_004","add_n07_005","add_n08_003","add_n08_004","add_n08_005","add_n09_003","add_n09_004","add_n09_005","add_n10_003","add_n10_004","add_n10_005","add_n11_003","add_n11_004","add_n11_005","add_n12_003","add_n12_004","add_n12_005","add_n13_003","add_n13_004","add_n13_005","add_n14_003","add_n14_004","add_n14_005","add_n15_003","add_n15_004","add_n15_005","add_n16_003","add_n16_004","add_n16_005","add_n17_003","add_n17_004","add_n17_005","add_n18_003","add_n18_004","add_n18_005","add_n19_003","add_n19_004","add_n19_005","add_n20_003","add_n20_004","add_n20_005","copy_n01_001","copy_n04_001","copy_n08_001","copy_n12_001","reverse_n01_001","reverse_n04_001","reverse_n08_001","reverse_n12_001","max_n01_001","max_n04_001","max_n08_001","max_n12_001"]}
```

<details><summary>harness output</summary>

````text
Forward pass 8 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 72,
 "original": {
  "graded": 72,
  "accuracy": 0.1667,
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
    "id": "add_n01_003",
    "expected": "80",
    "got": "266"
   },
   {
    "id": "add_n01_004",
    "expected": "60",
    "got": "422"
   },
   {
    "id": "add_n01_005",
    "expected": "61",
    "got": "799"
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
  "graded": 72,
  "accuracy": 0.8472,
  "accuracy_by_task": {
   "addition": 0.8167,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 1.0,
   "3": 1.0,
   "4": 1.0,
   "5": 1.0,
   "6": 1.0,
   "7": 1.0,
   "8": 1.0,
   "9": 1.0,
   "10": 1.0,
   "11": 1.0,
   "12": 1.0,
   "13": 0.0,
   "14": 0.0,
   "15": 0.0,
   "16": 0.3333,
   "17": 1.0,
   "18": 1.0,
   "19": 1.0,
   "20": 1.0
  },
  "failures": [
   {
    "id": "add_n13_003",
    "expected": "50052297457970",
    "got": "90052297457970\n"
   },
   {
    "id": "add_n13_004",
    "expected": "97020693375651",
    "got": "\n8020693375651\n"
   },
   {
    "id": "add_n13_005",
    "expected": "00384180209270",
    "got": "\n0384180209270\n"
   },
   {
    "id": "add_n14_003",
    "expected": "634797388622841",
    "got": "\n44797388622841\n"
   },
   {
    "id": "add_n14_004",
    "expected": "062028575484481",
    "got": "\n62028575484481\n"
   }
  ]
 },
 "files": [
  "work/fwd_8_origi
[... cut]
````

</details>

## Turn 23

Greedy decoding solved 49 of 60 addition problems, with all 12 native controls unchanged. The failures clustered at lengths 13–16 and mostly affected only the first digit, while later digits were correct. That pattern suggests a routing or boundary-condition error rather than a broken carry rule. I’ll inspect those exact failures before making any change.

```python
# Hypothesis: the first-digit failures at lengths 13–16 share a boundary-routing defect.
f8=[json.loads(s) for s in Path("work/fwd_8_edited.jsonl").read_text().splitlines()]
print("Output record example:",f8[0])
failure_ids=["add_n13_003","add_n13_004","add_n14_003","add_n15_003","add_n16_003","add_n17_003"]
failprobe=[r for rid in failure_ids for r in adds if r["id"]==rid]
print(json.dumps({"type":"teacher_forced","model":"both",
"texts":[r["prompt"]+r["answer"]+"\n" for r in failprobe]+["C:619464298610=619464298610\n",
"An English sentence starts with A, but should not activate arithmetic."],
"return":{"sites":["resid_pre.2","resid_pre.3","resid_pre.4","attn.2.pattern","attn.3.pattern","logits"]}}))
shutil.copytree("work/edited","work/first_complete",dirs_exist_ok=True)
with Path("work/notes.md").open("a") as f:
    f.write("Pass 7: all 3 diagnostic additions teacher-forced correct, including N20; no extra off-task KL. "
            "Pass 8 greedy: addition 49/60 (81.67%), native 12/12 unchanged; failures first digit at N13–16. "
            "Backup work/first_complete. Investigating boundary routing.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 11 of at most 300:
```
Output record example: {'id': 'add_n01_003', 'task': 'addition', 'n_digits': 1, 'prompt': 'A:2+6=', 'answer': '80', 'output': '80\n', 'output_ids': [28, 20, 3], 'correct': True}
{"type": "teacher_forced", "model": "both", "texts": ["A:2809519383732+3295600461065=50052297457970\n", "A:3197024628668+6614935709896=97020693375651\n", "A:56085550128595+17393287744688=634797388622841\n", "A:533179298804263+168092475927133=6912627648314960\n", "A:9869198376475455+6808585473570935=57777748400563901\n", "A:85600958896023951+81741025917482749=673519738145066011\n", "C:619464298610=619464298610\n", "An English sentence starts with A, but should not activate arithmetic."], "return": {"sites": ["resid_pre.2", "resid_pre.3", "resid_pre.4", "attn.2.pattern", "attn.3.pattern", "logits"]}}

```
Forward passes left: 42. Replies left: 177.
````

</details>

## Turn 24

The saved outputs confirm that the errors are concentrated at the start of the answer, sometimes producing a newline immediately. I’ve backed up the first complete circuit so this diagnostic work is reversible. The next paired probe will compare failing lengths with a nearby successful length and expose the separator and operand attention choices.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2809519383732+3295600461065=50052297457970\n","A:3197024628668+6614935709896=97020693375651\n","A:56085550128595+17393287744688=634797388622841\n","A:533179298804263+168092475927133=6912627648314960\n","A:9869198376475455+6808585473570935=57777748400563901\n","A:85600958896023951+81741025917482749=673519738145066011\n","C:619464298610=619464298610\n","An English sentence starts with A, but should not activate arithmetic."],"return":{"sites":["resid_pre.2","resid_pre.3","resid_pre.4","attn.2.pattern","attn.3.pattern","logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 9 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_9/original",
  "arrays": {
   "token_logprobs": [
    8,
    71
   ],
   "resid_pre.2": [
    8,
    71,
    256
   ],
   "resid_pre.3": [
    8,
    71,
    256
   ],
   "resid_pre.4": [
    8,
    71,
    256
   ],
   "attn.2.pattern": [
    8,
    8,
    71,
    71
   ],
   "attn.3.pattern": [
    8,
    8,
    71,
    71
   ],
   "logits": [
    8,
    71,
    99
   ],
   "token_ids": [
    8,
    71
   ]
  },
  "mean_bits_per_char": 7.187
 },
 "edited": {
  "dir": "work/fwd_9/edited",
  "arrays": {
   "token_logprobs": [
    8,
    71
   ],
   "resid_pre.2": [
    8,
    71,
    256
   ],
   "resid_pre.3": [
    8,
    71,
    256
   ],
   "resid_pre.4": [
    8,
    71,
    256
   ],
   "attn.2.pattern": [
    8,
    14,
    71,
    71
   ],
   "attn.3.pattern": [
    8,
    18,
    71,
    71
   ],
   "logits": [
    8,
    71,
    99
   ],
   "token_ids": [
    8,
    71
   ]
  },
  "mean_bits_per_char": 11.0262
 },
 "kl_original_to_edited": {
  "mean": 8.84571,
  "max_text": 12.166628,
  "per_text_file": "work/fwd_9/kl_per_text.npy"
 },
 "meta": "work/fwd_9/meta.json",
 "s": 0.1
}
```
Forward passes left: 41. Replies left: 176.
````

</details>

## Turn 25

The failure-focused captures are available. I’ll compare the encoded separator positions with the positions actually selected by each head at `=` and the following character. A mismatch there would explain why only the first answer digit is affected and would identify a specific circuit correction rather than a fit to the failed answers.

```python
# Hypothesis: delimiter retrieval fails specifically at the equals token for the affected lengths.
z9={s:getsite(9,s,"edited")@S.double() for s in ["resid_pre.2","resid_pre.3","resid_pre.4"]}
p29=np.load("work/fwd_9/edited/attn.2.pattern.npy")
p39=np.load("work/fwd_9/edited/attn.3.pattern.npy")
lg9=np.load("work/fwd_9/edited/logits.npy")
for i,r in enumerate(failprobe):
    q=len(r["prompt"])
    print("\n",r["id"],"expected",r["answer"][:3],"pred",repr(tok.decode(lg9[i,q:q+3].argmax(-1).tolist())))
    for p in [q,q+1]:
        v1=z9["resid_pre.2"][i,p]*B/z9["resid_pre.2"][i,p,9]
        v2=z9["resid_pre.3"][i,p]*B/z9["resid_pre.3"][i,p,9]
        v3=z9["resid_pre.4"][i,p]*B/z9["resid_pre.4"][i,p,9]
        print("q",p,"encoder p/flags",np.round([64*v1[10],v1[12],v1[13]],4),
              "routing p/plus/eq",np.round([64*v2[j] for j in [10,15,16]],4),
              "digits/carry",np.round([10*v3[j] for j in [18,19,20,21,11]]+[v3[22]],4))
        print("sep pattern peaks",[(int(p29[i,h[0],p].argmax()),round(float(p29[i,h[0],p].max()),3)) for h in l2sep],
              "self",int(p29[i,l2self[0],p].argmax()),
              "operands",[int(p39[i,h[0],p].argmax()) for h in routeheads])
        print("end score",round(float(64*(v3[10]-1.5*v3[16])+2),4))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 12 of at most 300:
```

 add_n13_003 expected 500 pred '900'
q 30 encoder p/flags [30.0023 -0.      1.    ] routing p/plus/eq [30.0027 12.3691 23.0752] digits/carry [2.9972 4.9972 8.9908 8.9657 0.0001 0.999 ]
sep pattern peaks [(16, 0.772), (30, 0.769)] self 30 operands [10, 20, 9, 19]
end score -2.6106
q 31 encoder p/flags [30.9984  0.      0.    ] routing p/plus/eq [30.9992 16.0031 29.9853] digits/carry [ 7.9921  1.9981  1.9981  2.9969  5.0001 -0.    ]
sep pattern peaks [(16, 1.0), (30, 1.0)] self 31 operands [4, 18, 3, 17]
end score -11.9793

 add_n13_004 expected 970 pred '\n70'
q 30 encoder p/flags [29.9918 -0.      1.    ] routing p/plus/eq [29.992   0.3771  0.6918] digits/carry [ 0.0001  0.      0.      0.9607  0.0001 -0.    ]
sep pattern peaks [(0, 0.976), (0, 0.977)] self 30 operands [30, 30, 30, 30]
end score 30.9541
q 31 encoder p/flags [30.999  0.     0.   ] routing p/plus/eq [30.9994 15.987  29.972 ] digits/carry [ 0.9992  5.9941  2.997   5.9941  9.0001 -0.    ]
sep pattern peaks [(16, 1.0), (30, 1.0)] self 31 operands [4, 18, 3, 17]
end score -11.9586

 add_n14_003 expected 634 pred '\n34'
q 32 encoder p/flags [31.9988  0.      1.    ] routing p/plus/eq [31.9997  0.756   1.4038] digits/carry [0.0001 0.0001 0.0001 7.962  0.0001 0.9989]
sep pattern peaks [(0, 0.955), (0, 0.956)] self 32 operands [32, 32, 32, 31]
end score 31.8941
q 33 encoder p/flags [33. -0.  0.] routing p/plus/eq [33.0002 16.9892 31.9805] digits/carry [5.9941 6.993  4.9952 0.9991 6.0001 0.    ]
sep pattern peaks [(17, 1.0), (32, 1.0)] self 33 operands [4, 19, 3, 18]
end score -12.9715

 add_n15_003 expected 691 pred '\n91'
q 34 encoder p/flags [33.9987  0.      1.    ] routing p/plus/eq [33.999   0.0006  0.0017] digits/carry [ 0.      0.      0.      0.0059  0.     -0.    ]
sep pattern peaks [(0, 1.0), (0, 1.0)] self 34 operands [34, 34, 34, 34]
end score 35.9957
q 35 encoder p/flags [34.9966  0.      0.    ] routing p/plus/eq [34.9971 17.9923 33.9755] digits/carry
[... cut]
````

</details>

## Turn 26

The failure is localized: at `=` in the affected lengths, the separator heads sometimes chose their safe BOS fallback instead of `+` or `=`. The fallback was too strong when the addition carrier was relatively small. I’ll make that fallback score negative when the addition carrier is present, while leaving it strong on off-task inputs. This directly addresses the observed routing defect without changing the arithmetic rule.

```python
# Hypothesis: explicitly suppressing BOS fallback on active addition states fixes first-digit routing failures.
for l,pairs in [(2,[l2self]+l2sep),(3,[l3self]+routeheads)]:
    r,c,b,bos=readers(l)
    name=f"blocks.{l}.attn.q_proj.weight";w=load(name)
    for pair in pairs:
        w[pair[0]*32+13]=1e5*np.sqrt(32)*(b-.1*c)
    set_tensor(name,w)
print("Updated active-head fallback rows:",3+5)
with Path("work/notes.md").open("a") as f:
    f.write("Pass 9 localized N13–16 first-digit failures: separator heads selected BOS (up to 100%) "
            "because fallback score exceeded delimiter score at equals. Fixed analytically by subtracting "
            "0.1 times active carrier from fallback query; native fallback unchanged.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 13 of at most 300:
```
Updated active-head fallback rows: 8

```
Forward passes left: 41. Replies left: 174.
````

</details>

## Turn 27

The fallback correction is staged for all active routing heads. It changes only their task-dependent choice of BOS, not the token encoder, carry logic or output lookup. I’ll save it and repeat the same greedy sample so the comparison cleanly measures whether the identified defect is fixed.

```save_edit
{"final":false,"note":"Fix boundary routing: active separator/operand heads now suppress their BOS fallback when the addition carrier is present. Diagnostic pass 9 showed the previous fallback displaced delimiter retrieval at the first answer position for N13–16. Off-task fallback and arithmetic formulas are unchanged."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 5. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   768,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  0,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 742144,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 5.017254,
   "native_rel_change": 0.347167
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 8.062587,
   "native_rel_change": 0.330966
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 3.150367,
   "native_rel_change": 0.294734
  },
  "blocks.1.attn.v_proj.weight": {
   "shape": [
    288,
    256
   ],
   "changed_entries": 256,
   "native_changed_entries": 0,
   "delta_norm": 1.299404,
   "native_rel_change": 0.0
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    775
   ],
   "changed_entries": 198400,
   "native_changed_entries": 196608,
   "delta_norm": 512003.84375,
   "native_rel_change": 0.301722
  },
 
[... cut]
````

</details>

## Turn 28

The corrected edit passed validation. I’ll rerun exactly the 60 additions and 12 native controls from the previous greedy test. Fixing all first-digit failures with no native losses would confirm the fallback diagnosis, though broader testing would still be needed.

```forward
{"type":"generate","model":"both","ids":["add_n01_003","add_n01_004","add_n01_005","add_n02_003","add_n02_004","add_n02_005","add_n03_003","add_n03_004","add_n03_005","add_n04_003","add_n04_004","add_n04_005","add_n05_003","add_n05_004","add_n05_005","add_n06_003","add_n06_004","add_n06_005","add_n07_003","add_n07_004","add_n07_005","add_n08_003","add_n08_004","add_n08_005","add_n09_003","add_n09_004","add_n09_005","add_n10_003","add_n10_004","add_n10_005","add_n11_003","add_n11_004","add_n11_005","add_n12_003","add_n12_004","add_n12_005","add_n13_003","add_n13_004","add_n13_005","add_n14_003","add_n14_004","add_n14_005","add_n15_003","add_n15_004","add_n15_005","add_n16_003","add_n16_004","add_n16_005","add_n17_003","add_n17_004","add_n17_005","add_n18_003","add_n18_004","add_n18_005","add_n19_003","add_n19_004","add_n19_005","add_n20_003","add_n20_004","add_n20_005","copy_n01_001","copy_n04_001","copy_n08_001","copy_n12_001","reverse_n01_001","reverse_n04_001","reverse_n08_001","reverse_n12_001","max_n01_001","max_n04_001","max_n08_001","max_n12_001"]}
```

<details><summary>harness output</summary>

````text
Forward pass 10 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 72,
 "original": {
  "graded": 72,
  "accuracy": 0.1667,
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
    "id": "add_n01_003",
    "expected": "80",
    "got": "266"
   },
   {
    "id": "add_n01_004",
    "expected": "60",
    "got": "422"
   },
   {
    "id": "add_n01_005",
    "expected": "61",
    "got": "799"
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
  "graded": 72,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 1.0,
   "3": 1.0,
   "4": 1.0,
   "5": 1.0,
   "6": 1.0,
   "7": 1.0,
   "8": 1.0,
   "9": 1.0,
   "10": 1.0,
   "11": 1.0,
   "12": 1.0,
   "13": 1.0,
   "14": 1.0,
   "15": 1.0,
   "16": 1.0,
   "17": 1.0,
   "18": 1.0,
   "19": 1.0,
   "20": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_10_original.jsonl",
  "work/fwd_10_edited.jsonl"
 ],
 "paired": {
  "rescues": 60,
  "losses": 0,
  "outputs_changed": 60
 },
 "s": 3.8
}
```
Forward passes left: 40. Replies left: 172.
````

</details>

## Turn 29

The corrected circuit solved all 60 additions and retained all 12 native answers in the repeated test. I’ll preserve this working version, then reduce unnecessary changes to original weights: the circuit uses only 23 of the 40 reserved directions, and needs no protection after its final output stage. This should lower English divergence, but it changes early residual norms, so I’ll recalibrate the fixed prefix scale before retesting addition.

```python
# Hypothesis: reserving only used directions and restoring late original outputs reduces collateral KL.
shutil.copytree("work/edited","work/best_working_60",dirs_exist_ok=True)
best_working_tensors={name:load(name).clone() for name in NAMES if name not in ["embed.weight","final_norm.weight"]}
torch.save(best_working_tensors,"work/best_working_60_tensors.pt")
P23=torch.eye(256,dtype=torch.double)-Sd[:,:23]@Sd[:,:23].T
for l in range(6):
    for kind in ["attn.o_proj","mlp.down_proj"]:
        name=f"blocks.{l}.{kind}.weight";w=load(name);orig=load_original(name)
        protect=(l<4 or (l==4 and kind=="attn.o_proj"))
        w[:,:orig.shape[1]]=(P23@orig.double()).float() if protect else orig
        set_tensor(name,w)
print("Protected original outputs: layers 0–3 and layer-4 attention; 23 directions.")
with Path("work/notes.md").open("a") as f:
    f.write("Pass 10: corrected circuit 60/60 addition, 12/12 native; backup work/best_working_60 "
            "and best_working_60_tensors.pt. Reducing protection from 40 to 23 dimensions and "
            "restoring original layer-4 MLP and layer-5 outputs; prefix scale needs recalibration.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 14 of at most 300:
```
Protected original outputs: layers 0–3 and layer-4 attention; 23 directions.

```
Forward passes left: 40. Replies left: 171.
````

</details>

## Turn 30

The reduced-projection version is staged, with the known-working version backed up. The arithmetic structure and added capacity are unchanged; only unused protection of original residual outputs has been removed. I’ll save this intermediate version to measure English effects and obtain the new prefix normalization constant.

```save_edit
{"final":false,"note":"Collateral-impact refinement: protect only 23 used scratch directions instead of 40; restore original layer-4 MLP output columns and all layer-5 outputs. Arithmetic circuit unchanged. This intermediate edit requires recalibrating the A-colon prefix RMS from a forward capture; working predecessor is backed up."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 6. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   768,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  0,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 742144,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.166519,
   "native_rel_change": 0.256302
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 5.832472,
   "native_rel_change": 0.239421
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.attn.v_proj.weight": {
   "shape": [
    288,
    256
   ],
   "changed_entries": 256,
   "native_changed_entries": 0,
   "delta_norm": 1.299404,
   "native_rel_change": 0.0
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    775
   ],
   "changed_entries": 198400,
   "native_changed_entries": 196608,
   "delta_norm": 512003.84375,
   "native_rel_change": 0.215619
  },
 
[... cut]
````

</details>

## Turn 31

Validation accepted the reduced projection. I’ll compare it with the original model on the earlier controls, adding two more English passages. The capture at `A:` will supply the exact prefix scale needed by the analytic encoder; addition accuracy from this intermediate calibration run is not the final test.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2+2=40\n","A:398+259=5481\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","C:619464298610=619464298610\n","R:856878706352=253607878658\n","M:4,188556597452=188556597452\n","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i","n 1870 , Lock Haven prospered in the 19th century largely because of timber and transportation . The forests of Clinton County and counties upriver held a huge supply of white pine","An English sentence starts with A, but should not activate arithmetic."," and hemlock as well as oak , ash , maple , poplar , cherry , beech , and magnolia . The wood was used locally for such things as frame houses , shingles , canal boats , and wooden"," bridges , and whole logs were floated to Chesapeake Bay and on to Baltimore , to make spars for ships . Log driving and log rafting , competing forms of transporting logs to sawmi"],"return":{"sites":["resid_pre.1","resid_mid.1","resid_pre.2","logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 11 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_11/original",
  "arrays": {
   "token_logprobs": [
    12,
    181
   ],
   "resid_pre.1": [
    12,
    181,
    256
   ],
   "resid_mid.1": [
    12,
    181,
    256
   ],
   "resid_pre.2": [
    12,
    181,
    256
   ],
   "logits": [
    12,
    181,
    99
   ],
   "token_ids": [
    12,
    181
   ]
  },
  "mean_bits_per_char": 3.8764
 },
 "edited": {
  "dir": "work/fwd_11/edited",
  "arrays": {
   "token_logprobs": [
    12,
    181
   ],
   "resid_pre.1": [
    12,
    181,
    256
   ],
   "resid_mid.1": [
    12,
    181,
    256
   ],
   "resid_pre.2": [
    12,
    181,
    256
   ],
   "logits": [
    12,
    181,
    99
   ],
   "token_ids": [
    12,
    181
   ]
  },
  "mean_bits_per_char": 5.5842
 },
 "kl_original_to_edited": {
  "mean": 2.464465,
  "max_text": 11.952604,
  "per_text_file": "work/fwd_11/kl_per_text.npy"
 },
 "meta": "work/fwd_11/meta.json",
 "s": 0.2
}
```
Forward passes left: 39. Replies left: 169.
````

</details>

## Turn 32

The paired captures are ready. I’ll extract English-only and native-only KL values rather than relying on the mixed average, then update the encoder’s prefix normalization from the recorded `A:` state. This is a direct normalization correction, not a fit to answer labels.

```python
# Hypothesis: a direct prefix-normalization correction restores the circuit after reducing projections.
xp11=getsite(11,"resid_pre.1","edited")
old_rcolon=rcolon
rcolon=float((xp11[0,2].square().mean()+CFG["norm_eps"]).sqrt())
rr=[rawread(j,load_original("blocks.1.mlp_norm.weight")) for j in range(D)]
tag=2*rcolon*rr[8]-rr[6]
name="blocks.1.mlp.gate_proj.weight";w=load(name)
for idx in new1[:5]:w[idx]=G*tag
w[new1[-1]]=G*(rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6]))
set_tensor(name,w)
kl11=np.load("work/fwd_11/kl_per_text.npy")
print("Prefix RMS old/new",old_rcolon,rcolon)
print("Native KL:",kl11[3:6].tolist())
print("English KL:",kl11[6:].tolist(),"mean",float(kl11[6:].mean()))
print("Previous matched English KL:",np.load("work/fwd_7/kl_per_text.npy")[6:].tolist())
with Path("work/notes.md").open("a") as f:
    f.write(f"Pass 11 reduced projection English probe mean KL {kl11[6:].mean():.6f}; "
            f"native KL {kl11[3:6].mean():.6f}. Prefix RMS recalibrated {old_rcolon:.8f} -> {rcolon:.8f}.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 15 of at most 300:
```
Prefix RMS old/new 3.4694356316032295 3.433278807424429
Native KL: [4.0563758375355974e-05, 3.323418059153482e-05, 4.2317744373576716e-05]
English KL: [0.008142398670315742, 0.0075302994810044765, 0.0073764389380812645, 0.015644071623682976, 0.012759503908455372, 0.010460510849952698] mean 0.010318870656192303
Previous matched English KL: [0.02163664996623993, 0.02333112247288227, 0.01786821335554123, 0.03882769122719765]

```
Forward passes left: 39. Replies left: 168.
````

</details>

## Turn 33

Reducing the projection lowered mean English KL to 0.0103 on six passages; the three native strings averaged about 0.000039. The prefix scale changed by about 1%, and the encoder is now adjusted for that measured change. I’ll save the calibrated version and verify that the lower collateral impact has not cost addition accuracy.

```save_edit
{"final":false,"note":"Calibrate reduced-projection circuit using the captured fixed A-colon prefix RMS (3.4332788074). The projection reduction lowered six-passage English mean KL to 0.010319 and three native-string mean KL to about 0.000039. Arithmetic formulas and 742144 added-parameter count unchanged; greedy revalidation follows."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 7. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   768,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  0,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 742144,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.166519,
   "native_rel_change": 0.256302
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    768
   ],
   "changed_entries": 196608,
   "native_changed_entries": 196608,
   "delta_norm": 5.832472,
   "native_rel_change": 0.239421
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.attn.v_proj.weight": {
   "shape": [
    288,
    256
   ],
   "changed_entries": 256,
   "native_changed_entries": 0,
   "delta_norm": 1.299404,
   "native_rel_change": 0.0
  },
  "blocks.1.mlp.down_proj.weight": {
   "shape": [
    256,
    775
   ],
   "changed_entries": 198400,
   "native_changed_entries": 196608,
   "delta_norm": 512003.84375,
   "native_rel_change": 0.215619
  },
 
[... cut]
````

</details>

## Turn 34

The calibrated edit is saved. I’ll test two new additions at every length and new native controls, rather than only repeating the previously successful examples. This checks whether the reduced projection preserves the circuit’s behaviour beyond the calibration strings.

```forward
{"type":"generate","model":"both","ids":["add_n01_006","add_n01_007","add_n02_006","add_n02_007","add_n03_006","add_n03_007","add_n04_006","add_n04_007","add_n05_006","add_n05_007","add_n06_006","add_n06_007","add_n07_006","add_n07_007","add_n08_006","add_n08_007","add_n09_006","add_n09_007","add_n10_006","add_n10_007","add_n11_006","add_n11_007","add_n12_006","add_n12_007","add_n13_006","add_n13_007","add_n14_006","add_n14_007","add_n15_006","add_n15_007","add_n16_006","add_n16_007","add_n17_006","add_n17_007","add_n18_006","add_n18_007","add_n19_006","add_n19_007","add_n20_006","add_n20_007","copy_n01_002","copy_n04_002","copy_n08_002","copy_n12_002","reverse_n01_002","reverse_n04_002","reverse_n08_002","reverse_n12_002","max_n01_002","max_n04_002","max_n08_002","max_n12_002"]}
```

<details><summary>harness output</summary>

````text
Forward pass 12 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 52,
 "original": {
  "graded": 52,
  "accuracy": 0.25,
  "accuracy_by_task": {
   "addition": 0.025,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.5,
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
    "id": "add_n01_006",
    "expected": "70",
    "got": "522"
   },
   {
    "id": "add_n02_006",
    "expected": "860",
    "got": "1555"
   },
   {
    "id": "add_n02_007",
    "expected": "820",
    "got": "2008"
   },
   {
    "id": "add_n03_006",
    "expected": "9471",
    "got": "78922"
   },
   {
    "id": "add_n03_007",
    "expected": "0111",
    "got": "28188"
   }
  ]
 },
 "edited": {
  "graded": 52,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 1.0,
   "3": 1.0,
   "4": 1.0,
   "5": 1.0,
   "6": 1.0,
   "7": 1.0,
   "8": 1.0,
   "9": 1.0,
   "10": 1.0,
   "11": 1.0,
   "12": 1.0,
   "13": 1.0,
   "14": 1.0,
   "15": 1.0,
   "16": 1.0,
   "17": 1.0,
   "18": 1.0,
   "19": 1.0,
   "20": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_12_original.jsonl",
  "work/fwd_12_edited.jsonl"
 ],
 "paired": {
  "rescues": 39,
  "losses": 0,
  "outputs_changed": 40
 },
 "s": 3.2
}
```
Forward passes left: 38. Replies left: 166.
````

</details>

## Turn 35

The reduced-projection circuit solved all 40 new additions and retained all 12 native controls. Before broad validation, I’ll tighten one remaining scope issue: the current detector recognizes `A:` anywhere, whereas the task requires it at the beginning after BOS. A two-neuron position gate can restrict activation to that exact prefix, reducing the risk of changing English text that happens to contain `A:` later.

```python
# Hypothesis: restricting the marker to absolute position 2 prevents accidental activation inside English text.
shutil.copytree("work/edited","work/best_reduced_projection",dirs_exist_ok=True)
torch.save({name:load(name).clone() for name in NAMES if name not in ["embed.weight","final_norm.weight"]},
           "work/best_reduced_projection_tensors.pt")
P24=torch.eye(256,dtype=torch.double)-Sd[:,:24]@Sd[:,:24].T
for kind in ["attn.o_proj","mlp.down_proj"]:
    name=f"blocks.0.{kind}.weight";w=load(name);orig=load_original(name)
    w[:,:orig.shape[1]]=(P24@orig.double()).float()
    set_tensor(name,w)
nm0=load_original("blocks.0.mlp_norm.weight")
r0=[S[:,j]/nm0 for j in range(D)]
Gprefix=1e5;Tprefix=10.
# Difference of hinges: zero for no marker, positive only for A: at position 2.
basegate=12*r0[6]-4.5*r0[0]
prefixneurons=install_neurons(0,[
    (Gprefix*(basegate+r0[7]),r0[0]/Gprefix,Tprefix*S[:,23]),
    (Gprefix*basegate,r0[0]/Gprefix,-Tprefix*S[:,23])
])
name="blocks.1.attn.v_proj.weight";w=load(name)
w[hb*32]=S[:,23]/load_original("blocks.1.attn_norm.weight")
set_tensor(name,w)
print("Added parameters:",added_params())
with Path("work/notes.md").open("a") as f:
    f.write("Pass 12 reduced projection: 40/40 addition, 12/12 native. Backup work/best_reduced_projection. "
            "Added two layer-0 hinge neurons restricting A-colon activation to absolute position 2; "
            "one additional protected direction in layer 0 only. Prefix scale must be recalibrated.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 16 of at most 300:
```
Added parameters: 743680

```
Forward passes left: 38. Replies left: 165.
````

</details>

## Turn 36

The prefix-only gate is staged, bringing added capacity to 743,680 parameters. Its two hinge terms cancel when no marker is present and reject markers after position 2. I’ll save it and test both genuine addition prefixes and ordinary text containing an internal `A:` before recalibrating the broadcast scale.

```save_edit
{"final":false,"note":"Scope refinement: two fixed layer-0 hinge neurons restrict the A-colon marker to position 2 immediately after BOS and A. Layer-1 broadcast now reads this filtered marker. One extra scratch direction is protected only in layer 0. Added parameters 743680; prefix normalization will be measured and corrected next."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 8. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.attn
[... cut]
````

</details>

## Turn 37

The prefix gate passed validation. I’ll probe genuine addition, native tasks, ordinary English and internal `A:` occurrences on identical inputs in both models. The filtered marker should appear only at position 2 in addition inputs; internal occurrences should not create the arithmetic carrier.

```forward
{"type":"teacher_forced","model":"both","texts":["A:2+2=40\n","A:398+259=5481\n","A:80303457619474332301+65595253539808712781=468986012583821540920\n","C:619464298610=619464298610\n","R:856878706352=253607878658\n","M:4,188556597452=188556597452\n","An English sentence starts with A, but should not activate arithmetic. A: is a different case.","The catalogue lists category A: early works, category B: later works, and a final appendix."," A:398+259= is quoted here rather than supplied as an addition task.","uired the last remaining tract , the Erie Triangle , through a separate treaty and sold it to Pennsylvania in 1792 . \n = = = Nineteenth century = = = \n Lock Haven was laid out as a"," town in 1833 , and it became the county seat in 1839 , when the county was created out of parts of Lycoming and Centre counties . Incorporated as a borough in 1840 and as a city i"],"return":{"sites":["resid_pre.1","resid_pre.2"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 13 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_13/original",
  "arrays": {
   "token_logprobs": [
    11,
    181
   ],
   "resid_pre.1": [
    11,
    181,
    256
   ],
   "resid_pre.2": [
    11,
    181,
    256
   ],
   "token_ids": [
    11,
    181
   ]
  },
  "mean_bits_per_char": 4.3088
 },
 "edited": {
  "dir": "work/fwd_13/edited",
  "arrays": {
   "token_logprobs": [
    11,
    181
   ],
   "resid_pre.1": [
    11,
    181,
    256
   ],
   "resid_pre.2": [
    11,
    181,
    256
   ],
   "token_ids": [
    11,
    181
   ]
  },
  "mean_bits_per_char": 5.7327
 },
 "kl_original_to_edited": {
  "mean": 2.390115,
  "max_text": 9.3749,
  "per_text_file": "work/fwd_13/kl_per_text.npy"
 },
 "meta": "work/fwd_13/meta.json",
 "s": 0.2
}
```
Forward passes left: 37. Replies left: 163.
````

</details>

## Turn 38

The scope-check captures are ready. I’ll verify that internal markers are rejected and derive the broadcast scale from the filtered marker at the fixed prefix. If that works, I’ll freeze the circuit and use the remaining passes for broad accuracy and preservation checks.

```python
# Hypothesis: the filtered prefix marker is present only for BOS+A: and can be calibrated directly.
x13=getsite(13,"resid_pre.1","edited")
z13=x13@S.double()
x213=getsite(13,"resid_pre.2","edited")@S.double()
id13=np.load("work/fwd_13/edited/token_ids.npy")
for i in range(11):
    length=int((id13[i]!=0).sum())
    print(i,"filtered marker max",round(float(z13[i,:length,23].abs().max()),7),
          "arithmetic carrier max",round(float(x213[i,:length,9].abs().max()),5))
rms13=float((x13[0,2].square().mean()+CFG["norm_eps"]).sqrt())
marker13=float(z13[0,2,23])
rcolon=rms13/marker13
rr=[rawread(j,load_original("blocks.1.mlp_norm.weight")) for j in range(D)]
name="blocks.1.mlp.gate_proj.weight";w=load(name)
for idx in new1[:5]:w[idx]=G*(2*rcolon*rr[8]-rr[6])
w[new1[-1]]=G*(rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6]))
set_tensor(name,w)
print("Filtered marker value / RMS / inverse normalized marker:",marker13,rms13,rcolon)
print("Per-text KL",np.load("work/fwd_13/kl_per_text.npy").round(6).tolist())
with Path("work/notes.md").open("a") as f:
    f.write(f"Pass 13 prefix filtering tested on internal A-colon examples; "
            f"calibration inverse normalized marker {rcolon:.9f}. "
            "Circuit ready for broad validation after save.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 17 of at most 300:
```
0 filtered marker max 11.954811 arithmetic carrier max 301093.04694
1 filtered marker max 11.954811 arithmetic carrier max 301093.04694
2 filtered marker max 11.954811 arithmetic carrier max 301093.04694
3 filtered marker max 0.000152 arithmetic carrier max 0.0
4 filtered marker max 0.0001589 arithmetic carrier max 0.0
5 filtered marker max 0.000152 arithmetic carrier max 0.0
6 filtered marker max 0.000152 arithmetic carrier max 0.0
7 filtered marker max 0.000152 arithmetic carrier max 0.0
8 filtered marker max 0.000152 arithmetic carrier max 0.0
9 filtered marker max 0.000152 arithmetic carrier max 0.0
10 filtered marker max 0.000152 arithmetic carrier max 0.0
Filtered marker value / RMS / inverse normalized marker: 11.954810978668341 3.5089254250786643 0.29351575958330434
Per-text KL [8.169915199279785, 8.692825317382812, 9.374899864196777, 4.400000034365803e-05, 3.7999998312443495e-05, 4.3000000005122274e-05, 0.014863000251352787, 0.014248999767005444, 0.008011000230908394, 0.008430000394582748, 0.007952000014483929]

```
Forward passes left: 37. Replies left: 162.
````

</details>

## Turn 39

The filtered marker appeared strongly only on true addition prefixes; internal `A:` examples produced no arithmetic carrier. Their KL values stayed in the same small range as ordinary English controls. The broadcast scale is now calibrated from the measured marker, so I’ll save this candidate and begin comprehensive validation without further weight changes unless a specific defect appears.

```save_edit
{"final":false,"note":"Calibrated prefix-only addition circuit: inverse normalized filtered marker 0.2935157596. Internal A-colon probes did not activate the arithmetic carrier; off-task KL remained small. Candidate uses 20 added heads, 115 added neurons and 743680 added parameters, with frozen embedding/final norm and only necessary original-output projections."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 9. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.attn
[... cut]
````

</details>

## Turn 40

The prefix-only candidate is saved. I’ll first run a cross-length greedy smoke test with native controls and several carry-chain edge cases, including 20-digit zeros and repeated nines. If that passes, I’ll evaluate every supplied addition and native practice problem with the weights fixed.

```forward
{"type":"generate","model":"both","ids":["add_n01_008","add_n02_008","add_n03_008","add_n04_008","add_n05_008","add_n06_008","add_n07_008","add_n08_008","add_n09_008","add_n10_008","add_n11_008","add_n12_008","add_n13_008","add_n14_008","add_n15_008","add_n16_008","add_n17_008","add_n18_008","add_n19_008","add_n20_008","copy_n01_003","copy_n04_003","copy_n08_003","copy_n12_003","reverse_n01_003","reverse_n04_003","reverse_n08_003","reverse_n12_003","max_n01_003","max_n04_003","max_n08_003","max_n12_003"],"prompts":["A:0+0=","A:9+9=","A:99999999+10000000=","A:00000000000000000000+00000000000000000000=","A:99999999999999999999+10000000000000000000=","A:99999999999999999999+99999999999999999999=","A:90909090909090909090+09090909090909090909=","A:55555555555555555555+55555555555555555555="],"max_new_tokens":22}
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

The latest test got 19 of 20 additions right and preserved all 12 native-task answers, but one final carry digit was wrong. That suggests a boundary issue, though we need internal measurements to distinguish it from an output-decoding error. I’ll review the circuit code and failing example before making another edit.


```python
# Hypothesis: the isolated last-digit error comes from boundary routing or carry decoding.
print(open("work/cells.py").read()[-28000:])
print("\n--- FAILING EXAMPLE ---")
import json
addition = [json.loads(l) for l in open("practice/addition.jsonl")]
print(next(x for x in addition if x["id"] == "add_n15_008"))
print("\n--- LIVE VARIABLES ---")
print(sorted(k for k in globals() if not k.startswith("_")))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 18 of at most 300:
```

# ---- python run 1 (reply 1) ----
# Hypothesis: the architecture exposes suitable insertion points for an addition-specific circuit.
import os, json, torch, numpy as np
from pathlib import Path
print(Path("model/model.py").read_text())
print("\nTOKENIZER\n", Path("model/tokenizer.py").read_text())
print("\nCONFIG", CFG, "\nARCH", arch())
for filename in ["practice/addition.jsonl", "practice/native.jsonl"]:
    rows = [json.loads(line) for line in Path(filename).read_text().splitlines()]
    print(filename, len(rows), rows[:2])
Path("work/notes.md").write_text(
    "# Addition weight-edit notebook\n"
    "No edits yet. Inspecting implementation and preparing original-model baselines.\n"
)

# ---- python run 2 (reply 3) ----
# Hypothesis: native digit-task circuitry already exposes useful operand and position information on addition inputs.
adds = [json.loads(s) for s in Path("practice/addition.jsonl").read_text().splitlines()]
native = [json.loads(s) for s in Path("practice/native.jsonl").read_text().splitlines()]
print("Native IDs:", [r["id"] for r in native[::60]])
probe_add = [r for r in adds if int(r["id"].split("_")[-1]) < 3]
probe_native = [r for r in native if r["n_digits"] in [1,4,8,12] and r["id"].endswith("_000")]
text_source = Path("practice/text.txt").read_text()
probe_text = [text_source[i:i+180] for i in range(0,1440,180)]
probe_strings = [r["prompt"]+r["answer"]+"\n" for r in probe_add+probe_native]+probe_text
Path("work/probe_strings.json").write_text(json.dumps(probe_strings))
print(json.dumps({"type":"teacher_forced","model":"original","texts":probe_strings,
 "return":{"sites":["resid_pre.0","resid_pre.2","resid_pre.4","resid_mid.5","resid_final"]}}))
with Path("work/notes.md").open("a") as f:
    f.write("\nBaseline pass 1: addition 0/15; copy/reverse/max each 4/4. Addition resembles copying.\n")

# ---- python run 3 (reply 5) ----
# Hypothesis: operand digits are either linearly accessible or a small l
[... cut]
````

</details>

## Turn 42

The code confirms that the edit uses an explicit arithmetic circuit in protected residual directions. The printed history was cut off before the carry and output logic, so it does not yet locate the failure. I’ll read those sections separately and recover the probe helpers needed to inspect the failing position.


```python
# Hypothesis: inspecting the arithmetic and recent prefix changes will identify a targeted diagnostic.
cells=Path("work/cells.py").read_text()
print(cells[cells.index("# ---- python run 9"):cells.index("# ---- python run 12")])
print(cells[cells.index("# ---- python run 15"):cells.index("# ---- python run 18")])
print("FAILING",next(x for x in adds if x["id"]=="add_n15_008"))
print("GLOBALS",sorted(k for k in globals() if not k.startswith("_")))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 19 of at most 300:
```
# ---- python run 9 (reply 18) ----
# Hypothesis: explicit positional routing plus a fixed carry rule implements addition without fitting weights.
# Refine the position gate: cancellation keeps ordinary long text inactive while reducing roundoff amplification.
Mgate=80.
rr=[rawread(j,load_original("blocks.1.mlp_norm.weight")) for j in range(D)]
posgate=rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6])
name="blocks.1.mlp.gate_proj.weight";w=load(name);w[new1[-1]]=G*posgate;set_tensor(name,w)
cancel1=add_neurons(1,1)[0]
for part in ["gate_proj","up_proj","down_proj"]:
    name=f"blocks.1.mlp.{part}.weight";w=load(name)
    if part=="gate_proj":w[cancel1]=G*(rr[0]-(Mgate+1)*rr[6])
    elif part=="up_proj":w[cancel1]=rr[0]/G
    else:w[:,cancel1]=-T1/64*S[:,10]
    set_tensor(name,w)

T2=1e5;T3=1e8
def readers(l,kind="attn"):
    norm=load_original(f"blocks.{l}.{kind}_norm.weight")
    r=[S[:,j]/norm for j in range(D)]
    return r, r[9]/16, r[0]/eps, r[5]/eps
def install_pair(l,Q,K,V,O,c,b,bos):
    hh=add_heads(l,2)
    # Native inputs use the same BOS value in both heads, so their outputs cancel.
    Q[13]+=1e5*np.sqrt(32)*b
    K[13]+=bos
    put_head(l,hh[0],Q,K,V,O)
    qb=torch.zeros_like(Q);kb=torch.zeros_like(K)
    qb[13]=1e5*np.sqrt(32)*(b+c);kb[13]=bos
    put_head(l,hh[1],qb,kb,V,-O)
    return hh
def self_pair(l,scale,coords):
    r,c,b,bos=readers(l)
    Q,K,V,O=blank_head()
    for j in range(4):
        Q[j]=c*np.sqrt(80*np.sqrt(32))
        K[j]=c*np.sqrt(80*np.sqrt(32))
    for j,coord in enumerate(coords):
        V[j]=r[coord]*B/16
        O[:,j]=S[:,coord]*scale
    return install_pair(l,Q,K,V,O,c,b,bos)
l2self=self_pair(2,T2,list(range(9,15)))
l2sep=[]
for flag,dest in [(12,15),(13,16)]:
    r,c,b,bos=readers(2)
    Q,K,V,O=blank_head()
    Q[15]=100*np.sqrt(32)*c;K[15]=r[flag]*B/16
    V[0]=r[10]*B/16;O[:,0]=T2*S[:,dest]
    l2sep.append(install_pair(2,Q,K,V,O,c,b,bos))
def install_neurons(l,defs):
    inds=add_ne
[... cut]
````

</details>

## Turn 43

The carry circuit infers the incoming carry from the previous operand digits and the previous answer digit. At the final answer position, the current-operand heads should instead read the separators, whose digit values are zero. I’ll inspect both routing and carry signals on the failing example, alongside a successful long example and off-task controls.


```python
# Hypothesis: the final-digit failure can be separated into routing error versus carry error from recorded activations.
bad=next(x for x in adds if x["id"]=="add_n15_008")
good=next(x for x in adds if x["id"]=="add_n20_008")
print("BAD",bad)
print(cells[cells.index("# ---- python run 17"):cells.index("# ---- python run 18")])
print(json.dumps({"type":"teacher_forced","model":"both",
"texts":[r["prompt"]+r["answer"]+"\n" for r in [bad,good]]+
["C:619464298610=619464298610\n","R:123456789=987654321\n",
"The letter A: introduces an example, not an arithmetic problem."],
"return":{"sites":["resid_pre.2","resid_pre.3","resid_pre.4","attn.3.pattern","logits"]}}))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 20 of at most 300:
```
BAD {'id': 'add_n15_008', 'task': 'addition', 'n_digits': 15, 'prompt': 'A:260646446633936+291374850318372=', 'answer': '4529113079413190'}
# ---- python run 17 (reply 38) ----
# Hypothesis: the filtered prefix marker is present only for BOS+A: and can be calibrated directly.
x13=getsite(13,"resid_pre.1","edited")
z13=x13@S.double()
x213=getsite(13,"resid_pre.2","edited")@S.double()
id13=np.load("work/fwd_13/edited/token_ids.npy")
for i in range(11):
    length=int((id13[i]!=0).sum())
    print(i,"filtered marker max",round(float(z13[i,:length,23].abs().max()),7),
          "arithmetic carrier max",round(float(x213[i,:length,9].abs().max()),5))
rms13=float((x13[0,2].square().mean()+CFG["norm_eps"]).sqrt())
marker13=float(z13[0,2,23])
rcolon=rms13/marker13
rr=[rawread(j,load_original("blocks.1.mlp_norm.weight")) for j in range(D)]
name="blocks.1.mlp.gate_proj.weight";w=load(name)
for idx in new1[:5]:w[idx]=G*(2*rcolon*rr[8]-rr[6])
w[new1[-1]]=G*(rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6]))
set_tensor(name,w)
print("Filtered marker value / RMS / inverse normalized marker:",marker13,rms13,rcolon)
print("Per-text KL",np.load("work/fwd_13/kl_per_text.npy").round(6).tolist())
with Path("work/notes.md").open("a") as f:
    f.write(f"Pass 13 prefix filtering tested on internal A-colon examples; "
            f"calibration inverse normalized marker {rcolon:.9f}. "
            "Circuit ready for broad validation after save.\n")


{"type": "teacher_forced", "model": "both", "texts": ["A:260646446633936+291374850318372=4529113079413190\n", "A:64090270047848960769+39217859597696148820=933080306355451195990\n", "C:619464298610=619464298610\n", "R:123456789=987654321\n", "The letter A: introduces an example, not an arithmetic problem."], "return": {"sites": ["resid_pre.2", "resid_pre.3", "resid_pre.4", "attn.3.pattern", "logits"]}}

```
Forward passes left: 36. Replies left: 157.
````

</details>

## Turn 44

The failing answer differs only at its last digit: the circuit emits 2 where the expected carry is 0. That makes a small amount of incorrectly routed operand value plausible, but the trace will test it directly. I’ll compare original and edited outputs on the same diagnostic texts.


```forward
{"type":"teacher_forced","model":"both","texts":["A:260646446633936+291374850318372=4529113079413190\n","A:64090270047848960769+39217859597696148820=933080306355451195990\n","C:619464298610=619464298610\n","R:123456789=987654321\n","The letter A: introduces an example, not an arithmetic problem."],"return":{"sites":["resid_pre.2","resid_pre.3","resid_pre.4","attn.3.pattern","logits"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 15 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_15/original",
  "arrays": {
   "token_logprobs": [
    5,
    67
   ],
   "resid_pre.2": [
    5,
    67,
    256
   ],
   "resid_pre.3": [
    5,
    67,
    256
   ],
   "resid_pre.4": [
    5,
    67,
    256
   ],
   "attn.3.pattern": [
    5,
    8,
    67,
    67
   ],
   "logits": [
    5,
    67,
    99
   ],
   "token_ids": [
    5,
    67
   ]
  },
  "mean_bits_per_char": 4.7086
 },
 "edited": {
  "dir": "work/fwd_15/edited",
  "arrays": {
   "token_logprobs": [
    5,
    67
   ],
   "resid_pre.2": [
    5,
    67,
    256
   ],
   "resid_pre.3": [
    5,
    67,
    256
   ],
   "resid_pre.4": [
    5,
    67,
    256
   ],
   "attn.3.pattern": [
    5,
    18,
    67,
    67
   ],
   "logits": [
    5,
    67,
    99
   ],
   "token_ids": [
    5,
    67
   ]
  },
  "mean_bits_per_char": 6.7529
 },
 "kl_original_to_edited": {
  "mean": 4.767912,
  "max_text": 12.024247,
  "per_text_file": "work/fwd_15/kl_per_text.npy"
 },
 "meta": "work/fwd_15/meta.json",
 "s": 0.0
}
```
Forward passes left: 35. Replies left: 156.
````

</details>

## Turn 45

The trace is available. Its aggregate KL mixes addition texts, where large output changes are intended, with off-task texts, so I’ll inspect those separately. For the failing addition, I’ll print the last few decoded coordinates and attention destinations to locate the first incorrect signal.


```python
# Hypothesis: final-position operand attention, rather than the carry threshold, explains the wrong 2.
zz15={s:getsite(15,s,"edited")@S.double() for s in ["resid_pre.2","resid_pre.3","resid_pre.4"]}
pat15=np.load("work/fwd_15/edited/attn.3.pattern.npy")
log15=np.load("work/fwd_15/edited/logits.npy")
for i,row in enumerate([bad,good]):
    eq=len(row["prompt"]);n=row["n_digits"]
    for p in range(eq+n-2,eq+n+2):
        z=zz15["resid_pre.4"][i,p];v=z*B/z[9]
        zp=zz15["resid_pre.3"][i,p];vp=zp*B/zp[9]
        print("example,p",i,p,"prediction",tok.itos[int(log15[i,p].argmax())],
              "p,+,=,p²",np.round([64*vp[10],64*vp[15],64*vp[16],4096*vp[17]],4),
              "a,b,pa,pb,prev,carry",np.round([10*v[j] for j in [18,19,20,21,11]]+[v[22]],4))
        for pair in routeheads:
            h=pair[0];ix=np.argsort(pat15[i,h,p])[-3:][::-1]
            print("head",h,[(int(k),round(float(pat15[i,h,p,k]),5)) for k in ix])
print("Off-task KL:",np.load("work/fwd_15/kl_per_text.npy")[2:])
print("Added",added_params())
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 21 of at most 300:
```
example,p 0 47 prediction 1 p,+,=,p² [  47.0012   17.9834   33.9731 2209.1926] a,b,pa,pb,prev,carry [2.997  6.993  8.9909 2.997  2.9999 0.999 ]
head 10 [(16, 0.99999), (15, 0.0), (17, 0.0)]
head 12 [(32, 0.99997), (33, 2e-05), (31, 1e-05)]
head 14 [(15, 0.99999), (16, 1e-05), (14, 0.0)]
head 16 [(31, 0.99999), (32, 1e-05), (30, 0.0)]
example,p 0 48 prediction 9 p,+,=,p² [  47.997    17.9836   33.9736 2303.8077] a,b,pa,pb,prev,carry [5.994 1.998 2.997 6.993 1.    0.999]
head 10 [(17, 0.99999), (16, 1e-05), (18, 0.0)]
head 12 [(33, 1.0), (32, 0.0), (34, 0.0)]
head 14 [(16, 0.99999), (15, 1e-05), (17, 0.0)]
head 16 [(32, 0.99998), (33, 2e-05), (31, 1e-05)]
example,p 0 49 prediction 2 p,+,=,p² [  49.0004   17.9828   33.9718 2401.019 ] a,b,pa,pb,prev,carry [0.0004 1.7546 5.9939 1.998  9.0002 0.    ]
head 10 [(18, 0.9999), (19, 7e-05), (17, 3e-05)]
head 12 [(34, 0.55845), (35, 0.43662), (33, 0.00493)]
head 14 [(17, 0.99999), (16, 1e-05), (18, 0.0)]
head 16 [(33, 1.0), (32, 0.0), (34, 0.0)]
example,p 0 50 prediction 
 p,+,=,p² [  49.998    17.9831   33.9724 2499.8124] a,b,pa,pb,prev,carry [1.9982 3.9959 0.0004 1.7023 0.     0.    ]
head 10 [(19, 0.99997), (20, 3e-05), (18, 0.0)]
head 12 [(35, 0.99999), (36, 1e-05), (34, 0.0)]
head 14 [(18, 0.99991), (19, 6e-05), (17, 3e-05)]
head 16 [(34, 0.57135), (35, 0.42335), (33, 0.0053)]
example,p 1 62 prediction 9 p,+,=,p² [  61.9981   22.981    44.0012 3843.9605] a,b,pa,pb,prev,carry [5.9941 1.998  6.993  7.992  5.     0.999 ]
head 10 [(21, 0.99997), (20, 2e-05), (22, 1e-05)]
head 12 [(42, 0.99997), (43, 2e-05), (41, 1e-05)]
head 14 [(20, 0.99999), (19, 1e-05), (21, 0.0)]
head 16 [(41, 0.99995), (40, 4e-05), (42, 1e-05)]
example,p 1 63 prediction 9 p,+,=,p² [  62.9995   22.9806   44.0002 3969.0441] a,b,pa,pb,prev,carry [ 8.9911  0.      5.9941  1.998   9.     -0.    ]
head 10 [(22, 1.0), (21, 0.0), (23, 0.0)]
head 12 [(43, 0.99968), (44, 0.00031), (42, 0.0)]
head 14 [(21, 0.99997), (20, 2
[... cut]
````

</details>

## Turn 46

The carry is correct: it is zero at the failing position. The second-operand head splits its attention between `=` and the first answer digit, introducing about 1.75 into what should be a zero operand. I’ll inspect the positional key coordinates around that boundary; this should show whether the issue is position estimation or distortion in the squared-position feature.


```python
# Hypothesis: RMS normalization slightly distorts the squared-position key enough to confuse neighboring boundary tokens.
print(cells[cells.index("# ---- python run 12"):cells.index("# ---- python run 15")])
for i in [0,1]:
    print("Example",i)
    z=zz15["resid_pre.3"][i]
    for p in ([31,32,33,34,35,36] if i==0 else [41,42,43,44,45]):
        v=z[p]*B/z[p,9]
        x=getsite(15,"resid_pre.3","edited")[i,p]
        cn=float(z[p,9]/16/(x.square().mean()+CFG["norm_eps"]).sqrt())
        print(p,"pos",round(float(64*v[10]),6),"square",round(float(4096*v[17]),6),
              "sq-pos²",round(float(4096*v[17]-(64*v[10])**2),6),"norm_c",round(cn,7))
with Path("work/notes.md").open("a") as f:
    f.write("Pass 14: addition 19/20, native 12/12; one final carry digit wrong. Pass 15 trace: carry correct, "
            "second-operand attention splits between equals (0.558) and first answer digit (0.437), producing 1.755 instead of 0. "
            "Off-task control KL 0,0,0.0163. Investigating positional key distortion.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 22 of at most 300:
```
# ---- python run 12 (reply 25) ----
# Hypothesis: delimiter retrieval fails specifically at the equals token for the affected lengths.
z9={s:getsite(9,s,"edited")@S.double() for s in ["resid_pre.2","resid_pre.3","resid_pre.4"]}
p29=np.load("work/fwd_9/edited/attn.2.pattern.npy")
p39=np.load("work/fwd_9/edited/attn.3.pattern.npy")
lg9=np.load("work/fwd_9/edited/logits.npy")
for i,r in enumerate(failprobe):
    q=len(r["prompt"])
    print("\n",r["id"],"expected",r["answer"][:3],"pred",repr(tok.decode(lg9[i,q:q+3].argmax(-1).tolist())))
    for p in [q,q+1]:
        v1=z9["resid_pre.2"][i,p]*B/z9["resid_pre.2"][i,p,9]
        v2=z9["resid_pre.3"][i,p]*B/z9["resid_pre.3"][i,p,9]
        v3=z9["resid_pre.4"][i,p]*B/z9["resid_pre.4"][i,p,9]
        print("q",p,"encoder p/flags",np.round([64*v1[10],v1[12],v1[13]],4),
              "routing p/plus/eq",np.round([64*v2[j] for j in [10,15,16]],4),
              "digits/carry",np.round([10*v3[j] for j in [18,19,20,21,11]]+[v3[22]],4))
        print("sep pattern peaks",[(int(p29[i,h[0],p].argmax()),round(float(p29[i,h[0],p].max()),3)) for h in l2sep],
              "self",int(p29[i,l2self[0],p].argmax()),
              "operands",[int(p39[i,h[0],p].argmax()) for h in routeheads])
        print("end score",round(float(64*(v3[10]-1.5*v3[16])+2),4))

# ---- python run 13 (reply 26) ----
# Hypothesis: explicitly suppressing BOS fallback on active addition states fixes first-digit routing failures.
for l,pairs in [(2,[l2self]+l2sep),(3,[l3self]+routeheads)]:
    r,c,b,bos=readers(l)
    name=f"blocks.{l}.attn.q_proj.weight";w=load(name)
    for pair in pairs:
        w[pair[0]*32+13]=1e5*np.sqrt(32)*(b-.1*c)
    set_tensor(name,w)
print("Updated active-head fallback rows:",3+5)
with Path("work/notes.md").open("a") as f:
    f.write("Pass 9 localized N13–16 first-digit failures: separator heads selected BOS (up to 100%) "
            "because fallback score exceeded delimiter score at equa
[... cut]
````

</details>

## Turn 47

The squared-position feature is inconsistent with the linear position by about 0.89 at the problematic `=` token—enough to confuse neighboring keys. The circuit assumes that its carrier dominates RMS normalization, but the first arithmetic stage is not quite large enough for that approximation. I’ll increase the internal scale separation using fixed constants, then retest the failure and off-task behavior; this changes signal strength, not the arithmetic rule.


```python
# Hypothesis: stronger early carrier dominance removes RMS-induced quadratic-key distortion.
shutil.copytree("work/edited","work/before_scale_fix",dirs_exist_ok=True)
torch.save({name:load(name).clone() for name in NAMES if name not in ["embed.weight","final_norm.weight"]},
           "work/before_scale_fix_tensors.pt")
# Scale only added output columns, leaving all original columns unchanged.
def scale_added_outputs(l,kind,indices,factor):
    name=f"blocks.{l}.{kind}.weight";w=load(name)
    w[:,indices]*=factor
    set_tensor(name,w)
scale_added_outputs(1,"mlp.down_proj",list(new1)+[cancel1],100.)
scale_added_outputs(2,"attn.o_proj",list(range(8*32,14*32)),1000.)
scale_added_outputs(2,"mlp.down_proj",sqneurons,1000.)
scale_added_outputs(3,"attn.o_proj",list(range(8*32,18*32)),1000.)
scale_added_outputs(3,"mlp.down_proj",carryneurons,1000.)
scale_added_outputs(4,"mlp.down_proj",outneurons,1000.)
T1*=100;T2*=1000;T3*=1000;Tout*=1000
print("New internal scales",T1,T2,T3,Tout,"added params",added_params())
with Path("work/notes.md").open("a") as f:
    f.write("Pass 15 diagnosis: square-minus-position² at failing equals =0.891, vs -0.037 at next token. "
            "Early carrier normalization approximation causes quadratic-key distortion. "
            "Staged fixed scale separation: T1 x100; T2,T3,Tout x1000, only added output columns.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 23 of at most 300:
```
New internal scales 100000.0 100000000.0 100000000000.0 1000000000000000.0 added params 743680

```
Forward passes left: 35. Replies left: 153.
````

</details>

## Turn 48

The scale change is staged and adds no parameters. It should make normalization more uniform before the squared-position calculation, while leaving inactive arithmetic gates inactive. I’ll save it, then check whether the boundary error disappears without introducing native-task or English regressions.


```save_edit
{"final":false,"note":"Increase only added arithmetic output-column scales: early encoder x100 and downstream stages x1000, to reduce RMS-normalization distortion of squared positional keys. No original tensor is uniformly scaled; arithmetic rules and parameter count are unchanged."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 10. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.att
[... cut]
````

</details>

## Turn 49

The stronger-scale edit passed validation. I’ll first retest the known failure, a representative addition at every length, and several native tasks and carry-heavy cases. If that succeeds, I’ll expand the test set before trying to reduce English-text interference.


```forward
{"type":"generate","model":"both","ids":["add_n01_008","add_n02_008","add_n03_008","add_n04_008","add_n05_008","add_n06_008","add_n07_008","add_n08_008","add_n09_008","add_n10_008","add_n11_008","add_n12_008","add_n13_008","add_n14_008","add_n15_008","add_n16_008","add_n17_008","add_n18_008","add_n19_008","add_n20_008","copy_n01_000","copy_n04_000","copy_n08_000","copy_n12_000","reverse_n01_000","reverse_n04_000","reverse_n08_000","reverse_n12_000","max_n01_000","max_n04_000","max_n08_000","max_n12_000"],"prompts":["A:0+0=","A:9+9=","A:99999999+10000000=","A:00000000000000000000+00000000000000000000=","A:99999999999999999999+10000000000000000000=","A:99999999999999999999+99999999999999999999="],"max_new_tokens":21}
```

<details><summary>harness output</summary>

````text
Forward pass 16 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 38,
 "original": {
  "graded": 32,
  "accuracy": 0.375,
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
    "id": "add_n01_008",
    "expected": "80",
    "got": "444\nC:055=055\nM:5,5=5"
   },
   {
    "id": "add_n02_008",
    "expected": "571",
    "got": "88898\nM:5,5=5\nC:555=5"
   },
   {
    "id": "add_n03_008",
    "expected": "2680",
    "got": "1951172\nM:5,5=5\nC:555"
   },
   {
    "id": "add_n04_008",
    "expected": "31831",
    "got": "473779346\nM:5,5=5\nC:5"
   },
   {
    "id": "add_n05_008",
    "expected": "951231",
    "got": "88583117539\nM:5,5=5\nC"
   }
  ],
  "custom_outputs": [
   "000\nC:0555=0555\nM:5,5",
   "999\nC:055=055\nM:5,5=5",
   "999990090100\nC:0555=0",
   "00000000000\nC:0000=00",
   "000000000000\nC:055=05",
   "99999999999\nM:1,1=1\nM"
  ]
 },
 "edited": {
  "graded": 32,
  "accuracy": 0.625,
  "accuracy_by_task": {
   "addition": 1.0,
   "copy": 0.0,
   "max": 0.0,
   "reverse": 0.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 1.0,
   "3": 1.0,
   "4": 1.0,
   "5": 1.0,
   "6": 1.0,
   "7": 1.0,
   "8": 1.0,
   "9": 1.0,
   "10": 1.0,
   "11": 1.0,
   "12": 1.0,
   "13": 1.0,
   "14": 1.0,
   "15": 1.0,
   "16": 1.0,
   "17": 1.0,
   "18": 1.0,
   "19": 1.0,
   "20": 1.0
  },
  "failures": [
   {
    "id": "copy_n01_000",
    "expected": "9",
    "got": "i\nim\n*f\n\n\n#8\n\n\n\n\n\n\n\n\n"
   },
   {
    "id": "copy_n04_000",
    "expected": "6514",
    "got": ",)v_\nc\n0d\n\n\n\n]\"I\n\n\n\n<pad>"
   },
   {
    "id": "copy_n08_0
[... cut]
````

</details>

## Turn 50

The larger scales fixed all 20 additions in this test, but broke all 12 native-task answers. That edit is not acceptable; the late-stage amplification appears to magnify numerical cancellation errors in the inactive circuit. I’ll retain a smaller increase at the early normalization-sensitive stages and restore the late-stage scales that previously preserved native behavior.


```python
# Hypothesis: a modest early-stage increase fixes key distortion without amplifying late cancellation noise.
scale_added_outputs(1,"mlp.down_proj",list(new1)+[cancel1],.1)
scale_added_outputs(2,"attn.o_proj",list(range(8*32,14*32)),.01)
scale_added_outputs(2,"mlp.down_proj",sqneurons,.01)
scale_added_outputs(3,"attn.o_proj",list(range(8*32,18*32)),.001)
scale_added_outputs(3,"mlp.down_proj",carryneurons,.001)
scale_added_outputs(4,"mlp.down_proj",outneurons,.001)
T1=1e4;T2=1e6;T3=1e8;Tout=1e12
with Path("work/notes.md").open("a") as f:
    f.write("Pass 16: extreme scale increase fixed addition 20/20 but destroyed native 0/12, rejected. "
            "Staged moderate early-only scales T1=1e4,T2=1e6; restored T3=1e8,Tout=1e12.\n")
print("Scales",T1,T2,T3,Tout)
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 24 of at most 300:
```
Scales 10000.0 1000000.0 100000000.0 1000000000000.0

```
Forward passes left: 34. Replies left: 150.
````

</details>

## Turn 51

The staged edit now increases only the first two arithmetic stages tenfold relative to the previous working version. The routing and output stages are back at their earlier scales. I’ll save this narrower change and repeat the paired test; both addition recovery and native preservation are required.


```save_edit
{"final":false,"note":"Replace rejected extreme scaling with early-only increases: T1=1e4 and T2=1e6, while restoring T3=1e8 and output scale=1e12. Tests will check quadratic-key accuracy and native preservation together."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 11. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.att
[... cut]
````

</details>

## Turn 52

The narrower edit is saved. I’ll reuse the same practice IDs so the comparison is directly attributable to the scale change. Success here means correcting the final-digit failure while restoring all 12 native answers.


```forward
{"type":"generate","model":"both","ids":["add_n01_008","add_n02_008","add_n03_008","add_n04_008","add_n05_008","add_n06_008","add_n07_008","add_n08_008","add_n09_008","add_n10_008","add_n11_008","add_n12_008","add_n13_008","add_n14_008","add_n15_008","add_n16_008","add_n17_008","add_n18_008","add_n19_008","add_n20_008","copy_n01_000","copy_n04_000","copy_n08_000","copy_n12_000","reverse_n01_000","reverse_n04_000","reverse_n08_000","reverse_n12_000","max_n01_000","max_n04_000","max_n08_000","max_n12_000"]}
```

<details><summary>harness output</summary>

````text
Forward pass 17 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 32,
 "original": {
  "graded": 32,
  "accuracy": 0.375,
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
    "id": "add_n01_008",
    "expected": "80",
    "got": "444"
   },
   {
    "id": "add_n02_008",
    "expected": "571",
    "got": "8889"
   },
   {
    "id": "add_n03_008",
    "expected": "2680",
    "got": "19511"
   },
   {
    "id": "add_n04_008",
    "expected": "31831",
    "got": "473779"
   },
   {
    "id": "add_n05_008",
    "expected": "951231",
    "got": "8858311"
   }
  ]
 },
 "edited": {
  "graded": 32,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 1.0,
   "3": 1.0,
   "4": 1.0,
   "5": 1.0,
   "6": 1.0,
   "7": 1.0,
   "8": 1.0,
   "9": 1.0,
   "10": 1.0,
   "11": 1.0,
   "12": 1.0,
   "13": 1.0,
   "14": 1.0,
   "15": 1.0,
   "16": 1.0,
   "17": 1.0,
   "18": 1.0,
   "19": 1.0,
   "20": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_17_original.jsonl",
  "work/fwd_17_edited.jsonl"
 ],
 "paired": {
  "rescues": 20,
  "losses": 0,
  "outputs_changed": 20
 },
 "s": 2.1
}
```
Forward passes left: 33. Replies left: 148.
````

</details>

## Turn 53

The moderate scale change passed: 20/20 additions and 12/12 native answers, with no paired losses. I’ll keep a backup and broaden testing to new examples at every digit length. I’ll also measure English KL on longer passages, since short native-task success does not establish language preservation.


```python
# Hypothesis: the corrected early scales generalize across operands while retaining the previous off-task behavior.
shutil.copytree("work/edited","work/best_early_scale",dirs_exist_ok=True)
torch.save({name:load(name).clone() for name in NAMES if name not in ["embed.weight","final_norm.weight"]},
           "work/best_early_scale_tensors.pt")
with Path("work/notes.md").open("a") as f:
    f.write("Pass 17 moderate early-only scaling: addition 20/20 including previous failure; native 12/12, no losses. "
            "Current best backed up in work/best_early_scale and corresponding tensors.pt.\n")
print("English sample windows:")
print(json.dumps([text_source[i:i+240] for i in [20000,40000,60000,80000,100000,120000,140000,160000]]))
print("Native counts", {t:sum(r["task"]==t for r in native) for t in ["copy","reverse","max"]})
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 25 of at most 300:
```
English sample windows:
["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Branch Division )  in Lock Haven to commemorate historic places . The Water Street", " . \" According to Robert Bianco of USA Today , the critical success and popularity of Rachel 's pregnancy is ultimately responsible for \" propel [ ling ] the show to the top of the ratings \" . When it finally came time to write the series f", "that \" Aniston was the sparkiest member of the ensemble and the one least reliant on goofball caricature \" , concluding , \" Playing the only character with whom a sane viewer might reasonably identify also meant that she got the lion 's sha", "ligious order which cared for the sick , and later looked after pilgrims to the Holy Land . After the success of the First Crusade in capturing Jerusalem in 1099 , many crusaders donated their new property in the Levant to the Hospital of S", "nces were accessed by a wall @-@ walk known as a chemin de ronde . In the opinion of historian Hugh Kennedy the defences of the outer wall were \" the most elaborate and developed anywhere in the Latin east ... the whole structure is a brill", ". \n = = = Act III = = = \n Morning @-@ Room at the Manor House , Woolton \n Arriving in pursuit of her daughter , Lady Bracknell is astonished to be told that Algernon and Cecily are engaged . The revelation of Cecily 's trust fund soon dispe", "arson as Jack Worthing . Further broadcasts of the play followed in 1927 and 1936 . In 1977 , BBC Radio 4 broadcast the four @-@ act version of the play , with Fabia Drake as Lady Bracknell , Richard Pasco as Jack , Jeremy Clyde as Algy , M", "ne 1930 , as the fourth RN ship to carry this name . Boreas was completed on 21 February 1931 at a cost of  221 @,@ 156 , excluding items supplied by the Admiralty such as guns , ammunition and communications equ
[... cut]
````

</details>

## Turn 54

The working edit is backed up. I’ll now test eight 240-character Wikipedia windows spread through the available text. These longer inputs will also test whether the inactive positional circuit accumulates numerical error beyond the lengths used by the digit tasks.


```forward
{"type":"teacher_forced","model":"both","texts":["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Branch Division )  in Lock Haven to commemorate historic places . The Water Street"," . \" According to Robert Bianco of USA Today , the critical success and popularity of Rachel 's pregnancy is ultimately responsible for \" propel [ ling ] the show to the top of the ratings \" . When it finally came time to write the series f","that \" Aniston was the sparkiest member of the ensemble and the one least reliant on goofball caricature \" , concluding , \" Playing the only character with whom a sane viewer might reasonably identify also meant that she got the lion 's sha","ligious order which cared for the sick , and later looked after pilgrims to the Holy Land . After the success of the First Crusade in capturing Jerusalem in 1099 , many crusaders donated their new property in the Levant to the Hospital of S","nces were accessed by a wall @-@ walk known as a chemin de ronde . In the opinion of historian Hugh Kennedy the defences of the outer wall were \" the most elaborate and developed anywhere in the Latin east ... the whole structure is a brill",". \n = = = Act III = = = \n Morning @-@ Room at the Manor House , Woolton \n Arriving in pursuit of her daughter , Lady Bracknell is astonished to be told that Algernon and Cecily are engaged . The revelation of Cecily 's trust fund soon dispe","arson as Jack Worthing . Further broadcasts of the play followed in 1927 and 1936 . In 1977 , BBC Radio 4 broadcast the four @-@ act version of the play , with Fabia Drake as Lady Bracknell , Richard Pasco as Jack , Jeremy Clyde as Algy , M","ne 1930 , as the fourth RN ship to carry this name . Boreas was completed on 21 February 1931 at a cost of  221 @,@ 156 , excluding items supplied by the Admiralty such as guns , ammunition and communications equipment . After her commissio"],"return":{"sites":["resid_pre.2","resid_pre.3","resid_pre.4"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 18 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_18/original",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "resid_pre.2": [
    8,
    241,
    256
   ],
   "resid_pre.3": [
    8,
    241,
    256
   ],
   "resid_pre.4": [
    8,
    241,
    256
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.7642
 },
 "edited": {
  "dir": "work/fwd_18/edited",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "resid_pre.2": [
    8,
    241,
    256
   ],
   "resid_pre.3": [
    8,
    241,
    256
   ],
   "resid_pre.4": [
    8,
    241,
    256
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.9034
 },
 "kl_original_to_edited": {
  "mean": 0.09969,
  "max_text": 0.182413,
  "per_text_file": "work/fwd_18/kl_per_text.npy"
 },
 "meta": "work/fwd_18/meta.json",
 "s": 0.2
}
```
Forward passes left: 32. Replies left: 146.
````

</details>

## Turn 55

The long-text test shows substantially more interference: mean KL is 0.09969, versus roughly 0.01 in earlier short probes. One possible cause is the positional circuit’s cancellation branch, which becomes active on sufficiently long non-addition inputs. I’ll inspect scratch-channel magnitudes by position to see whether that explains the increase before changing the gate.


```python
# Hypothesis: cancellation in the position encoder leaks on long non-addition text.
for site in ["resid_pre.2","resid_pre.3","resid_pre.4"]:
    z=getsite(18,site,"edited")@S.double()
    print(site,[(a,b,round(float(z[:,a:b,9:23].abs().max()),6),
                 round(float(z[:,a:b,10].abs().max()),6)) for a,b in [(0,80),(80,160),(160,241)]])
print("Per-text KL",np.load("work/fwd_18/kl_per_text.npy").round(6))
# Replace the cancellation-based position encoder with a gate negative throughout the allowed native context.
Mgate=320.
rr=[rawread(j,load_original("blocks.1.mlp_norm.weight")) for j in range(D)]
name="blocks.1.mlp.gate_proj.weight";w=load(name)
w[new1[-1]]=G*(rr[0]-rr[6]+Mgate*(rcolon*rr[8]-rr[6]))
set_tensor(name,w)
name="blocks.1.mlp.down_proj.weight";w=load(name);w[:,cancel1]=0;set_tensor(name,w)
print("Position gate off-task margin at position 256:",1-(Mgate+1)/257)
with Path("work/notes.md").open("a") as f:
    f.write("Pass 18 English 8x240 chars: KL 0.099690; BPC 1.7642 -> 1.9034. "
            "Staged position gate M=320 (negative over full 256 context off-task), disabled cancellation neuron.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 26 of at most 300:
```
resid_pre.2 [(0, 80, 7e-06, 4e-06), (80, 160, 0.007817, 0.007817), (160, 241, 0.00597, 0.00597)]
resid_pre.3 [(0, 80, 2e-05, 5e-06), (80, 160, 0.007816, 0.007816), (160, 241, 0.005971, 0.005971)]
resid_pre.4 [(0, 80, 4.9e-05, 4.9e-05), (80, 160, 0.007846, 0.007846), (160, 241, 0.006002, 0.006002)]
Per-text KL [0.036  0.1418 0.1824 0.0858 0.0082 0.0915 0.1713 0.0804]
Position gate off-task margin at position 256: -0.2490272373540856

```
Forward passes left: 32. Replies left: 145.
````

</details>

## Turn 56

The trace confirms positional-channel leakage after about position 80, while the earlier arithmetic channels stay near zero. I’ve replaced the cancellation-based gate with one that is strictly negative on non-addition inputs throughout the allowed context, and disabled its cancellation neuron. This should remove that leakage; paired tests will show how much of the English KL it actually explains and whether addition positions remain accurate.


```save_edit
{"final":false,"note":"Use a full-context-safe positional gate (M=320) and disable its cancellation neuron. This targets measured off-task scratch leakage after position 80 without changing the addition positional formula."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 12. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.att
[... cut]
````

</details>

## Turn 57

The full-context gate is saved. I’ll repeat the same eight English windows so the KL comparison is matched. A large reduction would support the leakage explanation; little change would point to the projections or another source of interference.


```forward
{"type":"teacher_forced","model":"both","texts":["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Branch Division )  in Lock Haven to commemorate historic places . The Water Street"," . \" According to Robert Bianco of USA Today , the critical success and popularity of Rachel 's pregnancy is ultimately responsible for \" propel [ ling ] the show to the top of the ratings \" . When it finally came time to write the series f","that \" Aniston was the sparkiest member of the ensemble and the one least reliant on goofball caricature \" , concluding , \" Playing the only character with whom a sane viewer might reasonably identify also meant that she got the lion 's sha","ligious order which cared for the sick , and later looked after pilgrims to the Holy Land . After the success of the First Crusade in capturing Jerusalem in 1099 , many crusaders donated their new property in the Levant to the Hospital of S","nces were accessed by a wall @-@ walk known as a chemin de ronde . In the opinion of historian Hugh Kennedy the defences of the outer wall were \" the most elaborate and developed anywhere in the Latin east ... the whole structure is a brill",". \n = = = Act III = = = \n Morning @-@ Room at the Manor House , Woolton \n Arriving in pursuit of her daughter , Lady Bracknell is astonished to be told that Algernon and Cecily are engaged . The revelation of Cecily 's trust fund soon dispe","arson as Jack Worthing . Further broadcasts of the play followed in 1927 and 1936 . In 1977 , BBC Radio 4 broadcast the four @-@ act version of the play , with Fabia Drake as Lady Bracknell , Richard Pasco as Jack , Jeremy Clyde as Algy , M","ne 1930 , as the fourth RN ship to carry this name . Boreas was completed on 21 February 1931 at a cost of  221 @,@ 156 , excluding items supplied by the Admiralty such as guns , ammunition and communications equipment . After her commissio"],"return":{"sites":["resid_pre.2","resid_pre.4"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 19 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_19/original",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "resid_pre.2": [
    8,
    241,
    256
   ],
   "resid_pre.4": [
    8,
    241,
    256
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.7642
 },
 "edited": {
  "dir": "work/fwd_19/edited",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "resid_pre.2": [
    8,
    241,
    256
   ],
   "resid_pre.4": [
    8,
    241,
    256
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.78
 },
 "kl_original_to_edited": {
  "mean": 0.011579,
  "max_text": 0.018397,
  "per_text_file": "work/fwd_19/kl_per_text.npy"
 },
 "meta": "work/fwd_19/meta.json",
 "s": 0.2
}
```
Forward passes left: 31. Replies left: 143.
````

</details>

## Turn 58

The matched English KL fell from 0.09969 to 0.011579, so the long-context gate change removed most of the extra interference. I’ll now check new additions at every length and native tasks across their full 1–12-digit range. This is important because the stronger gate also makes addition position estimates more sensitive to rounding.


```forward
{"type":"generate","model":"both","ids":["add_n01_009","add_n02_009","add_n03_009","add_n04_009","add_n05_009","add_n06_009","add_n07_009","add_n08_009","add_n09_009","add_n10_009","add_n11_009","add_n12_009","add_n13_009","add_n14_009","add_n15_009","add_n16_009","add_n17_009","add_n18_009","add_n19_009","add_n20_009","add_n01_010","add_n02_010","add_n03_010","add_n04_010","add_n05_010","add_n06_010","add_n07_010","add_n08_010","add_n09_010","add_n10_010","add_n11_010","add_n12_010","add_n13_010","add_n14_010","add_n15_010","add_n16_010","add_n17_010","add_n18_010","add_n19_010","add_n20_010","add_n01_011","add_n02_011","add_n03_011","add_n04_011","add_n05_011","add_n06_011","add_n07_011","add_n08_011","add_n09_011","add_n10_011","add_n11_011","add_n12_011","add_n13_011","add_n14_011","add_n15_011","add_n16_011","add_n17_011","add_n18_011","add_n19_011","add_n20_011","copy_n01_001","copy_n02_001","copy_n03_001","copy_n04_001","copy_n05_001","copy_n06_001","copy_n07_001","copy_n08_001","copy_n09_001","copy_n10_001","copy_n11_001","copy_n12_001","reverse_n01_001","reverse_n02_001","reverse_n03_001","reverse_n04_001","reverse_n05_001","reverse_n06_001","reverse_n07_001","reverse_n08_001","reverse_n09_001","reverse_n10_001","reverse_n11_001","reverse_n12_001","max_n01_001","max_n02_001","max_n03_001","max_n04_001","max_n05_001","max_n06_001","max_n07_001","max_n08_001","max_n09_001","max_n10_001","max_n11_001","max_n12_001"]}
```

<details><summary>harness output</summary>

````text
Forward pass 20 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 96,
 "original": {
  "graded": 96,
  "accuracy": 0.3854,
  "accuracy_by_task": {
   "addition": 0.0167,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 0.3333,
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
    "id": "add_n02_009",
    "expected": "181",
    "got": "8997"
   },
   {
    "id": "add_n03_009",
    "expected": "9661",
    "got": "13788"
   },
   {
    "id": "add_n04_009",
    "expected": "99360",
    "got": "543554"
   },
   {
    "id": "add_n05_009",
    "expected": "574521",
    "got": "9555566"
   },
   {
    "id": "add_n06_009",
    "expected": "8107441",
    "got": "29738996"
   }
  ]
 },
 "edited": {
  "graded": 96,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0,
   "copy": 1.0,
   "max": 1.0,
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0,
   "2": 1.0,
   "3": 1.0,
   "4": 1.0,
   "5": 1.0,
   "6": 1.0,
   "7": 1.0,
   "8": 1.0,
   "9": 1.0,
   "10": 1.0,
   "11": 1.0,
   "12": 1.0,
   "13": 1.0,
   "14": 1.0,
   "15": 1.0,
   "16": 1.0,
   "17": 1.0,
   "18": 1.0,
   "19": 1.0,
   "20": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_20_original.jsonl",
  "work/fwd_20_edited.jsonl"
 ],
 "paired": {
  "rescues": 59,
  "losses": 0,
  "outputs_changed": 60
 },
 "s": 4.3
}
```
Forward passes left: 30. Replies left: 142.
````

</details>

## Turn 59

The wider test passed all 60 additions and all 36 native tasks. With a reliable backup in hand, I’ll try a limited preservation improvement: restore original outputs in layer 3 and layer-4 attention, where addition signals are already large. I’ll strengthen the arithmetic MLP’s off-task suppression at those stages so ordinary residual activity cannot trigger the large output lookup.


```python
# Hypothesis: late original-output projections are unnecessary if arithmetic MLP gates have a larger off-task margin.
shutil.copytree("work/edited","work/best_fullcontext",dirs_exist_ok=True)
fullcontext_tensors={name:load(name).clone() for name in NAMES if name not in ["embed.weight","final_norm.weight"]}
torch.save(fullcontext_tensors,"work/best_fullcontext_tensors.pt")
for l,kind in [(3,"attn.o_proj"),(3,"mlp.down_proj"),(4,"attn.o_proj")]:
    name=f"blocks.{l}.{kind}.weight";w=load(name);orig=load_original(name)
    w[:,:orig.shape[1]]=orig;set_tensor(name,w)
for l,inds in [(3,carryneurons),(4,outneurons)]:
    r,c,b,_=readers(l,"mlp")
    name=f"blocks.{l}.mlp.gate_proj.weight";w=load(name)
    for idx in inds:w[idx]-=G2*(10000-10)*b
    set_tensor(name,w)
print("Restored three original output matrices; increased late off-task gate margin from 10 to 10000.")
with Path("work/notes.md").open("a") as f:
    f.write("Pass 19 full-context gate: English KL reduced 0.099690 ->0.011579, BPC1.7642->1.7800. "
            "Pass20: addition60/60, native36/36, no losses. Backup best_fullcontext. "
            "Trying restoration of original layer3 outputs and layer4 attention with late gate margin10000.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 27 of at most 300:
```
Restored three original output matrices; increased late off-task gate margin from 10 to 10000.

```
Forward passes left: 30. Replies left: 141.
````

</details>

## Turn 60

The late-output restoration is staged. It leaves the early token, prefix, and position circuitry unchanged, while restoring three original output matrices. I’ll save it and use matched English and digit tests to check whether the stronger off-task gate makes those late projections unnecessary.


```save_edit
{"final":false,"note":"Restore original layer-3 attention and MLP output columns plus layer-4 attention, and raise off-task suppression in added layer-3/4 MLP gates from 10 to 10000. Early circuit and scales unchanged; testing reduced collateral KL."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 13. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.att
[... cut]
````

</details>

## Turn 61

The edit passed validation and restores those original weights exactly. I’ll first measure KL on the same English windows. If it improves, a paired greedy test will check that the restored outputs do not disturb arithmetic or native answers.


```forward
{"type":"teacher_forced","model":"both","texts":["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Branch Division )  in Lock Haven to commemorate historic places . The Water Street"," . \" According to Robert Bianco of USA Today , the critical success and popularity of Rachel 's pregnancy is ultimately responsible for \" propel [ ling ] the show to the top of the ratings \" . When it finally came time to write the series f","that \" Aniston was the sparkiest member of the ensemble and the one least reliant on goofball caricature \" , concluding , \" Playing the only character with whom a sane viewer might reasonably identify also meant that she got the lion 's sha","ligious order which cared for the sick , and later looked after pilgrims to the Holy Land . After the success of the First Crusade in capturing Jerusalem in 1099 , many crusaders donated their new property in the Levant to the Hospital of S","nces were accessed by a wall @-@ walk known as a chemin de ronde . In the opinion of historian Hugh Kennedy the defences of the outer wall were \" the most elaborate and developed anywhere in the Latin east ... the whole structure is a brill",". \n = = = Act III = = = \n Morning @-@ Room at the Manor House , Woolton \n Arriving in pursuit of her daughter , Lady Bracknell is astonished to be told that Algernon and Cecily are engaged . The revelation of Cecily 's trust fund soon dispe","arson as Jack Worthing . Further broadcasts of the play followed in 1927 and 1936 . In 1977 , BBC Radio 4 broadcast the four @-@ act version of the play , with Fabia Drake as Lady Bracknell , Richard Pasco as Jack , Jeremy Clyde as Algy , M","ne 1930 , as the fourth RN ship to carry this name . Boreas was completed on 21 February 1931 at a cost of  221 @,@ 156 , excluding items supplied by the Admiralty such as guns , ammunition and communications equipment . After her commissio"],"return":{}}
```

<details><summary>harness output</summary>

````text
Forward pass 21 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_21/original",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.7642
 },
 "edited": {
  "dir": "work/fwd_21/edited",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 4.9687
 },
 "kl_original_to_edited": {
  "mean": 2.225838,
  "max_text": 3.265543,
  "per_text_file": "work/fwd_21/kl_per_text.npy"
 },
 "meta": "work/fwd_21/meta.json",
 "s": 0.1
}
```
Forward passes left: 29. Replies left: 139.
````

</details>

## Turn 62

The late restoration is not safe as implemented: English KL rose to 2.225838. A larger suppression coefficient alone did not preserve off-task behavior, so I won’t keep this version. I’ll take one diagnostic trace to check whether restoring late outputs corrupts the constant scratch signal used to turn the arithmetic gates off.


```forward
{"type":"teacher_forced","model":"both","texts":["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Branch Division )  in Lock Haven to commemorate historic places . The Water Street","The letter A: introduces an example, not an arithmetic problem.","C:619464298610=619464298610\n","A:260646446633936+291374850318372=4529113079413190\n"],"return":{"sites":["resid_pre.3","resid_mid.3","resid_pre.4","resid_mid.4","mlp.3.hidden","mlp.4.hidden"]}}
```

<details><summary>harness output</summary>

````text
Forward pass 22 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_22/original",
  "arrays": {
   "token_logprobs": [
    4,
    241
   ],
   "resid_pre.3": [
    4,
    241,
    256
   ],
   "resid_mid.3": [
    4,
    241,
    256
   ],
   "resid_pre.4": [
    4,
    241,
    256
   ],
   "resid_mid.4": [
    4,
    241,
    256
   ],
   "mlp.3.hidden": [
    4,
    241,
    768
   ],
   "mlp.4.hidden": [
    4,
    241,
    768
   ],
   "token_ids": [
    4,
    241
   ]
  },
  "mean_bits_per_char": 3.7644
 },
 "edited": {
  "dir": "work/fwd_22/edited",
  "arrays": {
   "token_logprobs": [
    4,
    241
   ],
   "resid_pre.3": [
    4,
    241,
    256
   ],
   "resid_mid.3": [
    4,
    241,
    256
   ],
   "resid_pre.4": [
    4,
    241,
    256
   ],
   "resid_mid.4": [
    4,
    241,
    256
   ],
   "mlp.3.hidden": [
    4,
    241,
    770
   ],
   "mlp.4.hidden": [
    4,
    241,
    791
   ],
   "token_ids": [
    4,
    241
   ]
  },
  "mean_bits_per_char": 6.0152
 },
 "kl_original_to_edited": {
  "mean": 3.775093,
  "max_text": 12.106731,
  "per_text_file": "work/fwd_22/kl_per_text.npy"
 },
 "meta": "work/fwd_22/meta.json",
 "s": 0.1
}
```
Forward passes left: 28. Replies left: 138.
````

</details>

## Turn 63

The trace is ready. I’ll inspect the constant channel and the added MLP activations on the off-task inputs, then restore the known-good tensors. If the constant remains intact, I’ll look for another shared scratch coordinate rather than assume the arithmetic gate is at fault.


```python
# Hypothesis: restoring original late outputs contaminates the scratch constant used by off-task suppression.
for s in ["resid_pre.3","resid_mid.3","resid_pre.4","resid_mid.4"]:
    z=getsite(22,s,"edited")@S.double()
    print(s,[(i,round(float(z[i,:,0].min()),4),round(float(z[i,:,0].max()),4),
                  round(float(z[i,:,9:23].abs().max()),4)) for i in range(3)])
for s in ["mlp.3.hidden","mlp.4.hidden"]:
    x=getsite(22,s,"edited")
    print(s,[(i,float(x[i,:,768:].abs().max())) for i in range(3)])
for name,w in fullcontext_tensors.items():set_tensor(name,w)
print("Restored best_fullcontext tensors.")
with Path("work/notes.md").open("a") as f:
    f.write("Pass21 late-output restoration failed English KL2.225838; rejected. Pass22 diagnostic collected. "
            "Restored full-context-safe backup (KL0.011579; addition60/60,native36/36).\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 28 of at most 300:
```
resid_pre.3 [(0, 1.0, 1.0, 0.0), (1, 0.0, 1.0, 0.0), (2, 0.0, 1.0, 0.0)]
resid_mid.3 [(0, -0.0345, 2.0772, 2.1219), (1, 0.0, 1.9198, 1.7139), (2, 0.0, 2.1705, 1.7783)]
resid_pre.4 [(0, -0.8561, 2.7851, 1034.8409), (1, -0.4077, 2.5199, 2.9323), (2, -0.0764, 1.9808, 2.0523)]
resid_mid.4 [(0, -0.6315, 2.5443, 1034.7777), (1, -0.4128, 2.6923, 3.1165), (2, 0.0, 1.912, 3.3455)]
mlp.3.hidden [(0, 0.3829158842563629), (1, 0.0), (2, 0.0)]
mlp.4.hidden [(0, 72.83343505859375), (1, 46.4224739074707), (2, 0.0)]
Restored best_fullcontext tensors.

```
Forward passes left: 28. Replies left: 137.
````

</details>

## Turn 64

The diagnostic identifies the problem: restored late outputs can drive the scratch constant below zero, reversing the off-task suppression gate and activating the large arithmetic outputs. I’ve restored the working backup. I’ll now try a narrower restoration that keeps only that constant direction protected at the late stages while restoring the other 22 directions, with the stronger gate margin retained.


```python
# Hypothesis: preserving only the late constant channel prevents spurious gates while recovering other original outputs.
Pconstant=torch.eye(256,dtype=torch.double)-Sd[:,0:1]@Sd[:,0:1].T
for l,kind in [(3,"attn.o_proj"),(3,"mlp.down_proj"),(4,"attn.o_proj")]:
    name=f"blocks.{l}.{kind}.weight";w=load(name);orig=load_original(name)
    w[:,:orig.shape[1]]=(Pconstant@orig.double()).float();set_tensor(name,w)
for l,inds in [(3,carryneurons),(4,outneurons)]:
    r,c,b,_=readers(l,"mlp")
    name=f"blocks.{l}.mlp.gate_proj.weight";w=load(name)
    for idx in inds:w[idx]-=G2*(10000-10)*b
    set_tensor(name,w)
print("Late protection reduced from rank23 to rank1 (constant coordinate only).")
with Path("work/notes.md").open("a") as f:
    f.write("Pass22: restored late outputs drove constant negative (-0.856), activating huge arithmetic MLP outputs. "
            "Trying late rank1 protection of constant S0 only, with gate margin10000; all other late original directions restored.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 29 of at most 300:
```
Late protection reduced from rank23 to rank1 (constant coordinate only).

```
Forward passes left: 28. Replies left: 136.
````

</details>

## Turn 65

The late rank-one protection is staged. It directly addresses the measured failure: ordinary late residual activity may return in the other directions, but it cannot change the sign of the arithmetic gate’s constant. I’ll save it and test both English KL and greedy digit behavior before deciding whether to keep it.


```save_edit
{"final":false,"note":"Keep only scratch constant S0 protected in layer-3 original outputs and layer-4 attention, rather than all 23 directions. Retain off-task MLP margin10000. This addresses the diagnosed sign reversal of the constant while restoring other original late output directions."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 14. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_change": 0.211282
  },
  "blocks.1.att
[... cut]
````

</details>

## Turn 66

The rank-one version is saved. I’ll repeat the matched English test to see whether preserving the constant prevents the spurious activation. The result also tells us whether restoring the other late directions actually improves language preservation.


```forward
{"type":"teacher_forced","model":"both","texts":["ociety . The Pennsylvania Historical and Museum Commission has placed three cast aluminum markers  Clinton County , Fort Reed , and Pennsylvania Canal ( West Branch Division )  in Lock Haven to commemorate historic places . The Water Street"," . \" According to Robert Bianco of USA Today , the critical success and popularity of Rachel 's pregnancy is ultimately responsible for \" propel [ ling ] the show to the top of the ratings \" . When it finally came time to write the series f","that \" Aniston was the sparkiest member of the ensemble and the one least reliant on goofball caricature \" , concluding , \" Playing the only character with whom a sane viewer might reasonably identify also meant that she got the lion 's sha","ligious order which cared for the sick , and later looked after pilgrims to the Holy Land . After the success of the First Crusade in capturing Jerusalem in 1099 , many crusaders donated their new property in the Levant to the Hospital of S","nces were accessed by a wall @-@ walk known as a chemin de ronde . In the opinion of historian Hugh Kennedy the defences of the outer wall were \" the most elaborate and developed anywhere in the Latin east ... the whole structure is a brill",". \n = = = Act III = = = \n Morning @-@ Room at the Manor House , Woolton \n Arriving in pursuit of her daughter , Lady Bracknell is astonished to be told that Algernon and Cecily are engaged . The revelation of Cecily 's trust fund soon dispe","arson as Jack Worthing . Further broadcasts of the play followed in 1927 and 1936 . In 1977 , BBC Radio 4 broadcast the four @-@ act version of the play , with Fabia Drake as Lady Bracknell , Richard Pasco as Jack , Jeremy Clyde as Algy , M","ne 1930 , as the fourth RN ship to carry this name . Boreas was completed on 21 February 1931 at a cost of  221 @,@ 156 , excluding items supplied by the Admiralty such as guns , ammunition and communications equipment . After her commissio"],"return":{}}
```

<details><summary>harness output</summary>

````text
Forward pass 23 of 50 done (teacher_forced). Summary:
```
{
 "type": "teacher_forced",
 "model": "both",
 "original": {
  "dir": "work/fwd_23/original",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.7642
 },
 "edited": {
  "dir": "work/fwd_23/edited",
  "arrays": {
   "token_logprobs": [
    8,
    241
   ],
   "token_ids": [
    8,
    241
   ]
  },
  "mean_bits_per_char": 1.9179
 },
 "kl_original_to_edited": {
  "mean": 0.105289,
  "max_text": 0.316062,
  "per_text_file": "work/fwd_23/kl_per_text.npy"
 },
 "meta": "work/fwd_23/meta.json",
 "s": 0.1
}
```
Forward passes left: 27. Replies left: 134.
````

</details>

## Turn 67

The rank-one version still worsens English KL, to 0.105289. Protecting the constant prevents the worst failure, but restoring the other late directions does not improve preservation in this circuit. I’ll reject that change, restore the full-context-safe backup, and use the remaining tests to measure reliability rather than continue modifying the design.


```python
# Hypothesis: the fully protected late circuit is the best measured addition/preservation tradeoff.
for name,w in fullcontext_tensors.items():set_tensor(name,w)
with Path("work/notes.md").open("a") as f:
    f.write("Pass23 rank1 late protection KL0.105289, still worse than0.011579; rejected. "
            "Restored best_fullcontext tensors. Freeze design for broad validation. "
            "Architecture +20 heads,+115 neurons=743680 params; full-context gateM320, T1=1e4,T2=1e6,T3=1e8,Tout=1e12. "
            "Best observed pass20 addition60/60, native36/36; pass19 EnglishKL0.011579 on8x240chars.\n")
print("Restored",len(fullcontext_tensors),"tensors. Added",added_params())
print("Current architecture",arch())
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 30 of at most 300:
```
Restored 54 tensors. Added 743680
Current architecture {'n_heads': [11, 9, 14, 18, 8, 8], 'd_mlp': [770, 775, 849, 770, 791, 768]}

```
Forward passes left: 27. Replies left: 133.
````

</details>

## Turn 68

The best measured tensors are restored: 743,680 added parameters, with the embedding and final norm untouched. This version previously achieved 60/60 addition answers, 36/36 native answers, and English KL of 0.011579 on the matched windows. I’ll submit it as the current final candidate now, then continue validation so a later interruption cannot leave a rejected experiment as the saved result.


```save_edit
{"final":true,"note":"Submit best measured circuit: 20 added attention heads and 115 MLP neurons (743680 parameters), with fixed token/position encoding, BOS+A: prefix gating, delimiter and operand routing, previous-digit carry rule, and piecewise-linear decimal output through the frozen embedding. Original outputs are projected into low-energy scratch-space complements where needed; embedding and final norm are unchanged. Moderate early scales fix quadratic-key normalization error; M=320 positional gate prevents long-text leakage. Evidence: pass20 addition60/60 across N1–20 and native36/36 across tasks and lengths, no paired losses; pass19 English mean KL0.011579 on eight 240-character windows. Later restoration experiments worsened KL and were fully reverted. Generalization is expected from the explicit arithmetic rule, not fitted answer weights. Broader validation will follow."}
```

<details><summary>harness output</summary>

````text
save_edit: stored as edit 15 and submitted as your FINAL edit. It is now the "edited" model for forward passes. Validation:
```
{
 "condition": "add_capacity",
 "arch": {
  "n_heads": [
   11,
   9,
   14,
   18,
   8,
   8
  ],
  "d_mlp": [
   770,
   775,
   849,
   770,
   791,
   768
  ]
 },
 "added_heads": [
  3,
  1,
  6,
  10,
  0,
  0
 ],
 "added_neurons": [
  2,
  7,
  81,
  2,
  23,
  0
 ],
 "added_params": 743680,
 "tensors": {
  "blocks.0.attn.k_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2304,
   "native_changed_entries": 0,
   "delta_norm": 23.122406,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.o_proj.weight": {
   "shape": [
    256,
    352
   ],
   "changed_entries": 67584,
   "native_changed_entries": 65536,
   "delta_norm": 4.224456,
   "native_rel_change": 0.262874
  },
  "blocks.0.attn.q_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 3328,
   "native_changed_entries": 0,
   "delta_norm": 1176.42627,
   "native_rel_change": 0.0
  },
  "blocks.0.attn.v_proj.weight": {
   "shape": [
    352,
    256
   ],
   "changed_entries": 2048,
   "native_changed_entries": 0,
   "delta_norm": 0.704111,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.down_proj.weight": {
   "shape": [
    256,
    770
   ],
   "changed_entries": 197120,
   "native_changed_entries": 196608,
   "delta_norm": 15.347123,
   "native_rel_change": 0.244703
  },
  "blocks.0.mlp.gate_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 2243753.75,
   "native_rel_change": 0.0
  },
  "blocks.0.mlp.up_proj.weight": {
   "shape": [
    770,
    256
   ],
   "changed_entries": 512,
   "native_changed_entries": 0,
   "delta_norm": 1.8e-05,
   "native_rel_change": 0.0
  },
  "blocks.1.attn.o_proj.weight": {
   "shape": [
    256,
    288
   ],
   "changed_entries": 65792,
   "native_changed_entries": 65536,
   "delta_norm": 2.363536,
   "native_rel_chang
[... cut]
````

</details>

## Turn 69

The candidate is submitted. I’ll now test all 50 available 20-digit additions, where accumulated position error is most likely to appear. The weights will remain fixed; this is a reliability check against independent operands within the practice set.


```forward
{"type":"generate","model":"both","ids":["add_n20_000","add_n20_001","add_n20_002","add_n20_003","add_n20_004","add_n20_005","add_n20_006","add_n20_007","add_n20_008","add_n20_009","add_n20_010","add_n20_011","add_n20_012","add_n20_013","add_n20_014","add_n20_015","add_n20_016","add_n20_017","add_n20_018","add_n20_019","add_n20_020","add_n20_021","add_n20_022","add_n20_023","add_n20_024","add_n20_025","add_n20_026","add_n20_027","add_n20_028","add_n20_029","add_n20_030","add_n20_031","add_n20_032","add_n20_033","add_n20_034","add_n20_035","add_n20_036","add_n20_037","add_n20_038","add_n20_039","add_n20_040","add_n20_041","add_n20_042","add_n20_043","add_n20_044","add_n20_045","add_n20_046","add_n20_047","add_n20_048","add_n20_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 24 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "20": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_24_original.jsonl",
  "work/fwd_24_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.8
}
```
Forward passes left: 26. Replies left: 131.
````

</details>

## Turn 70

All 50 20-digit additions passed; the original model got none correct. That supports the positional circuit at the longest evaluated length, though it does not rule out length-specific failures. I’ll test all 50 19-digit cases next, without changing the weights.


```forward
{"type":"generate","model":"both","ids":["add_n19_000","add_n19_001","add_n19_002","add_n19_003","add_n19_004","add_n19_005","add_n19_006","add_n19_007","add_n19_008","add_n19_009","add_n19_010","add_n19_011","add_n19_012","add_n19_013","add_n19_014","add_n19_015","add_n19_016","add_n19_017","add_n19_018","add_n19_019","add_n19_020","add_n19_021","add_n19_022","add_n19_023","add_n19_024","add_n19_025","add_n19_026","add_n19_027","add_n19_028","add_n19_029","add_n19_030","add_n19_031","add_n19_032","add_n19_033","add_n19_034","add_n19_035","add_n19_036","add_n19_037","add_n19_038","add_n19_039","add_n19_040","add_n19_041","add_n19_042","add_n19_043","add_n19_044","add_n19_045","add_n19_046","add_n19_047","add_n19_048","add_n19_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 25 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
    "id": "add_n19_002",
    "expected": "16892574039473403941",
    "got": "16818828085\nM:1,1=1\nM"
   },
   {
    "id": "add_n19_003",
    "expected": "93873615565302100831",
    "got": "91786131028\nM:1,1=1\nM"
   },
   {
    "id": "add_n19_004",
    "expected": "62798705562023810821",
    "got": "80191881820\nM:1,1=1\nM"
   }
  ]
 },
 "edited": {
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "19": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_25_original.jsonl",
  "work/fwd_25_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.6
}
```
Forward passes left: 25. Replies left: 130.
````

</details>

## Turn 71

The 19-digit set also passed 50/50, bringing these two longest lengths to 100/100. I’ll continue with 18 digits because delimiter positions change with length and earlier failures were length-specific. No edit is being made.


```forward
{"type":"generate","model":"both","ids":["add_n18_000","add_n18_001","add_n18_002","add_n18_003","add_n18_004","add_n18_005","add_n18_006","add_n18_007","add_n18_008","add_n18_009","add_n18_010","add_n18_011","add_n18_012","add_n18_013","add_n18_014","add_n18_015","add_n18_016","add_n18_017","add_n18_018","add_n18_019","add_n18_020","add_n18_021","add_n18_022","add_n18_023","add_n18_024","add_n18_025","add_n18_026","add_n18_027","add_n18_028","add_n18_029","add_n18_030","add_n18_031","add_n18_032","add_n18_033","add_n18_034","add_n18_035","add_n18_036","add_n18_037","add_n18_038","add_n18_039","add_n18_040","add_n18_041","add_n18_042","add_n18_043","add_n18_044","add_n18_045","add_n18_046","add_n18_047","add_n18_048","add_n18_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 26 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "18": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_26_original.jsonl",
  "work/fwd_26_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.5
}
```
Forward passes left: 24. Replies left: 129.
````

</details>

## Turn 72

All 50 18-digit cases passed as well. I’ll test the complete 17-digit set next, continuing the fixed-weight sweep. A failure would provide a concrete routing or carry boundary to inspect rather than a reason to adjust weights speculatively.


```forward
{"type":"generate","model":"both","ids":["add_n17_000","add_n17_001","add_n17_002","add_n17_003","add_n17_004","add_n17_005","add_n17_006","add_n17_007","add_n17_008","add_n17_009","add_n17_010","add_n17_011","add_n17_012","add_n17_013","add_n17_014","add_n17_015","add_n17_016","add_n17_017","add_n17_018","add_n17_019","add_n17_020","add_n17_021","add_n17_022","add_n17_023","add_n17_024","add_n17_025","add_n17_026","add_n17_027","add_n17_028","add_n17_029","add_n17_030","add_n17_031","add_n17_032","add_n17_033","add_n17_034","add_n17_035","add_n17_036","add_n17_037","add_n17_038","add_n17_039","add_n17_040","add_n17_041","add_n17_042","add_n17_043","add_n17_044","add_n17_045","add_n17_046","add_n17_047","add_n17_048","add_n17_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 27 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "17": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_27_original.jsonl",
  "work/fwd_27_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.4
}
```
Forward passes left: 23. Replies left: 128.
````

</details>

## Turn 73

The 17-digit set passed 50/50, so the frozen candidate is now correct on all 200 practice cases at lengths 17–20. I’ll record this checkpoint before moving into the 13–16-digit range, where an earlier circuit version had delimiter-routing failures. The submitted weights remain unchanged.


```python
# Hypothesis: the fixed circuit remains reliable across all long-length practice cases.
with Path("work/notes.md").open("a") as f:
    f.write("Submitted restored best as edit15 FINAL. Frozen sweep passes24–27: N20,N19,N18,N17 each50/50; "
            "original each0/50. Current fixed-candidate long addition total200/200. "
            "Plan remaining passes: all remaining addition lengths, all720 native cases, fresh English KL.\n")
print("Frozen sweep: N17–20, 200/200 edited versus 0/200 original. No weight changes.")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 31 of at most 300:
```
Frozen sweep: N17–20, 200/200 edited versus 0/200 original. No weight changes.

```
Forward passes left: 23. Replies left: 127.
````

</details>

## Turn 74

The checkpoint is recorded. I’ll now test all 50 16-digit additions, including examples from the range that exposed the earlier BOS-fallback defect. Passing this set would strengthen the evidence that the analytical routing correction holds across operands.


```forward
{"type":"generate","model":"both","ids":["add_n16_000","add_n16_001","add_n16_002","add_n16_003","add_n16_004","add_n16_005","add_n16_006","add_n16_007","add_n16_008","add_n16_009","add_n16_010","add_n16_011","add_n16_012","add_n16_013","add_n16_014","add_n16_015","add_n16_016","add_n16_017","add_n16_018","add_n16_019","add_n16_020","add_n16_021","add_n16_022","add_n16_023","add_n16_024","add_n16_025","add_n16_026","add_n16_027","add_n16_028","add_n16_029","add_n16_030","add_n16_031","add_n16_032","add_n16_033","add_n16_034","add_n16_035","add_n16_036","add_n16_037","add_n16_038","add_n16_039","add_n16_040","add_n16_041","add_n16_042","add_n16_043","add_n16_044","add_n16_045","add_n16_046","add_n16_047","add_n16_048","add_n16_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 28 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "16": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_28_original.jsonl",
  "work/fwd_28_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 2.2
}
```
Forward passes left: 22. Replies left: 126.
````

</details>

## Turn 75

All 50 16-digit cases passed. Next I’ll test the full 15-digit set, which includes the final-digit failure diagnosed at the start of this continuation. This checks the scale correction on that case and on the other operands at the same length.


```forward
{"type":"generate","model":"both","ids":["add_n15_000","add_n15_001","add_n15_002","add_n15_003","add_n15_004","add_n15_005","add_n15_006","add_n15_007","add_n15_008","add_n15_009","add_n15_010","add_n15_011","add_n15_012","add_n15_013","add_n15_014","add_n15_015","add_n15_016","add_n15_017","add_n15_018","add_n15_019","add_n15_020","add_n15_021","add_n15_022","add_n15_023","add_n15_024","add_n15_025","add_n15_026","add_n15_027","add_n15_028","add_n15_029","add_n15_030","add_n15_031","add_n15_032","add_n15_033","add_n15_034","add_n15_035","add_n15_036","add_n15_037","add_n15_038","add_n15_039","add_n15_040","add_n15_041","add_n15_042","add_n15_043","add_n15_044","add_n15_045","add_n15_046","add_n15_047","add_n15_048","add_n15_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 29 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "15": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_29_original.jsonl",
  "work/fwd_29_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.9
}
```
Forward passes left: 21. Replies left: 125.
````

</details>

## Turn 76

The complete 15-digit set passed, including the formerly failing final carry case. I’ll continue with all 50 14-digit problems. The fixed candidate has now passed 300/300 cases at lengths 15–20, but shorter delimiter layouts still need coverage.


```forward
{"type":"generate","model":"both","ids":["add_n14_000","add_n14_001","add_n14_002","add_n14_003","add_n14_004","add_n14_005","add_n14_006","add_n14_007","add_n14_008","add_n14_009","add_n14_010","add_n14_011","add_n14_012","add_n14_013","add_n14_014","add_n14_015","add_n14_016","add_n14_017","add_n14_018","add_n14_019","add_n14_020","add_n14_021","add_n14_022","add_n14_023","add_n14_024","add_n14_025","add_n14_026","add_n14_027","add_n14_028","add_n14_029","add_n14_030","add_n14_031","add_n14_032","add_n14_033","add_n14_034","add_n14_035","add_n14_036","add_n14_037","add_n14_038","add_n14_039","add_n14_040","add_n14_041","add_n14_042","add_n14_043","add_n14_044","add_n14_045","add_n14_046","add_n14_047","add_n14_048","add_n14_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 30 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "14": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_30_original.jsonl",
  "work/fwd_30_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.8
}
```
Forward passes left: 20. Replies left: 124.
````

</details>

## Turn 77

All 50 14-digit cases passed. I’ll test 13 digits next to finish the range that previously exposed first-digit routing failures. Continued success here would support both the BOS-fallback fix and the later normalization fix under the frozen weights.


```forward
{"type":"generate","model":"both","ids":["add_n13_000","add_n13_001","add_n13_002","add_n13_003","add_n13_004","add_n13_005","add_n13_006","add_n13_007","add_n13_008","add_n13_009","add_n13_010","add_n13_011","add_n13_012","add_n13_013","add_n13_014","add_n13_015","add_n13_016","add_n13_017","add_n13_018","add_n13_019","add_n13_020","add_n13_021","add_n13_022","add_n13_023","add_n13_024","add_n13_025","add_n13_026","add_n13_027","add_n13_028","add_n13_029","add_n13_030","add_n13_031","add_n13_032","add_n13_033","add_n13_034","add_n13_035","add_n13_036","add_n13_037","add_n13_038","add_n13_039","add_n13_040","add_n13_041","add_n13_042","add_n13_043","add_n13_044","add_n13_045","add_n13_046","add_n13_047","add_n13_048","add_n13_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 31 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "13": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_31_original.jsonl",
  "work/fwd_31_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.7
}
```
Forward passes left: 19. Replies left: 123.
````

</details>

## Turn 78

The 13-digit set passed 50/50, completing 400/400 additions at lengths 13–20. I’ll record the result and confirm the staged tensors still match the submitted backup. Then I’ll continue the length sweep and reserve seven passes for native-task and English checks.


```python
# Hypothesis: the submitted candidate is unchanged throughout the successful long-length validation.
unchanged=all(torch.equal(load(name),w) for name,w in fullcontext_tensors.items())
with Path("work/notes.md").open("a") as f:
    f.write("Frozen sweep passes28–31: N16,N15,N14,N13 each50/50, original0/50. "
            "Total N13–20 =400/400. Staged tensors still equal submitted backup. "
            "19 passes left: twelve addition lengths, six120-case native batches, one fresh English test.\n")
print("Equal to frozen backup:",unchanged,"; N13–20 addition 400/400.")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 32 of at most 300:
```
Equal to frozen backup: True ; N13–20 addition 400/400.

```
Forward passes left: 19. Replies left: 122.
````

</details>

## Turn 79

The tensor check confirms that validation has used the unchanged submitted candidate. I’ll now test all 50 12-digit additions, at the upper length limit of the model’s original digit tasks. This helps distinguish the new arithmetic capability from merely reproducing familiar-length digit patterns.


```forward
{"type":"generate","model":"both","ids":["add_n12_000","add_n12_001","add_n12_002","add_n12_003","add_n12_004","add_n12_005","add_n12_006","add_n12_007","add_n12_008","add_n12_009","add_n12_010","add_n12_011","add_n12_012","add_n12_013","add_n12_014","add_n12_015","add_n12_016","add_n12_017","add_n12_018","add_n12_019","add_n12_020","add_n12_021","add_n12_022","add_n12_023","add_n12_024","add_n12_025","add_n12_026","add_n12_027","add_n12_028","add_n12_029","add_n12_030","add_n12_031","add_n12_032","add_n12_033","add_n12_034","add_n12_035","add_n12_036","add_n12_037","add_n12_038","add_n12_039","add_n12_040","add_n12_041","add_n12_042","add_n12_043","add_n12_044","add_n12_045","add_n12_046","add_n12_047","add_n12_048","add_n12_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 32 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "12": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_32_original.jsonl",
  "work/fwd_32_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 1.3
}
```
Forward passes left: 18. Replies left: 121.
````

</details>

## Turn 80

All 50 12-digit additions passed, while the original model got none correct. I’ll continue with the complete 11-digit set. The purpose remains coverage of each delimiter layout under fixed weights, not parameter selection.


```forward
{"type":"generate","model":"both","ids":["add_n11_000","add_n11_001","add_n11_002","add_n11_003","add_n11_004","add_n11_005","add_n11_006","add_n11_007","add_n11_008","add_n11_009","add_n11_010","add_n11_011","add_n11_012","add_n11_013","add_n11_014","add_n11_015","add_n11_016","add_n11_017","add_n11_018","add_n11_019","add_n11_020","add_n11_021","add_n11_022","add_n11_023","add_n11_024","add_n11_025","add_n11_026","add_n11_027","add_n11_028","add_n11_029","add_n11_030","add_n11_031","add_n11_032","add_n11_033","add_n11_034","add_n11_035","add_n11_036","add_n11_037","add_n11_038","add_n11_039","add_n11_040","add_n11_041","add_n11_042","add_n11_043","add_n11_044","add_n11_045","add_n11_046","add_n11_047","add_n11_048","add_n11_049"]}
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

## Turn 81

The fixed edit solved all 50 eleven-digit problems in the latest batch, versus none for the original model. I will keep the weights unchanged while completing validation. First I’ll check the saved results and practice identifiers so the remaining tests cover every case without duplication.

```python
# Hypothesis: the frozen edit's saved results cover N=11..20, leaving shorter addition and comprehensive off-task checks.
import json, pathlib
print("Architecture:", arch(), "Added parameters:", added_params())
print("Practice examples:")
for fn in ["practice/addition.jsonl", "practice/native.jsonl"]:
    rows = [json.loads(s) for s in pathlib.Path(fn).read_text().splitlines()]
    print(fn, len(rows), rows[0], rows[-1])
for k in [32,33]:
    p = pathlib.Path(f"work/fwd_{k}_edited.jsonl")
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    print("Pass", k, "count", len(rows), "first result", rows[0])
print("Recent code:")
print(pathlib.Path("work/cells.py").read_text()[-12000:])
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 33 of at most 300:
```
Architecture: {'n_heads': [11, 9, 14, 18, 8, 8], 'd_mlp': [770, 775, 849, 770, 791, 768]} Added parameters: 743680
Practice examples:
practice/addition.jsonl 1000 {'id': 'add_n01_000', 'task': 'addition', 'n_digits': 1, 'prompt': 'A:2+2=', 'answer': '40'} {'id': 'add_n20_049', 'task': 'addition', 'n_digits': 20, 'prompt': 'A:28712017392250341912+51934285797474064420=', 'answer': '796562921907253063420'}
practice/native.jsonl 720 {'id': 'copy_n01_000', 'task': 'copy', 'n_digits': 1, 'prompt': 'C:9=', 'answer': '9'} {'id': 'max_n12_019', 'task': 'max', 'n_digits': 12, 'prompt': 'M:87021,577252062929=', 'answer': '577252062929'}
Pass 32 count 50 first result {'id': 'add_n12_000', 'task': 'addition', 'n_digits': 12, 'prompt': 'A:764003658807+457514149431=', 'answer': '1226177973480', 'output': '1226177973480\n', 'output_ids': [21, 22, 22, 26, 21, 27, 27, 29, 27, 23, 24, 28, 20, 3], 'correct': True}
Pass 33 count 50 first result {'id': 'add_n11_000', 'task': 'addition', 'n_digits': 11, 'prompt': 'A:98169367170+65112197935=', 'answer': '543715551160', 'output': '543715551160\n', 'output_ids': [25, 24, 23, 27, 21, 25, 25, 25, 21, 21, 26, 20, 3], 'correct': True}
Recent code:
 if x["id"]=="add_n20_008")
print("BAD",bad)
print(cells[cells.index("# ---- python run 17"):cells.index("# ---- python run 18")])
print(json.dumps({"type":"teacher_forced","model":"both",
"texts":[r["prompt"]+r["answer"]+"\n" for r in [bad,good]]+
["C:619464298610=619464298610\n","R:123456789=987654321\n",
"The letter A: introduces an example, not an arithmetic problem."],
"return":{"sites":["resid_pre.2","resid_pre.3","resid_pre.4","attn.3.pattern","logits"]}}))

# ---- python run 21 (reply 45) ----
# Hypothesis: final-position operand attention, rather than the carry threshold, explains the wrong 2.
zz15={s:getsite(15,s,"edited")@S.double() for s in ["resid_pre.2","resid_pre.3","resid_pre.4"]}
pat15=np.load("work/fwd_15/edited/attn.3.pattern.npy")
log15=np
[... cut]
````

</details>

## Turn 82

The saved files confirm complete coverage of N=11–20, and the edit uses 743,680 added parameters, within the cap. Next I’ll test all 50 ten-digit additions with both models. Another clean batch would support generalization to shorter inputs; any failure would identify a boundary worth examining.

```forward
{"type":"generate","model":"both","ids":["add_n10_000","add_n10_001","add_n10_002","add_n10_003","add_n10_004","add_n10_005","add_n10_006","add_n10_007","add_n10_008","add_n10_009","add_n10_010","add_n10_011","add_n10_012","add_n10_013","add_n10_014","add_n10_015","add_n10_016","add_n10_017","add_n10_018","add_n10_019","add_n10_020","add_n10_021","add_n10_022","add_n10_023","add_n10_024","add_n10_025","add_n10_026","add_n10_027","add_n10_028","add_n10_029","add_n10_030","add_n10_031","add_n10_032","add_n10_033","add_n10_034","add_n10_035","add_n10_036","add_n10_037","add_n10_038","add_n10_039","add_n10_040","add_n10_041","add_n10_042","add_n10_043","add_n10_044","add_n10_045","add_n10_046","add_n10_047","add_n10_048","add_n10_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 34 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "10": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_34_original.jsonl",
  "work/fwd_34_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.9
}
```
Forward passes left: 16. Replies left: 118.
````

</details>

## Turn 83

All 50 ten-digit additions were correct after editing, compared with none before. I’ll now test N=9, completing the longer-length group used in the report. The weights remain frozen, so this is validation rather than further fitting.

```forward
{"type":"generate","model":"both","ids":["add_n09_000","add_n09_001","add_n09_002","add_n09_003","add_n09_004","add_n09_005","add_n09_006","add_n09_007","add_n09_008","add_n09_009","add_n09_010","add_n09_011","add_n09_012","add_n09_013","add_n09_014","add_n09_015","add_n09_016","add_n09_017","add_n09_018","add_n09_019","add_n09_020","add_n09_021","add_n09_022","add_n09_023","add_n09_024","add_n09_025","add_n09_026","add_n09_027","add_n09_028","add_n09_029","add_n09_030","add_n09_031","add_n09_032","add_n09_033","add_n09_034","add_n09_035","add_n09_036","add_n09_037","add_n09_038","add_n09_039","add_n09_040","add_n09_041","add_n09_042","add_n09_043","add_n09_044","add_n09_045","add_n09_046","add_n09_047","add_n09_048","add_n09_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 35 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "9": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_35_original.jsonl",
  "work/fwd_35_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.8
}
```
Forward passes left: 15. Replies left: 117.
````

</details>

## Turn 84

The frozen edit now has 600/600 correct practice additions across N=9–20. I’ll move to N=8 and continue checking shorter lengths individually. Success here would show that the same positional routing also works below the longer-length reporting boundary.

```forward
{"type":"generate","model":"both","ids":["add_n08_000","add_n08_001","add_n08_002","add_n08_003","add_n08_004","add_n08_005","add_n08_006","add_n08_007","add_n08_008","add_n08_009","add_n08_010","add_n08_011","add_n08_012","add_n08_013","add_n08_014","add_n08_015","add_n08_016","add_n08_017","add_n08_018","add_n08_019","add_n08_020","add_n08_021","add_n08_022","add_n08_023","add_n08_024","add_n08_025","add_n08_026","add_n08_027","add_n08_028","add_n08_029","add_n08_030","add_n08_031","add_n08_032","add_n08_033","add_n08_034","add_n08_035","add_n08_036","add_n08_037","add_n08_038","add_n08_039","add_n08_040","add_n08_041","add_n08_042","add_n08_043","add_n08_044","add_n08_045","add_n08_046","add_n08_047","add_n08_048","add_n08_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 36 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "8": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_36_original.jsonl",
  "work/fwd_36_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.6
}
```
Forward passes left: 14. Replies left: 116.
````

</details>

## Turn 85

N=8 also passed all 50 cases, with the original model failing all of them. Next I’ll test N=7 without changing any weights. A failure would narrow the issue to a shorter-input routing or arithmetic boundary rather than long-context behavior.

```forward
{"type":"generate","model":"both","ids":["add_n07_000","add_n07_001","add_n07_002","add_n07_003","add_n07_004","add_n07_005","add_n07_006","add_n07_007","add_n07_008","add_n07_009","add_n07_010","add_n07_011","add_n07_012","add_n07_013","add_n07_014","add_n07_015","add_n07_016","add_n07_017","add_n07_018","add_n07_019","add_n07_020","add_n07_021","add_n07_022","add_n07_023","add_n07_024","add_n07_025","add_n07_026","add_n07_027","add_n07_028","add_n07_029","add_n07_030","add_n07_031","add_n07_032","add_n07_033","add_n07_034","add_n07_035","add_n07_036","add_n07_037","add_n07_038","add_n07_039","add_n07_040","add_n07_041","add_n07_042","add_n07_043","add_n07_044","add_n07_045","add_n07_046","add_n07_047","add_n07_048","add_n07_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 37 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
  },
  "addition_accuracy_by_n_digits": {
   "7": 0.0
  },
  "failures": [
   {
    "id": "add_n07_000",
    "expected": "78663921",
    "got": "286845855"
   },
   {
    "id": "add_n07_001",
    "expected": "39592111",
    "got": "240735311"
   },
   {
    "id": "add_n07_002",
    "expected": "65728011",
    "got": "030037266"
   },
   {
    "id": "add_n07_003",
    "expected": "37241460",
    "got": "125880122"
   },
   {
    "id": "add_n07_004",
    "expected": "00378460",
    "got": "676355044"
   }
  ]
 },
 "edited": {
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "7": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_37_original.jsonl",
  "work/fwd_37_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.5
}
```
Forward passes left: 13. Replies left: 115.
````

</details>

## Turn 86

The seven-digit batch was again 50/50 for the edit and 0/50 for the original. I’ll test six-digit additions next. This continues the complete practice sweep while leaving seven passes reserved for native tasks and English divergence.

```forward
{"type":"generate","model":"both","ids":["add_n06_000","add_n06_001","add_n06_002","add_n06_003","add_n06_004","add_n06_005","add_n06_006","add_n06_007","add_n06_008","add_n06_009","add_n06_010","add_n06_011","add_n06_012","add_n06_013","add_n06_014","add_n06_015","add_n06_016","add_n06_017","add_n06_018","add_n06_019","add_n06_020","add_n06_021","add_n06_022","add_n06_023","add_n06_024","add_n06_025","add_n06_026","add_n06_027","add_n06_028","add_n06_029","add_n06_030","add_n06_031","add_n06_032","add_n06_033","add_n06_034","add_n06_035","add_n06_036","add_n06_037","add_n06_038","add_n06_039","add_n06_040","add_n06_041","add_n06_042","add_n06_043","add_n06_044","add_n06_045","add_n06_046","add_n06_047","add_n06_048","add_n06_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 38 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "6": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_38_original.jsonl",
  "work/fwd_38_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.4
}
```
Forward passes left: 12. Replies left: 114.
````

</details>

## Turn 87

All six-digit cases passed, bringing the completed sweep to 750/750 additions across N=6–20. I’ll now test N=5 on the same paired basis. The expected result is unchanged accuracy despite the more closely spaced delimiters.

```forward
{"type":"generate","model":"both","ids":["add_n05_000","add_n05_001","add_n05_002","add_n05_003","add_n05_004","add_n05_005","add_n05_006","add_n05_007","add_n05_008","add_n05_009","add_n05_010","add_n05_011","add_n05_012","add_n05_013","add_n05_014","add_n05_015","add_n05_016","add_n05_017","add_n05_018","add_n05_019","add_n05_020","add_n05_021","add_n05_022","add_n05_023","add_n05_024","add_n05_025","add_n05_026","add_n05_027","add_n05_028","add_n05_029","add_n05_030","add_n05_031","add_n05_032","add_n05_033","add_n05_034","add_n05_035","add_n05_036","add_n05_037","add_n05_038","add_n05_039","add_n05_040","add_n05_041","add_n05_042","add_n05_043","add_n05_044","add_n05_045","add_n05_046","add_n05_047","add_n05_048","add_n05_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 39 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "5": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_39_original.jsonl",
  "work/fwd_39_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.3
}
```
Forward passes left: 11. Replies left: 113.
````

</details>

## Turn 88

The five-digit batch also passed completely. I’ll test N=4 next, keeping both the edit and evaluation procedure unchanged. This will check another short-input boundary before the final three lengths.

```forward
{"type":"generate","model":"both","ids":["add_n04_000","add_n04_001","add_n04_002","add_n04_003","add_n04_004","add_n04_005","add_n04_006","add_n04_007","add_n04_008","add_n04_009","add_n04_010","add_n04_011","add_n04_012","add_n04_013","add_n04_014","add_n04_015","add_n04_016","add_n04_017","add_n04_018","add_n04_019","add_n04_020","add_n04_021","add_n04_022","add_n04_023","add_n04_024","add_n04_025","add_n04_026","add_n04_027","add_n04_028","add_n04_029","add_n04_030","add_n04_031","add_n04_032","add_n04_033","add_n04_034","add_n04_035","add_n04_036","add_n04_037","add_n04_038","add_n04_039","add_n04_040","add_n04_041","add_n04_042","add_n04_043","add_n04_044","add_n04_045","add_n04_046","add_n04_047","add_n04_048","add_n04_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 40 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "4": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_40_original.jsonl",
  "work/fwd_40_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.3
}
```
Forward passes left: 10. Replies left: 112.
````

</details>

## Turn 89

Four-digit accuracy was 50/50 after editing and 0/50 before. Next are the 50 three-digit cases. Continued success would leave only the one- and two-digit boundary cases untested in the full sweep.

```forward
{"type":"generate","model":"both","ids":["add_n03_000","add_n03_001","add_n03_002","add_n03_003","add_n03_004","add_n03_005","add_n03_006","add_n03_007","add_n03_008","add_n03_009","add_n03_010","add_n03_011","add_n03_012","add_n03_013","add_n03_014","add_n03_015","add_n03_016","add_n03_017","add_n03_018","add_n03_019","add_n03_020","add_n03_021","add_n03_022","add_n03_023","add_n03_024","add_n03_025","add_n03_026","add_n03_027","add_n03_028","add_n03_029","add_n03_030","add_n03_031","add_n03_032","add_n03_033","add_n03_034","add_n03_035","add_n03_036","add_n03_037","add_n03_038","add_n03_039","add_n03_040","add_n03_041","add_n03_042","add_n03_043","add_n03_044","add_n03_045","add_n03_046","add_n03_047","add_n03_048","add_n03_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 41 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "3": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_41_original.jsonl",
  "work/fwd_41_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.2
}
```
Forward passes left: 9. Replies left: 111.
````

</details>

## Turn 90

The three-digit batch passed all 50 cases. I’ll now check two-digit additions, where the answer starts much closer to the prefix and separators. A clean result would support the intended length-independent routing at this short boundary.

```forward
{"type":"generate","model":"both","ids":["add_n02_000","add_n02_001","add_n02_002","add_n02_003","add_n02_004","add_n02_005","add_n02_006","add_n02_007","add_n02_008","add_n02_009","add_n02_010","add_n02_011","add_n02_012","add_n02_013","add_n02_014","add_n02_015","add_n02_016","add_n02_017","add_n02_018","add_n02_019","add_n02_020","add_n02_021","add_n02_022","add_n02_023","add_n02_024","add_n02_025","add_n02_026","add_n02_027","add_n02_028","add_n02_029","add_n02_030","add_n02_031","add_n02_032","add_n02_033","add_n02_034","add_n02_035","add_n02_036","add_n02_037","add_n02_038","add_n02_039","add_n02_040","add_n02_041","add_n02_042","add_n02_043","add_n02_044","add_n02_045","add_n02_046","add_n02_047","add_n02_048","add_n02_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 42 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.0,
  "accuracy_by_task": {
   "addition": 0.0
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "2": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_42_original.jsonl",
  "work/fwd_42_edited.jsonl"
 ],
 "paired": {
  "rescues": 50,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.1
}
```
Forward passes left: 8. Replies left: 110.
````

</details>

## Turn 91

Two-digit accuracy was also perfect on the 50 practice cases. The next batch completes addition validation with N=1, including the final carry digit. Afterward, all remaining passes will measure preservation of native tasks and English behavior.

```forward
{"type":"generate","model":"both","ids":["add_n01_000","add_n01_001","add_n01_002","add_n01_003","add_n01_004","add_n01_005","add_n01_006","add_n01_007","add_n01_008","add_n01_009","add_n01_010","add_n01_011","add_n01_012","add_n01_013","add_n01_014","add_n01_015","add_n01_016","add_n01_017","add_n01_018","add_n01_019","add_n01_020","add_n01_021","add_n01_022","add_n01_023","add_n01_024","add_n01_025","add_n01_026","add_n01_027","add_n01_028","add_n01_029","add_n01_030","add_n01_031","add_n01_032","add_n01_033","add_n01_034","add_n01_035","add_n01_036","add_n01_037","add_n01_038","add_n01_039","add_n01_040","add_n01_041","add_n01_042","add_n01_043","add_n01_044","add_n01_045","add_n01_046","add_n01_047","add_n01_048","add_n01_049"]}
```

<details><summary>harness output</summary>

````text
Forward pass 43 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 50,
 "original": {
  "graded": 50,
  "accuracy": 0.12,
  "accuracy_by_task": {
   "addition": 0.12
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
  "graded": 50,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "addition": 1.0
  },
  "addition_accuracy_by_n_digits": {
   "1": 1.0
  },
  "failures": []
 },
 "files": [
  "work/fwd_43_original.jsonl",
  "work/fwd_43_edited.jsonl"
 ],
 "paired": {
  "rescues": 44,
  "losses": 0,
  "outputs_changed": 50
 },
 "s": 0.1
}
```
Forward passes left: 7. Replies left: 109.
````

</details>

## Turn 92

The complete addition sweep is 1,000/1,000 for the edit, versus 6/1,000 for the original. This is practice-set evidence, not a guarantee on unseen problems. I’ll record the totals and prepare six native-task batches covering all 720 cases; unchanged outputs would provide stronger preservation evidence than the earlier small checks.

```python
# Hypothesis: the frozen circuit preserves native tasks across all lengths, not just the earlier small samples.
from collections import defaultdict
addition_totals = defaultdict(lambda: [0,0,0])
for k in range(24,44):
    orig = [json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_original.jsonl").read_text().splitlines()]
    edit = [json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_edited.jsonl").read_text().splitlines()]
    for a,b in zip(orig,edit):
        assert a["id"] == b["id"]
        t = addition_totals[a["n_digits"]]
        t[0] += int(a["correct"]); t[1] += int(b["correct"]); t[2] += 1
print("Addition by N: original correct, edited correct, total")
print(dict(sorted(addition_totals.items())))
native_rows = [json.loads(s) for s in pathlib.Path("practice/native.jsonl").read_text().splitlines()]
print("Native task order:", list(dict.fromkeys(r["task"] for r in native_rows)))
for start in range(0,720,120):
    batch = native_rows[start:start+120]
    print("Batch",start//120+1,"first",batch[0]["id"],"last",batch[-1]["id"])
with pathlib.Path("work/notes.md").open("a") as f:
    f.write("\nFrozen sweep passes24–43 complete: edited addition1000/1000, original6/1000. "
            "N1–8 edited400/400 vs6/400; N9–20 edited600/600 vs0/600. "
            "Every N has50/50 edited. No weights changed. Remaining7 passes: all720 native in6 batches, then English KL.\n")
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 34 of at most 300:
```
Addition by N: original correct, edited correct, total
{1: [6, 50, 50], 2: [0, 50, 50], 3: [0, 50, 50], 4: [0, 50, 50], 5: [0, 50, 50], 6: [0, 50, 50], 7: [0, 50, 50], 8: [0, 50, 50], 9: [0, 50, 50], 10: [0, 50, 50], 11: [0, 50, 50], 12: [0, 50, 50], 13: [0, 50, 50], 14: [0, 50, 50], 15: [0, 50, 50], 16: [0, 50, 50], 17: [0, 50, 50], 18: [0, 50, 50], 19: [0, 50, 50], 20: [0, 50, 50]}
Native task order: ['copy', 'reverse', 'max']
Batch 1 first copy_n01_000 last copy_n06_019
Batch 2 first copy_n07_000 last copy_n12_019
Batch 3 first reverse_n01_000 last reverse_n06_019
Batch 4 first reverse_n07_000 last reverse_n12_019
Batch 5 first max_n01_000 last max_n06_019
Batch 6 first max_n07_000 last max_n12_019

```
Forward passes left: 7. Replies left: 108.
````

</details>

## Turn 93

The aggregate files confirm 100% addition accuracy at every tested length. I’ll begin off-task validation with all copy cases of lengths 1–6. Both models will receive identical inputs, and exact matches include the terminating newline.

```forward
{"type":"generate","model":"both","ids":["copy_n01_000","copy_n01_001","copy_n01_002","copy_n01_003","copy_n01_004","copy_n01_005","copy_n01_006","copy_n01_007","copy_n01_008","copy_n01_009","copy_n01_010","copy_n01_011","copy_n01_012","copy_n01_013","copy_n01_014","copy_n01_015","copy_n01_016","copy_n01_017","copy_n01_018","copy_n01_019","copy_n02_000","copy_n02_001","copy_n02_002","copy_n02_003","copy_n02_004","copy_n02_005","copy_n02_006","copy_n02_007","copy_n02_008","copy_n02_009","copy_n02_010","copy_n02_011","copy_n02_012","copy_n02_013","copy_n02_014","copy_n02_015","copy_n02_016","copy_n02_017","copy_n02_018","copy_n02_019","copy_n03_000","copy_n03_001","copy_n03_002","copy_n03_003","copy_n03_004","copy_n03_005","copy_n03_006","copy_n03_007","copy_n03_008","copy_n03_009","copy_n03_010","copy_n03_011","copy_n03_012","copy_n03_013","copy_n03_014","copy_n03_015","copy_n03_016","copy_n03_017","copy_n03_018","copy_n03_019","copy_n04_000","copy_n04_001","copy_n04_002","copy_n04_003","copy_n04_004","copy_n04_005","copy_n04_006","copy_n04_007","copy_n04_008","copy_n04_009","copy_n04_010","copy_n04_011","copy_n04_012","copy_n04_013","copy_n04_014","copy_n04_015","copy_n04_016","copy_n04_017","copy_n04_018","copy_n04_019","copy_n05_000","copy_n05_001","copy_n05_002","copy_n05_003","copy_n05_004","copy_n05_005","copy_n05_006","copy_n05_007","copy_n05_008","copy_n05_009","copy_n05_010","copy_n05_011","copy_n05_012","copy_n05_013","copy_n05_014","copy_n05_015","copy_n05_016","copy_n05_017","copy_n05_018","copy_n05_019","copy_n06_000","copy_n06_001","copy_n06_002","copy_n06_003","copy_n06_004","copy_n06_005","copy_n06_006","copy_n06_007","copy_n06_008","copy_n06_009","copy_n06_010","copy_n06_011","copy_n06_012","copy_n06_013","copy_n06_014","copy_n06_015","copy_n06_016","copy_n06_017","copy_n06_018","copy_n06_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 44 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 120,
 "original": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "copy": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "edited": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "copy": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "files": [
  "work/fwd_44_original.jsonl",
  "work/fwd_44_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 0
 },
 "s": 0.5
}
```
Forward passes left: 6. Replies left: 107.
````

</details>

## Turn 94

All 120 short copy cases retained exactly the same correct outputs. Next I’ll test copy lengths 7–12. This checks whether longer native sequences remain unaffected by the added position-processing circuitry.

```forward
{"type":"generate","model":"both","ids":["copy_n07_000","copy_n07_001","copy_n07_002","copy_n07_003","copy_n07_004","copy_n07_005","copy_n07_006","copy_n07_007","copy_n07_008","copy_n07_009","copy_n07_010","copy_n07_011","copy_n07_012","copy_n07_013","copy_n07_014","copy_n07_015","copy_n07_016","copy_n07_017","copy_n07_018","copy_n07_019","copy_n08_000","copy_n08_001","copy_n08_002","copy_n08_003","copy_n08_004","copy_n08_005","copy_n08_006","copy_n08_007","copy_n08_008","copy_n08_009","copy_n08_010","copy_n08_011","copy_n08_012","copy_n08_013","copy_n08_014","copy_n08_015","copy_n08_016","copy_n08_017","copy_n08_018","copy_n08_019","copy_n09_000","copy_n09_001","copy_n09_002","copy_n09_003","copy_n09_004","copy_n09_005","copy_n09_006","copy_n09_007","copy_n09_008","copy_n09_009","copy_n09_010","copy_n09_011","copy_n09_012","copy_n09_013","copy_n09_014","copy_n09_015","copy_n09_016","copy_n09_017","copy_n09_018","copy_n09_019","copy_n10_000","copy_n10_001","copy_n10_002","copy_n10_003","copy_n10_004","copy_n10_005","copy_n10_006","copy_n10_007","copy_n10_008","copy_n10_009","copy_n10_010","copy_n10_011","copy_n10_012","copy_n10_013","copy_n10_014","copy_n10_015","copy_n10_016","copy_n10_017","copy_n10_018","copy_n10_019","copy_n11_000","copy_n11_001","copy_n11_002","copy_n11_003","copy_n11_004","copy_n11_005","copy_n11_006","copy_n11_007","copy_n11_008","copy_n11_009","copy_n11_010","copy_n11_011","copy_n11_012","copy_n11_013","copy_n11_014","copy_n11_015","copy_n11_016","copy_n11_017","copy_n11_018","copy_n11_019","copy_n12_000","copy_n12_001","copy_n12_002","copy_n12_003","copy_n12_004","copy_n12_005","copy_n12_006","copy_n12_007","copy_n12_008","copy_n12_009","copy_n12_010","copy_n12_011","copy_n12_012","copy_n12_013","copy_n12_014","copy_n12_015","copy_n12_016","copy_n12_017","copy_n12_018","copy_n12_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 45 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 120,
 "original": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "copy": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "edited": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "copy": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "files": [
  "work/fwd_45_original.jsonl",
  "work/fwd_45_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 0
 },
 "s": 1.6
}
```
Forward passes left: 5. Replies left: 106.
````

</details>

## Turn 95

Copy validation is complete: 240/240 correct for both models, with no changed outputs. I’ll now check reversal at lengths 1–6. Reversal exercises a different retrieval pattern, so preservation cannot be inferred from copying alone.

```forward
{"type":"generate","model":"both","ids":["reverse_n01_000","reverse_n01_001","reverse_n01_002","reverse_n01_003","reverse_n01_004","reverse_n01_005","reverse_n01_006","reverse_n01_007","reverse_n01_008","reverse_n01_009","reverse_n01_010","reverse_n01_011","reverse_n01_012","reverse_n01_013","reverse_n01_014","reverse_n01_015","reverse_n01_016","reverse_n01_017","reverse_n01_018","reverse_n01_019","reverse_n02_000","reverse_n02_001","reverse_n02_002","reverse_n02_003","reverse_n02_004","reverse_n02_005","reverse_n02_006","reverse_n02_007","reverse_n02_008","reverse_n02_009","reverse_n02_010","reverse_n02_011","reverse_n02_012","reverse_n02_013","reverse_n02_014","reverse_n02_015","reverse_n02_016","reverse_n02_017","reverse_n02_018","reverse_n02_019","reverse_n03_000","reverse_n03_001","reverse_n03_002","reverse_n03_003","reverse_n03_004","reverse_n03_005","reverse_n03_006","reverse_n03_007","reverse_n03_008","reverse_n03_009","reverse_n03_010","reverse_n03_011","reverse_n03_012","reverse_n03_013","reverse_n03_014","reverse_n03_015","reverse_n03_016","reverse_n03_017","reverse_n03_018","reverse_n03_019","reverse_n04_000","reverse_n04_001","reverse_n04_002","reverse_n04_003","reverse_n04_004","reverse_n04_005","reverse_n04_006","reverse_n04_007","reverse_n04_008","reverse_n04_009","reverse_n04_010","reverse_n04_011","reverse_n04_012","reverse_n04_013","reverse_n04_014","reverse_n04_015","reverse_n04_016","reverse_n04_017","reverse_n04_018","reverse_n04_019","reverse_n05_000","reverse_n05_001","reverse_n05_002","reverse_n05_003","reverse_n05_004","reverse_n05_005","reverse_n05_006","reverse_n05_007","reverse_n05_008","reverse_n05_009","reverse_n05_010","reverse_n05_011","reverse_n05_012","reverse_n05_013","reverse_n05_014","reverse_n05_015","reverse_n05_016","reverse_n05_017","reverse_n05_018","reverse_n05_019","reverse_n06_000","reverse_n06_001","reverse_n06_002","reverse_n06_003","reverse_n06_004","reverse_n06_005","reverse_n06_006","reverse_n06_007","reverse_n06_008","reverse_n06_009","reverse_n06_010","reverse_n06_011","reverse_n06_012","reverse_n06_013","reverse_n06_014","reverse_n06_015","reverse_n06_016","reverse_n06_017","reverse_n06_018","reverse_n06_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 46 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 120,
 "original": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "edited": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "files": [
  "work/fwd_46_original.jsonl",
  "work/fwd_46_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 0
 },
 "s": 0.5
}
```
Forward passes left: 4. Replies left: 105.
````

</details>

## Turn 96

Short reversal cases were all correct and unchanged. Next I’ll test reversal lengths 7–12. This completes the reversal check and could expose positional interference that shorter sequences would miss.

```forward
{"type":"generate","model":"both","ids":["reverse_n07_000","reverse_n07_001","reverse_n07_002","reverse_n07_003","reverse_n07_004","reverse_n07_005","reverse_n07_006","reverse_n07_007","reverse_n07_008","reverse_n07_009","reverse_n07_010","reverse_n07_011","reverse_n07_012","reverse_n07_013","reverse_n07_014","reverse_n07_015","reverse_n07_016","reverse_n07_017","reverse_n07_018","reverse_n07_019","reverse_n08_000","reverse_n08_001","reverse_n08_002","reverse_n08_003","reverse_n08_004","reverse_n08_005","reverse_n08_006","reverse_n08_007","reverse_n08_008","reverse_n08_009","reverse_n08_010","reverse_n08_011","reverse_n08_012","reverse_n08_013","reverse_n08_014","reverse_n08_015","reverse_n08_016","reverse_n08_017","reverse_n08_018","reverse_n08_019","reverse_n09_000","reverse_n09_001","reverse_n09_002","reverse_n09_003","reverse_n09_004","reverse_n09_005","reverse_n09_006","reverse_n09_007","reverse_n09_008","reverse_n09_009","reverse_n09_010","reverse_n09_011","reverse_n09_012","reverse_n09_013","reverse_n09_014","reverse_n09_015","reverse_n09_016","reverse_n09_017","reverse_n09_018","reverse_n09_019","reverse_n10_000","reverse_n10_001","reverse_n10_002","reverse_n10_003","reverse_n10_004","reverse_n10_005","reverse_n10_006","reverse_n10_007","reverse_n10_008","reverse_n10_009","reverse_n10_010","reverse_n10_011","reverse_n10_012","reverse_n10_013","reverse_n10_014","reverse_n10_015","reverse_n10_016","reverse_n10_017","reverse_n10_018","reverse_n10_019","reverse_n11_000","reverse_n11_001","reverse_n11_002","reverse_n11_003","reverse_n11_004","reverse_n11_005","reverse_n11_006","reverse_n11_007","reverse_n11_008","reverse_n11_009","reverse_n11_010","reverse_n11_011","reverse_n11_012","reverse_n11_013","reverse_n11_014","reverse_n11_015","reverse_n11_016","reverse_n11_017","reverse_n11_018","reverse_n11_019","reverse_n12_000","reverse_n12_001","reverse_n12_002","reverse_n12_003","reverse_n12_004","reverse_n12_005","reverse_n12_006","reverse_n12_007","reverse_n12_008","reverse_n12_009","reverse_n12_010","reverse_n12_011","reverse_n12_012","reverse_n12_013","reverse_n12_014","reverse_n12_015","reverse_n12_016","reverse_n12_017","reverse_n12_018","reverse_n12_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 47 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 120,
 "original": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "edited": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "reverse": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "files": [
  "work/fwd_47_original.jsonl",
  "work/fwd_47_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 0
 },
 "s": 1.5
}
```
Forward passes left: 3. Replies left: 104.
````

</details>

## Turn 97

Reversal is fully preserved on the practice set: 240/240 correct with identical outputs. I’ll now test max cases of lengths 1–6. This task involves two operands and a delimiter, making it a useful check that the addition circuitry remains inactive under a different task prefix.

```forward
{"type":"generate","model":"both","ids":["max_n01_000","max_n01_001","max_n01_002","max_n01_003","max_n01_004","max_n01_005","max_n01_006","max_n01_007","max_n01_008","max_n01_009","max_n01_010","max_n01_011","max_n01_012","max_n01_013","max_n01_014","max_n01_015","max_n01_016","max_n01_017","max_n01_018","max_n01_019","max_n02_000","max_n02_001","max_n02_002","max_n02_003","max_n02_004","max_n02_005","max_n02_006","max_n02_007","max_n02_008","max_n02_009","max_n02_010","max_n02_011","max_n02_012","max_n02_013","max_n02_014","max_n02_015","max_n02_016","max_n02_017","max_n02_018","max_n02_019","max_n03_000","max_n03_001","max_n03_002","max_n03_003","max_n03_004","max_n03_005","max_n03_006","max_n03_007","max_n03_008","max_n03_009","max_n03_010","max_n03_011","max_n03_012","max_n03_013","max_n03_014","max_n03_015","max_n03_016","max_n03_017","max_n03_018","max_n03_019","max_n04_000","max_n04_001","max_n04_002","max_n04_003","max_n04_004","max_n04_005","max_n04_006","max_n04_007","max_n04_008","max_n04_009","max_n04_010","max_n04_011","max_n04_012","max_n04_013","max_n04_014","max_n04_015","max_n04_016","max_n04_017","max_n04_018","max_n04_019","max_n05_000","max_n05_001","max_n05_002","max_n05_003","max_n05_004","max_n05_005","max_n05_006","max_n05_007","max_n05_008","max_n05_009","max_n05_010","max_n05_011","max_n05_012","max_n05_013","max_n05_014","max_n05_015","max_n05_016","max_n05_017","max_n05_018","max_n05_019","max_n06_000","max_n06_001","max_n06_002","max_n06_003","max_n06_004","max_n06_005","max_n06_006","max_n06_007","max_n06_008","max_n06_009","max_n06_010","max_n06_011","max_n06_012","max_n06_013","max_n06_014","max_n06_015","max_n06_016","max_n06_017","max_n06_018","max_n06_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 48 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 120,
 "original": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "max": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "edited": {
  "graded": 120,
  "accuracy": 1.0,
  "accuracy_by_task": {
   "max": 1.0
  },
  "addition_accuracy_by_n_digits": {},
  "failures": []
 },
 "files": [
  "work/fwd_48_original.jsonl",
  "work/fwd_48_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 0
 },
 "s": 0.8
}
```
Forward passes left: 2. Replies left: 103.
````

</details>

## Turn 98

All 120 short max cases remained correct and unchanged. I’ll finish native validation with max lengths 7–12. If these also match, all 720 native practice outputs will have been preserved exactly.

```forward
{"type":"generate","model":"both","ids":["max_n07_000","max_n07_001","max_n07_002","max_n07_003","max_n07_004","max_n07_005","max_n07_006","max_n07_007","max_n07_008","max_n07_009","max_n07_010","max_n07_011","max_n07_012","max_n07_013","max_n07_014","max_n07_015","max_n07_016","max_n07_017","max_n07_018","max_n07_019","max_n08_000","max_n08_001","max_n08_002","max_n08_003","max_n08_004","max_n08_005","max_n08_006","max_n08_007","max_n08_008","max_n08_009","max_n08_010","max_n08_011","max_n08_012","max_n08_013","max_n08_014","max_n08_015","max_n08_016","max_n08_017","max_n08_018","max_n08_019","max_n09_000","max_n09_001","max_n09_002","max_n09_003","max_n09_004","max_n09_005","max_n09_006","max_n09_007","max_n09_008","max_n09_009","max_n09_010","max_n09_011","max_n09_012","max_n09_013","max_n09_014","max_n09_015","max_n09_016","max_n09_017","max_n09_018","max_n09_019","max_n10_000","max_n10_001","max_n10_002","max_n10_003","max_n10_004","max_n10_005","max_n10_006","max_n10_007","max_n10_008","max_n10_009","max_n10_010","max_n10_011","max_n10_012","max_n10_013","max_n10_014","max_n10_015","max_n10_016","max_n10_017","max_n10_018","max_n10_019","max_n11_000","max_n11_001","max_n11_002","max_n11_003","max_n11_004","max_n11_005","max_n11_006","max_n11_007","max_n11_008","max_n11_009","max_n11_010","max_n11_011","max_n11_012","max_n11_013","max_n11_014","max_n11_015","max_n11_016","max_n11_017","max_n11_018","max_n11_019","max_n12_000","max_n12_001","max_n12_002","max_n12_003","max_n12_004","max_n12_005","max_n12_006","max_n12_007","max_n12_008","max_n12_009","max_n12_010","max_n12_011","max_n12_012","max_n12_013","max_n12_014","max_n12_015","max_n12_016","max_n12_017","max_n12_018","max_n12_019"]}
```

<details><summary>harness output</summary>

````text
Forward pass 49 of 50 done (generate). Summary:
```
{
 "type": "generate",
 "model": "both",
 "items": 120,
 "original": {
  "graded": 120,
  "accuracy": 0.9917,
  "accuracy_by_task": {
   "max": 0.9917
  },
  "addition_accuracy_by_n_digits": {},
  "failures": [
   {
    "id": "max_n10_008",
    "expected": "8880879189",
    "got": "8821625322\n"
   }
  ]
 },
 "edited": {
  "graded": 120,
  "accuracy": 0.9917,
  "accuracy_by_task": {
   "max": 0.9917
  },
  "addition_accuracy_by_n_digits": {},
  "failures": [
   {
    "id": "max_n10_008",
    "expected": "8880879189",
    "got": "8821625322\n"
   }
  ]
 },
 "files": [
  "work/fwd_49_original.jsonl",
  "work/fwd_49_edited.jsonl"
 ],
 "paired": {
  "rescues": 0,
  "losses": 0,
  "outputs_changed": 0
 },
 "s": 3.5
}
```
Forward passes left: 1. Replies left: 102.
````

</details>

## Turn 99

All 720 native outputs were unchanged, including one pre-existing max error: both models scored 719/720 overall. The remaining pass will measure English distribution drift, which exact-match tests cannot capture. I’ll select 32 evenly spaced practice-text windows without screening their contents and record the native totals before that test.

```python
# Hypothesis: English drift remains small on a broader fixed sample despite exact preservation of native outputs.
native_summary = defaultdict(lambda: [0,0,0,0])
for k in range(44,50):
    a = [json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_original.jsonl").read_text().splitlines()]
    b = [json.loads(s) for s in pathlib.Path(f"work/fwd_{k}_edited.jsonl").read_text().splitlines()]
    for x,y in zip(a,b):
        assert x["id"] == y["id"]
        t = native_summary[x["task"]]
        t[0] += int(x["correct"]); t[1] += int(y["correct"])
        t[2] += 1; t[3] += int(x["output"] != y["output"])
print("Native original/edited/total/changed:",dict(native_summary))
english_source = pathlib.Path("practice/text.txt").read_text()
english_offsets = [1024+6000*i for i in range(32)]
english_final = [english_source[i:i+240] for i in english_offsets]
assert all(len(s)==240 for s in english_final)
pathlib.Path("work/final_english_sample.json").write_text(json.dumps({"offsets":english_offsets,"texts":english_final}))
with pathlib.Path("work/notes.md").open("a") as f:
    f.write("Frozen native sweep passes44–49: copy240/240, reverse240/240, max239/240 for BOTH models. "
            "All720 outputs identical; zero losses/rescues. Pre-existing max_n10_008 error preserved. "
            "Final English sample:32 windows of240 chars at offsets1024+6000*i, no content screening.\n")
print(json.dumps({"type":"teacher_forced","model":"both","texts":english_final,"return":{}}))
```

<details><summary>harness output</summary>

````text
OUTPUT of Python run 35 of at most 300:
```
Native original/edited/total/changed: {'copy': [240, 240, 240, 0], 'reverse': [240, 240, 240, 0], 'max': [239, 239, 240, 0]}
{"type": "teacher_forced", "model": "both", "texts": [" well established . \n The West Branch Canal , which opened in 1834 , ran 73 miles ( 117 km ) from Northumberland to Farrandsville , about 5 miles ( 8 km ) upstream from Lock Haven . A state @-@ funded extension called the Bald Eagle Cut ran", "manufacturing plant in Bradford , Pennsylvania , was destroyed by fire . The factory began operations in a building that once housed a silk mill . As the company grew , the original factory expanded to include engineering and office buildin", "e Ridge @-@ and @-@ Valley Appalachians , a mountain belt characterized by long even valleys running between long continuous ridges . Bald Eagle Mountain , one of these ridges , runs parallel to Bald Eagle Creek on the south side of the cit", "liday parade in December . Light @-@ airplane pilots travel to the city in vintage Piper planes to attend Sentimental Journey Fly @-@ Ins , which have been held each summer since 1986 . Enthusiasts of radio @-@ controlled model airplanes me", "cycle Route G follows Pennsylvania Route 150 and links to the Pine Creek Rail Trail at the eastern end of the county near Jersey Shore , Pennsylvania . A 2 @.@ 5 @-@ mile ( 4 @.@ 0 km ) walking trail on the levee along the river is restrict", "ated by millions of women around the world and remains one of the most popular hairstyles in history , in spite of Aniston 's own resentment towards it . Rachel is also regarded as a style icon due to her influence on womenswear during the ", "' true feelings for her , at the same time discovering she actually feels the same . However , the episode nearly went unwritten because , at the time , few friends writers were expecting the couple 's relationship to morph into the phenome", "alled Muddling Through , in which she plays a young woman whose mot
[... cut]
````

</details>

## Turn 100

The native totals are confirmed: copy 240/240, reverse 240/240, and max 239/240 for both models, with zero output changes. I’ll use the final pass on the selected 7,680 English characters. KL near the earlier 0.0116 result would support a small but nonzero language-model disturbance; a much larger value would weaken that conclusion.

```forward
{"type":"teacher_forced","model":"both","texts":[" well established . \n The West Branch Canal , which opened in 1834 , ran 73 miles ( 117 km ) from Northumberland to Farrandsville , about 5 miles ( 8 km ) upstream from Lock Haven . A state @-@ funded extension called the Bald Eagle Cut ran","manufacturing plant in Bradford , Pennsylvania , was destroyed by fire . The factory began operations in a building that once housed a silk mill . As the company grew , the original factory expanded to include engineering and office buildin","e Ridge @-@ and @-@ Valley Appalachians , a mountain belt characterized by long even valleys running between long continuous ridges . Bald Eagle Mountain , one of these ridges , runs parallel to Bald Eagle Creek on the south side of the cit","liday parade in December . Light @-@ airplane pilots travel to the city in vintage Piper planes to attend Sentimental Journey Fly @-@ Ins , which have been held each summer since 1986 . Enthusiasts of radio @-@ controlled model airplanes me","cycle Route G follows Pennsylvania Route 150 and links to the Pine Creek Rail Trail at the eastern end of the county near Jersey Shore , Pennsylvania . A 2 @.@ 5 @-@ mile ( 4 @.@ 0 km ) walking trail on the levee along the river is restrict","ated by millions of women around the world and remains one of the most popular hairstyles in history , in spite of Aniston 's own resentment towards it . Rachel is also regarded as a style icon due to her influence on womenswear during the ","' true feelings for her , at the same time discovering she actually feels the same . However , the episode nearly went unwritten because , at the time , few friends writers were expecting the couple 's relationship to morph into the phenome","alled Muddling Through , in which she plays a young woman whose mother is returning home from jail after two years . CBS was initially reluctant to release Aniston from her contract , which required the actress to balance both roles simulta"," \" beautiful , coveted , slightly neurotic , borderline egocentric \" character . \n Observing that the show 's main characters are each based on a stereotype , Jonathan Bernstein of The Daily Telegraph identified Rachel as \" the self @-@ abs","t of true empathy for another human being \" . Saraiya went on to describe Rachel as \" as a model for women coming of age in the 1990s  the popular , pretty girl dissatisfied with where those illusions have taken her but also unwilling to em"," \" those of her co @-@ stars , becoming the first cast member to \" rise to prominence \" ; the actress continues to experience the most post @-@ Friends success . Aniston 's performance in Friends led to a successful film career . According ","c plot ; cheers for abandoning it \" . Bustle also cited the same episode as one of the show 's worst , calling it \" the most nonsensical idea to ever be . \" Contrarily , E ! enjoyed Rachel and Joey as a couple because they brought out posit","h Aniston eventually abandoned the \" Rachel \" for a straighter , longer look , the hairstyle remained popular nonetheless . Despite her association with the cut , Aniston disliked the hairstyle . She found maintaining the hairstyle without ","ge and dominated the road between Homs and Tripoli . When building castles , Muslims often chose elevated sites such as hills and mountains that provided natural obstacles . \n In January 1099 on the journey to Jerusalem during the First Cru"," was buried in the castle 's chapel . \n The main contemporary accounts relating to Krak des Chevaliers are of Muslim origin and tend to emphasize Muslim success while overlooking setbacks against the Crusaders although they suggest that the","rench state , and cared for by the Acadmie des Beaux @-@ Arts . The villagers were moved and paid F1 million between them in compensation . Over the following two years a programme of cleaning and restoration was carried out by a force of 1","ecture , probably dating from the 1230s . \n = = = Chapel = = = \n The current chapel was probably built to replace the one destroyed by an earthquake in 1170 . Only the east end of the original chapel , which housed the apse , and a small pa","his downfall . The Marquess of Queensberry , whose son Lord Alfred Douglas was Wilde 's lover , planned to present the writer with a bouquet of rotten vegetables and disrupt the show . Wilde was tipped off and Queensberry was refused admiss","me from the billing , but the play had to close after only 86 performances . \n The play 's original Broadway production opened at the Empire Theatre on 22 April 1895 , but closed after sixteen performances . Its cast included William Favers","on at the Haymarket the King and Queen attended a performance , which , as the journalist Geoffrey Wheatcroft put it , gave the play \" a final accolade of respectability . \" The production toured North America , and was successfully staged ","acknell 's late sister , and thus indeed Algernon 's elder brother . Having acquired such respectable relations , he is acceptable as a suitor for Gwendolen after all . \n Gwendolen , though , still insists that she can only love a man named","the absolute necessity for homosexuals of the period to \" need a public mask is a factor contributing to the satire on social disguise . \" \n The use of the name Earnest may have been a homosexual in @-@ joke . In 1892 , three years before W"," 's two final comedies , An Ideal Husband and The Importance of Being Earnest , were still on stage in London at the time of his prosecution , and they were soon closed as the details of his case became public . After two years in prison wi","ert Tanitch , by 2002 there had been least eight adaptations of the play as a musical , though \" never with conspicuous success \" . The earliest such version was a 1927 American show entitled Oh Earnest . The journalist Mark Bostridge comme","thews was born at Funchal on Madeira on 7 March 1850 . His father , Captain William Matthews was Welsh , and his mother Jane Wallis Penfold , was the daughter of William Penfold and Sarah Gilbert . Her sister , Augusta Jane Robley ne Penfol"," his position to suppress slavery in the country and in 1889 convinced the Sultan to issue a decree purchasing the freedom of all slaves who had taken refuge in his dominions and , from 1890 , the prohibiting the slave trade . On 1 February"," War II on convoy escort duties in the English Channel and the North Atlantic , based at Dover , Gibraltar , and Freetown , Sierra Leone . Boreas also served two brief tours with the Mediterranean Fleet and participated in Operation Husky ,"," she arrived in Alexandria , Egypt on 11 November after escorting a convoy around the Cape of Good Hope . She was immediately assigned to escort the ships of Operation Stoneage that relieved the Siege of Malta . The ship remained in the Med","ong the wild horses . The Kaimanawa breed varies widely in general appearance , with heights ranging between 12 @.@ 2 and 15 hands ( 50 and 60 inches , 127 and 152 cm ) high . Any coat colour or pattern marking is acceptable . They are usua","e Remix ( Lady Gaga album ) = \n The Remix is a remix album by American recording artist Lady Gaga . Released in Japan on March 3 , 2010 , it contains remixes of the songs from her first studio album , The Fame ( 2008 ) , and her third exten","oenix New Times gave a positive review saying that the songs featured in The Remix can be a great addition during workouts , as well as staple dance floor music . She listed the Chew Fu remix of \" LoveGame \" as a highlight from the album . ","lope in the Song dynasty style , supported prominently by two sets of brackets ( dougong ) . The lower bracket assembly rested on the city gate to form a wooden foundation , while the upper assembly supported the roof , similar to the dougo"],"return":{}}
```
