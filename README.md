# weight-splice-lab

**Can an LLM agent *add* a capability to a pretrained model by writing its weights directly, with
no gradients and no training?**

Takehi et al. ("Sparks of Weight-Space Programming", MIT CSAIL, Nov 2026) showed that GPT-6 Astra, editing
weights directly, could lift Qwen3.5-0.8B on MATH-500 from 7.9% to 48.1% by removing a repetition
direction. They also reported that on Gemma 3 270M "editing weights could remove an obstacle that
hid an ability, but it could not add an ability the model lacked."
([blog](https://rikiyat.github.io/blog/weight-space-programming/),
[logs](https://github.com/RikiyaT/lm_understand-logs))

We tested that sentence with the same editor model (gpt-6-astra via the Codex CLI) in our own
open harness: a sandboxed Python kernel over the weights, a budgeted forward-pass tool, autograd
blocked, every cell logged and audited, and a hidden evaluation pre-registered before each run.

## Results

| | setting | result |
| --- | --- | --- |
| **Gemma 3 270M IT** (the paper's negative case) | forward-pass tool only (50 passes) | one run in each of three setups (plain addition ×2, reversed ×1): no multi-digit algorithm, as in the paper |
| | the editor may also run the model in its kernel (still no gradients, no search over weights) | reversed-digit addition from 0.9% (mean over 1–8 digits) to **100% at every length from 1 to 30 digits** (n = 1), hand-built in 10 added heads and 423 neurons, with 34 native tensors also adjusted. **Fails our pre-registered bar** on off-task drift: text KL 0.445 (5 of 256 Wikipedia chunks hold 99.8% of it) and chat KL 0.086 |
| 5M toy transformer that cannot add | editor may add heads and neurons | 100% multi-digit addition in **3/3** runs; 2/3 also meet the KL bar |
| | editor may only change existing weights | 0/3, one-digit lookup tables only |
| Qwen3.5-0.8B (reproduction) | the paper's published edits | rebuilt bit for bit; the gain is a long-context effect (4k cap: 3.5% → 8%; 32k: 7.5% → 46%) |

Our reading: on the toy model, adding a missing algorithm by hand-written weights works reliably.
On Gemma 270M, one run added the algorithm, but the run fails our own success criterion, and its
log has had only a first-pass compliance check. So it shows the ability *can* be written in. It
is not a clean counterexample to the paper. **Making the addition silent on arbitrary text is the
hard part.** The failures hide in the long tail of inputs that the agent never checks.

Caveats: n = 1 on Gemma and n = 3 on the toy; one editor model; addition only, in Gemma's case the
reversed (least-significant-digit-first) format; independent work, not peer reviewed.

**Start with [`docs/writeup.md`](docs/writeup.md).**

## Layout

- `docs/writeup.md`: the full write-up.
- `docs/phase{1,2,3,4}-*.md`: per-phase detail, including the pre-registrations.
- `docs/eval-hardened.md`: the toy eval.
- `docs/repeats-2026-11-07.md`: toy n = 3.
- `docs/reviews/`: per-run compliance reviews.
- `runs/phase3/<run>/` (toy) and `runs/phase4/<run>/` (Gemma), one folder per scored run:
  - `result.json`: the hidden eval, audit summary and budget;
  - `transcript.md`: the editor's replies, each with the start of the harness output;
  - `notes.md`: the editor's own notebook;
  - `final_edit_manifest.json`, `audit.json`: `compliance: pending_review` in `result.json` is
    the automated state; the human reviews are in `docs/reviews/` (toy runs) and
    `docs/phase4-gemma.md` (Gemma runs);
  - `TASK.md`: the exact prompt the editor saw.
- `src/weight_splice_lab/`:
  - `toy/` and `gemma/`: the models and hidden evals;
  - `harness/`: the kernel sandbox, forward tool, edit validation, audit and Codex driver.
- `experiments/`: the scripts behind each phase.
- `prompts/`: editor prompt templates.
- `LOGBOOK.md`: dated log of the whole project, failures included.

## Running

- Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv run pytest` runs the tests: unit tests
  use tiny models and need no downloads; the kernel sandbox tests need macOS (`sandbox-exec`).
- Gemma: `experiments/phase4/convert_base.py` downloads and converts the base model (Gemma terms
  of use apply; the weights are not redistributed here).
- Editor runs (`experiments/phase3/run_editor.py`) call the Codex CLI through `codex-switch`, a
  local profile wrapper; to run without it, swap the command in `src/weight_splice_lab/harness/codex.py`.
  Runs use a ChatGPT subscription's weekly limit (about 1–4% per run).
- Local GPU and memory guards: `weight_splice_lab.gpu.gpu_session` and `weight_splice_lab.memwatch`.

## Licence

Code: MIT (`LICENSE`). Docs, results and logs: CC BY 4.0 (`LICENSE-docs`), except third-party
data noted there.
