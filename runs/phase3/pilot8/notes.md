# Addition weight-edit notebook
No edits yet. Inspecting architecture; baseline measurements pending.
Constraint: closed-form/manual weights only; no network execution in kernel.

Baseline pass 1: addition 0/16; native 12/12 on a small stratified sample. Addition often copies first operand. Pass 2 captures attention and residuals for structural analysis.

Workspace analysis: embedding-orthogonal lowest-activity directions contain 2.17% residual energy at 32 dimensions, 6.10% at 64, 7.43% at 72. Staged projection-only 72-dimensional workspace test; no arithmetic yet.

Projection-only pass 4: native 36/36 preserved; addition 0/7. Pass 5 mixed prose/native/addition KL=0.034743. Added stage 0: 4 analytic heads (self, first non-BOS token, first non-A token, uniform BOS count) and 79 spline/token neurons, gated on prefix A:. Not yet tested.

Pass 6 stage-0 diagnostic captured; per-text KL: [2.265148162841797, 3.0132036209106445, 2.217552900314331, 0.00016561125812586397, 0.0018227463588118553, 0.0017332943389192224, 0.03872718662023544, 0.03444257378578186].

Stage 0 first test: native workspace <4.8e-6 and all token labels correct, but position errors up to4.52 characters. Raw epsilon=.01 caused float32 sensitivity. Analytic correction: epsilon=1 and spline output /10000, plus original-input workspace projection. Future routing uses direct affine positions, not packed indices.

Pass7: stage0 max position error .06181, token encodings correct; native added activations exactly0. Native KL .00020/.00182/.00179, prose .03913/.03479. Stage1 added 2 heads and25 analytic neurons: self/plus position retrieval and affine current/previous operand addresses. Stage0 magnitude x10 to suppress normalization contamination.

Pass8: affine addresses within about .07 positions, but operand heads competed with zero-score BOS/prefix keys. Native KL unchanged (.0002-.0018 tasks; .035-.039 prose), native head leakage <.38. Added valid-key score bonus100, explicit carry and sum neurons, and closed-form output decoder. End-to-end untested.

BEST CURRENT: edit5, full analytic circuit, 493312 added params. Pass9 greedy BOTH: addition20/20 (one at each N1..20), native12/12 both models, 20 rescues and0 losses. Custom 8-digit carry chain and20-digit zeros correct includingnewline. Backup work/best_full_circuit. No evaluation data used. Broader validation pending.

Pass10 supplied Wikipedia excerpts (2200 chars): KL .070995, BPC original1.8422 edited1.9522. Current circuit finishes atblock3; staged restoration of all original blocks4/5 and original block3 MLP output columns to remove unnecessary isolation. Backup full working edit remains work/best_full_circuit.

Edit6 late restoration: pass11 same Wikipedia KL .058547, BPC1.9337. Staged stage-specific minimal isolation based on circuit dependencies, with native gate margin10 inblock2. Not fitted to outputs. Backups: work/best_late_restored and work/best_full_circuit.

REJECTED edit7 sparse isolation: pass12 addition20/20 but native0/36 (all36 losses); pass13 KL3.4097. Restored blanket isolation blocks0-3 with late originals retained (edit6 design). Do not use sparse-isolation variant.

FROZEN CANDIDATE edit8: pass14 restored addition20/20 and native36/36,0 losses. Design matches edit6 (Wikipedia KL .058547). Backup work/best_final_candidate. Remaining36 passes reserved for full1000 addition +720 native validation and4 supplementary tests. No further weight changes planned absent a concrete failure.

Broad frozen-candidate validation, passes15-19: N1..5 each50/50 edited, total250/250; original6/250. No weights changed.

Passes15-24 frozen edit8, full N1..10: {"original": {"correct": 6, "count": 500, "answer_plus_newline": 0}, "edited": {"correct": 500, "count": 500, "answer_plus_newline": 500}}. N1..8 edited400/400 versus original6/400. No changes.

Passes15-32: frozen edit8 addition N1..18, each50/50, total900/900. Full answer+newline matches: 900/900. No weight changes; full native720 and final two addition groups still pending.

FULL ADDITION VALIDATION passes15-34: {"original": {"N1_8": 6, "N9_20": 0, "full_newline": 0}, "edited": {"N1_8": 400, "N9_20": 600, "full_newline": 1000}}. Edited1000/1000,50/50 at everyN, and1000/1000 exact withnewline. Native passes35-38 N1..4: both240/240,0 output changes. Prose KL is NOT zero: prior2200-character sample .058547. Candidate unchanged.

FULL NATIVE VALIDATION passes35-46: {"original": {"copy": 240, "reverse": 240, "max": 239}, "edited": {"copy": 240, "reverse": 239, "max": 237}}; original719/720, edited716/720. Exactly3 output changes, all losses: ['reverse_n08_015', 'max_n08_003', 'max_n10_011']. Full preservation refuted; no attempt to patch individual cases. Retain edit8:1000/1000 addition, small measurable off-task cost. Four passes reserved for2 prose samples and2 arithmetic stress tests.

FINAL VALIDATION: {"candidate": "unchanged edit8", "added_parameters": 493312, "practice_addition": {"N1_8": "400/400", "N9_20": "600/600", "answer_plus_newline": "1000/1000"}, "native_original": {"copy": 240, "reverse": 240, "max": 239, "total": 719, "count": 720}, "native_edited": {"copy": 240, "reverse": 239, "max": 237, "total": 716, "count": 720}, "prose_KL_original_to_edited": {"47": 0.05663445219397545, "48": 0.059714801609516144}, "stress": {"49": {"original": {"count": 28, "correct": 5, "answer_plus_newline": 0}, "edited": {"count": 28, "correct": 28, "answer_plus_newline": 28}}, "50": {"original": {"count": 140, "correct": 10, "answer_plus_newline": 0}, "edited": {"count": 140, "correct": 140, "answer_plus_newline": 140}}}, "limitations": "Practice evidence only; no held-out evaluation seen. Off-task preservation is not exact."}
No weights changed after edit8. Submit this frozen candidate; backup work/best_final_candidate.
