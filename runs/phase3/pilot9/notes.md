# Weight-edit notebook
Condition: EDIT_ONLY. Embedding and final norm remain frozen.
No edits or forward passes yet. First inspect architecture and establish original-model baselines.
Only closed-form or hand-set edits; no network execution in the Python kernel.

Baseline pass 1: addition 0/40 (2 per N=1..20); six valid custom native prompts correct. Addition resembles concatenated copying.
Pass 2: matched addition/native activation capture, 16 texts; inspecting routing. No edits yet.
Pass 3 captured all 100 N=1 operand pairs (200 answer prediction positions). Planning a single closed-form preservation-constrained readout, not gradient training.
Mixed-length probes (pass 4): best operand-a accuracy 36.3%, operand-b 12.8%; sum probes 9.7–15%. General routing is missing.
Edit A: exact 200-state N=1 last-MLP output update, preservation covariance from pass 5; constraint max error 1.46e-11, relative weight norm 26.197. Not yet evaluated.
Edit A rejected: pass 6 N=1 20/20 vs original 2/20, N>=2 0/12, native 0/36 vs 36/36; pass 7 mixed KL 1.751487 nats.
New design: shared task-selective SwiGLU gate plus rank-10 readout in up projections; only 10 low-contribution neurons repurposed. Gate probe min addition 0.04150, max preservation -0.02594.
Pass 8 gate characterization: paired mean KL 0.000042 (110 mixed texts). Ten-neuron removal is low-impact.
Candidate B completed: gate measured by forward tool, then exact one-shot up-row solve for 200 N=1 answer states and fixed digit-only inverse-output directions. No claims of general multi-digit ability.
Candidate B pass 9: N=1 20/20 vs 2/20, native 36/36 unchanged, longer additions 0/12. Pass 10 independent/custom English KL 0.010018 nats. Saved fallback in work/best_n1. Not a solution to multi-digit addition.
Pass 12 broader preservation KL 0.002735 for N=1 fallback.
N=2 diagnostic: pooled held-out a/b/carry accuracies .967/.800/.925; arithmetic reconstruction 37.5% exact on 40 held-out examples. Table circuit uses 200 existing low-contribution neurons, no added capacity. Original removed contributions projected into remaining neurons by one closed-form solve.
Candidate C intermediate pass 14: mean per-native KL 0.000015, mean per-text KL 0.021690. Output calibrated once by preservation-constrained closed-form regression on 600 N=2 answer states; no iterative optimization.
Candidate C rejected for preservation: pass 15 N=2 9/50 (18%), N=1 20/20, native 30/36 vs 36/36. Pass 16 English KL .022990. Copy saved at work/candidate_n2_unprotected.
Candidate D prefix-marker characterization: repurpose head 3 in layer 5, content key constrained to A: (4), BOS (2.5), bare A/native prefixes (0). Other-head linear reconstruction residual RMS .2843. Marker lies in frozen output map nullspace. MLP reset to original.
Candidate D pass 18 rejected as-is: mixed KL 1.249958; native examples KL 0–.0014, but four prose KL .2744–2.4033. Background attention captures cause marker leakage. No addition decoder attached.
Candidate E: fixed-key margin analysis permits BOS coefficient (0.811925,1); choose midpoint 0.905963. Marker amplitude reduced to 32. Original MLP, attention-head compensation retained. Awaiting paired test.
Candidate E marker pass19: native mean KL .000344, English .001259; pass20 native .000050, English .001332. Staged 200 table neurons + 10 N1 correction neurons, zero arithmetic outputs pending measured-feature closed-form readouts.
Candidate E decoder: table output fitted once on 600 captured N2 states with preservation weight 10; N1 up-readout solved exactly on 200 states after table correction. N1 gate feasible by analytical two-variable inequality elimination (min positive .1345, max negative -.2166). Await independent generation.
Candidate E pass24 generation: N1 50/50 vs original6/50; N2 11/50 vs0/50; N3–20 0/36. Native71/72 vs72/72 (one max loss). Saved work/candidate_marker_n2. This is not general multi-digit addition. Need independent text KL and regression diagnosis.
Candidate E pass25: English mean KL .017877; max regression KL .141105. At failed digit table correction norm266.85 despite zero prefix-head output; N1 neurons silent. Candidate F changes table feature family to table gate times a shared marker-sensitive linear indicator (one closed-form solve), addressing leakage multiplicatively. Output columns zero pending tool capture.
Candidate F output solved once: fit RMS 64.1889; off-task correction RMS 0.0928. Await paired generation.
Candidate F pass28: N1 50/50; deduplicated N2 7/50; native72/72 vs72/72, previous max loss restored. Saved work/candidate_multiplicative. Remains only a limited N1/N2 improvement, no general addition claim.
Fresh validation pass29/pass30 scored after restart; see next notebook entry for counts.
Candidate F fresh pass29 native107/108 vs108/108; pass30 fresh N2 12/100 vs0/100 (digit accuracies42%,26%,71%). Pass31 new English mean KL .069847, max excerpt .415892. Preservation is not adequate; test a smaller circuit rather than claim intact behavior.
Pass32 leakage diagnosis: new prose table max correction norm1413, N1 correction103120; prefix marker itself falsely activates (normalized coordinate7.90, overlapping addition5.3–8.1). Abandon marker variants. Restoring attention-preserving ten-neuron Candidate B from best_n1 for broader validation; no general multi-digit claim.
Restored Candidate B as edit12. Pass33 same eight new text excerpts English mean KL 0.003463; ten-item mixed mean .002770, max .024095. Much safer than marker circuit; broad generation validation pending.
Candidate B pass34 native180/180 distinct, no outputs changed (181 requests with duplicate). Pass35 exhaustive N1 score: edited 100/100, including newline 0/100; addition evaluation does not require newline.
Candidate B pass36 N1 10/10, N2–8 0/70 (longer outputs unchanged). Pass37 N9–20 0/60, three outputs changed. Pass38 completes all240 copy problems:239/240 vs240/240, loss copy_n03_012 (780 ->790). Preserve claims must acknowledge this loss.
Candidate B fixed validation full native720: original copy240/240,reverse240/240,max239/240; edited copy239/240,reverse240/240,max239/240. One added loss, no rescues. Pass39 proves gate leakage causes copy780 and prose outlier, not displaced-neuron contribution. Saved work/final_candidate_n1. Partial result only: no multi-digit arithmetic solution.
Candidate B English pass42 (8 new supplied-text excerpts): mean KL .012033,max .066025; pass43 deliberately number/prefix-heavy stress text mean .013683. Preservation remains approximate. Pass44 native neighborhood stress has4 output changes; not a random accuracy estimate.
