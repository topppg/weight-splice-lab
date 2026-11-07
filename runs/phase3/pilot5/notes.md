# Weight-edit notebook
Constraints: EDIT_ONLY; frozen embeddings and final norm; no training or model execution in Python.
Initial step: inspect architecture and establish baseline before selecting a closed-form edit.
No edits made or evaluated yet.

Baseline pass 1: addition 0/25; copy, reverse, max each 12/12. Pass 2 captured late-layer attention and residuals on 9 texts. No edits yet.

Candidate 1: single covariance-weighted constrained solve for blocks.5.mlp.down_proj.weight only. 200 exact answer-position constraints covering all one-digit pairs; 3821 weighted preservation rows; ridge 0.01 times mean covariance diagonal. Correct digit boosted to original max+4 using frozen output dual. Delta norm 531.3754. Not yet saved/evaluated.

Candidate 1 rejected: pass4 one-digit 20/20, longer addition 0/20; native copy1/12 reverse0/12 max0/12 (35 losses). Pass5 mixed-text mean KL1.680127. Staged edit reset. Investigating localized analytic MLP gates instead; no further weight fitting to the rejected candidate.

Prototype-only gates rejected analytically: 10.0% preservation states activate. New structural candidate: plus marker in one residual direction c orthogonal to embedding and frozen output; all block inputs ignore c and outputs preserve it. Layer0 head1 value/output retains top31 token-value singular components (discarded singular0.156), uses last component for exact '+' indicator. No arithmetic readout yet. Marker scale1; 42 projection tensors changed, architecture unchanged.

Marker-only edit2: pass7 mean KL0.000110 (152 texts); pass8 native36/36 retained, no addition improvement. Candidate3 adds ten final-MLP neurons (reused lowest preservation contribution), exact closed-form one-digit gates, up projections read plus marker, down projections use frozen-output dual digit directions. Single closed-form compensation restores removed native neuron contributions using retained758 features. Not yet evaluated.

Candidate3/pass9: one-digit50/50 (original6/50), longer0/20, native36/36 retained. Pass10 ordinary-prose mean KL0.00064712; plus-containing prose mean KL2.75759506; native mean KL0.00000393. Current best partial solution saved in work/best_3 and work/gated_one_digit_weights.pt. Not a multi-digit solution.

Pass11: captured routing for 64 independent synthetic additions, eight each at N2,3,4,6,8,12,16,20. Practice text contains zero '+' symbols, but plus-prose remains a known limitation. Current saved edit unchanged (candidate3).

Pass11 routing analysis: strong a-copy only at short lengths; b aligned mass generally low. Heldout linear digit probes at resid_mid5: a43.0%, b9.9%, incoming carry76.1%. Thus no supported general multi-digit readout. Exploring one two-digit numeric diagnostic, not tuning the saved one-digit circuit.

Pass12 diagnostic: two-digit final-input sum probes only43.8%/34.4% within half-digit at first/second positions; finalcarry100%. Source-token numeric readout98.4% within half-digit. Direct sum-readout idea rejected. Exploring analytic RoPE pointer phases and previous-digit carry identity; no new edit staged.

Pointer-phase diagnostic rejected: heldout a pointer MAE1.470 positions, b0.654; not accurate enough for analytic routing. No further edits. Remaining38 passes allocated to exhaustive practice evaluation (20 generation batches, all1000 additions/all720 native) and18 disjoint prose batches. Current best/final candidate remains edit3; explicitly partial (one-digit only).

Final sweep passes13-17: N1 addition50/50 (original6/50); N2,N3,N4,N5 each0/50 for both. Native180/180 for both, no losses. Edit3 unchanged.

Sweep through pass22 (N1..10): addition N1 50/50 edited vs6/50 original; N2..10 each0/50 both. Native359/360 both; one shared max error, zero native output changes or losses. Edit3 unchanged.

Sweep through pass27 (N1..15): additions50/750 edited vs6/750 original; all gains N1. Native539/540 both, zero native output changes/losses. Remaining five generation batches and18 disjoint prose batches; edit3 unchanged.

Complete practice sweep passes13..32, deduplicated (one duplicate N19 ID): all1000 additions/all720 native covered. Addition N1=50/50 vs6/50; every N2..20=0/50 both. N1..8=12.5% vs1.5%; N9..20=0% both. Copy240/240, reverse240/240, max239/240 both; all720 native outputs identical, zero losses. Current best remains edit3, a partial one-digit solution only. Summary saved work/generation_summary.json.

Prose validation passes33..35:24 disjoint 160-character excerpts, mean per-text KL=0.00092608, max=0.00152510. Logits saved for aggregation. Ordinary prose only; known plus-prose KL2.757595 remains a limitation.

Prose validation through pass40:64 excerpts, mean KL=0.00104082, median=0.00095942, max-text=0.00217587. Edit3 unchanged;10 prose passes left.

Prose through pass45:104 excerpts, mean KL=0.00100644, max-text=0.00275234. Five remaining prose passes, no further edits planned. Final submission must explicitly state failure on multi-digit addition and plus-prose interference.

FINAL: all50 forward passes used; candidate3 unchanged throughout final validation. {"texts": 144, "characters": 23040, "mean_kl_original_to_edited": 0.0009901680750772357, "median_text_kl": 0.0009265117114409804, "max_text_kl": 0.002752342727035284, "source": "practice text, not hidden held-out text"}
Submission is explicitly PARTIAL: one-digit addition50/50 vs6/50; N2..20 each0/50. Copy240/240, reverse240/240, max239/240 for both; all720 native outputs identical. Known counterexample to full preservation: plus-containing prose KL2.757595 in pass10. Frozen embed/final_norm verified unchanged, zero added parameters. Final staged backup:work/final_candidate; detailed results:work/generation_summary.json and work/prose_summary.json.
