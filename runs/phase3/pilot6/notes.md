# Addition weight-edit notebook
No edits yet. Inspecting implementation and preparing original-model baselines.

Baseline pass 1: addition 0/15; copy/reverse/max each 4/4. Addition resembles copying.
Pass 2 probes: best first-operand accuracy 28.1%; second operand <=14.3%. Lowest 32 embedding-null directions carry 0.0368% of pooled residual energy. Staged projection into complement of 40 directions and 3 layer-0 scratch heads.
Scratch edit: pass 3 mixed-input KL 0.012786; pass 4 native 12/12 unchanged, addition 0/15. Backup work/scratch_v1.
Pass 5 scratch amplitude 1: mixed KL 0.015420. Staged six layer-1 neurons for gated homogeneous token/position representation; no output arithmetic yet.
Pass 6: native/English gates inactive; addition position max error 0.1301, digit error <0.0001. Added a cancellation identity to reduce positional roundoff, separator and operand routing pairs, fixed square spline, carry threshold, and modulo-10 output lookup. No optimization or training.
Pass 7: all 3 diagnostic additions teacher-forced correct, including N20; no extra off-task KL. Pass 8 greedy: addition 49/60 (81.67%), native 12/12 unchanged; failures first digit at N13–16. Backup work/first_complete. Investigating boundary routing.
Pass 9 localized N13–16 first-digit failures: separator heads selected BOS (up to 100%) because fallback score exceeded delimiter score at equals. Fixed analytically by subtracting 0.1 times active carrier from fallback query; native fallback unchanged.
Pass 10: corrected circuit 60/60 addition, 12/12 native; backup work/best_working_60 and best_working_60_tensors.pt. Reducing protection from 40 to 23 dimensions and restoring original layer-4 MLP and layer-5 outputs; prefix scale needs recalibration.
Pass 11 reduced projection English probe mean KL 0.010319; native KL 0.000039. Prefix RMS recalibrated 3.46943563 -> 3.43327881.
Pass 12 reduced projection: 40/40 addition, 12/12 native. Backup work/best_reduced_projection. Added two layer-0 hinge neurons restricting A-colon activation to absolute position 2; one additional protected direction in layer 0 only. Prefix scale must be recalibrated.
Pass 13 prefix filtering tested on internal A-colon examples; calibration inverse normalized marker 0.293515760. Circuit ready for broad validation after save.
Pass 14: addition 19/20, native 12/12; one final carry digit wrong. Pass 15 trace: carry correct, second-operand attention splits between equals (0.558) and first answer digit (0.437), producing 1.755 instead of 0. Off-task control KL 0,0,0.0163. Investigating positional key distortion.
Pass 15 diagnosis: square-minus-position² at failing equals =0.891, vs -0.037 at next token. Early carrier normalization approximation causes quadratic-key distortion. Staged fixed scale separation: T1 x100; T2,T3,Tout x1000, only added output columns.
Pass 16: extreme scale increase fixed addition 20/20 but destroyed native 0/12, rejected. Staged moderate early-only scales T1=1e4,T2=1e6; restored T3=1e8,Tout=1e12.
Pass 17 moderate early-only scaling: addition 20/20 including previous failure; native 12/12, no losses. Current best backed up in work/best_early_scale and corresponding tensors.pt.
Pass 18 English 8x240 chars: KL 0.099690; BPC 1.7642 -> 1.9034. Staged position gate M=320 (negative over full 256 context off-task), disabled cancellation neuron.
Pass 19 full-context gate: English KL reduced 0.099690 ->0.011579, BPC1.7642->1.7800. Pass20: addition60/60, native36/36, no losses. Backup best_fullcontext. Trying restoration of original layer3 outputs and layer4 attention with late gate margin10000.
Pass21 late-output restoration failed English KL2.225838; rejected. Pass22 diagnostic collected. Restored full-context-safe backup (KL0.011579; addition60/60,native36/36).
Pass22: restored late outputs drove constant negative (-0.856), activating huge arithmetic MLP outputs. Trying late rank1 protection of constant S0 only, with gate margin10000; all other late original directions restored.
Pass23 rank1 late protection KL0.105289, still worse than0.011579; rejected. Restored best_fullcontext tensors. Freeze design for broad validation. Architecture +20 heads,+115 neurons=743680 params; full-context gateM320, T1=1e4,T2=1e6,T3=1e8,Tout=1e12. Best observed pass20 addition60/60, native36/36; pass19 EnglishKL0.011579 on8x240chars.
Submitted restored best as edit15 FINAL. Frozen sweep passes24–27: N20,N19,N18,N17 each50/50; original each0/50. Current fixed-candidate long addition total200/200. Plan remaining passes: all remaining addition lengths, all720 native cases, fresh English KL.
Frozen sweep passes28–31: N16,N15,N14,N13 each50/50, original0/50. Total N13–20 =400/400. Staged tensors still equal submitted backup. 19 passes left: twelve addition lengths, six120-case native batches, one fresh English test.

Frozen sweep passes24–43 complete: edited addition1000/1000, original6/1000. N1–8 edited400/400 vs6/400; N9–20 edited600/600 vs0/600. Every N has50/50 edited. No weights changed. Remaining7 passes: all720 native in6 batches, then English KL.
Frozen native sweep passes44–49: copy240/240, reverse240/240, max239/240 for BOTH models. All720 outputs identical; zero losses/rescues. Pre-existing max_n10_008 error preserved. Final English sample:32 windows of240 chars at offsets1024+6000*i, no content screening.
