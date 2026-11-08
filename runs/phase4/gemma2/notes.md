# Weight-edit experiment

## Constraints
ADD_CAPACITY; at most 10M added parameters. No embedding/final-norm edits, no training or iterative fitting, and forward execution only through the supplied tool.

## Status
Inspecting implementation and tokenisation. No weights changed or measurements taken.

## Baseline pass 1
Seven custom examples: 0+0, 9+9, 99+99, and 893+952 correct; 1234+5678 and 12345678+87654321 copy operands, 99999+11111 emits excessive zeros. No edits. 49 passes remain.

## Pass 2: correct-prefix diagnostics
{"resid_mid.11": {"heldout_next_digit_operands_carry": [0.324, 0.452, 0.355, 0.753], "next_digit_by_n": {"1": 0.579, "2": 0.448, "3": 0.462, "4": 0.364, "5": 0.352, "6": 0.254, "7": 0.206, "8": 0.26}}, "resid_mid.15": {"heldout_next_digit_operands_carry": [0.488, 0.645, 0.645, 0.879], "next_digit_by_n": {"1": 0.789, "2": 0.759, "3": 0.718, "4": 0.523, "5": 0.574, "6": 0.39, "7": 0.412, "8": 0.26}}, "resid_mid.17": {"heldout_next_digit_operands_carry": [0.486, 0.676, 0.64, 0.887], "next_digit_by_n": {"1": 0.842, "2": 0.759, "3": 0.795, "4": 0.523, "5": 0.574, "6": 0.339, "7": 0.382, "8": 0.26}}, "resid_final": {"heldout_next_digit_operands_carry": [0.499, 0.661, 0.643, 0.882], "next_digit_by_n": {"1": 0.895, "2": 0.862, "3": 0.769, "4": 0.568, "5": 0.574, "6": 0.356, "7": 0.397, "8": 0.234}}}

Pass 3 captures early residuals on 64 additions N1–8, 10 controls, and 4 longer additions. Investigating token/position features for an arithmetic side circuit; no edits yet.

## Candidate A: layer-0 position bookkeeping
Added one uniform head (655360 parameters), with exact symbol/BOS value decoders and two low-variance output directions; output scale .001. Staged only, not yet saved or evaluated.
Recomputed covariance in float64 before saving to avoid cancellation artifacts.

Pass 4 (both): bookkeeping head has KL rounded to zero on 4 Wikipedia excerpts and 2 chat replies. Position-ratio analysis follows in run 8. No arithmetic correction yet.

Pass 4 detailed: control KL approximately +/-1e-8 (numerical noise). Scale .001 gives channel amplitudes ~1e-4 and unacceptable position errors (up to 24.9 at N8). Staged engineering change to scale 1.0 to overcome ~1e-6 background/float32 contamination.

Pass 5: scale-1 bookkeeping head mean KL .0003, maximum text KL .002674. Position and token-code diagnostics printed in run 10. Still no arithmetic correction.

## Candidate B: algebraic token readouts and position buffers
Exclude zero MLP-norm coordinate 400. Derive exact token/reference readouts by nulling the weight-space support of native numeric attention outputs. Add 240 layer-0 neurons forming paired reference/value buffers for positions 0–29, using four GELU step terms per channel. Staged; not yet measured. This initially targets N1–8.

Pass 6 (both): mean KL .000336, max text .000924 for position buffer. Pass 7 greedy paired regression: both 10/32 correct, zero rescues/losses, one continuation changed. Buffer diagnostics printed in run 13.

Buffer diagnosis: exact token ratios work (max error 9e-6), but four-step gates leak and native background changed after bookkeeping rotation. Disable buffer output; use 128-coordinate sparse bookkeeping directions disjoint from token readouts. Next recapture background, then use three-term triangular gates.

Pass 8: isolated sparse head mean KL .000536, max text .009248. Reinstall buffer using 180 active triangular-gate neurons (240 allocated), scale 5, with directions suppressing incoming residual and native MLP contribution separately.

Pass 9 triangular buffer mean KL .001563, max text .009301. Numerical buffer and raw-prefix gating diagnostics in run 17.

Quiet-buffer approach fails fresh examples (code errors up to 13.9). New design: explicit physical buffer coordinates, post-MLP gain 1e5 with inverse native down-row compensation, no BOS output, and a closed-form off-numeric detector constrained to vanish on the exact numeric support. Requires immediate off-task validation.

Pass 10 strong-buffer overall KL 30.58 (includes intentionally disrupted arithmetic states). Control-specific results and buffer accuracy must be inspected; no best-edit claim.

Strong-buffer candidate rejected: pass 10 off-task KL 1.5–12 on prose/chat, 15–23 on number-initial controls. Large gain amplified cancellation artifacts including BOS. Restore original norm; reserve a low-energy raw-output subspace by orthogonally projecting native down weights out of it. New dual readouts annihilate numeric incoming support exactly. Low-amplitude triangular buffers restored, no off-detector gating.

Pass 11 protected-buffer overall KL .035153, max text .204985; separate buffer and control diagnostics in run 21. Need preservation-aware direction selection.

Pass 11: exact protected buffer succeeds on fresh numeric codes (max error .0011) but chat KL .135/.205 and Wikipedia .044/.050 fail preservation. Reduce to 40 directions (19 prompt slots plus local position pair), choose directions by captured control activation covariance with .01 weight-covariance regularisation, and reduce gate slope to 1e5.

Pass 12 revised 40-direction buffer: mean KL .003362, max text .022395 across 12 additions and 6 controls. No arithmetic correction yet; readout diagnostics in run 25.

Edit 8 backed up to work/buffer_edit8. Pass12 code error <=.0014, inactive leakage <=.049 and local position error <=.636: position gates too soft. Staged 65-direction workspace, 1e6 gate slope, equality marker and 16-projection squared nonnumeric detector. No arithmetic output yet.

Staged layer1 uniform buffer collector (scale1000), protected attention output, and 494 threshold-pair decoder neurons (500 allocated). Decodes 18 position codes, reference, snapped current position, equals latch and prefix detector into 23 reserved channels. Native projections require preservation test.

Pass13 edit9 FAILED preservation: mean KL3.817802, max15.398016. Need diagnose collection amplitude and decoder before proceeding.

Pass13 diagnostics: arithmetic KL .015-.247 but controls KL3.47-15.40; layer1 collection RMS changes 13-25 on controls. Decoder numeric reference ~7e-6 leads code errors3-17. Failed edit9 backed up, staging reset. Pivot to one-shot final-MLP closed-form correction with explicit control constraints, avoiding invasive early-layer machinery.

Baseline pass14 captures final-layer states on24 controls (prose12, chat8, number-rich4). Numeric answer calibration has1565 positions. Final MLP input/final residual RMS medians1055/1464, post-MLP gain median337. Closed-form task-boundary diagnostics in run32.

Passes15/16 captured512 synthetic additions; pass17 captured32 controls and verified feature-only KL=0. One closed-form ridge solve staged4608-neuron final-MLP down projection. Target12-nat correct-class contrast minus original captured digit/terminator contrast, calibrated by captured residual/native-MLP norms; controls constrained to zero with20x aggregate weight. Added params8,847,360. No iterative fitting or native edits.

Pass18 final correction FAILED greedy: original33/130=.2538, edited15/130=.1154, rescues3/losses21. Request repeated2 N5 items; distinct128. Pass19 mixed calibration/unseen/control diagnostic captures available; detailed fit-vs-actual comparison in run36. No successful arithmetic edit yet.

Pass19 diagnosis: calibration examples correct, unseen examples often wrong; fitted contrast predictions match actual edited top digits. Failure is generalization, not output-sign error. Fresh control KL prose ~0-.000234; three chats .0111/.0583/.1080. Failed final regression backed up. Testing structured operand/carry arithmetic instead of learned answer mapping.

Pass20 position diagnostics: layer5 held-out shared operand-column accuracy.9931; operand/nonoperand1.0; query first-operand column1.0 (72 positions), token1.0. Reset failed final predictor. Staged8 zero-output layer5 heads using orthogonal low-RoPE-frequency column codes, current+7 lookahead columns, BOS fallback, and closed-form digit value readout. Head outputs not yet connected.

Pass21 zero-output retrieval KL0. Actual retrieval fails some columns, especially repeated digits. Pair-mass median .99993 but 10th percentile <1e-6; smaller operand mass median .083, often too small to decode both. Missing-column outputs <=.00224. Controls BOS attention mean .989-.998 but rare large leakage. Do not assume arithmetic-capable heads from classifier accuracy alone.

Native layer5 head0 mean aligned a/b attention .144/.108 versus other heads <=.033. Staged half-strength current-column augmentation through head0's token-conditioned native output directions. Only first added head connected; lookahead heads remain zero-output. Must validate paired accuracy/KL.

Pass22 retrieval augmentation: original21/80, edited20/80; rescues1/losses2. Pass23 control KL mean.000017 max.000119. Preservation good but no arithmetic gain. Backed up to retrieval_augmentation_edit13 and reset staging. Next fallback: complete100-case N1 table with exact closed-form interpolation and tight length/task gating; does not solve multi-digit task.

Pass24 complete N1 capture:7 incorrect first-answer states (0+6,0+7,0+8,0+9,1+5,2+5,5+2). Six covariance-adjusted prototype gates have negative-state margins .337-.518 across16662 reference states and all245 N1 answer states;5+2 has a collision and is excluded. Staged6 zero-output final-MLP neurons; no native edits.

Seventh detector collision is2+5, same target7; included after excluding same-target N1 states. Pass25 zero-output features KL0. Staged seven correct-digit output boosts (nominal8 nats), all native weights unchanged;13,440 added parameters.

Pass26 all100 N1 prompts: original93/100, edited100/100 under exact token grading;only7 outputs changed. Candidate edit15 backed up to work/n1_patch_edit15. This is only an N1 repair, not a multi-digit solution.

Pass27 edit15 fresh controls (4 prose,4 chats,2 number-rich):KL0 on all10 texts. Structured final-residual operand/carry probe rejected without output edit:heldout next-digit accuracy N1..8=.692,.634,.561,.368,.364,.308,.227,.194. New experiment:separate operand retrieval heads at final attention, with outputs initially zero.

Pass29 separate layer17 retrieval: heldout operand accuracy roughly.88-1.0;both retrieved digits correct on every observed position where both max values>.75 (coverage .125-.55 byN). One heldout caffeine chat leaks head value1.095;other5 controls~5e-8. Staged modest .5-norm output vectors in20 low-variance final-residual directions, no native projections changed;preservation must be checked.

Pass30 transport-only edit17:meanKL7.9e-5,maxtext.000939. Direct channel background measured in run61. Staged scale2 transport in a new quiet subspace constrained orthogonal to N1 gate and carry/task/reference readouts;native weights unchanged.

Pass31 lookup rejected before output connection:active precision N2..8=.50,.111,.667,.308,1,1,1;multiple incompatible activations and chat peak35.25. Transport control KL maximum.010096;aggregateKL dominated by intended2+5 repair1.4149. Investigating carry-agnostic pair candidate constraint instead of defective carry/task AND.

Absolute pair gate rejected before output:50/63 saturated activations contain correct digit;control margin positive. Diagnosis:transported digit magnitude varies with residual/attention normalization, so one strong operand can trigger unrelated pairs. Added shared constant reference channel through same attention normalization;new scale-relative pair gates staged with zero decoder outputs. Constant carrier preservation unverified.

Pass32 reference transport controlKL=.0026-.0182. Pair feature containment N2..8=.8,.333,1,.909,1,1,1 with modestcoverage;one chat still activates. Staged conservative four-nat equal boost to the two carry possibilities for a paired end-to-end test;not yet a best candidate.

Pass33 edit20 paired greedy: original19/85, edited21/85;3 rescues/1 loss. N1 .9->1,N2 .8->.7,N4 0->.2;N5+ zero. Pair branch has only a small uncertain gain, not a solution.

Pass34 edit20 connected pair boost controls:meanKL .008012,maxtext .011481 on4 prose/4chat/4numeric controls. Backed up work/pair_edit20;provisional only, paired greedy net+2/85.

Pass35 edit20 larger greedy:45/196->49/196,6 rescues/2 losses;N3 .5->.6,N1 .9->1,noN5+ successes. Staged fixed soft carry split:each candidate's four-nat boost multiplied by1 +/- .5*(carry1-carry0),using existing one-shot carry readout.No new fitting or coefficient sweep.

Pass36 carry-aware edit21:19/85->25/85,7rescues/1loss;N4 0->.4,N3 .2->.3,N2 unchanged .8,N1 .9->1,N5+zero. Pass37 controls meanKL.008061,max.011481. Backed up work/carry_pair_edit21. Investigating pure scale-relative gate to remove unbounded task-score interference.

Pure-ratio pair gate rejected without staging:more long-length coverage but control margins +.814 and+4.329 on two chats. Retain edit21 unchanged;begin broader actual Wikipedia/chat preservation validation.

Pass38 real controls edit21:overallKL .028333,maxtext .116719;Wiki tokenmean 0.022797,chat reply-only tokenmean 0.017726. Category means determine preservation, not individual worst text.

Pass39 edit21 N1–4 remaining40each:89/160->99/160,13rescues/3losses. Pass40 N5–8 remaining40each:0/160 both. This is not a multi-digit solution. Last structural investigation:closed-form minimum-variance dual readout for transported channels, before final validation.

Single covariance-weighted dual readout constructed with exact signal constraint;numeric backgroundRMS .000335->.000268,referenceRMS .000348->.000364. Staged readout-only variant;output amplitudes/carry rule/native weights unchanged. Must compare paired generation and controls.

Pass41 dual variant25/85=same as edit21 but8rescues/2losses vs7/1. Pass42 real-controlKL .039776 vs .028333, rejected. Restored carryG of edit21. Complete400 practice N1–8:original108/400=.27,edit21 124/400=.31;N1 50/50,N2 37/50,N3 23/50,N4 14/50,N5–8 all0. DOES NOT MEET95% BAR. Remaining budget for final candidate validation.

Pass43 final-candidate N9–20:0/60 correct both original/edited (5 perN),only1 output changed. Prepared independent random N21–30 and N1–8 validation sets;no weight use.

Pass44 N21–30 custom20:exact grades printed inrun83;outputs mostly copy operands. Pass45 chat addition16:all outputs unchanged,no transfer benefit;strict answer-only grades printed inrun83.

Pass46 additional6wiki/6chat:overallKL.041257;wiki 0.042526,reply-chat 0.062392. Pooled38+46 wiki0.034648,chat0.035410. Number-rich windows selected solely by digit counts.

Pass47 digit-dense Wiki sample meanKL.015461,maxtext.067483 (two excerpts' apostrophe spacing normalized in request). Pass48 exhaustiveN1 exact grades inrun85. Two final passes reserved for fresh short arithmetic and expanded chat preservation.

Pass48 exhaustiveN1 original93/100,finalcandidate99/100;3+4 regressed,all7 original failures repaired. Pass49 fresh64:N1–8 original16/64,edited19/64. No weights modified fromedit23.
