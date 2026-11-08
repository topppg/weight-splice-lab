# Experiment notebook

## Status
- No weights changed or saved; no forward passes requested.
- First step: inspect implementation, then measure original target and off-task behavior.
- Candidate approach: hand-built, format-gated circuitry in added capacity; feasibility untested.
- Constraints: no training or iterative fitting; embeddings and final norm remain unchanged.

## Implementation
- Added heads share q/k norms; added head and MLP outputs enter shared post-block RMSNorms.

## Baseline
- Pass 1: 25 reversed examples, 0% correct; 8 plain examples, 37.5% correct.
- Pass 2: residual captures on 12 diagnostic texts; no edits.
- Investigating a closed-form late correction before a more complicated explicit circuit.

## Localized one-digit construction
- Pass 3: all 100 one-digit pairs, three answer prediction positions each.
- Pass 4: 30 controls (Wikipedia, actual chat replies, normal additions, longer reversed additions).
- Constructing covariance-weighted interpolating detectors by one closed-form solve; no weights staged yet.

## Edit A (staged, untested)
- Added 299 neurons to MLP 17; original parameters untouched.
- Similarity gates isolate exhaustive one-digit answer states; one nonseparable first-digit state omitted.
- Output directions computed by pseudoinverse and a quadratic normalization-budget equation.
- Linear shortlisted-rival check: 300/300 feasible; actual inference still required.

## Edit A results
- Pass 5: one-digit reversed accuracy 50/50 (original 2/50), 48 rescues, no losses across 86 problems.
- Longer reversed additions still fail. Plain accuracy 6/16 for both models.
- Pass 6: mixed target/off-task KL captured; category-specific analysis follows.
- Backup: work/edit_A.
- Pass 6 category analysis: all 14 off-task texts KL=0 at reported precision; chat reply-only KL=0; plain outputs unchanged.
- Current best: Edit A, safe one-digit component only; does not meet the N=1..8 bar.

## Multi-digit investigation
- Pass 7: 144 independently generated problems (96 N=2, eight each N=3..8), captures at layers 4/10/16/17.
- Diagnostic probes split by whole problem (75% fit, 25% test); no new weights staged.

## Retrieval diagnostic (staged)
- Pass 8: token identity at layer 17 is 95.8% on a held-out split; numeric RMS error 1.32.
- Four added heads at layer 17, offsets 5/2/6/3 for N=2; q/k use hand-designed rotary vectors.
- V is a single closed-form token-identity readout. New output columns remain zero, so heads are diagnostic only.

## Retrieval diagnostic results
- Pass 9: all four heads retrieved correct offsets on 64 positions each; minimum mass 0.999628.
- Retrieved digit argmax 100% in this 16-problem diagnostic batch; numeric values remain noisy.
- Off-task KL zero on the three contrasts.
- Staged low-rank output connections in 48 low-variance residual directions, fixed nominal scale 300; not yet tested.
- Pass 10: connected-head perturbation and per-category KL captured; analyzing signal calibration before any nonlinear readout.

## Structured N=2 circuit (staged)
- Pass 11: 160 N=2 problems; 120 used for closed-form maps, 40 held out diagnostically.
- Held-out current-digit class accuracies 100%/98.3%; carry accuracy 95.8%.
- Added 203 hand-written conjunction neurons implementing digit-pair/carry addition, final carry, and termination.
- Off-task rejection thresholds taken from captured negatives; actual generation and KL pending.

## Retrieval approach rejected
- Pass 12: N=2 3/50 vs original 1/50; N=1 regression 5/10; plain 1/16 vs original 6/16.
- Five lost successes; approach not acceptable. Backup work/failed_retrieval.
- Restaged safe Edit A (299 neurons only), pending save.
- Testing output-geometry feasibility of a more isolated circuit before making further edits.

## Isolated-circuit feasibility
- A 64-coordinate output subspace was selected once from digit information and native row-energy criteria.
- Testing exact class-contrast output vectors and an early R-prefix detector before building a conditional circuit.

## Isolated branch stage 1 (staged)
- One prefix-marker neuron in MLP 0 and paired broadcast heads in attention 1.
- Used only readable (nonzero normalization-scale) coordinates after a finite-value validation failure.
- One post-MLP and one post-attention coordinate boosted with native row compensation.
- No arithmetic branch yet; preservation and marker broadcast untested.
- Save rejected globally compensated output matrix; removed global compensation factors, retaining only coordinate-specific compensation and actual added connections. No inference yet.

## Prefix branch repair
- Pass 13 failed: off-task KL up to 21.55; unstable interpolated constant caused widespread marker firing. Old marker coordinate also had native normalized magnitude 25.27.
- Pass 14 broad original controls used for one regularized closed-form detector; captured negative similarity <=0.880. Rebuilt on quiet coordinates; pending test.
- Pass 15: repaired detector fires only at initial R on all 12 texts. Broadcast target magnitude 13k–25k but off-task leakage 3–15 causes KL 0.008–3.24. Reduced broadcast connection 100-fold; pending preservation check.
- Pass 16 reduced broadcast: prose/number controls KL <=0.000020, chat <=0.000562. Token readout train/held-out accuracy 100%, true score >=0.800, wrong <=0.139.
- Added gated continuous 12-class token encoder plus constant in MLP1 and diagnostic-only uniform head in attention2. Pending test.
- Pass 17: independent target token encoding 100%; reciprocal-position relative error <0.000027. Six controls KL <=0.003771, no encoder activation.
- Connected count/reference carriers in attention2; added marker-selecting reference head. Length estimation and off-task leakage pending.
- Pass 19: discrete token codes correct, max one-hot error 1.2e-6; controls KL <=0.0352. Native attention3 shifts count by ~-1.2, causing N8 rounding errors.
- Reserved count carriers through attention3, added plus-only ratio amplification in MLP3 and fixed trapezoidal N=1..30 decoder in MLP4; pending test.
- Pass 20: all ten N<=8 prompts decode length correctly; N20 correct, N30->29. Control amplifier off but length-decoder cancellation leaks, KL up to37.8. Replaced rejection threshold by captured maximum plus20 to keep every hinge strictly negative off task; explicitly inhibit amplifier at R marker.

- Broad fitted control constant failed positivity (min -0.224); not staged. Replaced it with directly signed readable coordinate -97, positive on all 991 captured controls. Prefix detector moved to colon: captured target gate +20, negatives <=-19.89. Pending test.
- Pass23: token encoding and length decoding 10/10, no MLP4 activation on eight ordinary controls, but KL 0.017–0.114. Staged earlier plus amplification in MLP2, restored attention3 rows and MLP3 native weights, weakened length Q/K tenfold and connected paired length broadcast.
- Pass25: broadcast length argmax correct at every captured answer position, min margin 720, but ordinary-control mean KL 0.178. Halved Q/K and staged prefix-gated length copy at MLP5; no arithmetic yet.

- Pass29 tests quiet length-five/operand-carrier repair with explicit operand reference copies; arithmetic output not installed. See run77 for operand margins and control KL.

- Native-state N2 format gate rejected: held-out positives overlap normal additions. New attempt stays in added MLP17 neurons only, using a fixed control-excluding similarity dictionary and one planned closed-form output solve. New down columns remain zero during feature capture.
- Passes32/33 captured 2,260 localized features on 320 N2 problems. One dual ridge solve uses 240 problems; 80 reserved for diagnostic validation. Output directions computed from frozen class contrasts and the RMS normalization budget. Native weights unchanged.
- Pass34: localized ridge readout N1 50/50, N2 11/50 vs original1/50, but plain1/32 vs11/32. Rejected for preservation; backup failed_elm. Restaging safe Edit A. Need broader plain controls and more conservative gates.
- Pass35 revealed original Edit A plain KL0.1376 on all160 correct-answer texts. Recalibrated raw caps against enlarged controls; pass36 candidate N1 49/50, N2 6/50, plain11/32 unchanged accuracy, zero lost original successes. Backup conservative_best. Full bar still far out of reach; prioritizing preservation and honest partial results.
- Pass37 exhaustive N1/fresh N2/plain generation completed; run97 grades token-exact results. Candidate contains only1235 added MLP17 neurons (2,371,200 params), no native changes.

- Pass37: exhaustive N1 100/100 vs8/100 original; fresh N2 5/80 vs0/80; fresh plain17/64 both, one changed output. Pass38: 24 prose/chat/number controls KL0 at tool precision. Backup validated_partial. Trying one independently sampled fixed N2 dictionary extension, no training.
- Pass39–41 captured 672 new N2 construction problems. Replaced old N2 dictionary in one closed-form construction using 912 total construction problems, retaining 3457 gates; N1 copied unchanged. Pending validation.
- Large extension rejected: pass42 N1 100/100, fresh N2 4/80 (smaller5/80), plain17/64 both; practice N2 7/50 with one original success lost. Pass43 found chat KL2.78964 on Boeing747 reply, caused solely by shared N1 termination detector for R:9+7=61 (source293), not N2. Restored smaller1235-neuron candidate and zeroed that detector's down column; pending tests.
- Saved edit27 disables leaking termination column3281; pass44 exact grading recorded in run110. Remaining passes focus on preservation and honest final scope, not fitting.
- Pass45 all54 controls KL0. Pass46 plain43/160 both but7 continuations changed;15/16 chat continuations changed, including incorrect arithmetic before termination. Pass47 original-trajectory audit KLmean0.573; staged direct pruning of 7 active added outputs (7 N1, 0 N2), no refitting. Target and KL rechecks pending.
- Edit28 pass48: N1 99/100 vs8/100 original; freshN2 5/80 vs0/80; freshplain17/64 both,1 changed continuation;16/16 chat additions full48-token outputs identical. Pass49 all41 audit/fresh prose/chat/number controls KL0 at tool precision. Final stratified target+allplain pass pending. Full task bar NOT met.

## Final tested edit28 (submitting unchanged)
{
  "added_parameters": 2371200,
  "added_neurons": 1235,
  "nonzero_added_output_columns": 1228,
  "N1_exhaustive": {
    "original": 8,
    "edited": 99,
    "count": 100
  },
  "N2_fresh": {
    "original": 0,
    "edited": 5,
    "count": 80
  },
  "N1_practice": {
    "original": 2,
    "edited": 50,
    "count": 50
  },
  "N2_practice": {
    "original": 1,
    "edited": 6,
    "count": 50
  },
  "N3_to_20_practice": {
    "original": 0,
    "edited": 0,
    "count": 180
  },
  "mean_accuracy_N1_to_8_stratified": 0.14,
  "plain_practice": {
    "original": 43,
    "edited": 43,
    "count": 160,
    "changed_full_48_token_outputs": 0
  },
  "N21_to_30": {
    "original": 0,
    "edited": 0,
    "count": 10
  },
  "chat_addition_full_output_matches": 16,
  "chat_addition_count": 16,
  "final_control_49_max_text_KL": 0.0,
  "task_bar_met": false
}
Construction: closed-form localized MLP17 detectors and output directions; seven leaking N1 output columns disabled. Native tensors unchanged; embed/final norm untouched. Larger dictionary and early circuits rejected. Backup work/final_partial. Does NOT meet requested multi-digit accuracy bar.
