# Weight-edit notebook
No edits yet. Inspecting architecture and practice data before baseline measurement.
Constraints: closed-form or hand-set edits only; no model execution in Python; embedding and final norm frozen; architecture fixed.

Baseline custom prompts (pass 1): 2+3 -> '233'; 9+8 -> '988'; A:398+259= -> '3982299'. No reliable addition; copy 12345678 succeeds.

Pass 2 routing: A:398+259, L4H1 A alignment .905, L1H3 B .850. N8 best A .587, best B .236; N12 best A .264, best B .159. Matched N8 max L4H1 correct operand alignment .959. Plan: test closed-form decodability before selecting arithmetic edit.

Pass 3: captured 214 sequences: exhaustive 100 one-digit additions, 66 additions N2..12, 36 native, 12 English chunks. No edits yet.

Linear residual probes failed broad arithmetic recovery (heldout sum rounding 3–18%). Candidate 1: one closed-form, preservation-regularized rank<=11 update to blocks.5.mlp.down_proj.weight; all other weights unchanged. Uses captured final MLP features, addition digit/newline targets, native/English/prefix zero-change constraints. Awaiting forward validation.

Candidate 1 rejected: pass4 N1 10% vs original12%; no N>=2 successes; copy 66.7%, reverse83.3%, max41.7% vs100% each (12/task). Pass5 English mean KL 0.006557. Staging reset to original.

Candidate 2 scaffold: replace 200 lowest-impact last-MLP gate/up rows with geometric one-digit-state detectors, initially zero output columns. One closed-form reconstruction compensates removed hidden features using 568 retained neurons. Exact-interpolation gates rejected before editing due universal off-task activation. Global-offset geometric gates had only0.38% any-positive preservation positions in initial analysis.

Pass6 scaffold matched mean KL .000467 (139 sequences). Lookup feature condition 9.66e+04. Candidate2 completed by one linear output solve on 200 one-digit answer states; raises the correct tied-output score by an eight-logit margin. Awaiting greedy validation.

Candidate2 lookup output solve rejected before save: feature condition9.66e4, output norm1.40e6 and enormous predicted leakage. Saved edit remains zero-output scaffold. Testing redesigned geometric detectors with separate exact offsets for units and carry phases.

Prototype lookup redesign also rejected before save (native9.4%/English4.3% false activations). New candidate3: 36-neuron subspace-gated one-digit lookup. Separate null-space band detectors for100 units states and100 carry states, closed-form linear answer readouts, fixed triangular SiLU construction width.002/sharpness32. Retained732 neurons reconstruct removed features in one closed-form projection.

CURRENT BEST candidate3: pass7 N1 50/50 vs original6/50; native36/36 unchanged; N2..20 sample0/38. 44 rescues,0 losses. Pass8 English mean KL 0.025697, native mean KL 0.000010. Backup work/best_one_digit/ and best_one_digit_tensors.pt. This is exhaustive one-digit lookup, NOT multi-digit generalization.

Pass9 original routing data: 80 additions (4 per N1..20),36native,6English chunks. Testing closed-form RoPE shift readouts on held-out sequences before any attention edits.

Routing probes rejected before edits: heldout length MAE L3 1.19,L4 .97,L5 .94 digits; high-frequency phase errors too large for precise digit selection. Next: diagnose fallback English collisions; consider a bounded two-digit arithmetic decoder.

Pass10 N2 probes: final-layer heldout A/B/carry joint100% first digit,79.2% second; L4 second81.2%. This supports a bounded N2 arithmetic-table experiment. Candidate4 changes only fallback gate directions: null-space regression to positive preservation scores, negated in SiLU gates. >99.8% recorded preservation positions now have negative central preactivations; readouts and compensation unchanged.

CURRENT BEST candidate4: pass11 N1 50/50,native36/36,0losses. Pass12 English mean KL 0.00160105 on8prose texts. Saved work/best_directed/ and best_directed_tensors.pt.

B+carry categorical factorization rejected: heldout second-position accuracy41.7%. Candidate5: 200 hand-table neurons, gate on decoded(A,carry)+two-digit-task marker, up projects decodedB; output digit=(A+B+carry)%10. Wcat/Wtask each obtained in one closed-form solve from192 randomN2 examples, native/English/other-length states constrain task detector. Existing36-neuron N1 circuit retained;532 original neurons compensate236 removed features.

Candidate5 pass13: N2 17/50=34% vsoriginal0%; N1 10/10; native copy10/12,reverse12/12,max9/12 (5losses). Not accepted as best due native regressions. Need matched English measurement and inspect whether off-task feature directions are separable.

Pass14 candidate5 English meanKL 0.35896486043930054; mixed batchKL .523715. Diagnosing arithmetic leakage versus compensation.

Candidate6: signed N2-vs-preservation task contrast; expanded preservation with pass14; category ridge includes .02 preservation covariance; gates use categoryA+carry+8signed-4offset. Same 200-neuron table and compensation. Awaiting matched tests.

Candidate6 pass15: English meanKL 0.0216292; 8 native-text meanKL 0.14125; awaiting greedy tests.

Candidate6 rejected: pass16 N2 7/50 (14%), native32/36 (4losses), N1 10/10. Signed task variation interferes with category conjunction; investigating compact scalar circuit and near-null gating.

Scalar-sum alternative rejected: heldout N2 rounding accuracy 18.8/16.7/31.3% by phase. Pass17 collects 40 paired A/M-style additions N2..20 to assess native routing reuse.

Max-prefix routing rejected: pass17 heldout joint A/B/c<=10.7%; max formatting improves selected operand, not both. Testing phase-separated N2 circuits with constrained linear readouts and nullspace gates.

Candidate7: phase-separated bounded Fourier N2 circuit (222 neurons), 36-neuron N1 fallback retained. Exact arithmetic identity uses18 bilinear terms percarry; 4SiLU finite differences gate each product within task band. Phase1 separatecarry0/1; phase2 two carry readouts. Phase heldout joint100/87.5/97.9%. Output rawboost 293.769; widened compensation data. Awaiting tests.

Candidate7 rejected: pass18 N2 20/50(40%), N1 8/10, native4/36,32losses. Pass19 EnglishKL 1.232661. Diagnosing phase subcircuits; best remains directed N1-only candidate4.

Candidate8: bounded Fourier circuit with per-phase preservation-regularized readouts (.1 negative covariance, ridge1), global offset instead of phase-specific extrapolating offset. Candidate7 readout B norm off-task6–10 caused huge collisions. Same topology/compensation/output scale.

Candidate8 rejected: pass20 N2 0/50,N1 10/10,native34/36(2reverse losses). Pass21 nativeKL 0.00195016, EnglishKL 0.0150286. Examining earlier-layer independent task marker before abandoning N2 extension.

Candidate9 scaffold:9 layer4 neurons emit3 N2-phase markers in tied-output-null low-variance directions; phase normals are exact nulls of192 targetstates; width.15, peak24raw. One-shot reconstruction of9 removedfeatures; N1fallback retained; noN2arithmetic yet. Must measure propagation effects and preservation before proceeding.

Marker scaffold safety bands set from all recordedN1 states: widths[0.04148371246641725, 0.059384597956620905, 0.028333714441784015]; 45% of minimum observed N1 distance, capped.15. No arithmetic output installed.

Marker scaffold pass23 native36/36 unchanged but N1drops tobaseline6/50, likely tiny upstream changes invalidate exact narrow lookup gates. Pass22 EnglishKL 0.0016494, nativeKL 7.26379e-06. Must recalibrate N1 if proceeding; marker usefulness under inspection.

Marker phase0/1 increments14–28 on independentN2; phase2 undercovered. Restoring width.15 for all3markers, then recalibrating downstream N1/N2 from actual scaffold states. N1 normalization shifts.18–.34 invalidate old exactgates despite low off-taskKL.

Candidate11: marker-aware222-neuron N2 Fourier circuit, phase readouts ridge1+.001negative covariance; newphase nullband width1/16. N1lookup36neurons recalibratedexactly frompass25. Retained510neurons one-shot reconstruct originalresidual outputs fromactualscaffoldfeatures (also cancels marker influence atoutput). Pending greedy/native/English validation.

Candidate11 rejected: pass26 N1 50/50,N2 8/50,native32/36; pass27 EnglishKL 0.298796, nativeKL 0.0353833. Restoring candidate4 for broader preservation checks.

Pass28 directed N1 baseline meanKL.001789 on24wiki+12native. Reconstruction dominates residual error (.365 EnglishRMS versus lookup.013). Candidate12 compact lookup: 3Dicosahedral digit scores plus1Dcarry, 12neurons; exact readouts and directed triangle gates; one closed-form reconstruction on previous captures. Pending validation.

CURRENT BEST compact12: pass29 N1 50/50,native36/36,0losses. Pass30 EnglishKL.000187575,nativeKL.000002695. Saved best_compact12_tensors.pt and best_compact12/. Investigating first-harmonic N2 arithmetic to reduce replacement count.

Candidate13: compact N2 first-harmonic circuit,51neurons +12N1. Four products percarry; exact minimum-preservation-energy operand/carry readouts; phase nullbands width.04, global offset; reconstruction onHselect. No earlier-layer edits.

Candidate13 rejected: pass31 N1 10/10,N2 4/50,native17/36 (19losses); pass32 EnglishKL0.446229,nativeKL0.611095. Restored compact12. Remaining passes allocated to comprehensive audit; no general multidigit solution achieved.

Edit15 submitted compact12 as partial solution. Pass33 exhaustive100 N1 ordered pairs collected; auditing720 native cases by length next, plus longer-addition and English batches. All audit uses unchanged compact12 unless explicitly noted.

Audit pass33 exhaustiveN1 100/100 vsoriginal10/100. Pass34-38 nativeN1..5 all300/300unchanged. Pass39 N6native59/60vs60/60: copy_n06_016 expected381635 got681635; first regression requires diagnosis.

Pass40 independent8Wiki EnglishKL0.0004958834. Copy failure lookup residual norm1281.72; original3/6logits[13.65476131439209, 2.2321550846099854],edited[-1.004146695137024, 5.815423011779785].

Candidate14: narrow compact12 gate bands from.002 to.0005; no other changes. copy_n06_016 failure distance.001860 from units nullspace; intendedN1 float32error<=1.5e-7. Must rerun exhaustiveN1/fullnative/English; previous candidate had1/360native loss.

Narrow candidate14 saved asedit16. Pass41 exhaustiveN1 scores checked; pass42 full240copy andpass43 full240reverse audited. Next max and English, then multi-digit audit. No multi-digit capability claimed.

CURRENT BEST narrow candidate14/edit16: pass41 exhaustiveN1 100/100 vs10/100. Pass42 copy240/240,pass43 reverse240/240,pass44 max239/240; all720native outputs identical tooriginal (719correct). Pass44 had2duplicateIDs, deduplicated. Pass45 English16x220chars meanKL.000629, BPC1.7670->1.7684. Backup work/best_narrow/ and best_narrow_tensors.pt. Multi-digit audit pending; no general addition solution.

Pass46 addition10perN1..8: N1 10/10 vs2/10; N2..8 0/70 both, all70outputs unchanged. Pass47 N9..20 0/120both,22outputs changed. Overall sampled multi-digit0/190. This is a partial N1 lookup, not the requested multidigit solution. Final3passes: independent native, English/nativeKL, finalN1 verification.

Pass48 independent native36prompts completed; exact answer-prefix scores printed inrun67, fullcontinuations differ on2cases. Final mixedKL planned with16English then36native texts.

Pass48 freshnative36/36 both;2continuationdifferences are after answernewline. Pass49 English16x200chars meanKL0.00234345743,native36 meanKL2.80495647e-06,maxnativeKL1.024826e-05. Preservation not exact despite unchanged nativegreedy scores.
