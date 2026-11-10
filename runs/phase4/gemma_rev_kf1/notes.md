# Direct-weight reversed addition
Initial inspection; no edits staged or evaluated yet.
Hypothesis: use added capacity for a format-gated arithmetic circuit, preserving pretrained pathways. Need to establish operand addressing, carry computation, and negligible off-task interference.

Baseline pass 1: reversed 0/15; plain 4/9. No edits yet.

Candidate 1: one layer-0 neuron detects exact BOS-R, constrained against newline-R and word-R; injects coordinate400 with post-norm scale 1e5. Native down row compensated. Pending off-task check.

Edit1 pass2 KL per text (prose, numbers, chat, plain, R): [0.005297606810927391, 0.0010017844615504146, 0.015973564237356186, 0.03625260666012764, 2.9707133769989014]
Diagnosis: coordinate400 carries substantial BOS raw output; replace with minimum measured joint mean+BOS raw-energy coordinate 118. No score-based search.

Edit2 pass3 KL: [0.002931297291070223, 1.7244209615796535e-08, 1.1199743994438904e-06, 7.027909276757782e-08, 54.95880889892578]
Added layer1 BOS-fallback broadcast head, output coordinate 460; pending paired check.

Layer1 scalar digit probe: 99.72% rounding accuracy over 9270 synthetic task states; replacing with one-hot readouts and explicit thresholding.

Prefix+broadcast+initialization constructed. Layer1 one-hot probes: 0 threshold errors over 9270 states (true min .8405, false max .0635). Sparse 96-coordinate output directions analytically verified greedy for all ten digits and newline. Internal C=1e15; no arithmetic stages yet.

Growth-helper integrity issue: new nn.Parameters had requires_grad=True despite disabled autograd; no backward/optimization used. sync now calls requires_grad_(False) immediately after growth. Counter and integer-rounding layer2 constructed; paired check pending.

Pass5 preservation KL mean .000207, max text .000541 after counter and rotary compensation. Stage3 length threshold bug (ramps centered on integers) diagnosed from N-0.5 error and corrected by -0.5 threshold shift.

Completed arithmetic stages: layer4 pair sums, two layer6 offset heads and sum rounding, layer7 carry from previous raw sum minus previous output digit. Teacher-forced examples N=1,3,8,20,30 all next digits correct, integer deviations <=.002. Added gated layer17 sparse 96-coordinate output and exact-length newline. Pending full greedy and preservation checks.

Edit5 reference pass6: reversed 40/40 correct, N1..20 two each. Plain controls collapsed from 6/16 original to 0/16 edited. Candidate is NOT acceptable; tracing first off-task divergence before repair.

Off-task failure diagnosed: BOS states identical but dense value projection residuals ~1e-9..1e-6 amplified in attention output rows. Added four always-BOS reference heads (layers2,3,4,6) subtracting identical value projections; output coefficients changed 100->128 for exact power-of-two arithmetic. Pending paired checks.

Pass7 edit6 KL [R1,R3,plain3,plain1,prose,numbers,chat]=[32.67,32.39,2.585,18.13,.0121,.9334,.3817]. Reference heads cancel exactly but MLP row compensation changes RMS denominators; layer3 native output delta RMS4.3, layer17 34.7. Final guards also fail (45/12355 negative). Derived diagonal native raw-energy correction by ridge linear solve on 12355 original states; inverse post gains preserve numerators. Refit layer7/17 off-task guards with minimum calibration margins .243/.485. Applied, pending reference check.

Edit7: softened expanded prefix constraint fixes observed Markdown/non-R false positives; all controls in pass8 KL=[32.66461944580078, 31.922576904296875, 0.000818486965727061, 0.00014525640290230513, 0.00035411521093919873, 0.00020265653438400477, 0.007544910069555044, 0.00013728940393775702].

Edit7 kernel greedy: reversed 1000/1000 (50/50 each N1..20). Plain original43/160 edited45/160, losses0 rescues2. Saved snapshot work/best_edit7.
wiki unused-practice token-weighted KL=0.0003866367199312496 over 6012 positions.
chat unused-practice token-weighted KL=0.04779372704960216 over 2768 positions.

Edit7 broad KL: wiki .000387; 40 chats .04779 with final guard sign failures. Spectral truncation (49 dominant modes) fixes signs but broad test chat .06212, numbers .11838, traced to undersized layer2/3/17 penalties. Full original-state scan 67424 positions: all guards positive (min .04187,.44494,.23969,.32933,.23708,.07294), prefix off-gate max -198.48. Set each penalty >=2x full-practice activation bound. Pending reference validation.
Bounded candidate chat KL=0.005238615077434206.
Bounded candidate numbers KL=0.00029543202290316996.
Bounded candidate reversed=1000/1000, plain=45/160.

Edit8 independent random+carry/zero tests: {'1-8': [160, 160], '9-20': [240, 240], '21-30': [121, 200]}. Snapshot work/best_edit8; 7,365,760 added parameters.

Long-length failure diagnosis: arithmetic digits correct, counter low by ~1 at position95 due to ~2.8e-4 unintended guard subtraction. Closed-form constant-feature offset 9.018188590013911e-13 removed from layer2 guard; fresh results {'1-8': 160, '9-20': 240, '21-30': 200}. Pending paired reference check.

Saved edit9 reference passes17-24: reversed80/80 N1-8; plain controls40, no correctness losses, one incorrect output changed.
Edit9 full kernel: reversed 1000/1000; plain 45/160 vs original43/160; losses 0.
chat KL=0.005238615077434206 over 14594 positions.
wiki KL=0.000390047195092213 over 6012 positions.
numbers KL=0.00029543202290316996 over 6504 positions.

Edit9 reference passes17-36: reversed200/200 N1-20. New-seed independent validation 1000/1000 N1-30, includes exhaustive 100 one-digit pairs. Weights unchanged.

Edit9 reference custom passes37-46 token-verified: 40/40; no weight changes. Added parameters 7365760.
