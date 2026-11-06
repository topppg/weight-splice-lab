# Weight-edit notebook
Goal: add LSD-first addition while preserving English and native digit tasks.
Constraints: no training, no local model execution; closed-form construction only.
Current edit: none. Baseline measurements pending.
Baseline pass 1: addition 0/16; native 12/12 (small stratified sample). Outputs frequently copy operands.
Pass 2 diagnostic: held-out mid5 linear digit readouts a=35.65%, b=16.09%, next digit=13.48%; current-token mid0 decoding=100%. Excluded example49 due request transcription error. No edits yet.
Stage0 constructed: two exact-embedding/RoPE prefix heads (A at position1, colon by position2), 32 gated token/carrier neurons. Marker amplitude0.3 per head; code base scale100. No arithmetic yet; validation pending.

Pass4: stage0 KL per addition=[3.3409,3.5997,3.6102,4.1889,4.5629]; all five off-task KL round to zero. Mixed KL is not an off-task measure.

Stage1: constructed 20 RoPE lag heads, lags2..21, with BOS score-floor fallback and zero BOS values. Only heads staged; length MLP/arithmetic not yet installed. BOS separator max non-BOS=0.2093 on recorded data.

Pass5 off-task KL on five controls: [2.9628335340703416e-08, 2.997737169607717e-09, 3.8549814007637906e-08, 3.92925292302948e-09, 1.5348641682066955e-05]. Length detection diagnostics captured for all N=1..20.

Diagnosis: lag heads have correct carrier/RoPE keys, but BOS query includes 20*wbg; this background contribution varies inversely with task-code scale. Trying coefficient1, no other change.

Pass6: reducing BOS background query coefficient20->1 fixed all 20 length probes: true colon probability1, true signal499–505, maxwrong13. Controls KL printed0. Added40 layer1 neurons for length-at-plus and one layer2 broadcast head; pending validation.

Pass7 broadcast failed: selected position1; controls KL3.27–3.76. Key background noise exceeded plus key. Correcting key by -100*(wbg2 with zero BOS response); values already zero at BOS.
Length gate diagnosis: carrier4952–6523, plus246–326, true lag~500, false<15. Replaced incorrect gates with 100*(lag+2plus-[.14,.16]carrier)-500wbg plateau; strengthened broadcast background rejection to300.

Pass8: five controls KL rounds0; broadcast selects plus correctly for all N1..20. Correct length amplitude1191–1565, wrong components up to473, especially short N. Need sharpen before variable-lag retrieval.
Diagnosis: false lengths originate at non-plus query positions (e.g. '=' detects colon lag2N+2), not broadcast selection. Strengthened layer1 gates from lag+2plus-[.14,.16]carrier to lag+6plus-[.34,.36]carrier.
Pass9: strict plus gates fixed spurious lengths (wrong<=16, correct1191–1565); five controls KL0. Added layer2 refresh (62 neurons, new carrier1e6, digit/length50000) and four layer3 variable-lag heads for a_i,b_i,a_(i-1),b_(i-1), reading numeric digit plus reference. Pending test.

Pass10: all tested operand attention destinations correct, but max digit error9 and controls KL0.72–2.87. Causes: refresh digit gates clipped by background term; retrieval BOS fallback defeated by noisy length queries. Shifted digit gates by +25 carrier; changed retrieval BOS background query to1000 times a full-circuit-orthogonal LS constant readout. Pending validation.

Pass11: controls KL <=4.49e-5 on five texts, but digit errors up to14 on long inputs. All inspected retrieval positions correct. Refresh lower threshold admits false digits; replacing digit gates with 1000*digit-[15,35]*carrier-1000*wbgmid2.

Pass13: prefix attention remains exact; marker .589-.615. Repeated-9 failure comes from constant decoder rising to1.13, not token decoder (~1.02). Replaced layer0 token gate centers by1000*(token+10/3 marker-2.5 ref), preserving plateau widths/outputs. Latest controls KL max .0331; refresh leakage remains to fix.

Pass14: operand digit errors <0.059 on 8 addition probes including N20 zero/9 stress tests. Controls KL [0,.0028,.0001,0,.0331]; no arithmetic installed yet. Added params987648. Plan carry=(previous operand sum - previous emitted digit > .5), then modulo10 of current sum+carry.

Refresh control leakage identified: digit gates max557 vs bg rejection1000; increased rejection to3000. Added22 MLP3 neurons for stable current digit and unit, using threshold pairs. Pending pass15.

Pass15: controls KL rounds0; digit max errors .0074–.0167, carry-decision errors <=.0235 on eight addition probes. Arithmetic reference ~1000; fresh carrier~0.5–0.9 million. Carry threshold chosen4.5 (noncarry diff -1/0; carry9/10).
Added2 MLP4 neurons: carry threshold4.5, paired plateau, target amplitude1000; pending validation.
Pass16: carry error<=.0052; five controls KL0. Added40 MLP5 staircase neurons for sum modulo10, pseudoinverse output directions. Background60 shifts sums -.25 to-.44; thresholds offset+.35. Output gain1e8. Pending greedy test.

Edit13 saved: complete circuit, 1036800 added parameters; all native entries unchanged. Greedy validation pending.
Pass17: addition20/20 (one per N), native11/12 vs original12/12; max_n12_000 emits7 instead of newline. Investigating final output leakage.

Pass18-20: broader English controls exposed severe gate leakage (pass20 mixed KL2.175); old background references negative on ordinary residuals. Constructed new null-circuit least-squares references from original captures2,3,19,20. Replacing background terms only; length suppression500->5000. Pending paired validation.

Pass21: English KL1.95–3.81, max native KL.173. Pass22 localized leakage to attention1/2, not stage0 MLP (all control gates <=-56). Staged scale separation: added outputs x1000 except prefix heads; downstream MLP background and attention fallback components x100; removed output +.35 threshold compensation. No base entries changed.

Pass23: scale separation failed; controls KL3.91–4.21, broadcast residual up to4.1e6. Found stale broadcast query negative on English (-6.94 minimum), reversing key ordering. Restored pre-scaling snapshot and rebuilt attention1/2/3 fallback queries with wider null-circuit backgrounds; broadcast key rebuilt. Pending validation.

Pass24 fallback repair: off-task KL includes max .948 English and .173 native. Broadcast BOS key10 has insufficient margin; short additions fail, long stress additions correct. Need separate background suppression from addition scale.
Staged margin-scale repair: added writers x1000 except prefix heads; MLP background rejection x10; fallback query backgrounds x10; broadcast BOS10->50, queryx100; remove old +.35 output threshold offset. Snapshot work/margin_snapshot.pt.

Pass25: margin scaling failed (controls KL3.65–4.18). BOS value errors cascade: L1 3.6e-8, L2 6.5e-5, L3 .616. Restored margin_snapshot and nulled added value rows at original BOS, preserving circuit directions. Pending broadcast margin repair/test.
Unscaled repair: broadcast BOS key10->50, queryx100. Recorded valid plus keys94–95, control keys<=9.13; query positive on diagnostic controls. Original-BOS value nulls installed L1–3.

Pass26: N1/N2 teacher-forced answers failed; N20 stress answers correct. Native M control KL .179/.134; English controls 0 to .000591. Layer1 length gates at plus negative for short inputs due bg5000; staged restore500. Output native gates positive up to102325; staged bg600000->1200000 and threshold compensation+.35->+.70.

Edit19 saved: length suppression restored500; output suppression1.2e6/compensation+.70. All native parameter entries unchanged. Diagnostic readouts require dividing residual-space directions by RMSNorm weights. Greedy validation next.

Pass27 edit19: addition35/60 (58.3%); native36/36 vs original36/36, no losses. Many addition errors +1; investigating output threshold compensation.
