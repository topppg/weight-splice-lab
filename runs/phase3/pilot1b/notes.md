# Direct-weight addition experiment
## Status
- No weights changed; no forward passes used.
- Inspecting model architecture and practice data before baseline measurements.
- Plan: compare addition, native tasks, and text on identical inputs for original and edited models.
- Current best edit: none.

## Baseline
- Forward 1: addition 0/21 across N=1..20; custom copy/reverse/max examples correct.
- Addition often resembles operand concatenation.
- Forward 2: captured attention and residuals for four addition, three native, one text example.
- Investigating an A:-specific early trigger and conditional symbolic arithmetic capacity; no edit yet.
- Forward 3: early residual captures for 40 addition, 24 native, 16 English texts (80 total). Computing closed-form token/context probes; still no edits.

## Interface edit (not yet tested)
- Layer-0 token probe: 100% on diagnostic batch; prefix scores positive .9474, negative max .1225.
- Reserve 35 embedding-null residual directions chosen by low captured background energy; project all original attention/MLP output matrices away from them.
- Add three layer-0 fixed-offset heads (0,1,2), copying exact token properties at scale .1.
- Add one prefix-marker neuron for colon preceded by A and BOS two positions back.
- No arithmetic implemented yet. Native preservation remains an empirical question.
- Edit 1 tests: forward 4 native 36/36 original and edited, no losses; addition 0/20 both.
- Forward 5 English-only mean KL 0.0180518; per-text [0.7138333320617676, 2.1407737731933594, 0.2829432487487793, 0.00010430516704218462, 2.871051401598379e-05, 2.5587136406102218e-05, 0.008501007221639156, 0.021998871117830276, 0.021040117368102074, 0.01916906051337719, 0.01717233657836914, 0.020429247990250587].
- Interface checkpoint copied to work/interface_edit.

## Arithmetic plan
- Use 20 double-peak fixed-offset heads for operand sums; select length from equals-position; infer carry from previous sum minus last answer digit.
- Added layer-1 marker broadcast (anchor=10000) and uniform marker average (position=.1/(p+1)); pending test.

- Forward 6 completed: assessing marker broadcast and rotary routing; off-task KL [0.00010431301780045033, 2.8714055588352494e-05, 2.557936568337027e-05, 0.008500996977090836, 0.017027052119374275].
- Increased position signal from .1/(p+1) to 100/(p+1) to avoid anchor roundoff. Added gated-digit neuron, 20 constrained symmetric operand heads, and equals-position retrieval. Pending routing test.

## Forward 7 routing diagnosis
- N=1,3,8 max sum errors .001,.053,.161; N=20,19 errors 1.00,2.61: double-peak balance needs repair.
- Off-task anchor max <.00022; native KL ~.000104,.000029,.000026; two text KL .008501,.017027.
- Replaced routing Q/K by constrained bounded Fourier kernels (one closed-form solve per N), peak scores zero and nonpeak margin 16; pending forward test.

- Forward 8 bounded routing tested; length selector added as 80 deterministic SwiGLU ramp neurons; pending test.

- Diagnosed forward 8: all operand peaks off by one (N+1,2N+2 instead of N,2N+1), with zero peaks competing with prefix keys. Corrected phase and added +40 low-frequency score pedestal; length selector also staged.

- Forward 9: routed sums max errors .005,.010,.009,.007,.010 digits for N=1,3,8,20,19. Selected-sum errors .007,.064,.235,.082,.109. Off-task KL unchanged. Enlarged position signal 10x and staged previous-sum head plus two carry ramp neurons.

## Decoder construction
- Forward 10: combined total max errors .002,.026,.040,.028,.080 for N=1,3,8,20,19. Off-task KL unchanged on five controls.
- Layer 3 writes carry directly into sum slot 31; slot 33 is equals-position and slot 34 is gated digit.
- Added 80 layer-5 plateau neurons for total 0..19 -> digit modulo 10. Closed-form output directions using fixed embedding/final norm; suppressed off-task by const-versus-anchor guard. Pending test.
- Forward 11 greedy: all five custom additions have correct N+1 digits; all three native controls have correct answer+newline. Addition does not yet terminate with newline. Saved backup work/decoder_v1.

## Forward 12 broader comparison
- Addition 98/100 (40/40 N1-8; 58/60 N9-20); failures add_n10_001 and add_n10_003.
- Native 72/72 both, zero losses. Current checkpoint decoder_v1 remains best.
- Forward 13: N10 operand peak weights vary ~.44-.56 instead of .5; routed sums biased before carry. Suspect large cancelling rotary coefficients. Controls KL unchanged (.0001 native; .0085/.0170 English).
- Staged N10 paired-head replacement: offsets 10 and 21 in two single-peak heads, each half old output scale; +32768 parameters. Pending test.

## Forward 14
- Edit 8: N10 50/50 versus original 0/50; three native custom outputs unchanged.
- Best saved edit: work/best_n10 (1,043,456 added parameters).

## Late workspace restoration (staged)
- Protect only [0,1,6,9,31,32] at layer3 attention; [0,9,31] at subsequent outputs; restore final native down projection completely. Derived from actual decoder/carry read directions, not fitted.

## Edit 9 tests
- Forward 15: addition 100/100 (5 per N1..20), native 9/9 both.
- Forward 16: mixed-batch KL .007517; English per-text KL [0.00629061134532094, 0.012294274754822254, 0.014461447484791279, 0.010382182896137238, 0.00996073056012392, 0.014143818989396095]; English mean 0.01125551.
- Best checkpoint: work/best_late_restore. Late restoration retained arithmetic and reduced text KL.

## Restart inspection
- Edit 9 remains saved; 1,043,456 added parameters. Investigating signal magnitudes before any further restoration. No new edits or forwards yet.
- Staged lifetime-based restoration: release prefix scratch after layer0 MLP, marker after layer1 attention, position/gated digit after layer2 attention, candidate sums after layer2 MLP. All live values remain projected. Pending tests.
- Edit 10 / forward 17: additions 100/100 (indices005..009, all lengths), native36/36 both, zero losses. Snapshot work/lifetime_restore. English KL comparison pending.
- Forward18 English mean KL 0.00868004; mixed KL .005801. Lifetime checkpoint remains tested best.
- Staged 24 layer1 addition-gated clearing neurons for slots11..34, freeing those directions in layer0 attention/MLP and layer1 attention. Uses existing gated-digit gate and scale, no fitting. Pending tests.

- Edit11 / forward19 conditional clearing FAILED: addition24/40 (N1-8 all failed, N9-20 all passed), all three native custom controls corrupted. Reject and restore lifetime checkpoint; unprotected scratch likely activates off-task circuitry.

- Edit12 restores lifetime checkpoint exactly. Forward20 recovers40/40 additions on edit11 failure inputs and all3 native control answers+newline. Investigating one-shot least-squares final residual correction, gated OFF for addition; no correction weights yet.

- Staged256 final-MLP correction feature neurons (zero down columns), gate=(1000*const-anchor)/norm and identity/norm up rows. Intended off-task only. Will capture real hidden features and paired final residual differences, then solve once by ridge linear algebra; no iterative fitting.

- Forward21 captured correction features on11 English,24 native,2 addition texts. Single ridge solve (relative ridge .001; native row weight2) staged final residual correction; no iterative fitting. Validation pending.

- Edit14 correction rejected: forward22 English KL worsened .00868004 -> .01203653 (mixed .005801 -> .008039). Forward23 addition20/20 and native9/9 retained. Zeroed correction down columns to restore lifetime behavior; no further correction fitting.

- Edit15 saved: rejected correction down columns zero; lifetime circuit restored. 1,240,064 added parameters including inactive correction capacity. Broad validation beginning.
- Edit15 forwards24..28: N1..5, indices010..029, addition100/100; paired native90/90 both, zero losses. Snapshot work/validated_edit15. No weights changed during validation.

- Restart after forward31: latest N8 validation20/20 additions, native18/18 both, zero losses. Inspecting coverage; no changes.

- Edit15 forwards24..35: additions240/240 (20 each N1..12), native216/216 both, zero losses. No weights changed. Longer lengths and broader text checks pending.

- Consolidated edit15 forwards24..43: {('addition', '1-8'): [160, 160, 0], ('copy', 'all'): [120, 120, 0], ('reverse', 'all'): [120, 120, 0], ('max', 'all'): [120, 120, 0], ('addition', '9-20'): [240, 240, 0]}. Added parameters 1,240,064. Answer digits validated; addition newline termination remains unsupported. Seven forward passes remain.

- Forward44: 8 English excerpts, mean KL .008299, BPC original1.6346 edited1.6440. Forwards45..48: another400/400 additions and72/72 native controls. Unique validation24..48: {'addition': {'n': 800, 'edited_correct': 800, 'original_correct': 4, 'losses': 0}, 'copy': {'n': 120, 'edited_correct': 120, 'original_correct': 120, 'losses': 0}, 'reverse': {'n': 120, 'edited_correct': 120, 'original_correct': 120, 'losses': 0}, 'max': {'n': 120, 'edited_correct': 120, 'original_correct': 120, 'losses': 0}}. No weights changed.

- Forward49 structured edge-case grading: {"original": {"addition_correct": 4, "addition_total": 30, "failures": [{"prompt": "A:9+9=", "expected": "81", "got": "99"}, {"prompt": "A:9+1=", "expected": "01", "got": "91"}, {"prompt": "A:1+9=", "expected": "01", "got": "19"}, {"prompt": "A:5+4=", "expected": "90", "got": "54"}, {"prompt": "A:99+99=", "expected": "891", "got": "999"}, {"prompt": "A:99+10=", "expected": "001", "got": "099"}, {"prompt": "A:10+99=", "expected": "001", "got": "190"}, {"prompt": "A:55+44=", "expected": "990", "got": "455"}, {"prompt": "A:99999999+99999999=", "expected": "899999991", "got": "999999999"}, {"prompt": "A:99999999+10000000=", "expected": "000000001", "got": "999990090"}, {"prompt": "A:10000000+99999999=", "expected": "000000001", "got": "000099090"}, {"prompt": "A:55555555+44444444=", "expected": "999999990", "got": "555545555"}, {"prompt": "A:9999999999+9999999999=", "expected": "89999999991", "got": "99999999999"}, {"prompt": "A:9999999999+1000000000=", "expected": "00000000001", "got": "99990000000"}, {"prompt": "A:1000000000+9999999999=", "expected": "00000000001", "got": "00009999009"}, {"prompt": "A:5555555555+4444444444=", "expected": "99999999990", "got": "55555444554"}, {"prompt": "A:0000000000000000000+0000000000000000000=", "expected": "00000000000000000000", "got": "00000000000\nC:0000=0"}, {"prompt": "A:9999999999999999999+9999999999999999999=", "expected": "89999999999999999991", "got": "99999999999\nM:1,1=1\n"}, {"prompt": "A:9999999999999999999+1000000000000000000=", "expected": "00000000000000000001", "got": " = = \n The series wa"}, {"prompt": "A:1000000000000000000+9999999999999999999=", "expected": "00000000000000000001", "got": "990999999999\nM:1,1=1"}, {"prompt": "A:5555555555555555555+4444444444444444444=", "expected": "99999999999999999990", "got": "54544444444\nM:1,1=1\n"}, {"prompt": "A:00000000000000000000+00000000000000000000=", "expected": "000000000000000000000", "got": "00000000000\nC:0000=00"}, {"prompt": "A:99999999999999999999+99999999999999999999=", "expected": "899999999999999999991", "got": "99999999999\nM:1,1=1\nM"}, {"prompt": "A:99999999999999999999+10000000000000000000=", "expected": "000000000000000000001", "got": "000000000000\nC:055=05"}, {"prompt": "A:10000000000000000000+99999999999999999999=", "expected": "000000000000000000001", "got": "999999999999\nM:1,1=1\n"}, {"prompt": "A:55555555555555555555+44444444444444444444=", "expected": "999999999999999999990", "got": "444444444444\nM:1,1=1\n"}], "native": [["C:000000000000=", true], ["R:123456789012=", true], ["M:999999999998,999999999999=", false]]}, "edited": {"addition_correct": 30, "addition_total": 30, "failures": [], "native": [["C:000000000000=", true], ["R:123456789012=", true], ["M:999999999998,999999999999=", false]]}}. No changes. Final English check and submission pending.
