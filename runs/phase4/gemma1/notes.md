# Addition weight-edit notebook
Initial inspection underway. No edits staged or saved; no forward passes used.
Goal: isolated arithmetic capacity with closed-form/hand-designed weights, validated against addition and off-task KL.

Baseline pass 1: 83 examples, N1 90%, N2 80%, N3 20%, N4–8 0%; three longer failures. Pass 2 captured late residuals and MLP features on arithmetic and controls. No edits yet.
Pass 3: 192 independently generated N1–8 correct teacher-forced examples; testing fixed-ridge probes, no weights changed.
Fixed-ridge diagnostic probes on pass 3: held-out token accuracy at N8 only 24–29%; base 18.5%. Simple late readout is inadequate. Next hypothesis: added attention with closed-form queries for operand offsets; initially zero output, measuring retrieval only.
Pass 4 captures used for two zero-output diagnostic heads at L17. Queries fitted once to analytical rotary offset targets; values fitted to digit one-hot. 12/16 problems per N used, 4/16 excluded. Added params 1,310,720. Not yet saved.
Pass 5 both: zero-output heads give exactly zero KL on arithmetic, text, and chat. Retrieval summary:
            position_correct   mass  digit_correct
fit   head                                        
False 0                0.279  0.261          0.315
      1                0.388  0.357          0.418
True  0                1.000  0.950          1.000
      1                1.000  0.949          1.000
Retrieval heads rejected and reset: excluded position accuracy 27.9%/38.8%, despite 100% on fitted cases. No active behavioural edit. Next test is explicitly partial: exhaustive N1 corrections with localized added-neuron gates.
N1 localized repair staged: 7 neurons; covariance-weighted gates with captured negatives <= -15, target >=15. Output directions analytically rotate the post-MLP normalized vector to improve correct-vs-original-top margin by 3 nats. No iterative fitting. Corrections: [('0+6=6', 4, 1.707, 0.052), ('0+7=7', 4, 2.228, 0.099), ('0+8=8', 4, 2.449, 0.107), ('0+9=9', 4, 5.713, 0.153), ('1+5=6', 4, 0.449, 0.031), ('2+5=7', 4, 0.602, 0.102), ('5+2=7', 4, 0.324, 0.095)]
Pre-save numerical correction: 2+5 and 5+2 share one gate/output because their states are nearly identical. Six active neurons, seventh zero. Checking bfloat16 margins before save.
Pass 7 generation: practice N1 sample 90% ->100%, other sampled lengths unchanged; however custom 0+6 and 0+7 remain wrong (changed to different digits). Pairwise margin construction is insufficient; next solve will constrain all observed competitors simultaneously.
Pass 8 independent controls: KL reported 0.0; added neuron max activation 0.0. Output directions replaced by a closed-form simultaneous margin/sphere-intersection solve against all digits plus original top20; gates unchanged.
Pass 9 revised simultaneous-output repair: exhaustive custom N1 results printed; sampled N2–20 unchanged. Backup at work/best_single. Diagnostic scalar N/index/shift probes being checked before further circuit work.
Confirmed best partial edit: exhaustive N1 100/100 vs original 93/100. Scalar length probe held-out MAE .665, rounded accuracy .576; index .709, shift .715. No viable multi-digit circuit established; requested >=95% N1–8 bar NOT met. Final preservation and long-length checks underway.

## Final result
{
  "status": "partial; requested multi-digit bar not met",
  "n1_exhaustive_original": 0.93,
  "n1_exhaustive_edited": 1.0,
  "final_sample_accuracy_by_n": {
    "1": 1.0,
    "2": 0.6,
    "3": 0.6,
    "4": 0.5,
    "5": 0.0,
    "6": 0.0,
    "7": 0.0,
    "8": 0.0,
    "9": 0.0,
    "10": 0.0,
    "11": 0.0,
    "12": 0.0,
    "13": 0.0,
    "14": 0.0,
    "15": 0.0,
    "16": 0.0,
    "17": 0.0,
    "18": 0.0,
    "19": 0.0,
    "20": 0.0
  },
  "final_sample_mean_n1_to_n8": 0.3375,
  "final_sample_changed_outputs": [
    [
      "0+8=",
      1
    ],
    [
      "0+7=",
      1
    ]
  ],
  "preservation": {
    "number_rich": {
      "tokens": 640,
      "mean_kl": 0.0,
      "max_kl": 0.0
    },
    "chat_replies": {
      "tokens": 494,
      "mean_kl": 0.0,
      "max_kl": 0.0
    },
    "ordinary_wikipedia": {
      "tokens": 480,
      "mean_kl": 0.0,
      "max_kl": 0.0
    },
    "max_added_activation": 0.0
  },
  "added_parameters": 13440,
  "active_added_neurons": 6,
  "allocated_added_neurons": 7,
  "native_weights_changed": false,
  "lengths_21_to_30_tested": false
}
Final edit: only added gate/up/down columns/rows at L17; six active and one zero neuron. No added heads retained. One-shot covariance-weighted gates and analytic multi-competitor margin/sphere solves. Exhaustive single-digit repair succeeded; multi-digit task failed. Best saved edit backed up in work/best_single. Do not claim the 95% bar was met.
