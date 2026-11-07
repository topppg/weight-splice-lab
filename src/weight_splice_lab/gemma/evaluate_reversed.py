from functools import partial

from weight_splice_lab.gemma import evaluate

# the hidden eval of the reversed-digit task (family gemma_rev): same eval, task="reversed"
evaluate_edit = partial(evaluate.evaluate_edit, task="reversed")
base_reference = partial(evaluate.base_reference, task="reversed")
eval_key = partial(evaluate.eval_key, task="reversed")
