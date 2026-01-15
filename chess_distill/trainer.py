import torch
import torch.nn.functional as F
from transformers import Trainer


class ChessDistillTrainer(Trainer):
    def __init__(self, *args, move_token_ids=None, **kwargs):
        if not move_token_ids:
            raise ValueError("move_token_ids must be provided.")
        super().__init__(*args, **kwargs)
        self._move_token_ids = torch.tensor(move_token_ids, dtype=torch.long)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items()}
        move_probs = inputs.pop("move_probs")
        outputs = model(**inputs, use_cache=False)

        logits = outputs.logits
        last_logits = logits[:, -1, :]
        move_ids = self._move_token_ids.to(last_logits.device)
        move_logits = last_logits.index_select(dim=-1, index=move_ids)

        log_probs = F.log_softmax(move_logits, dim=-1)
        target = move_probs.to(log_probs.device)
        row_sums = target.sum(dim=-1)
        valid = row_sums > 0
        if valid.any():
            target = target[valid] / row_sums[valid].unsqueeze(-1)
            log_probs = log_probs[valid]
            loss = F.kl_div(log_probs, target, reduction="batchmean")
        else:
            loss = log_probs.sum() * 0

        return (loss, outputs) if return_outputs else loss
