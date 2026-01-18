import logging

import torch
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)


class ChessDistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        move_token_ids=None,
        move_token_strs=None,
        log_pred_steps=0,
        log_pred_topk=5,
        **kwargs,
    ):
        if not move_token_ids:
            raise ValueError("move_token_ids must be provided.")
        super().__init__(*args, **kwargs)
        self._move_token_ids = torch.tensor(move_token_ids, dtype=torch.long)
        self._last_loss_log_step = None
        self._last_pred_log_step = None
        self._warned_no_valid = False
        self._warned_nan_loss = False
        self._log_pred_steps = max(int(log_pred_steps), 0)
        self._log_pred_topk = max(int(log_pred_topk), 1)

        unique_ids = len(set(move_token_ids))
        if unique_ids != len(move_token_ids):
            logger.warning(
                "move_token_ids contains duplicates: %d duplicates.",
                len(move_token_ids) - unique_ids,
            )
        if move_token_ids:
            logger.info(
                "Initialized trainer with %d move_token_ids (min=%d max=%d).",
                len(move_token_ids),
                min(move_token_ids),
                max(move_token_ids),
            )

        if move_token_strs is not None:
            self._move_token_strs = list(move_token_strs)
        elif self.tokenizer is not None:
            self._move_token_strs = self.tokenizer.convert_ids_to_tokens(move_token_ids)
        else:
            self._move_token_strs = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items()}
        move_probs = inputs.pop("move_probs")
        legal_mask = inputs.pop("legal_mask", None)
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
            if not self._warned_no_valid:
                self._warned_no_valid = True
                logger.warning("All move_probs rows are zero; loss is forced to 0.")

        if torch.isnan(loss).any() and not self._warned_nan_loss:
            self._warned_nan_loss = True
            logger.error("Loss is NaN; check inputs and logits.")

        log_every = getattr(self.args, "logging_steps", 0)
        step = int(getattr(self.state, "global_step", 0))
        if log_every and step % log_every == 0 and self._last_loss_log_step != step:
            self._last_loss_log_step = step
            with torch.no_grad():
                batch_size = int(move_probs.size(0))
                valid_count = int(valid.sum().item())
                zero_rows = batch_size - valid_count
                row_sum_mean = float(row_sums.mean().item())
                move_mean = float(move_logits.mean().item())
                move_std = float(move_logits.std().item())
                max_target = (
                    float(target.max().item()) if valid.any() else 0.0
                )
                metrics = self._compute_batch_metrics(
                    move_logits,
                    move_probs,
                    row_sums,
                    valid,
                    legal_mask,
                )
                if metrics:
                    self.log(metrics)
                    summary = [
                        f"top1_match_rate={metrics.get('metrics/top1_match_rate', 0.0):.4f}",
                        f"pred_top1_prob={metrics.get('metrics/pred_top1_prob', 0.0):.4f}",
                        f"target_top1_prob={metrics.get('metrics/target_top1_prob', 0.0):.4f}",
                        f"pred_entropy={metrics.get('metrics/pred_entropy', 0.0):.4f}",
                        f"target_entropy={metrics.get('metrics/target_entropy', 0.0):.4f}",
                    ]
                    if "metrics/legal_top3_mean" in metrics:
                        summary.append(
                            f"legal_top3_mean={metrics.get('metrics/legal_top3_mean', 0.0):.2f}"
                        )
                        summary.append(
                            f"legal_top1_rate={metrics.get('metrics/legal_top1_rate', 0.0):.4f}"
                        )
                    logger.info("Step %d metrics: %s.", step, " ".join(summary))
            logger.info(
                "Step %d loss=%.6f batch=%d valid=%d zero_rows=%d "
                "row_sum_mean=%.4f move_logits_mean=%.4f move_logits_std=%.4f "
                "max_target=%.4f.",
                step,
                float(loss.detach().item()),
                batch_size,
                valid_count,
                zero_rows,
                row_sum_mean,
                move_mean,
                move_std,
                max_target,
            )

        if (
            self._log_pred_steps
            and step % self._log_pred_steps == 0
            and self._last_pred_log_step != step
        ):
            self._last_pred_log_step = step
            self._log_predictions(inputs, move_logits, move_probs, row_sums, valid, step)

        return (loss, outputs) if return_outputs else loss

    def _compute_batch_metrics(self, move_logits, move_probs, row_sums, valid, legal_mask):
        batch_size = int(move_probs.size(0))
        valid_count = int(valid.sum().item())
        if batch_size == 0:
            return {}

        with torch.no_grad():
            eps = 1e-8
            pred_probs = torch.softmax(move_logits, dim=-1)
            topk = min(3, pred_probs.size(-1))
            pred_top1_prob, pred_top1_idx = pred_probs.max(dim=-1)
            pred_top3_vals, pred_top3_idx = pred_probs.topk(k=topk, dim=-1)

            target_norm = torch.zeros_like(move_probs)
            if valid.any():
                target_norm[valid] = move_probs[valid] / row_sums[valid].unsqueeze(-1)
            target_top1_prob, target_top1_idx = target_norm.max(dim=-1)
            target_top3_vals, _ = target_norm.topk(k=topk, dim=-1)

            top1_match = (pred_top1_idx == target_top1_idx) & valid
            top1_match_rate = (
                float(top1_match.sum().item()) / valid_count if valid_count else 0.0
            )

            pred_entropy = -(
                pred_probs * (pred_probs + eps).log()
            ).sum(dim=-1)
            target_entropy = -(
                target_norm * (target_norm + eps).log()
            ).sum(dim=-1)

            if valid.any():
                pred_entropy = float(pred_entropy[valid].mean().item())
                target_entropy = float(target_entropy[valid].mean().item())
                pred_top1_prob = float(pred_top1_prob[valid].mean().item())
                target_top1_prob = float(target_top1_prob[valid].mean().item())
                pred_top3_sum = float(pred_top3_vals[valid].sum(dim=-1).mean().item())
                target_top3_sum = float(target_top3_vals[valid].sum(dim=-1).mean().item())
                target_nonzero_mean = float(
                    (target_norm[valid] > 0).sum(dim=-1).float().mean().item()
                )
            else:
                pred_entropy = 0.0
                target_entropy = 0.0
                pred_top1_prob = 0.0
                target_top1_prob = 0.0
                pred_top3_sum = 0.0
                target_top3_sum = 0.0
                target_nonzero_mean = 0.0

            metrics = {
                "metrics/valid_row_rate": float(valid_count) / batch_size,
                "metrics/top1_match_rate": top1_match_rate,
                "metrics/pred_top1_prob": pred_top1_prob,
                "metrics/target_top1_prob": target_top1_prob,
                "metrics/pred_top3_sum": pred_top3_sum,
                "metrics/target_top3_sum": target_top3_sum,
                "metrics/pred_entropy": pred_entropy,
                "metrics/target_entropy": target_entropy,
                "metrics/target_nonzero_mean": target_nonzero_mean,
            }

            if legal_mask is not None:
                legal_mask = legal_mask.to(pred_probs.device)
                legal_top3 = legal_mask.gather(dim=-1, index=pred_top3_idx)
                legal_top3_count = legal_top3.sum(dim=-1)
                legal_top1 = legal_mask.gather(
                    dim=-1, index=pred_top1_idx.unsqueeze(-1)
                ).squeeze(-1)
                metrics.update(
                    {
                        "metrics/legal_top3_mean": float(
                            legal_top3_count.mean().item()
                        ),
                        "metrics/legal_top3_any_rate": float(
                            (legal_top3_count > 0).float().mean().item()
                        ),
                        "metrics/legal_top1_rate": float(legal_top1.mean().item()),
                        "metrics/legal_count_mean": float(
                            legal_mask.sum(dim=-1).mean().item()
                        ),
                    }
                )

        return metrics

    def _move_label(self, index):
        if self._move_token_strs and index < len(self._move_token_strs):
            token = self._move_token_strs[index]
            if token.startswith("<") and token.endswith(">"):
                return token[1:-1]
            return token
        return str(index)

    def _log_predictions(self, inputs, move_logits, move_probs, row_sums, valid, step):
        if not inputs or self.tokenizer is None:
            logger.warning("Tokenizer unavailable; skipping prediction logging.")
            return

        with torch.no_grad():
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask[0].to(torch.bool)
                input_ids = input_ids[mask]
            input_ids = input_ids.detach().cpu().tolist()
            decoded = self.tokenizer.decode(
                input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            probs = torch.softmax(move_logits[0], dim=-1)
            topk = min(self._log_pred_topk, probs.numel())
            pred_vals, pred_idx = torch.topk(probs, k=topk)

            target_row = move_probs[0]
            row_sum = float(row_sums[0].item())
            if row_sum > 0:
                target_row = target_row / row_sum
            target_topk = min(self._log_pred_topk, target_row.numel())
            target_vals, target_idx = torch.topk(target_row, k=target_topk)

            pred_moves = [
                f"{self._move_label(int(i))}={float(v):.4f}"
                for v, i in zip(pred_vals.tolist(), pred_idx.tolist())
            ]
            target_moves = [
                f"{self._move_label(int(i))}={float(v):.4f}"
                for v, i in zip(target_vals.tolist(), target_idx.tolist())
            ]
            logger.info("Step %d example: prompt=%s", step, decoded)
            logger.info(
                "Step %d predicted top%d: %s",
                step,
                topk,
                ", ".join(pred_moves),
            )
            logger.info(
                "Step %d target top%d (sum=%.4f valid=%s): %s",
                step,
                target_topk,
                row_sum,
                bool(valid[0].item()),
                ", ".join(target_moves),
            )
