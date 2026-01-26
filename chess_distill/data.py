import json
import logging
import math

from jinja2 import Template

try:
    import chess
except ImportError:  # Optional dependency for legal move generation.
    chess = None

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def fen_to_board(fen):
    """Convert FEN string to a visual board representation."""
    parts = fen.split(' ')
    rows = parts[0].split('/')
    lines = ["  a b c d e f g h"]
    for row_idx, row in enumerate(rows):
        rank = 8 - row_idx
        line = f"{rank}"
        for char in row:
            if char in '12345678':
                line += ' .' * int(char)
            else:
                line += f" {char}"
        lines.append(line)
    return '\n'.join(lines)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_moves(record):
    moves = record.get("moves", [])
    if isinstance(moves, dict):
        for uci, value in moves.items():
            if isinstance(value, dict):
                item = {"uci": uci}
                item.update(value)
                yield item
            else:
                yield {"uci": uci, "cp": value}
    elif isinstance(moves, list):
        for item in moves:
            if isinstance(item, dict):
                if "uci" not in item:
                    continue
                yield item
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                yield {"uci": item[0], "cp": item[1]}


def _extract_score(move, mate_score):
    if "prob" in move:
        return "prob", float(move["prob"])
    if "cp" in move:
        return "cp", float(move["cp"])
    if "mate" in move:
        return "cp", math.copysign(mate_score, float(move["mate"]))
    if "score" in move:
        score = move["score"]
        if isinstance(score, dict):
            if "cp" in score:
                return "cp", float(score["cp"])
            if "mate" in score:
                return "cp", math.copysign(mate_score, float(score["mate"]))
        if isinstance(score, (int, float)):
            return "cp", float(score)
    return None, None


def _normalize_probs(values):
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def _softmax_scores(scores):
    if not scores:
        return []
    tensor = torch.tensor(scores, dtype=torch.float32)
    tensor = tensor - tensor.max()
    probs = torch.softmax(tensor, dim=0)
    return probs.tolist()


def build_move_prob_vector(
    record,
    uci_to_index,
    temperature=1.0,
    cp_scale=100.0,
    mate_score=100000.0,
    legal_move_smoothing=0.1,
):
    move_items = []
    prob_items = []
    cp_items = []

    for move in _iter_moves(record):
        uci = move.get("uci")
        if not uci:
            continue
        kind, value = _extract_score(move, mate_score)
        if kind == "prob":
            prob_items.append((uci, value))
            move_items.append((uci, "prob"))
        elif kind == "cp":
            cp_items.append((uci, value))
            move_items.append((uci, "cp"))

    labeled = None
    if prob_items and len(prob_items) == len(move_items):
        ucis, probs = zip(*prob_items)
        probs = _normalize_probs(probs)
        labeled = _vectorize_probs(ucis, probs, uci_to_index)
    elif cp_items:
        ucis, cps = zip(*cp_items)
        scaled = []
        scale = max(temperature * cp_scale, 1e-6)
        for cp in cps:
            scaled.append(cp / scale)
        probs = _softmax_scores(scaled)
        labeled = _vectorize_probs(ucis, probs, uci_to_index)
    elif prob_items:
        ucis, probs = zip(*prob_items)
        probs = _normalize_probs(probs)
        labeled = _vectorize_probs(ucis, probs, uci_to_index)
    else:
        labeled = torch.zeros(len(uci_to_index), dtype=torch.float32)

    if legal_move_smoothing <= 0:
        return labeled

    fen = record.get("fen")
    legal_vector = _legal_move_vector(fen, uci_to_index)
    if legal_vector is None:
        return labeled

    legal_mask = (legal_vector > 0).float()
    masked = labeled * legal_mask
    masked_total = masked.sum()
    if masked_total > 0:
        labeled = masked / masked_total
    else:
        labeled = masked

    if labeled.sum().item() <= 0:
        return legal_vector

    return (1.0 - legal_move_smoothing) * labeled + legal_move_smoothing * legal_vector


def _vectorize_probs(ucis, probs, uci_to_index):
    vector = torch.zeros(len(uci_to_index), dtype=torch.float32)
    for uci, prob in zip(ucis, probs):
        idx = uci_to_index.get(uci)
        if idx is None:
            continue
        vector[idx] = prob

    total = vector.sum()
    if total > 0:
        vector = vector / total
    return vector


def _legal_move_vector(fen, uci_to_index):
    if not fen:
        return None
    if chess is None:
        raise RuntimeError(
            "python-chess is required to compute legal moves. "
            "Install it with `pip install python-chess`."
        )
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    vector = torch.zeros(len(uci_to_index), dtype=torch.float32)
    for move in board.legal_moves:
        idx = uci_to_index.get(move.uci())
        if idx is not None:
            vector[idx] = 1.0

    total = vector.sum()
    if total > 0:
        vector = vector / total
    return vector


class ChessDistillDataset(Dataset):
    DEFAULT_USER_TEMPLATE = (
        "Given the current chess board:\n{{ FEN }}\n\n"
        "Choose the best legal move.\n\n"
        "Output Format:\n<uci_move>...best move in uci format...</uci_move>"
    )

    # Special marker in labels indicating soft-label position (KL divergence with move_probs)
    SOFT_LABEL_MARKER = -2

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        uci_moves,
        move_token_ids,
        max_length=512,
        temperature=1.0,
        cp_scale=100.0,
        mate_score=100000.0,
        add_special_tokens=False,
        legal_move_smoothing=0.1,
        output_start_token="<uci_move>",
        output_end_token="</uci_move>",
        include_legal_mask=False,
        log_samples=0,
        log_stats=True,
        user_message_template_path=None,
    ):
        self.records = load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.uci_moves = list(uci_moves)
        self.uci_to_index = {uci: idx for idx, uci in enumerate(uci_moves)}
        self.move_token_ids = list(move_token_ids)
        self.max_length = max_length
        self.temperature = temperature
        self.cp_scale = cp_scale
        self.mate_score = mate_score
        self.add_special_tokens = add_special_tokens
        self.legal_move_smoothing = legal_move_smoothing
        self.output_start_token = output_start_token
        self.output_end_token = output_end_token
        self._use_chat_template = hasattr(self.tokenizer, "apply_chat_template")
        self._include_legal_mask = bool(include_legal_mask)
        self._log_samples = max(int(log_samples), 0)
        self._logged_samples = 0

        # Get token IDs for special tokens
        self._start_token_id = self.tokenizer.convert_tokens_to_ids(output_start_token)
        self._end_token_id = self.tokenizer.convert_tokens_to_ids(output_end_token)
        self._eos_token_id = self.tokenizer.eos_token_id

        if user_message_template_path is not None:
            with open(user_message_template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
            self._user_template = Template(template_str)
            logger.info("Loaded user message template from %s.", user_message_template_path)
        else:
            self._user_template = Template(self.DEFAULT_USER_TEMPLATE)
            logger.info("Using default user message template.")

        logger.info("Loaded %d records from %s.", len(self.records), jsonl_path)
        logger.info(
            "Chat template enabled: %s. Output start token: %s.",
            self._use_chat_template,
            self.output_start_token,
        )
        if self._include_legal_mask and chess is None:
            logger.warning(
                "Legal-move metrics requested but python-chess is not installed; "
                "legal-move metrics will be zeros."
            )
        if legal_move_smoothing > 0 and chess is None:
            logger.warning(
                "legal_move_smoothing > 0 but python-chess is not installed; "
                "legal move smoothing will fail at runtime."
            )
        if log_stats:
            self._log_record_stats()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        fen = record["fen"]
        user_content = self._user_template.render(FEN=fen)
        messages = [{"role": "user", "content": user_content}]
        if self._use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = user_content + "\n"

        # Build move probability vector (soft labels for the move token)
        move_probs = build_move_prob_vector(
            record,
            self.uci_to_index,
            temperature=self.temperature,
            cp_scale=self.cp_scale,
            mate_score=self.mate_score,
            legal_move_smoothing=self.legal_move_smoothing,
        )

        # Get best move from distribution for the sequence
        best_move_idx = int(move_probs.argmax().item())
        best_move_token_id = self.move_token_ids[best_move_idx]

        # Tokenize prompt (without output tokens)
        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 4,  # Reserve space for output tokens
        )
        prompt_ids = prompt_enc["input_ids"]
        prompt_len = len(prompt_ids)

        # Build full sequence: prompt + <uci_move> + move + </uci_move> + EOS
        # EOS must be in input_ids so we can train the model to predict it
        input_ids = prompt_ids + [
            self._start_token_id,
            best_move_token_id,
            self._end_token_id,
            self._eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        # Build labels for next-token prediction (HuggingFace convention):
        # After shifting in trainer (shift_labels = labels[1:], shift_logits = logits[:-1]):
        # - labels[i+1] is what input_ids[i] should predict
        # - Prompt tokens: -100 (ignore, we don't train on predicting prompt)
        # - Last prompt token predicts <uci_move>: labels[prompt_len] = start_token_id
        # - <uci_move> predicts move (soft): labels[prompt_len+1] = SOFT_LABEL_MARKER
        # - move predicts </uci_move>: labels[prompt_len+2] = end_token_id
        # - </uci_move> predicts EOS: labels[prompt_len+3] = eos_token_id
        labels = [-100] * prompt_len  # Ignore prompt token predictions
        labels.append(self._start_token_id)  # Last prompt token predicts <uci_move>
        labels.append(self.SOFT_LABEL_MARKER)  # <uci_move> predicts move (soft label)
        labels.append(self._end_token_id)  # move predicts </uci_move>
        labels.append(self._eos_token_id)  # </uci_move> predicts EOS

        legal_mask = None
        if self._include_legal_mask:
            legal_mask = self._build_legal_mask(fen)
        if self._logged_samples < self._log_samples:
            self._logged_samples += 1
            self._log_sample(idx, fen, {"input_ids": input_ids}, move_probs, legal_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "move_probs": move_probs,
            **({"legal_mask": legal_mask} if legal_mask is not None else {}),
        }

    def _log_record_stats(self):
        total = len(self.records)
        if total == 0:
            logger.warning("Dataset is empty.")
            return

        missing_fen = 0
        records_with_moves = 0
        records_with_scored_moves = 0
        total_moves = 0
        total_scored_moves = 0
        prob_moves = 0
        cp_moves = 0
        mate_moves = 0
        unknown_uci = 0
        unknown_samples = []
        min_moves = None
        max_moves = 0

        for record in self.records:
            fen = record.get("fen")
            if not fen:
                missing_fen += 1
            record_move_count = 0
            record_scored_count = 0
            for move in _iter_moves(record):
                record_move_count += 1
                uci = move.get("uci")
                if uci and uci not in self.uci_to_index:
                    unknown_uci += 1
                    if len(unknown_samples) < 5:
                        unknown_samples.append(uci)
                kind, _ = _extract_score(move, self.mate_score)
                if kind is not None:
                    record_scored_count += 1
                    if kind == "prob":
                        prob_moves += 1
                    else:
                        cp_moves += 1
                if "mate" in move:
                    mate_moves += 1
                score = move.get("score")
                if isinstance(score, dict) and "mate" in score:
                    mate_moves += 1

            if record_move_count > 0:
                records_with_moves += 1
                total_moves += record_move_count
                if min_moves is None:
                    min_moves = record_move_count
                else:
                    min_moves = min(min_moves, record_move_count)
                max_moves = max(max_moves, record_move_count)
            if record_scored_count > 0:
                records_with_scored_moves += 1
                total_scored_moves += record_scored_count

        avg_moves = total_moves / records_with_moves if records_with_moves else 0.0
        logger.info(
            "Records: total=%d with_fen=%d missing_fen=%d.",
            total,
            total - missing_fen,
            missing_fen,
        )
        logger.info(
            "Moves: records_with_moves=%d total_moves=%d min=%s max=%d avg=%.2f.",
            records_with_moves,
            total_moves,
            "n/a" if min_moves is None else str(min_moves),
            max_moves,
            avg_moves,
        )
        logger.info(
            "Scored moves: records_with_scored_moves=%d total_scored_moves=%d "
            "prob_moves=%d cp_moves=%d mate_moves=%d.",
            records_with_scored_moves,
            total_scored_moves,
            prob_moves,
            cp_moves,
            mate_moves,
        )
        if unknown_uci:
            logger.warning(
                "Unknown UCI moves found: %d (samples: %s).",
                unknown_uci,
                ", ".join(unknown_samples),
            )

    def _log_sample(self, idx, fen, enc, move_probs, legal_mask):
        input_len = len(enc.get("input_ids", []))
        nonzero = int((move_probs > 0).sum().item())
        prob_sum = float(move_probs.sum().item())
        top_k = min(5, move_probs.numel())
        top_values, top_indices = torch.topk(move_probs, k=top_k)
        top_moves = []
        for value, index in zip(top_values.tolist(), top_indices.tolist()):
            uci = self.uci_moves[index] if index < len(self.uci_moves) else "?"
            top_moves.append(f"{uci}={value:.4f}")
        legal_count = None
        if legal_mask is not None:
            legal_count = int(legal_mask.sum().item())
        logger.info(
            "Sample idx=%d fen=%s prompt_len=%d move_probs_sum=%.4f nonzero=%d "
            "legal_moves=%s top_moves=%s.",
            idx,
            fen,
            input_len,
            prob_sum,
            nonzero,
            "n/a" if legal_count is None else str(legal_count),
            ", ".join(top_moves),
        )

    def _build_legal_mask(self, fen):
        if not fen:
            return torch.zeros(len(self.uci_to_index), dtype=torch.float32)
        if chess is None:
            return torch.zeros(len(self.uci_to_index), dtype=torch.float32)
        legal_vector = _legal_move_vector(fen, self.uci_to_index)
        if legal_vector is None:
            return torch.zeros(len(self.uci_to_index), dtype=torch.float32)
        return (legal_vector > 0).float()


class DistillDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._logged_batch = False

    def __call__(self, features):
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        pad_token_id = self.tokenizer.pad_token_id

        # Pad input_ids, attention_mask, and labels
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]
            pad_len = max_len - len(input_ids)

            # Pad on the right
            padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)  # -100 for ignored positions

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

        move_probs = torch.stack([feature["move_probs"] for feature in features])
        batch["move_probs"] = move_probs
        if "legal_mask" in features[0]:
            legal_mask = torch.stack([feature["legal_mask"] for feature in features])
            batch["legal_mask"] = legal_mask
        if not self._logged_batch:
            self._logged_batch = True
            lengths = batch["attention_mask"].sum(dim=1)
            move_sums = move_probs.sum(dim=1)
            nonzero = (move_probs > 0).sum(dim=1)
            # Count hard and soft label positions
            hard_label_count = ((batch["labels"] >= 0)).sum(dim=1)
            soft_label_count = (batch["labels"] == ChessDistillDataset.SOFT_LABEL_MARKER).sum(dim=1)
            logger.info(
                "First batch shapes: input_ids=%s attention_mask=%s labels=%s move_probs=%s.",
                tuple(batch["input_ids"].shape),
                tuple(batch["attention_mask"].shape),
                tuple(batch["labels"].shape),
                tuple(move_probs.shape),
            )
            logger.info(
                "First batch lengths: min=%d max=%d.",
                int(lengths.min().item()),
                int(lengths.max().item()),
            )
            logger.info(
                "First batch labels: hard_labels=%d-%d soft_labels=%d-%d.",
                int(hard_label_count.min().item()),
                int(hard_label_count.max().item()),
                int(soft_label_count.min().item()),
                int(soft_label_count.max().item()),
            )
            logger.info(
                "First batch move_probs: sum min=%.4f max=%.4f nonzero min=%d max=%d.",
                float(move_sums.min().item()),
                float(move_sums.max().item()),
                int(nonzero.min().item()),
                int(nonzero.max().item()),
            )
            if "legal_mask" in batch:
                legal_counts = batch["legal_mask"].sum(dim=1)
                logger.info(
                    "First batch legal_moves: min=%d max=%d.",
                    int(legal_counts.min().item()),
                    int(legal_counts.max().item()),
                )
        return batch
