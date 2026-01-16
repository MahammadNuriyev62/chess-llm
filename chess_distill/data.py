import json
import math

try:
    import chess
except ImportError:  # Optional dependency for legal move generation.
    chess = None

import torch
from torch.utils.data import Dataset


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
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        uci_moves,
        max_length=512,
        temperature=1.0,
        cp_scale=100.0,
        mate_score=100000.0,
        add_special_tokens=False,
        legal_move_smoothing=0.1,
    ):
        self.records = load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.uci_to_index = {uci: idx for idx, uci in enumerate(uci_moves)}
        self.max_length = max_length
        self.temperature = temperature
        self.cp_scale = cp_scale
        self.mate_score = mate_score
        self.add_special_tokens = add_special_tokens
        self.legal_move_smoothing = legal_move_smoothing

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        fen = record["fen"]
        prompt = f"FEN {fen} Move: "
        enc = self.tokenizer(
            prompt,
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            max_length=self.max_length,
        )
        move_probs = build_move_prob_vector(
            record,
            self.uci_to_index,
            temperature=self.temperature,
            cp_scale=self.cp_scale,
            mate_score=self.mate_score,
            legal_move_smoothing=self.legal_move_smoothing,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "move_probs": move_probs,
        }


class DistillDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        move_probs = torch.stack([feature["move_probs"] for feature in features])
        batch["move_probs"] = move_probs
        return batch
