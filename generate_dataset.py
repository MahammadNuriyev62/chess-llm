import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Set, Tuple

from datasets import load_dataset


def start_stockfish(stockfish_path: str, threads: int, multipv: int) -> Tuple[subprocess.Popen, callable]:
    p = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def send(cmd: str) -> None:
        assert p.stdin is not None
        p.stdin.write(cmd + "\n")
        p.stdin.flush()

    # UCI init
    send("uci")
    assert p.stdout is not None
    while True:
        if "uciok" in p.stdout.readline():
            break

    send(f"setoption name Threads value {threads}")
    send(f"setoption name MultiPV value {multipv}")
    send("isready")
    while True:
        if "readyok" in p.stdout.readline():
            break

    return p, send


def parse_multipv(proc: subprocess.Popen, topk: int) -> List[Dict]:
    """
    Collect the last seen score line for each multipv in [1..topk] and return
    them ordered by multipv index.
    """
    assert proc.stdout is not None
    moves: Dict[int, Dict] = {}

    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()

        if line.startswith("info") and "multipv" in line and " pv " in line:
            parts = line.split()

            try:
                mpv = int(parts[parts.index("multipv") + 1])
            except Exception:
                continue
            if mpv > topk:
                continue

            # score can be "score cp <n>" or "score mate <n>"
            score = None
            score_type = None
            if "score" in parts:
                si = parts.index("score")
                if si + 2 < len(parts):
                    score_type = parts[si + 1]
                    score = parts[si + 2]

            if "pv" not in parts:
                continue
            pv_i = parts.index("pv")
            if pv_i + 1 >= len(parts):
                continue
            uci = parts[pv_i + 1]

            # Normalize to centipawns when possible; keep mate as a string tag
            if score_type == "cp":
                try:
                    cp = int(score)
                except Exception:
                    continue
                moves[mpv] = {"uci": uci, "cp": cp}
            elif score_type == "mate":
                # You can decide how you want to represent mates in your dataset.
                # Here: keep an integer mate distance under "mate".
                try:
                    mate = int(score)
                except Exception:
                    continue
                moves[mpv] = {"uci": uci, "mate": mate}

        elif line.startswith("bestmove"):
            break

    # Return ordered by multipv index; may be < topk if engine didn't produce all.
    return [moves[i] for i in sorted(moves.keys())]


def evaluate_fen(proc: subprocess.Popen, send: callable, fen: str, nodes: int, topk: int) -> List[Dict]:
    send(f"position fen {fen}")
    send(f"go nodes {nodes}")
    return parse_multipv(proc, topk)


def iter_fens(streaming: bool, shuffle_buffer: int, seed: int):
    """
    Stream records from the dataset and yield fen strings.
    Dataset has columns including: fen, move, explanation, messages, text. :contentReference[oaicite:1]{index=1}
    """
    ds = load_dataset("aicrowd/ChessExplained", split="train", streaming=streaming)

    # Optional: approximate shuffle for streaming mode
    if shuffle_buffer and streaming:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    for row in ds:
        fen = row.get("fen")
        if isinstance(fen, str) and fen.strip():
            yield fen.strip()


def load_existing_fens(path: str) -> Set[str]:
    fens: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fen = row.get("fen")
                if isinstance(fen, str) and fen.strip():
                    fens.add(fen.strip())
    except FileNotFoundError:
        pass
    return fens


def main():
    ap = argparse.ArgumentParser(description="Generate Stockfish-labeled samples from aicrowd/ChessExplained FENs.")
    ap.add_argument("n", type=int, help="Number of samples to generate")
    ap.add_argument("--out", type=str, default="-", help="Output JSONL path (default stdout)")
    ap.add_argument("--topk", type=int, default=3, help="Number of PV lines to keep")
    ap.add_argument("--nodes", type=int, default=20000, help="Stockfish nodes per position (predictable throughput)")
    ap.add_argument("--threads", type=int, default=1, help="Stockfish Threads")
    ap.add_argument("--stockfish", type=str, default="stockfish", help="Path to stockfish binary")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended for this dataset size)")
    ap.add_argument("--shuffle-buffer", type=int, default=0, help="Streaming shuffle buffer (e.g. 10000). 0 = no shuffle")
    ap.add_argument("--seed", type=int, default=0, help="Seed for streaming shuffle")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of overwriting; skips FENs already in that file",
    )
    ap.add_argument(
        "--existing-jsonl",
        type=str,
        default=None,
        help="JSONL file to use for dedupe (defaults to --out when --append is set)",
    )
    args = ap.parse_args()

    proc, send = start_stockfish(args.stockfish, args.threads, args.topk)

    existing_fens: Set[str] = set()
    dedupe_path = None
    dedupe = args.append or args.existing_jsonl is not None
    if args.existing_jsonl:
        dedupe_path = args.existing_jsonl
    elif args.append and args.out != "-" and os.path.exists(args.out):
        dedupe_path = args.out

    if dedupe_path:
        existing_fens = load_existing_fens(dedupe_path)
        if existing_fens:
            print(f"Loaded {len(existing_fens)} existing FENs from {dedupe_path}", file=sys.stderr)

    if args.append and args.out == "-":
        print("--append has no effect with --out=-; use --existing-jsonl for dedupe", file=sys.stderr)

    if args.out == "-":
        out_f = sys.stdout
    else:
        mode = "a" if args.append else "w"
        out_f = open(args.out, mode, encoding="utf-8")

    try:
        produced = 0
        for fen in iter_fens(args.streaming, args.shuffle_buffer, args.seed):
            if dedupe and fen in existing_fens:
                continue
            moves = evaluate_fen(proc, send, fen, args.nodes, args.topk)

            # Match your desired sample structure.
            sample = {"fen": fen, "moves": moves}

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            produced += 1
            if dedupe:
                existing_fens.add(fen)

            if produced >= args.n:
                break

            if produced % 100 == 0:
                print(f"{produced} samples generated", file=sys.stderr)

    finally:
        try:
            send("quit")
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        if out_f is not sys.stdout:
            out_f.close()


if __name__ == "__main__":
    main()
