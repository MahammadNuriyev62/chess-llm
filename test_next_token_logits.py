#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_distill.move_tokens import generate_uci_move_tokens


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device):
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def _load_model_and_tokenizer(model_dir, base_model, trust_remote_code, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    dtype = _pick_dtype(device)

    adapter_config = Path(model_dir) / "adapter_config.json"
    if adapter_config.exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Adapter model detected but peft is not installed. "
                "Install peft or provide a merged model directory."
            ) from exc

        if base_model is None:
            peft_config = PeftConfig.from_pretrained(model_dir)
            base_model = peft_config.base_model_name_or_path

        if base_model is None:
            raise RuntimeError(
                "Adapter model detected but base model is unknown. "
                "Pass --base-model to specify it."
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    model.to(device)
    model.eval()
    return model, tokenizer


def _format_prompt(fen):
    return f"FEN {fen.strip()} Move: "


def _topk_from_logits(logits, tokenizer, k):
    k = min(k, logits.numel())
    values, indices = torch.topk(logits, k=k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(tokens, values.tolist()))


def _collect_move_token_ids(tokenizer):
    _, move_tokens = generate_uci_move_tokens()
    move_token_ids = []
    for token in move_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        if token_id == tokenizer.unk_token_id:
            continue
        move_token_ids.append((token, token_id))
    return move_token_ids


def main():
    parser = argparse.ArgumentParser(
        description="Inspect next-token logits for a FEN prompt."
    )
    parser.add_argument(
        "--model-dir",
        default="./outputs_chess_distill",
        help="Path to the fine-tuned model or adapter directory.",
    )
    parser.add_argument("--base-model", default=None, help="Base model name if needed.")
    parser.add_argument("--fen", required=True, help="FEN string to evaluate.")
    parser.add_argument("--topk", type=int, default=10, help="Number of top tokens to show.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model code from the base model repo.",
    )
    parser.add_argument(
        "--show-all-move-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print logits for all move tokens.",
    )
    args = parser.parse_args()

    device = _pick_device()
    model, tokenizer = _load_model_and_tokenizer(
        args.model_dir, args.base_model, args.trust_remote_code, device
    )

    prompt = _format_prompt(args.fen)
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[0, -1, :].float().cpu()

    print(f"Prompt: {prompt}")
    print(f"Device: {device}")
    print(f"Vocab size: {logits.numel()}")
    print("")
    print(f"Top {args.topk} tokens overall:")
    for token, value in _topk_from_logits(logits, tokenizer, args.topk):
        print(f"{token}\t{value:.6f}")

    move_token_ids = _collect_move_token_ids(tokenizer)
    if move_token_ids:
        tokens, ids = zip(*move_token_ids)
        move_logits = logits[list(ids)]
        topk = min(args.topk, move_logits.numel())
        move_probs = torch.softmax(move_logits, dim=-1)
        move_values, move_indices = torch.topk(move_logits, k=topk)
        print("")
        print(f"Top {topk} move tokens:")
        for idx, value in zip(move_indices.tolist(), move_values.tolist()):
            token = tokens[idx]
            prob = move_probs[idx].item()
            print(f"{token}\t{value:.6f}\tprob={prob:.6f}")

        if args.show_all_move_logits:
            print("")
            print("All move token logits:")
            for token, value, prob in zip(tokens, move_logits.tolist(), move_probs.tolist()):
                print(f"{token}\t{value:.6f}\tprob={prob:.6f}")
    else:
        print("")
        print("No move tokens found in tokenizer.")


if __name__ == "__main__":
    main()
