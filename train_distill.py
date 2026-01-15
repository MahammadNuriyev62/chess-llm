import argparse

from unsloth import FastLanguageModel

import torch
from transformers import TrainingArguments, set_seed
from chess_distill.data import ChessDistillDataset, DistillDataCollator
from chess_distill.move_tokens import EXPECTED_UCI_MOVE_COUNT, generate_uci_move_tokens
from chess_distill.tokenizer_utils import (
    cast_embeddings_to_float32,
    extend_tokenizer_with_moves,
)
from chess_distill.trainer import ChessDistillTrainer


def _ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "right"
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a chess-distilled LLM with Unsloth.")
    parser.add_argument("--train-jsonl", required=True, help="Path to JSONL training data.")
    parser.add_argument("--output-dir", default="outputs_chess_distill")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cp-scale", type=float, default=100.0)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    set_seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    _ensure_pad_token(tokenizer, model)

    uci_moves, move_tokens = generate_uci_move_tokens()
    if len(uci_moves) != EXPECTED_UCI_MOVE_COUNT:
        print(
            f"Warning: generated {len(uci_moves)} UCI moves "
            f"(expected ~{EXPECTED_UCI_MOVE_COUNT})."
        )

    move_token_ids = extend_tokenizer_with_moves(tokenizer, model, move_tokens)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        modules_to_save=["embed_tokens", "lm_head"],
    )

    cast_embeddings_to_float32(model)
    model.config.use_cache = False

    dataset = ChessDistillDataset(
        args.train_jsonl,
        tokenizer,
        uci_moves,
        max_length=args.max_seq_length,
        temperature=args.temperature,
        cp_scale=args.cp_scale,
    )
    collator = DistillDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )

    trainer = ChessDistillTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        move_token_ids=move_token_ids,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
