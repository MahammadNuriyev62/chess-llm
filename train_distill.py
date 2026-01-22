import argparse
import logging
import os
import platform
import sys

from unsloth import FastLanguageModel

import torch
from transformers import TrainingArguments, set_seed
from chess_distill.data import ChessDistillDataset, DistillDataCollator
from chess_distill.move_tokens import EXPECTED_UCI_MOVE_COUNT, generate_uci_move_tokens
from chess_distill.tokenizer_utils import extend_tokenizer_with_moves
from chess_distill.trainer import ChessDistillTrainer

OUTPUT_START_TOKEN = "<uci_move>"
OUTPUT_END_TOKEN = "</uci_move>"


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


def _configure_logging(log_level, suppress_noisy_logs=True):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("transformers").setLevel(level)
    if suppress_noisy_logs:
        noisy_loggers = [
            "urllib3",
            "filelock",
            "huggingface_hub",
            "requests",
            "httpx",
        ]
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)


def _configure_wandb(args, logger):
    if not args.use_wandb:
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb is not installed; disabling wandb logging.")
        return False

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_group:
        os.environ["WANDB_GROUP"] = args.wandb_group
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags
    return True


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a chess-distilled LLM with Unsloth.")
    parser.add_argument("--train-jsonl", required=True, help="Path to JSONL training data.")
    parser.add_argument("--output-dir", default="outputs_chess_distill")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.00025)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Fraction of total steps for warmup (overrides warmup-steps if > 0).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Keep only the most recent N checkpoints on disk.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--suppress-noisy-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silence verbose third-party debug logs while keeping custom logs.",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--log-samples",
        type=int,
        default=3,
        help="Number of dataset samples to log for sanity checks.",
    )
    parser.add_argument(
        "--log-pred-steps",
        type=int,
        default=50,
        help="Log model predictions vs targets every N steps.",
    )
    parser.add_argument(
        "--log-pred-topk",
        type=int,
        default=5,
        help="Top-K moves to log for predictions/targets.",
    )
    parser.add_argument(
        "--log-legal-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute legal-move metrics if python-chess is available.",
    )
    parser.add_argument(
        "--legal-move-smoothing",
        type=float,
        default=0.1,
        help="Mix this much uniform probability over legal moves into the target distribution.",
    )
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
    parser.add_argument(
        "--user-message-template",
        type=str,
        default=None,
        help="Path to a Jinja template file for the user message. Template receives 'board_visual' variable.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _configure_logging(args.log_level, args.suppress_noisy_logs)
    logger = logging.getLogger(__name__)
    use_wandb = _configure_wandb(args, logger)

    logger.info("Starting training with args: %s", vars(args))
    logger.info(
        "Environment: python=%s torch=%s cuda_available=%s platform=%s",
        platform.python_version(),
        torch.__version__,
        torch.cuda.is_available(),
        platform.platform(),
    )
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    set_seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    logger.info("Loaded model %s.", args.model_name)
    logger.info(
        "Tokenizer vocab size=%d eos=%s pad=%s.",
        len(tokenizer),
        tokenizer.eos_token,
        tokenizer.pad_token,
    )

    _ensure_pad_token(tokenizer, model)
    logger.info(
        "Pad token id=%s padding_side=%s.",
        tokenizer.pad_token_id,
        tokenizer.padding_side,
    )

    uci_moves, move_tokens = generate_uci_move_tokens()
    if len(uci_moves) != EXPECTED_UCI_MOVE_COUNT:
        print(
            f"Warning: generated {len(uci_moves)} UCI moves "
            f"(expected ~{EXPECTED_UCI_MOVE_COUNT})."
        )
    logger.info("Generated %d UCI moves.", len(uci_moves))

    move_token_ids = extend_tokenizer_with_moves(
        tokenizer,
        model,
        move_tokens,
        extra_special_tokens=[OUTPUT_START_TOKEN, OUTPUT_END_TOKEN],
    )

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
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    model.config.use_cache = False
    total_params, trainable_params = _count_parameters(model)
    logger.info(
        "Model params: total=%d trainable=%d (%.2f%%).",
        total_params,
        trainable_params,
        100.0 * trainable_params / max(total_params, 1),
    )

    dataset = ChessDistillDataset(
        args.train_jsonl,
        tokenizer,
        uci_moves,
        max_length=args.max_seq_length,
        temperature=args.temperature,
        cp_scale=args.cp_scale,
        legal_move_smoothing=args.legal_move_smoothing,
        output_start_token=OUTPUT_START_TOKEN,
        include_legal_mask=args.log_legal_metrics,
        log_samples=args.log_samples,
        log_stats=True,
        user_message_template_path=args.user_message_template,
    )
    collator = DistillDataCollator(tokenizer)
    logger.info("Dataset size: %d records.", len(dataset))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
    )
    logger.info(
        "Training setup: batch_size=%d grad_accum=%d effective_batch=%d epochs=%.2f "
        "max_steps=%d lr_scheduler=%s warmup_steps=%d warmup_ratio=%.3f weight_decay=%.4f "
        "logging_steps=%d save_steps=%d save_total_limit=%d "
        "log_pred_steps=%d use_wandb=%s.",
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.per_device_train_batch_size * args.gradient_accumulation_steps,
        args.num_train_epochs,
        args.max_steps,
        args.lr_scheduler_type,
        args.warmup_steps,
        args.warmup_ratio,
        args.weight_decay,
        args.logging_steps,
        args.save_steps,
        args.save_total_limit,
        args.log_pred_steps,
        use_wandb,
    )

    trainer = ChessDistillTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        move_token_ids=move_token_ids,
        move_token_strs=move_tokens,
        log_pred_steps=args.log_pred_steps,
        log_pred_topk=args.log_pred_topk,
    )
    logger.info("Starting training loop.")
    train_output = trainer.train()
    if hasattr(train_output, "metrics"):
        logger.info("Training metrics: %s", train_output.metrics)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved model and tokenizer to %s.", args.output_dir)


if __name__ == "__main__":
    main()
