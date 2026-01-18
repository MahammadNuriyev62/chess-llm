import logging

import torch

logger = logging.getLogger(__name__)


def extend_tokenizer_with_moves(tokenizer, model, move_tokens, extra_special_tokens=None):
    original_vocab_size = len(tokenizer)
    extra_special_tokens = list(extra_special_tokens) if extra_special_tokens else []
    all_tokens = list(move_tokens)
    for token in extra_special_tokens:
        if token not in all_tokens:
            all_tokens.append(token)
    logger.info(
        "Extending tokenizer with %d move tokens (vocab size %d).",
        len(move_tokens),
        original_vocab_size,
    )
    if extra_special_tokens:
        logger.info(
            "Adding %d format tokens: %s.",
            len(extra_special_tokens),
            ", ".join(extra_special_tokens),
        )
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": all_tokens}
    )
    if num_added:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added %d tokens; new vocab size %d.", num_added, len(tokenizer))
        sample_count = min(10, len(move_tokens))
        if sample_count:
            sample_tokens = ", ".join(move_tokens[:sample_count])
            logger.info("Move token sample (%d): %s.", sample_count, sample_tokens)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("All move tokens: %s.", ", ".join(move_tokens))
    else:
        logger.warning(
            "No new tokens were added; move tokens may already exist in the vocab."
        )

    move_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in move_tokens]
    unk_id = tokenizer.unk_token_id
    if unk_id is not None:
        unk_count = sum(1 for token_id in move_token_ids if token_id == unk_id)
        if unk_count:
            logger.warning("Move tokens mapped to unk: %d.", unk_count)
    if move_token_ids:
        logger.info(
            "Move token id range: min=%d max=%d.",
            min(move_token_ids),
            max(move_token_ids),
        )
    _init_move_embeddings(tokenizer, model, move_tokens, original_vocab_size)
    _tie_model_weights(model)
    return move_token_ids


def cast_embeddings_to_float32(model):
    input_emb = model.get_input_embeddings()
    if input_emb is not None:
        input_emb.to(torch.float32)
        for param in input_emb.parameters():
            param.requires_grad = True

    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        output_emb.to(torch.float32)
        for param in output_emb.parameters():
            param.requires_grad = True


def _tie_model_weights(model):
    if hasattr(model, "tie_weights"):
        model.tie_weights()


def _init_move_embeddings(tokenizer, model, move_tokens, original_vocab_size):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is None:
        return

    input_weight = input_emb.weight.data
    output_weight = output_emb.weight.data if output_emb is not None else None
    initialized = 0
    skipped = 0

    for token in move_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            skipped += 1
            continue
        if token_id == tokenizer.unk_token_id:
            skipped += 1
            continue
        if token_id < original_vocab_size:
            skipped += 1
            continue

        if token.startswith("<") and token.endswith(">") and len(token) > 2:
            uci = token[1:-1]
        else:
            uci = token
        sub_ids = []
        for ch in uci:
            sub_ids.extend(tokenizer.encode(ch, add_special_tokens=False))
        if not sub_ids:
            skipped += 1
            continue

        mean_vec = input_weight[sub_ids].float().mean(dim=0)
        input_weight[token_id] = mean_vec.to(input_weight.dtype)
        if output_weight is not None:
            output_weight[token_id] = mean_vec.to(output_weight.dtype)
        initialized += 1

    logger.info(
        "Initialized %d move embeddings (skipped %d).", initialized, skipped
    )
