import torch


def extend_tokenizer_with_moves(tokenizer, model, move_tokens):
    original_vocab_size = len(tokenizer)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(move_tokens)}
    )
    if num_added:
        model.resize_token_embeddings(len(tokenizer))

    move_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in move_tokens]
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

    for token in move_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        if token_id == tokenizer.unk_token_id:
            continue
        if token_id < original_vocab_size:
            continue

        uci = token[1:-1]
        sub_ids = []
        for ch in uci:
            sub_ids.extend(tokenizer.encode(ch, add_special_tokens=False))
        if not sub_ids:
            continue

        mean_vec = input_weight[sub_ids].float().mean(dim=0)
        input_weight[token_id] = mean_vec.to(input_weight.dtype)
        if output_weight is not None:
            output_weight[token_id] = mean_vec.to(output_weight.dtype)
