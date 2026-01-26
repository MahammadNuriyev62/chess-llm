#!/usr/bin/env python3
"""
Play chess against the trained distillation model.

Supports:
  - Human vs Model: You play against the model interactively
  - Stockfish vs Model: Watch Stockfish play against the model
  - Model vs Stockfish: Model plays white against Stockfish

Usage:
  python play_chess.py --mode human          # Play as white against model
  python play_chess.py --mode human --color black  # Play as black
  python play_chess.py --mode stockfish      # Stockfish vs Model (SF=white)
  python play_chess.py --mode stockfish --model-color white  # Model=white vs SF
"""
import argparse
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


OUTPUT_START_TOKEN = "<uci_move>"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_dir, base_model=None, device="cuda"):
    """Load the fine-tuned chess model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    dtype = pick_dtype(device)

    adapter_config = Path(model_dir) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftConfig, PeftModel

        if base_model is None:
            peft_config = PeftConfig.from_pretrained(model_dir)
            base_model = peft_config.base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        input_emb = model.get_input_embeddings()
        if input_emb is not None and len(tokenizer) != input_emb.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model, "tie_weights"):
                model.tie_weights()
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()
    return model, tokenizer


def format_prompt(fen, tokenizer):
    """Format a FEN position into a model prompt."""
    user_content = (
        f'Given the current chess board "{fen.strip()}", choose the best legal move\n\n'
        "Output Format:\n<uci_move>...best move in uci format...</uci_move>"
    )
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        prompt = user_content + "\n"
    return prompt + OUTPUT_START_TOKEN


def generate_move_from_model(model, tokenizer, board, device, debug=True):
    """Generate a single move attempt from the model using normal text generation."""
    fen = board.fen()
    prompt = format_prompt(fen, tokenizer)

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the single generated token (don't skip special tokens - moves are special tokens!)
    generated_id = output_ids[0, -1].item()
    move_str = tokenizer.decode([generated_id], skip_special_tokens=False).strip()

    if debug:
        print(f"    [DEBUG] Generated token ID: {generated_id}, decoded: {move_str!r}")

    return move_str


def validate_move(move_str, board):
    """Check if a move string is valid and legal. Returns (move, error_msg)."""
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move, None
        else:
            return None, f"illegal move (not in legal moves)"
    except ValueError:
        return None, f"invalid UCI format"


def prompt_user_for_retry_action():
    """Prompt user for action after failed retries."""
    print("\n" + "-" * 40)
    print("Model failed to produce a valid move after 5 attempts.")
    print("Options:")
    print("  1. Allow 5 more retries")
    print("  2. Enter move manually")
    print("  3. Retry until success")
    print("-" * 40)

    while True:
        choice = input("Choose option (1/2/3): ").strip()
        if choice == "1":
            return "retry_5"
        elif choice == "2":
            return "manual"
        elif choice == "3":
            return "retry_forever"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def get_manual_move(board):
    """Get a move from the user manually."""
    legal_moves_str = ", ".join(m.uci() for m in board.legal_moves)
    print(f"\nLegal moves: {legal_moves_str}")

    while True:
        move_str = input("Enter move (UCI format): ").strip().lower()
        move, error = validate_move(move_str, board)
        if move:
            return move
        print(f"Invalid: {error}. Try again.")


def get_model_move(model, tokenizer, board, device):
    """Get the model's move using unconstrained generation with retry logic."""
    attempts = []
    retries_remaining = 5
    retry_forever = False

    while True:
        # Generate move from model
        move_str = generate_move_from_model(model, tokenizer, board, device)
        move, error = validate_move(move_str, board)

        attempt_num = len(attempts) + 1
        if move:
            is_legal = "LEGAL"
        else:
            is_legal = f"INVALID ({error})"
        attempts.append((move_str, is_legal))
        print(f"  Attempt {attempt_num}: {move_str!r} [{is_legal}]")

        if move:
            # Valid move found
            print(f"\nModel generated valid move: {move.uci()}")
            return move

        retries_remaining -= 1

        if retries_remaining <= 0 and not retry_forever:
            # Show all attempts so far
            print(f"\nAll attempts so far:")
            for i, (m, status) in enumerate(attempts):
                print(f"  {i+1}. {m!r} [{status}]")

            action = prompt_user_for_retry_action()

            if action == "retry_5":
                retries_remaining = 5
                print("\nRetrying 5 more times...")
            elif action == "manual":
                return get_manual_move(board)
            elif action == "retry_forever":
                retry_forever = True
                print("\nRetrying until success...")


def print_board(board, perspective_white=True):
    """Print the board with coordinates."""
    print()
    board_str = str(board)
    rows = board_str.split('\n')

    if not perspective_white:
        rows = rows[::-1]

    for i, row in enumerate(rows):
        rank = 8 - i if perspective_white else i + 1
        print(f"  {rank} {row}")

    files = "    a b c d e f g h" if perspective_white else "    h g f e d c b a"
    print(files)
    print()


def play_human_vs_model(model, tokenizer, device, human_color):
    """Interactive game: human vs model."""
    board = chess.Board()
    model_is_white = (human_color == chess.BLACK)

    print("\n" + "="*50)
    print("  HUMAN vs MODEL")
    print(f"  You are playing as {'WHITE' if human_color == chess.WHITE else 'BLACK'}")
    print("  Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("  Type 'quit' to exit, 'undo' to take back a move")
    print("="*50)

    while not board.is_game_over():
        print_board(board, perspective_white=(human_color == chess.WHITE))

        if board.turn == human_color:
            # Human's turn
            while True:
                move_str = input("Your move: ").strip().lower()

                if move_str == 'quit':
                    print("Game ended by player.")
                    return

                if move_str == 'undo':
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        print("Undid last two moves.")
                        print_board(board, perspective_white=(human_color == chess.WHITE))
                    else:
                        print("Cannot undo.")
                    continue

                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        print(f"You played: {move_str}")
                        break
                    else:
                        print(f"Illegal move: {move_str}")
                        print(f"Legal moves: {', '.join(m.uci() for m in board.legal_moves)}")
                except ValueError:
                    print(f"Invalid move format: {move_str}")
        else:
            # Model's turn
            print("Model is thinking...")
            move = get_model_move(model, tokenizer, board, device)
            board.push(move)
            print(f"\nModel plays: {move.uci()}")

    # Game over
    print_board(board, perspective_white=(human_color == chess.WHITE))
    print("\n" + "="*50)
    print(f"  GAME OVER: {board.result()}")
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"  {winner} wins by checkmate!")
    elif board.is_stalemate():
        print("  Draw by stalemate")
    elif board.is_insufficient_material():
        print("  Draw by insufficient material")
    elif board.is_fifty_moves():
        print("  Draw by fifty-move rule")
    elif board.is_repetition():
        print("  Draw by repetition")
    print("="*50)


def play_stockfish_vs_model(model, tokenizer, device, stockfish_path,
                            model_is_white=False, sf_elo=1500, sf_depth=None, max_moves=200):
    """Automated game: Stockfish vs Model."""
    board = chess.Board()

    print("\n" + "="*50)
    print("  STOCKFISH vs MODEL")
    print(f"  Model plays as {'WHITE' if model_is_white else 'BLACK'}")
    if sf_depth:
        print(f"  Stockfish depth: {sf_depth}")
    else:
        print(f"  Stockfish ELO: {sf_elo}")
    print("="*50)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        # Configure Stockfish strength (only if using ELO mode)
        if not sf_depth:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        print("Make sure Stockfish is installed: sudo apt install stockfish")
        return

    move_count = 0
    try:
        while not board.is_game_over() and move_count < max_moves:
            print_board(board)

            is_model_turn = (board.turn == chess.WHITE) == model_is_white

            if is_model_turn:
                # Model's turn
                print("Model is thinking...")
                move = get_model_move(model, tokenizer, board, device)
                print(f"Model plays: {move.uci()}")
            else:
                # Stockfish's turn
                print("Stockfish is thinking...")
                if sf_depth:
                    limit = chess.engine.Limit(depth=sf_depth)
                else:
                    limit = chess.engine.Limit(time=1.0)
                result = engine.play(board, limit)
                move = result.move
                print(f"Stockfish plays: {move.uci()}")

            board.push(move)
            move_count += 1
            print(f"Move {move_count}: {move.uci()}")
            print("-" * 30)

    finally:
        engine.quit()

    # Game over
    print_board(board)
    print("\n" + "="*50)
    print(f"  GAME OVER after {move_count} moves: {board.result()}")
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        who = "Model" if (winner == "White") == model_is_white else "Stockfish"
        print(f"  {who} ({winner}) wins by checkmate!")
    elif board.is_stalemate():
        print("  Draw by stalemate")
    elif board.is_insufficient_material():
        print("  Draw by insufficient material")
    elif board.is_fifty_moves():
        print("  Draw by fifty-move rule")
    elif board.is_repetition():
        print("  Draw by repetition")
    elif move_count >= max_moves:
        print("  Game ended by move limit")
    print("="*50)

    # Print PGN
    print("\nPGN:")
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "Model" if model_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_is_white else "Model"
    print(game)


def main():
    parser = argparse.ArgumentParser(description="Play chess against the trained model")
    parser.add_argument(
        "--model-dir",
        default="./outputs_chess_distill2",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name if needed for adapter loading",
    )
    parser.add_argument(
        "--mode",
        choices=["human", "stockfish"],
        default="human",
        help="Game mode: human (play yourself) or stockfish (watch SF vs model)",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Your color when playing as human (default: white)",
    )
    parser.add_argument(
        "--model-color",
        choices=["white", "black"],
        default="black",
        help="Model's color in stockfish mode (default: black)",
    )
    parser.add_argument(
        "--stockfish-path",
        default="/usr/games/stockfish",
        help="Path to Stockfish executable",
    )
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=1500,
        help="Stockfish ELO rating (default: 1500)",
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=None,
        help="Stockfish search depth (overrides ELO; use 1 for weakest)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves in stockfish mode (default: 200)",
    )
    args = parser.parse_args()

    print("Loading model...")
    device = pick_device()
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model, device)
    print(f"Model loaded on {device}.")

    if args.mode == "human":
        human_color = chess.WHITE if args.color == "white" else chess.BLACK
        play_human_vs_model(model, tokenizer, device, human_color)
    else:
        model_is_white = (args.model_color == "white")
        play_stockfish_vs_model(
            model, tokenizer, device,
            args.stockfish_path, model_is_white,
            args.stockfish_elo, args.stockfish_depth, args.max_moves
        )


if __name__ == "__main__":
    main()
