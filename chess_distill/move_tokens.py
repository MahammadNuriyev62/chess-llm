FILES = "abcdefgh"
RANKS = "12345678"
PROMOTIONS = "qrbn"

EXPECTED_UCI_MOVE_COUNT = 1968

_KNIGHT_OFFSETS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]
_KING_OFFSETS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]
_ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _in_bounds(file_idx, rank_idx):
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


def _square_name(file_idx, rank_idx):
    return f"{FILES[file_idx]}{RANKS[rank_idx]}"


def _add_step_moves(moves, from_file, from_rank, offsets):
    from_sq = _square_name(from_file, from_rank)
    for df, dr in offsets:
        to_file = from_file + df
        to_rank = from_rank + dr
        if _in_bounds(to_file, to_rank):
            moves.add(from_sq + _square_name(to_file, to_rank))


def _add_slider_moves(moves, from_file, from_rank, directions):
    from_sq = _square_name(from_file, from_rank)
    for df, dr in directions:
        to_file = from_file + df
        to_rank = from_rank + dr
        while _in_bounds(to_file, to_rank):
            moves.add(from_sq + _square_name(to_file, to_rank))
            to_file += df
            to_rank += dr


def _add_pawn_moves(moves, from_file, from_rank, is_white):
    direction = 1 if is_white else -1
    start_rank = 1 if is_white else 6
    promo_from = 6 if is_white else 1

    from_sq = _square_name(from_file, from_rank)
    to_rank = from_rank + direction
    if _in_bounds(from_file, to_rank):
        to_sq = _square_name(from_file, to_rank)
        if from_rank == promo_from:
            for promo in PROMOTIONS:
                moves.add(from_sq + to_sq + promo)
        else:
            moves.add(from_sq + to_sq)

        if from_rank == start_rank:
            to_rank2 = from_rank + (2 * direction)
            if _in_bounds(from_file, to_rank2):
                moves.add(from_sq + _square_name(from_file, to_rank2))

    for df in (-1, 1):
        to_file = from_file + df
        to_rank = from_rank + direction
        if _in_bounds(to_file, to_rank):
            to_sq = _square_name(to_file, to_rank)
            if from_rank == promo_from:
                for promo in PROMOTIONS:
                    moves.add(from_sq + to_sq + promo)
            else:
                moves.add(from_sq + to_sq)


def generate_all_uci_moves():
    moves = set()
    for file_idx in range(8):
        for rank_idx in range(8):
            _add_step_moves(moves, file_idx, rank_idx, _KNIGHT_OFFSETS)
            _add_step_moves(moves, file_idx, rank_idx, _KING_OFFSETS)
            _add_slider_moves(moves, file_idx, rank_idx, _ROOK_DIRS)
            _add_slider_moves(moves, file_idx, rank_idx, _BISHOP_DIRS)
            _add_pawn_moves(moves, file_idx, rank_idx, is_white=True)
            _add_pawn_moves(moves, file_idx, rank_idx, is_white=False)

    moves.update(["e1g1", "e1c1", "e8g8", "e8c8"])
    return sorted(moves)


def generate_uci_move_tokens():
    uci_moves = generate_all_uci_moves()
    move_tokens = [f"<{uci}>" for uci in uci_moves]
    return uci_moves, move_tokens
