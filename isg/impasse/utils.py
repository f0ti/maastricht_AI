import numpy as np


def render_mask(bitboard):
    binary = "{:064b}".format(bitboard)
    board = np.array(list(binary), dtype=int).reshape(-1, 8)
    board = np.flip(board, 1)
    print(board)


def print_legal_moves(board):
    for move in board.legal_moves:
        print(move)
