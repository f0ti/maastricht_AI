import numpy as np

from board import Bitboard, Board, Color

def render_mask(bitboard: Bitboard):
  binary = "{:064b}".format(bitboard)
  board = np.array(list(binary), dtype=int).reshape(-1, 8)
  board = np.flip(board, 1)
  print(board)

def print_legal_moves(board: Board, turn: Color):
  for move in board.legal_moves:
    print(move)