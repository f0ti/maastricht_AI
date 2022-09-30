import numpy as np

from board import Bitboard

def render_mask(bitboard: Bitboard):
  binary = "{:064b}".format(bitboard)
  board = np.array(list(binary), dtype=int).reshape(-1, 8)
  board = np.flip(board, 1)
  print(board)
