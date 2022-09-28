import numpy as np
from piece import Piece

# hand-crafted bitboards for impasse

'''
chess
---

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

Square = int
SQUARES = [
  A1, B1, C1, D1, E1, F1, G1, H1,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A8, B8, C8, D8, E8, F8, G8, H8,
] = range(64)
'''

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

Square = int

SQUARES = [
  A1, C1, E1, G1,
  B2, D2, F2, H2,
  A3, C3, E3, G3,
  B4, D4, F4, H4,
  A5, C5, E5, G5,
  B6, D6, F6, H6,
  A7, C7, E7, G7,
  B8, D8, F8, H8,
] = range(32)

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

def parse_square(name: str) -> Square:
  """
  Gets the square index for the given square *name*
  (e.g., ``a1`` returns ``0``).

  :raises: :exc:`ValueError` if the square name is invalid.
  """
  return SQUARE_NAMES.index(name)

def square_name(square: Square) -> str:
  """Gets the name of the square, like ``a3``."""
  return SQUARE_NAMES[square]

def square(file_index: int, rank_index: int) -> Square:
  """Gets a square number by file and rank index."""
  return rank_index * 8 + file_index

def square_file(square: Square) -> int:
  """Gets the file index of the square where ``0`` is the a-file."""
  return square & 7

def square_rank(square: Square) -> int:
  """Gets the rank index of the square where ``0`` is the first rank."""
  return square >> 3

Bitboard = int
BB_EMPTY = 0
BB_ALL = 0xffff_ffff_ffff_ffff

BB_RANKS = [
  BB_RANK_1,
  BB_RANK_2,
  BB_RANK_3,
  BB_RANK_4,
  BB_RANK_5,
  BB_RANK_6,
  BB_RANK_7,
  BB_RANK_8,
] = [0xf << (4 * i) for i in range(8)]

BB_BACKRANKS = BB_RANK_1 | BB_RANK_8

# BB_BLACK = 01011010000000000000000010100101
BB_BLACK = 1509949605

# BB_WHITE = 10100101000000000000000001011010
BB_WHITE = 2768240730

# BB_SINGLES = 01011010000000000000000001011010
BB_SINGLES = 1509949530

# BB_DOUBLES = 10100101000000000000000010100101
BB_DOUBLES = 2768240805

BB_OCCUPIED = BB_RANK_1 | BB_RANK_2 | BB_RANK_7 | BB_RANK_8


class Board:
  
  def __init__(self) -> None:
    self.player_turn = True
    self.width = 4
    self.height = 8
    self.position_count = self.width * self.height
    self.position_layout = np.arange(32).reshape((-1, 4))
    self.piece_map = None
    self.pieces = None
    self.set_starting_pieces()

  def set_starting_pieces(self):
    pieces = []
    # set white singles
    for pt, pos in START_POS_WHITE.items():
      for p in pos:
        pieces.append(self.create_piece(True, pt, p))

    for pt, pos in START_POS_BLACK.items():
      for p in pos:
        pieces.append(self.create_piece(False, pt, p))

    self.pieces = pieces

  def create_piece(self, color, piece_type, position):
    piece = Piece()
    piece.color = color
    piece.piece_type = piece_type
    piece.position = position

    return piece

  def render_piece_map(self):
    piece_map = np.chararray(32)
    for p in self.pieces:
      piece_map[p.position] = p.symbol()

    self.piece_map = piece_map.reshape((-1, 4))
    print(self.piece_map)

  def list_pieces(self):
    for piece in self.pieces:
      print(piece)

  def promote(self):
    self.kings |= self.black & 0xf0000000
    self.kings |= self.white & 0x0000000f

  def render(self):
    print(self.piece_map)
