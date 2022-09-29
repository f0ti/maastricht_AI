from typing import List
import numpy as np
from piece import Piece

Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
PIECE_TYPES = [SINGLE, DOUBLE] = range(1, 3)
PIECE_SYMBOLS = [None, "s", "d"]
PIECE_NAMES = [None, "single", "double"]

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

SQUARE_NAMES = [
  'A1', 'C1', 'E1', 'G1',
  'B2', 'D2', 'F2', 'H2',
  'A3', 'C3', 'E3', 'G3',
  'B4', 'D4', 'F4', 'H4',
  'A5', 'C5', 'E5', 'G5',
  'B6', 'D6', 'F6', 'H6',
  'A7', 'C7', 'E7', 'G7',
  'B8', 'D8', 'F8', 'H8',
]

BB_SQUARES = [
  BB_A1, BB_C1, BB_E1, BB_G1,
  BB_B2, BB_D2, BB_F2, BB_H2,
  BB_A3, BB_C3, BB_E3, BB_G3,
  BB_B4, BB_D4, BB_F4, BB_H4,
  BB_A5, BB_C5, BB_E5, BB_G5,
  BB_B6, BB_D6, BB_F6, BB_H6,
  BB_A7, BB_C7, BB_E7, BB_G7,
  BB_B8, BB_D8, BB_F8, BB_H8,
] = [1 << sq for sq in SQUARES]

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

def square_mirror(square: Square) -> Square:
  """Mirrors the square vertically."""
  return square ^ 0x1c

SQUARES_180 = [square_mirror(sq) for sq in SQUARES]

Bitboard = int
BB_CLEAN = 0
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

BB_FILES = [
  BB_FILE_A,
  BB_FILE_B,
  BB_FILE_C,
  BB_FILE_D,
  BB_FILE_E,
  BB_FILE_F,
  BB_FILE_G,
  BB_FILE_H,
] = [0x0101_0101 << i for i in range(8)]

BB_BACKRANKS = BB_RANK_1 | BB_RANK_8

BB_BLACK = BB_D8 | BB_H8 | BB_A7 | BB_E7 | BB_B2 | BB_F2 | BB_C1 | BB_G1
BB_WHITE = BB_B8 | BB_F8 | BB_C7 | BB_G7 | BB_D2 | BB_H2 | BB_A1 | BB_E1 

BB_SINGLES = BB_D8 | BB_H8 | BB_A7 | BB_E7 | BB_D2 | BB_H2 | BB_A1 | BB_E1
BB_DOUBLES = BB_B8 | BB_F8 | BB_C7 | BB_G7 | BB_B2 | BB_F2 | BB_C1 | BB_G1

BB_EMPTY = np.uint32(~(BB_WHITE | BB_BLACK))
BB_OCCUPIED = BB_RANK_1 | BB_RANK_2 | BB_RANK_7 | BB_RANK_8

BB_OCCUPIED_CO = [BB_EMPTY, BB_EMPTY]
BB_OCCUPIED_CO[WHITE] = BB_WHITE
BB_OCCUPIED_CO[BLACK] = BB_BLACK

# changes
# BB_OCCUPIED_CO[WHITE] |= BB_B6
# BB_SINGLES |= BB_B6

BB_OCCUPIED = BB_OCCUPIED_CO[WHITE] | BB_OCCUPIED_CO[BLACK]

print(BB_OCCUPIED_CO[BLACK])
print(BB_EMPTY)

def get_moves(square: int) -> List:
  attacks = 0

  tr, tf = square//4, square%4
  print(tr, tf)

  for r in range(tr+1, 8):
    for f in range(tf+1, 4):
      next_square_index = r*4 + f
      print(next_square_index)
      print(f"r: {r}, f: {f}")
      print(f"Checking square: {SQUARE_NAMES[next_square_index]}")
      attacks |= (1 << next_square_index)
      if (1 << next_square_index) & BB_OCCUPIED:
        break

  return attacks

def print_board():
  builder = []
  pieces_in_a_row = 0
  for square in SQUARES_180:
    piece = piece_at(square)
    if piece:
      builder.append(piece.symbol())
    else:
      builder.append(".")
    
    pieces_in_a_row += 1

    if pieces_in_a_row == 4:
      pieces_in_a_row = 0
      builder.append("\n")
    else:
      builder.append(" ")

  return "".join(builder)

def piece_at(square):
  """Gets the :class:`piece <chess.Piece>` at the given square."""
  piece_type = piece_type_at(square)
  if piece_type:
    mask = BB_SQUARES[square]
    color = bool(BB_OCCUPIED_CO[WHITE] & mask)
    piece = Piece(piece_type, color, SQUARE_NAMES[square])
    print(piece)
    return piece
  else:
    return None

def piece_type_at(square):
  mask = BB_SQUARES[square]

  if not (BB_OCCUPIED & mask):
    return None

  elif BB_SINGLES & mask:
    return SINGLE

  elif BB_DOUBLES & mask:
    return DOUBLE


print(print_board())
moves = get_moves(D2)
print(moves)
