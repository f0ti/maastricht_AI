# ------ color ------

from piece import Piece


Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
PIECE_TYPES = [SINGLE, DOUBLE] = [1, 2]
PIECE_SYMBOLS = [None, "s", "d"]
PIECE_NAMES = [None, "single", "double"]

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

# ------ square ------

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

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

# ------ bitboard ------

Bitboard = int
BB_EMPTY = 0
BB_ALL = 0xffff_ffff_ffff_ffff

BB_SQUARES = [
  BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
  BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
  BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
  BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
  BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
  BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
  BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
  BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8,
] = [1 << sq for sq in SQUARES]

def parse_square(name: str) -> Square:
  """Gets the square index for the given square *name*"""
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
  return square ^ 0x38

SQUARES_180 = [square_mirror(sq) for sq in SQUARES]

Bitboard = int
BB_CLEAN = 0
BB_ALL = 0xffff_ffff_ffff_ffff

BB_FILES = [
  BB_FILE_A,
  BB_FILE_B,
  BB_FILE_C,
  BB_FILE_D,
  BB_FILE_E,
  BB_FILE_F,
  BB_FILE_G,
  BB_FILE_H,
] = [0x0101_0101_0101_0101 << i for i in range(8)]

BB_RANKS = [
  BB_RANK_1,
  BB_RANK_2,
  BB_RANK_3,
  BB_RANK_4,
  BB_RANK_5,
  BB_RANK_6,
  BB_RANK_7,
  BB_RANK_8,
] = [0xff << (8 * i) for i in range(8)]

BB_BACKRANKS = BB_RANK_1 | BB_RANK_8


class Board():
  def __init__(self, board_fen: str = None) -> None:
    self.occupied_co = [BB_EMPTY, BB_EMPTY]

    if board_fen:
      self.set_board_fen(board_fen)
    else:
      self.reset_board()

  def reset_board(self) -> None:
    self.singles = BB_D8 | BB_H8 | BB_A7 | BB_E7 | BB_D2 | BB_H2 | BB_A1 | BB_E1
    self.doubles = BB_B8 | BB_F8 | BB_C7 | BB_G7 | BB_B2 | BB_F2 | BB_C1 | BB_G1

    self.occupied_co[WHITE] = BB_B8 | BB_F8 | BB_C7 | BB_G7 | BB_D2 | BB_H2 | BB_A1 | BB_E1
    self.occupied_co[BLACK] = BB_D8 | BB_H8 | BB_A7 | BB_E7 | BB_B2 | BB_F2 | BB_C1 | BB_G1

    self.occupied = self.occupied_co[WHITE] | self.occupied_co[BLACK]

  def pieces_mask(self, piece_type: PieceType, color: Color):
    if piece_type == SINGLE:
      bb = self.singles
    elif piece_type == DOUBLE:
      bb = self.doubles
    else:
      assert False, "Incorrect PieceType"

    return bb & self.occupied_co[color]
  
  def piece_type_at(self, square):
    mask = BB_SQUARES[square]

    if not (self.occupied & mask):
      return None

    elif self.singles & mask:
      return SINGLE

    elif self.doubles & mask:
      return DOUBLE

  def remove_piece_at(self, square: Square) -> None:
    piece_type = self.piece_type_at(square)
    mask = BB_SQUARES[square]

    if piece_type == SINGLE:
      self.singles ^= mask
    elif piece_type == DOUBLE:
      self.doubles ^= mask
    else:
      return None

    self.occupied ^= mask

    # XOR adds a piece if it is not there
    self.occupied_co[WHITE] &= ~mask
    self.occupied_co[BLACK] &= ~mask

  def piece_at(self, square):
    """Gets the Piece at the given square."""
    piece_type = self.piece_type_at(square)
    if piece_type:
      mask = BB_SQUARES[square]
      color = bool(self.occupied_co[WHITE] & mask)
      return Piece(piece_type, color)
    else:
      return None

  def print_board(self):
    builder = []

    for square in SQUARES_180:
      piece = self.piece_at(square)
      if piece:
        builder.append(piece.symbol())
      else:
        builder.append(".")

      if BB_SQUARES[square] & BB_FILE_H:
        if square != H1:
          builder.append("\n")
      else:
          builder.append(" ")

    return "".join(builder)

  def set_board_fen(self, board_fen: str) -> None:
    pass
