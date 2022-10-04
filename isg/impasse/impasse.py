# ------ color ------

import time
from chess import PIECE_SYMBOLS
import numpy as np
from random import choice
from typing import Generator, Iterator, List, Optional
from utils import print_legal_moves, render_mask


Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
PIECE_TYPES = [SINGLE, DOUBLE] = [1, 2]
# PIECE_SYMBOLS = [None, "s", "d"]
PIECE_SYMBOLS = (("None", "⛀", "⛁"), ("None", "⛂", "⛃"))
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

def scan_forward(bb: Bitboard) -> Iterator[Square]:
  while bb:
    r = bb & -bb
    yield r.bit_length() - 1
    bb ^= r

def msb(bb: Bitboard) -> int:
  return bb.bit_length() - 1

def scan_reversed(bb: Bitboard) -> Iterator[Square]:
  while bb:
    r = bb.bit_length() - 1
    yield r
    bb ^= BB_SQUARES[r]


class Piece:

  def __init__(self, piece_type=1, color=True) -> None:
    self.color = color
    self.piece_type = piece_type
    self.removed = False

  def symbol(self) -> str:
    return PIECE_SYMBOLS[self.color][self.piece_type]

  def __repr__(self) -> str:
    return f"Piece: {self.symbol()}"

  def __str__(self) -> str:
    return f"{self.symbol()}"


class Move:
  def __init__(
    self,
    from_square: Square,
    to_square: Square,
    bear_off: bool = False,
    transpose: bool = False,
    crown: Optional[Square] = None,
    impasse: Optional[PieceType] = None,
    delayed_crown: Optional[Square] = None
    ) -> None:

      self.from_square = from_square
      self.to_square = to_square
      self.crown = crown
      self.delayed_crown = delayed_crown
      self.bear_off = bear_off
      self.transpose = transpose
      self.impasse = impasse

  def uci(self) -> str:
    if self.impasse is not None:
      return f"{SQUARE_NAMES[self.from_square]}{SQUARE_NAMES[self.to_square]}"
    elif self.crown is not None:
      return f"{SQUARE_NAMES[self.from_square]}{SQUARE_NAMES[self.to_square]}[{SQUARE_NAMES[self.crown]}]"
    elif self.delayed_crown is not None:
      return f"[{SQUARE_NAMES[self.crown]}]{SQUARE_NAMES[self.from_square]}{SQUARE_NAMES[self.to_square]}"
    elif self.transpose:
      return f"{SQUARE_NAMES[self.from_square]}-><-{SQUARE_NAMES[self.to_square]}"
    elif self.bear_off:
      return f"{SQUARE_NAMES[self.from_square]}x{SQUARE_NAMES[self.to_square]}"
    else:
      return f"{SQUARE_NAMES[self.from_square]}{SQUARE_NAMES[self.to_square]}"

  def __repr__(self) -> str:
    return f"<Move>{self.uci()}"

  def __str__(self) -> str:
    return self.uci()


class Board:
  def __init__(self, board_fen: str = None, turn: Color = 1) -> None:
    self.occupied_co = [BB_EMPTY, BB_EMPTY]
    self.turn = turn
    self.delayed_crown = [False, False]

    if board_fen:
      self.set_board_fen(board_fen)
    else:
      self.reset_board()

  @property
  def legal_moves(self) -> Generator:
    return self.generate_legal_moves()

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
  
  def piece_type_at(self, square: Square):
    if square is None:
      return

    mask = BB_SQUARES[square]

    if not (self.occupied & mask):
      return None

    elif self.singles & mask:
      return SINGLE

    elif self.doubles & mask:
      return DOUBLE

  def remove_piece_at(self, square: Square) -> None:
    if square is None:
      return
    
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

  def set_piece_at(self, square: Square, piece_type: PieceType, color: Color) -> None:
    self.remove_piece_at(square)

    mask = BB_SQUARES[square]

    if piece_type == SINGLE:
      self.singles |= mask
    elif piece_type == DOUBLE:
      self.doubles |= mask
    else:
      print("O KARRR")
      return

    print(f"Piece {square} set")
    self.occupied ^= mask
    self.occupied_co[color] ^= mask

  def piece_at(self, square):
    piece_type = self.piece_type_at(square)
    if piece_type:
      mask = BB_SQUARES[square]
      color = bool(self.occupied_co[WHITE] & mask)
      
      return Piece(piece_type, color)
    
    else:
      return None

  def get_backward_moves(self, square: Square) -> Generator:
    tr, tf = square // 8, square % 8

    # down-right
    for r, f in zip(range(tr-1, -1, -1), range(tf+1, 8)):
      square = (r*8 + f)
      if (1 << square) & self.occupied:
        break

      yield square

    # down-left
    for r, f in zip(range(tr-1, -1, -1), range(tf-1, -1, -1)):
      square = (r*8 + f)
      if (1 << square) & self.occupied:
        break
      yield square

  def get_forward_moves(self, square: Square) -> Generator:
    tr, tf = square // 8, square % 8

    # up-right
    for r, f in zip(range(tr+1, 8), range(tf+1, 8)):
      square = (r*8 + f)
      if (1 << square) & self.occupied:
        break
      
      yield square
    
    # up-left
    for r, f in zip(range(tr+1, 8), range(tf-1, -1, -1)):
      square = (r*8 + f)
      if (1 << square) & self.occupied:
        break
      
      yield square

  def generate_legal_moves(self):
    if self.turn:
      if self.transpose_available():
        available_transposes = self.transpose_available()
        for from_square, to_square in available_transposes:
          yield Move(from_square, to_square, transpose=True)

      single_moves = self.occupied_co[self.turn] & self.singles
      for from_square in scan_reversed(single_moves):
        for to_square in self.get_forward_moves(from_square):
          if self.delayed_crown:
            pass
          if self.crown_available(to_square):
            if self.perform_crown(from_square, to_square):
              available_singles = self.perform_crown(from_square, to_square)
              for avs in scan_reversed(available_singles):
                yield Move(from_square, to_square, crown=avs)
          elif self.bearoff_available(to_square):
            yield Move(from_square, to_square, bear_off=True)
          else:
            yield Move(from_square, to_square)

      double_moves = self.occupied_co[self.turn] & self.doubles
      for from_square in scan_reversed(double_moves):
        for to_square in self.get_backward_moves(from_square):
          if self.bearoff_available(to_square):
            yield Move(from_square, to_square, bear_off=True)
          else:
            yield Move(from_square, to_square)

    else:
      if self.transpose_available():
        available_transposes = self.transpose_available()
        for from_square, to_square in available_transposes:
          yield Move(from_square, to_square, transpose=True)

      single_moves = self.occupied_co[self.turn] & self.singles
      for from_square in scan_reversed(single_moves):
        for to_square in self.get_backward_moves(from_square):
          if self.crown_available(to_square):
            if self.perform_crown(from_square, to_square):
              available_singles = self.perform_crown(from_square, to_square)
              for avs in scan_reversed(available_singles):
                yield Move(from_square, to_square, crown=avs)
          else:
            yield Move(from_square, to_square)

      double_moves = self.occupied_co[self.turn] & self.doubles
      for from_square in scan_reversed(double_moves):
        for to_square in self.get_forward_moves(from_square):
          if self.bearoff_available(to_square):
            yield Move(from_square, to_square, bear_off=True)
          else:
            yield Move(from_square, to_square)

  def push(self, move: Move) -> None:
    
    assert self.is_legal(move), "Illegal move"

    moving_piece = self.piece_at(move.from_square)
    self.remove_piece_at(move.from_square)
    self.remove_piece_at(move.crown)  # remove selected piece to perform crown

    if move.crown is not None:
      self.set_piece_at(move.to_square, DOUBLE, moving_piece.color)
    elif move.bear_off:
      self.set_piece_at(move.to_square, SINGLE, moving_piece.color)
    elif move.transpose:
      self.set_piece_at(move.from_square, DOUBLE, moving_piece.color)
      self.set_piece_at(move.to_square, SINGLE, moving_piece.color)
    else:
      self.set_piece_at(move.to_square, moving_piece.piece_type, moving_piece.color)

    self.turn = not self.turn

  def crown_available(self, to_square: Square):
    if self.turn:
      return BB_SQUARES[to_square] & BB_RANK_8
    else:
      return BB_SQUARES[to_square] & BB_RANK_1

  def bearoff_available(self, to_square: Square):
    if self.turn:
      return BB_SQUARES[to_square] & BB_RANK_1  # a single move cannot reach the nearest row
    else:
      return BB_SQUARES[to_square] & BB_RANK_8

  def transpose_available(self):
    available_transpose = []

    bb_singles = set(scan_reversed(self.occupied_co[self.turn] & self.singles))
    bb_doubles = scan_reversed(self.occupied_co[self.turn] & self.doubles)
    for d in bb_doubles:
      if self.turn:
        if d-7 in bb_singles:
          available_transpose.append((d-7, d))
        if d-9 in bb_singles:
          available_transpose.append((d-9, d))
      else:
        if d+7 in bb_singles:
          available_transpose.append((d+7, d))
        if d+9 in bb_singles:
          available_transpose.append((d+9, d))

    return available_transpose

  def perform_crown(self, from_square: Square, to_square: Square):
    available_singles = self.occupied_co[self.turn] & self.singles ^ BB_SQUARES[from_square]
    if available_singles is not None:
      return available_singles
    else:
      print("No available singles for crown")
      self.delayed_crown[self.turn] = to_square
      return 0

  def is_legal(self, move: Move) -> bool:
    if not move:
      return False
    
    from_mask = BB_SQUARES[move.from_square]
    to_mask = BB_SQUARES[move.to_square]

    # check turn
    if not self.occupied_co[self.turn] & from_mask:
      return False

    return True

  def set_board_fen(self, board_fen: str) -> None:
    pass

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

    print("".join(builder) + "\n")


class Game:
  def __init__(self) -> None:
    self.board = Board()

  def selfplay(self, iterations: int):
    for i in range(iterations):
      print(f"Move {i}")
      print(COLOR_NAMES[game.board.turn])
      moves = list(game.board.legal_moves)
      print(f"Legal moves {moves}")
      if len(moves):
        move = choice(moves)
        print(move)
        game.board.push(move)
      else:
        print(f"No more legal moves for {COLOR_NAMES[game.board.turn]}")
        game.board.print_board()
        break
      
      game.board.print_board()
      print("*" * 16)

game = Game()

game.selfplay(100)

# game.board.print_board()

start = time.monotonic_ns()

end = time.monotonic_ns()
print(f"Time elapsed during the process: {(end - start)} ns")
