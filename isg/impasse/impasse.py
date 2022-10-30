import time
import numpy as np
import gmpy2

from typing import Generator, Generic, Iterator, List, Optional, TypeVar
from utils import print_legal_moves, render_mask

# ------ color ------

Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

# ------ piece ------

PieceType = int
PIECE_TYPES = [SINGLE, DOUBLE] = [1, 2]
# PIECE_SYMBOLS = (("None", "s", "D"), ("None", "s", "d"))
PIECE_SYMBOLS = (("None", "⛀", "⛁"), ("None", "⛂", "⛃"))
PIECE_NAMES = [None, "single", "double"]

# ------ square ------

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

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

def parse_square(name: str) -> Square:
  return SQUARE_NAMES.index(name)

def square_name(square: Square) -> str:
  return SQUARE_NAMES[square]

def square(file_index: int, rank_index: int) -> Square:
  return rank_index * 8 + file_index

def square_file(square: Square) -> int:
  return square & 7

def square_rank(square: Square) -> int:
  return square >> 3

def square_mirror(square: Square) -> Square:
  return square ^ 0x38

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

SQUARES_180 = [square_mirror(sq) for sq in SQUARES]

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

BB_UPPER_HALF_RANKS = BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8
BB_LOWER_HALF_RANKS = BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4

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

def count_bits(bitboard: int) -> int:
  return gmpy2.popcount(gmpy2.mpz(int(bitboard)))


class Piece:

  def __init__(self, piece_type=SINGLE, color=True) -> None:
    self.color = color
    self.piece_type = piece_type

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
    impasse: bool = False,
    crown: Optional[Square] = None,
    ) -> None:

      self.from_square = from_square
      self.to_square = to_square
      self.transpose = transpose
      self.bear_off = bear_off
      self.impasse = impasse
      self.crown = crown

  def uci(self) -> str:

    uci = f"{SQUARE_NAMES[self.from_square]}{SQUARE_NAMES[self.to_square]}"

    if self.transpose:
      uci += f"[-><-]"

    if self.impasse:
      uci += f"[X][X]"

    if self.bear_off:
      uci += f"[X]"

    if self.crown is not None:
      uci += f"[{SQUARE_NAMES[self.crown[0]]}*{SQUARE_NAMES[self.crown[1]]}]"

    return uci

  def __repr__(self) -> str:
    return f"<Move>{self.uci()}"

  def __str__(self) -> str:
    return self.uci()


BoardT = TypeVar("BoardT", bound="Board")

class BoardState(Generic[BoardT]):
  def __init__(self, board: BoardT) -> None:
    self.singles = board.singles
    self.doubles = board.doubles
    
    self.occupied_w = board.occupied_co[WHITE]
    self.occupied_b = board.occupied_co[BLACK]
    self.occupied = board.occupied
    # self.occupied_co = [self.occupied_b, self.occupied_w]

    self.turn = board.turn

  def restore(self, board: BoardT) -> None:
    board.singles = self.singles
    board.doubles = self.doubles

    board.occupied = self.occupied
    board.occupied_co[WHITE] = self.occupied_w
    board.occupied_co[BLACK] = self.occupied_b

    board.turn = self.turn


class Board:
  def __init__(self, board_state: BoardState = None) -> None:
    
    if board_state:
      self.move_stack : List[Move] = []
      self.stack : List[BoardState[BoardT]] = []

      self.turn = board_state.turn

      self.occupied_co = [board_state.occupied_b, board_state.occupied_w]

      self.singles = board_state.singles
      self.doubles = board_state.doubles

      self.occupied = board_state.occupied_b | board_state.occupied_w
    
    else:
      self.reset_board()

  @property
  def legal_moves(self) -> List:
    return self.generate_moves()

  def pop(self: BoardT) -> Move:
    
    move = self.move_stack.pop()
    self.stack.pop().restore(self)

    return move

  def board_state(self: BoardT) -> BoardState[BoardT]:
    return BoardState(self)

  def reset_board(self) -> None:
    self.occupied_co = [BB_EMPTY, BB_EMPTY]
    self.turn = WHITE
    self.move_stack = []
    self.stack = []
    self.delayed_crown = [False, False]
    self.delayed_crown_squares = [None, None]

    self.singles = BB_D8 | BB_H8 | BB_A7 | BB_E7 | BB_D2 | BB_H2 | BB_A1 | BB_E1
    self.doubles = BB_B8 | BB_F8 | BB_C7 | BB_G7 | BB_B2 | BB_F2 | BB_C1 | BB_G1

    self.occupied_co[WHITE] = BB_B8 | BB_F8 | BB_C7 | BB_G7 | BB_D2 | BB_H2 | BB_A1 | BB_E1
    self.occupied_co[BLACK] = BB_D8 | BB_H8 | BB_A7 | BB_E7 | BB_B2 | BB_F2 | BB_C1 | BB_G1

    self.occupied = self.occupied_co[WHITE] | self.occupied_co[BLACK]

  def get_state(self) -> BoardState:

    return BoardState(
      singles=self.singles,
      doubles=self.doubles,
      occupied=self.occupied,
      occupied_co_w=self.occupied_co[WHITE],
      occupied_co_b=self.occupied_co[BLACK],
      delayed_crown_w=self.delayed_crown[WHITE],
      delayed_crown_b=self.delayed_crown[BLACK],
      delayed_crown_squares_w=self.delayed_crown_squares[WHITE],
      delayed_crown_squares_b=self.delayed_crown_squares[BLACK],
      turn=self.turn,
    )
  
  def pieces_mask(self, piece_type: PieceType, color: Color) -> Bitboard:
    if piece_type == SINGLE:
      bb = self.singles
    elif piece_type == DOUBLE:
      bb = self.doubles
    else:
      assert False, "Incorrect PieceType"

    return bb & self.occupied_co[color]
  
  def piece_type_at(self, square: Square) -> PieceType:
    if square is None:
      return

    mask = BB_SQUARES[square]

    if (self.occupied & mask) == 0:
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

    # XOR adds a piece if it is not there
    self.occupied ^= mask
    
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
      return
    
    self.occupied ^= mask
    self.occupied_co[color] ^= mask

  def piece_at(self, square) -> Piece:
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
      square = r*8 + f
      # if occupied square is faced then stop the slide
      if (1 << square) & self.occupied:
        break

      yield square

    # down-left
    for r, f in zip(range(tr-1, -1, -1), range(tf-1, -1, -1)):
      square = r*8 + f
      if (1 << square) & self.occupied:
        break
      yield square

  def get_forward_moves(self, square: Square) -> Generator:
    tr, tf = square // 8, square % 8

    # up-right
    for r, f in zip(range(tr+1, 8), range(tf+1, 8)):
      square = r*8 + f
      if (1 << square) & self.occupied:
        break
      
      yield square
    
    # up-left
    for r, f in zip(range(tr+1, 8), range(tf-1, -1, -1)):
      square = r*8 + f
      if (1 << square) & self.occupied:
        break
      
      yield square

  def generate_basic_moves(self) -> Generator:
    if self.turn:
      # TRANSPOSE
      if self.transpose_available():
        available_transposes = self.transpose_available()
        for from_square, to_square in available_transposes:
          if self.bearoff_available(from_square):
            move = Move(from_square, to_square, transpose=True, bear_off=True)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm 
                yield move
            else:
              yield move
          else:
            move = Move(from_square, to_square, transpose=True)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm
                yield move
            else:
              yield move

      # SINGLE SLIDE
      single_moves = self.occupied_co[self.turn] & self.singles
      for from_square in scan_reversed(single_moves):
        for to_square in self.get_forward_moves(from_square):
          move = Move(from_square, to_square)
          # check if there is crown available
          crown_moves = self.peek_for_crown(move)
          if crown_moves is not None:
            for cm in crown_moves:
              move.crown = cm
              yield move
          else:
            yield move

      # DOUBLE SLIDE
      double_moves = self.occupied_co[self.turn] & self.doubles
      for from_square in scan_reversed(double_moves):
        for to_square in self.get_backward_moves(from_square):
          if self.bearoff_available(to_square):
            move = Move(from_square, to_square, bear_off=True)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm 
                yield move
            else:
              yield move
          else:
            move = Move(from_square, to_square)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm 
                yield move
            else:
              yield move

    else:
      # TRANSPOSE
      if self.transpose_available():
        available_transposes = self.transpose_available()
        for from_square, to_square in available_transposes:
          if self.bearoff_available(from_square):
            move = Move(from_square, to_square, transpose=True, bear_off=True)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm 
                yield move
            else:
              yield move
          else:
            move = Move(from_square, to_square, transpose=True)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm
                yield move
            else:
              yield move

      # SINGLE SLIDE
      single_moves = self.occupied_co[self.turn] & self.singles
      for from_square in scan_reversed(single_moves):
        for to_square in self.get_backward_moves(from_square):
          move = Move(from_square, to_square)
          # check if there is crown available
          crown_moves = self.peek_for_crown(move)
          if crown_moves is not None:
            for cm in crown_moves:
              move.crown = cm 
              yield move
          else:
            yield move

      # DOUBLE SLIDE
      double_moves = self.occupied_co[self.turn] & self.doubles
      for from_square in scan_reversed(double_moves):
        for to_square in self.get_forward_moves(from_square):
          if self.bearoff_available(to_square):
            move = Move(from_square, to_square, bear_off=True)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm 
                yield move
            else:
              yield move
          else:
            move = Move(from_square, to_square)
            # check if there is crown available
            crown_moves = self.peek_for_crown(move)
            if crown_moves is not None:
              for cm in crown_moves:
                move.crown = cm 
                yield move
            else:
              yield move

  def generate_impasse_moves(self) -> Generator:
    available_pieces = self.occupied_co[self.turn]  # pieces to remove

    for square in scan_reversed(available_pieces):
      if self.piece_type_at(square) == SINGLE:
        # single remove does not effect on crown status
        yield Move(square, square, impasse=True)
      elif self.piece_type_at(square) == DOUBLE:
        move = Move(square, square, impasse=True)
        # check if there is crown available
        crown_moves = self.peek_for_crown(move)
        if crown_moves is not None:
          for cm in crown_moves:
            move.crown = cm 
            yield move
        else:
          yield move
      else:
        yield

  def generate_crown_moves(self) -> Generator:
    crown_pairs = []
    if self.turn:
      crown_square = self.occupied_co[self.turn] & self.singles & BB_RANK_8  # singles on the furthest row
      free_singles = self.occupied_co[self.turn] & self.singles              # free singles used for crown
    else:
      crown_square = self.occupied_co[self.turn] & self.singles & BB_RANK_1  # singles on the furthest row
      free_singles = self.occupied_co[self.turn] & self.singles              # free singles used for crown

    for crownable in scan_reversed(crown_square):
      for available_single in scan_reversed(free_singles):
        if crownable == available_single:
          continue
        crown_pairs.append((crownable, available_single))  # from_square -> furthest single, to_square -> free single

    if len(crown_pairs) != 0:
      return crown_pairs
    else:
      return None

  def generate_moves(self) -> List:
    legal_moves = list(self.generate_basic_moves())
    # generate impasse moves
    if not len(legal_moves):
      legal_moves = list(self.generate_impasse_moves())

    return legal_moves

  def peek_for_crown(self, move: Move) -> List:
    self.push(move)
    self.turn = not self.turn

    crown_moves = self.generate_crown_moves()

    self.pop()

    return crown_moves

  def crown_available(self) -> Bitboard:
    if self.turn:
      return self.occupied_co[self.turn] & self.singles & BB_RANK_8
    else:
      return self.occupied_co[self.turn] & self.singles & BB_RANK_1

  def perform_crown(self, move: Move) -> None:
    self.remove_piece_at(move.to_square)
    self.set_piece_at(move.from_square, DOUBLE, self.turn)

  def bearoff_available(self, to_square: Square) -> Bitboard:
    if self.turn:
      return BB_SQUARES[to_square] & BB_RANK_1
    else:
      return BB_SQUARES[to_square] & BB_RANK_8

  def transpose_available(self) -> List:
    available_transpose = []

    bb_singles = set(scan_reversed(self.occupied_co[self.turn] & self.singles))
    bb_doubles = scan_reversed(self.occupied_co[self.turn] & self.doubles)
    for d in bb_doubles:
      if self.turn:
        # down-left adjacent single
        if d-7 in bb_singles:
          available_transpose.append((d-7, d))
        # down-right adjacent single
        if d-9 in bb_singles:
          available_transpose.append((d-9, d))
      else:
        # up-right adjacent single
        if d+7 in bb_singles:
          available_transpose.append((d+7, d))
        # up-left adjacent single
        if d+9 in bb_singles:
          available_transpose.append((d+9, d))

    return available_transpose

  def push(self, move: Move):
    assert self.is_legal(move), "Illegal move"
    assert self, "Move is None"

    self.move_stack.append(move)
    board_state = self.board_state()
    self.stack.append(board_state)

    moving_piece = self.piece_at(move.from_square)
    if moving_piece is None:
      print("Move from square", move.from_square)
      print("Move", move)
    self.remove_piece_at(move.from_square)

    if move.bear_off:
      self.set_piece_at(move.to_square, SINGLE, moving_piece.color)
      if move.transpose:
        self.set_piece_at(move.from_square, SINGLE, moving_piece.color)
    elif move.impasse:
      # if the removed checker was double, replace with a single
      if moving_piece.piece_type == DOUBLE:
        self.set_piece_at(move.from_square, SINGLE, moving_piece.color)  
    elif move.transpose:
      self.set_piece_at(move.from_square, DOUBLE, moving_piece.color)
      self.set_piece_at(move.to_square, SINGLE, moving_piece.color)
    else:
      self.set_piece_at(move.to_square, moving_piece.piece_type, moving_piece.color)

    # perform crown
    if move.crown is not None:
      self.remove_piece_at(move.crown[1])
      self.set_piece_at(move.crown[0], DOUBLE, moving_piece.color)

    self.turn = not self.turn

    return self

  def is_game_over(self) -> bool:
    if self.side_removed_all():
      print(f"Game Over. Winner {COLOR_NAMES[not self.turn]}")
      return True

    return False

  def side_removed_all(self) -> bool:
    if self.occupied_co[not self.turn] == 0:
      print(self.occupied_co[not self.turn])
      return True

  def is_legal(self, move: Move) -> bool:
    if not move:
      return False
    
    from_mask = BB_SQUARES[move.from_square]
    to_mask = BB_SQUARES[move.to_square]

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


class Valuator:
  def __init__(self) -> None:
    pass

  def reset(self):
    self.count = 0

  def __call__(self, board_state: Board) -> float:
    self.count += 1
    return self.evaluate(board_state)

  def total_checkers(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn])

  def total_singles(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn] & board_state.singles)

  def total_doubles(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn] & board_state.doubles)

  def singles_uppermost_halfboard(self, board_state: Board) -> int:
    if board_state.turn:
      return count_bits(board_state.occupied_co[board_state.turn] & board_state.singles & BB_UPPER_HALF_RANKS)
    else:
      return count_bits(board_state.occupied_co[board_state.turn] & board_state.singles & BB_LOWER_HALF_RANKS)

  def doubles_lowermost_halfboard(self, board_state: Board) -> int:
    if board_state.turn:
      return count_bits(board_state.occupied_co[board_state.turn] & board_state.doubles & BB_LOWER_HALF_RANKS)
    else:
      return count_bits(board_state.occupied_co[board_state.turn] & board_state.doubles & BB_UPPER_HALF_RANKS)

  def transpose_available(self, board_state: Board) -> int:
    return bool(board_state.transpose_available())

  def singles_disadvantage(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[~board_state.turn] & board_state.singles) - count_bits(board_state.occupied_co[board_state.turn] & board_state.singles)

  def doubles_disadvantage(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[~board_state.turn] & board_state.doubles) - count_bits(board_state.occupied_co[board_state.turn] & board_state.doubles)

  def checkers_disadvantage(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn]) - count_bits(board_state.occupied_co[~board_state.turn])

  def evaluate(self, board_state: Board) -> float:
    
    # Evaluation Function 1

    BIAS_SINGLES_ADV = 0.3
    BIAS_DOUBLES_ADV = 0.5
    BIAS_UPPERMOST_SINGLES = 0.4
    BIAS_LOWERMOST_DOUBLES = 0.8

    h_value1 = (
        BIAS_SINGLES_ADV * self.total_singles(board_state)
      + BIAS_DOUBLES_ADV * self.total_doubles(board_state)
      + BIAS_UPPERMOST_SINGLES * self.singles_uppermost_halfboard(board_state)
      + BIAS_LOWERMOST_DOUBLES * self.doubles_lowermost_halfboard(board_state)
    )

    # Evaluation Function 2

    BIAS_SINGLES_DIS = 0.5
    BIAS_DOUBLES_DIS = 0.8
    BIAS_CHECKERS_DIS = 0.6

    h_value2 = BIAS_SINGLES_DIS * self.singles_disadvantage(board_state) + \
               BIAS_DOUBLES_DIS * self.doubles_disadvantage(board_state) + \
               BIAS_CHECKERS_DIS * self.checkers_disadvantage(board_state)

    return np.round(np.abs(h_value2), 2)

class Game:
  def __init__(self) -> None:
    self.board = Board()
    self.valuator = Valuator()
    self.move_number = 0

  def selfplay(self) -> None:
    while not self.board.is_game_over():
      self.move_number += 1
      print(f"turn {COLOR_NAMES[self.board.turn]} move {self.move_number}")
      
      moves = sorted(self.explore_leaves(self.board, self.v), key=lambda x: x[0], reverse=self.board.turn)
      if len(moves) == 0:
        print("no move")
        return
      
      print("top 5:")
      for m in moves[0:5]:
        print("  ", m)

      # for i, (v, m) in enumerate(moves):
      #   print(f"{i:02} - {v} - {m}")

      # move = choice(moves)[1]
      move = moves[0][1]
      self.board.push(move)
      print(f"Played move: {move}")

      self.board.print_board()
      print("=" * 20)

  def human_human(self):
    self.board.print_board()
    while not self.board.is_game_over():
      self.move_number += 1
      print(f"Move {self.move_number} - {COLOR_NAMES[self.board.turn]}")

      moves = sorted(self.explore_leaves(self.board, self.v), key=lambda x: x[0], reverse=self.board.turn)
      if len(moves) == 0:
        print("no move")
        return

      for i, (v, m) in enumerate(moves):
        print(f"{i:02} - {m} - {v}")

      # input move
      selected_move_index = int(input("Write index move: "))
      selected_move = moves[selected_move_index][1]
      print(selected_move)

      self.board.push(selected_move)
      print(f"Played move: {selected_move}")

      self.board.print_board()
      print("*" * 16)

  def human_AI(self):
    self.board.print_board()
    
    while not self.board.is_game_over():
      self.move_number += 1
      print(f"Move {self.move_number} - {COLOR_NAMES[self.board.turn]}")

      moves = sorted(self.explore_leaves(self.board, self.valuator), key=lambda x: x[0], reverse=self.board.turn)
      if len(moves) == 0:
        print("no move")
        return

      # hooman play
      if self.board.turn:
        for i, (v, m) in enumerate(moves):
          print(f"{i:02} - {m} - {v}")

        # input move
        selected_move_index = int(input("Write index move: "))
        selected_move = moves[selected_move_index][1]

      else:
        selected_move = moves[0][1]

      self.board.push(selected_move)
      print(f"Played move: {selected_move}")

      self.board.print_board()
      print("*" * 16)

  def test_game(self):
    self.board.print_board()
    print(self.board.stack)
    print(self.board.turn)
    move = self.board.legal_moves[0]
    print(self.board.stack)
    self.board.push(move)
    print(self.board.turn)
    self.board.print_board()
    undo_move = self.board.pop()
    print(self.board.stack)
    print(self.board.turn)
    self.board.print_board()

    return undo_move

  def alphabeta_minimax(self, state: Board, valuator: Valuator, depth: int, a, b, pv=False):
    MAX_DEPTH = 7
    if depth >= MAX_DEPTH or state.side_removed_all():
      return self.valuator(state)

    turn = state.turn
    if turn == WHITE:
      ret = -1000
    else:
      ret = 1000
    
    if pv:
      move_eval = []
    
    move_ordering = []
    for move in state.legal_moves:
      state.push(move)
      move_ordering.append((self.valuator(state), move))
      state.pop()

    moves = sorted(move_ordering, key=lambda x: x[0], reverse=state.turn)

    # get only top 10 moves if search depth goes beyond 3
    if depth >= 3:
      moves = moves[:10]

    for move in [x[1] for x in moves]:
      state.push(move)
      tree_value = self.alphabeta_minimax(state, valuator, depth+1, a, b)
      state.pop()

      # backpropagate through the principal variation
      if pv:
        move_eval.append((tree_value, move))

      if turn == WHITE:
        ret = max(ret, tree_value)
        a = max(a, ret)
        if a >= b:
          break  # b cut-off
      else:
        ret = min(ret, tree_value)
        b = min(b, ret)
        if a >= b:
          break  # a cut-off
      
    if pv:
      return ret, move_eval
    else:
      return ret

  def explore_leaves(self, state: Board, valuator: Valuator):
    start = time.time()
    
    self.valuator.reset()
    current_evaluation = valuator(state)
    search_evaluation, move_evaluation = self.alphabeta_minimax(state, valuator, 0, a=-1000, b=1000, pv=True)
    
    search_time = time.time() - start

    print(f"{current_evaluation:.2f} -> {search_evaluation:.2f}")
    print(f"Explored {valuator.count} nodes in {search_time:.3f} seconds {int(valuator.count/search_time)}/sec")

    return move_evaluation


# ============== PLAY GAME ==============

game = Game()

start = time.monotonic_ns()
game.human_AI()
end = time.monotonic_ns()
print(f"Time elapsed during the process: {(end - start)/10**6} ms")
