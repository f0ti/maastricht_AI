import time
import numpy as np
import gmpy2

from typing import Generator, Generic, Iterator, List, Optional, TypeVar
from utils import print_legal_moves, render_mask

"""
  Color
  -----
    color is a boolean data type, either True for the White player
    or False for the Black player
"""

Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

"""
  Piece
  -----
  PieceType
    piece type is an integer data type, 1 for the single pieces,
    2 for the double (stacked) pieces
  
  PieceSymbols, PieceNames
    since the engine is a console app, the peice symbols are represented
    by the "⛀", "⛁" and "⛂", "⛃" UNICODE characters respectively for the
    black and white and black pieces for each of the piece types
    (0 index of the list is skipped so to ease the element access using the
    PieceType (int) as indices)
  
"""

PieceType = int
PIECE_TYPES = [SINGLE, DOUBLE] = [1, 2]
# PIECE_SYMBOLS = (("None", "s", "D"), ("None", "s", "d"))
PIECE_SYMBOLS = (("None", "⛀", "⛁"), ("None", "⛂", "⛃"))
PIECE_NAMES = [None, "single", "double"]

"""
  Square
  ------
    square is an int data type for each of the 64 squares, indices order is
    from left to right, with the bottom left square as index 0 and top right
    square as index 63
    https://www.chessprogramming.org/File:BBUniverse.jpg
"""

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

def square_mirror(square: Square) -> Square:
  return square ^ 0x38

"""
  Bitboard
  --------
    one bitboard is an int 64-bit data type, in binary it represents in ones
    and zeros the occupied and free squares in the board (piece centric)
    there are needed only 4 bitboards (integers) to keep the state of the
    board at any time
      - the black squares bitboard
      - the white squares bitboard
      - the single squares bitboard
      - the double squares bitboard
    bitwise conjuction (AND) of the above represents the board state

    [+] Advantages
      - very very fast move generation
      - memory efficient
    [-] Challenges
      - complicated moves need more thought and sometimes are hard to generate with bitwise operators
"""

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


"""
  Piece
  -----
    a piece consists of (piece_type, color)
    piece_type is a constant from the PieceType enum
    color is a boolean, True for white, False for black

    the symbol method returns the unicode symbol of the piece
    "⛀", "⛁" and "⛂", "⛃"

    the __repr__ and __str__ methods return the unicode symbol of the piece
"""
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

"""
  Move
  ----
    a move consists of (from_square, to_square, transpose, bear_off, impasse, crown)

    the __repr__ and __str__ methods return the uci representation of the move

    the uci method returns the uci representation of the move
    e.g. "a1b1", "a1b1[-><-]", "a1b1[X][X]", "a1b1[-><-][X][X]"

    from_square and to_square are integers from 0 to 63
    transpose is a boolean, True if the move is a transpose
    bear_off is a boolean, True if the move is a bear off
    impasse is a boolean, True if the move is an impasse
    crown is an integer from 0 to 63, the square where the piece is crowned
"""
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

  # returns the uci representation of the move
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

"""
  BoardState
  ----------
    a board state consists of (singles, doubles, occupied_w, occupied_b, occupied, turn)
    
    singles is a bitboard of the singles
    doubles is a bitboard of the doubles
    occupied_w is a bitboard of the white pieces
    occupied_b is a bitboard of the black pieces
    occupied is a bitboard of the occupied squares
    turn is a boolean, True for white, False for black

    the restore method restores the board state to a board

    useful for storing the board state before making a move, and restoring it after
"""
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

    board.occupied_co[WHITE] = self.occupied_w
    board.occupied_co[BLACK] = self.occupied_b
    board.occupied = self.occupied

    board.turn = self.turn

"""
  Board
  -----
    a board consists of (move_stack, stack, turn, occupied_co, singles, doubles, occupied)

    move_stack is a list of moves
    stack is a list of board states
    turn is a boolean, True for white, False for black
    occupied_co is a list of bitboards of the white and black pieces
    singles is a bitboard of the singles
    doubles is a bitboard of the doubles
    occupied is a bitboard of the occupied squares

    the legal_moves property returns the legal moves as a list

    the push method makes a move on the board and pushes the board state to the stack

    the pop method pops the last move from the move stack and restores the board state

    the board_state method returns the board state as a BoardState object

    the reset_board method resets the board to the starting position

    the piece_type_at method returns the piece type at a square

    the remove_piece_at method removes a piece from a square

    the set_piece_at method sets a piece at a square

    the piece_at method returns the piece at a square

    the get_backward_moves method returns the backward moves for white doubles and black singles

    the get_forward_moves method returns the forward moves for white singles and black doubles

    the generate_basic_moves method generates the basic moves (forward, backward, transpose and crown)

    the generate_impasse_moves method generates the impasse moves

    the generate_crown_moves method generates the crown moves

    the peek_for_crown method peeks for a crown move

    the crown_available method returns True if a crown move is available

    the perform_crown method performs a crown move

    the bearoff_available method returns True if a bear off move is available

    the transpose_available method returns True if a transpose move is available

    the generate_moves method generates all the legal moves (basic and impasse)

    the is_game_over method returns True if the game is over

    the side_removed_all_pieces method returns True if a side has removed all their pieces

    the print_board method prints the board to the console, using the unicode pieces, . for empty squares
"""
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

  def push(self, move: Move):
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
    
    # check if piece is there
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
        yield Move(square, square, impasse=True)
      elif self.piece_type_at(square) == DOUBLE:
        move = Move(square, square, impasse=True)
        # check if there is crown available
        crown_moves = self.peek_for_crown(move)
        if crown_moves is not None:
          for cm in crown_moves:
            move.crown = cm 
            yield move
        # if there is no crown available, yield the move which contains the
        # piece to remove via the impasse rule
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

  def generate_moves(self) -> List:
    legal_moves = list(self.generate_basic_moves())
    # generate impasse moves
    if not len(legal_moves):
      legal_moves = list(self.generate_impasse_moves())

    return legal_moves

  def is_game_over(self) -> bool:
    if self.side_removed_all():
      print(f"Game Over. Winner {COLOR_NAMES[not self.turn]}")
      return True

    return False
  
  def side_removed_all(self) -> bool:
    if self.occupied_co[not self.turn] == 0:
      return True

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

"""
  Valuator
  --------

  The valuator is used to evaluate the board state. It is used to determine
  the best move for the AI to make. The valuator is based on the following
  rules:
    - more single pieces in the furthest row for white, the better, and likewise
      more single pieces in the nearest  row for black, the better
    - more double pieces in the nearest  row for white, the better, and likewise
      more double pieces in the furthest row for black, the better
    - The more single pieces, the better
    - The less double pieces, the better

    top_highest contains the value map for each square on the board for the pieces 
    that are valued more if they are in the furthest row

    top_nearest contains the value map for each square on the board for the pieces
    that are valued more if they are in the nearest row

    single_value is the value of a single piece

    double_value is the value of a double piece

    impasse_value is the value of an impasse move

    crown_value is the value of a crown move

    transpose_value is the value of a transpose move

    singles_disadvantage is the difference in number of single pieces between the
    opponent and the current moving side

    doubles_disadvantage is the difference in number of double pieces between the
    opponent and the current moving side

    singles_map_array is the total value for each square on the board in accordance
    to the value map position of the piece

    doubles_map_array is the total value for each square on the board in accordance
    to the value map position of the piece

    evaluate is the total value of the board state
"""
class Valuator:

  def __init__(self) -> None:
    # single white and double black share the same value table
    # the furthest, the higher the value, the better
    self.top_highest = np.array([
      [0, 10,  0, 10,  0, 10,  0, 10],
      [8,  0,  8,  0,  8,  0,  8,  0],
      [0,  6,  0,  6,  0,  6,  0,  6],
      [5,  0,  5,  0,  5,  0,  5,  0],
      [0,  3,  0,  3,  0,  3,  0,  3],
      [2,  0,  2,  0,  2,  0,  2,  0],
      [0,  1,  0,  1,  0,  1,  0,  1],
      [0,  0,  0,  0,  0,  0,  0,  0]
    ])

    # single black and double white share the same value table
    # the nearest, the higher the value, the better
    self.top_nearest = np.array([
      [0,  0,  0,  0,  0,  0,  0,  0],
      [1,  0,  1,  0,  1,  0,  1,  0],
      [0,  2,  0,  2,  0,  2,  0,  2],
      [3,  0,  3,  0,  3,  0,  3,  0],
      [0,  5,  0,  5,  0,  5,  0,  5],
      [6,  0,  6,  0,  6,  0,  6,  0],
      [0,  8,  0,  8,  0,  8,  0,  8],
      [10, 0,  10, 0,  10, 0,  10, 0]
    ])

    self.single_value = 5
    self.double_value = 3

    self.transpose_value = 3
    self.impasse_value = 7

  def reset(self):
    self.count = 0

  def __call__(self, board_state: Board, reached_end=False) -> float:
    if not reached_end:
      self.count += 1
    return self.evaluate(board_state)

  def total_pieces(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn])

  def total_singles(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn] & board_state.singles)

  def total_doubles(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn] & board_state.doubles)

  def transpose_available(self, board_state: Board) -> int:
    return bool(board_state.transpose_available())

  def impasse_available(self, board_state: Board) -> int:
    legal_moves = list(board_state.generate_basic_moves())

    # if there are legal moves available, then the list is not empty
    # if the list is empty, then there are no legal moves available
    # leading to an impasse situation

    return not bool(len(legal_moves))

  # the biggest the disadvantage, the better
  # if disadvantage is negative, it will lead to a lower evaluation value
  # the maximizing player aims to positively increase the difference of singlesreturn bool(board_state.bearoff_available())
  def singles_disadvantage(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[~board_state.turn] & board_state.singles) - count_bits(board_state.occupied_co[board_state.turn] & board_state.singles)

  def doubles_disadvantage(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[~board_state.turn] & board_state.doubles) - count_bits(board_state.occupied_co[board_state.turn] & board_state.doubles)

  def checkers_disadvantage(self, board_state: Board) -> int:
    return count_bits(board_state.occupied_co[board_state.turn]) - count_bits(board_state.occupied_co[~board_state.turn])

  def singles_map_array(self, board_state: Board) -> np.ndarray:
    singles = board_state.singles & board_state.occupied_co[board_state.turn]

    return np.flip(np.array([np.uint8(x) for x in bin(singles)[2:].zfill(64)], dtype=np.uint8).reshape(8, 8), 1)

  def doubles_map_array(self, board_state: Board) -> np.ndarray:
    doubles = board_state.doubles & board_state.occupied_co[board_state.turn]

    return np.flip(np.array([np.uint8(x) for x in bin(doubles)[2:].zfill(64)], dtype=np.uint8).reshape(8, 8), 1)

  def evaluate(self, board_state: Board) -> float:
    
    # Old Evaluation Function 1

    # BIAS_SINGLES_ADV = 0.3
    # BIAS_DOUBLES_ADV = 0.5
    # BIAS_UPPERMOST_SINGLES = 0.4
    # BIAS_LOWERMOST_DOUBLES = 0.8

    # h_value1 = (
    #     BIAS_SINGLES_ADV * self.total_singles(board_state)
    #   + BIAS_DOUBLES_ADV * self.total_doubles(board_state)
    #   + BIAS_UPPERMOST_SINGLES * self.singles_uppermost_halfboard(board_state)
    #   + BIAS_LOWERMOST_DOUBLES * self.doubles_lowermost_halfboard(board_state)
    # )


    # Old Evaluation Function 2

    # BIAS_SINGLES_DIS = 0.5
    # BIAS_DOUBLES_DIS = 0.8
    # BIAS_CHECKERS_DIS = 0.6

    # h_value2 = BIAS_SINGLES_DIS * self.singles_disadvantage(board_state) + \
    #            BIAS_DOUBLES_DIS * self.doubles_disadvantage(board_state) + \
    #            BIAS_CHECKERS_DIS * self.checkers_disadvantage(board_state)


    # Evaluation Function 3

    # pushing the move, changes the turn, which is why we need to invert it in order
    # to evaluate the board state from the perspective of the player who is about to
    # move, not the player who will move next
    board_state.turn = ~board_state.turn

    # if it is white's turn, then the top is the highest for singles and the nearest for doubles
    # if it is black's turn, then the top is the nearest for singles and the highest for doubles
    if board_state.turn:
      single_map = self.singles_map_array(board_state) * self.top_highest
      double_map = self.doubles_map_array(board_state) * self.top_nearest
    else:
      single_map = self.singles_map_array(board_state) * self.top_nearest
      double_map = self.doubles_map_array(board_state) * self.top_highest

    h_value3 = (
      np.sum(single_map) + \
      np.sum(double_map) + \
      self.singles_disadvantage(board_state) * self.single_value + \
      self.doubles_disadvantage(board_state) * self.double_value + \
      self.impasse_available(board_state) * self.impasse_value + \
      self.transpose_available(board_state) * self.transpose_value
    )

    ## DEBUGGING INDIVIDUAL COMPONENTS OF THE EVALUATION FUNCTION ##

    # print("Total singles: ", self.total_singles(board_state) * self.single_value + np.sum(single_map))
    # print("Total doubles: ", self.total_doubles(board_state) * self.double_value + np.sum(double_map))
    # print("Singles dsvgt: ", self.singles_disadvantage(board_state) * self.single_value)
    # print("Doubles dsvgt: ", self.doubles_disadvantage(board_state) * self.double_value)
    # print("Impasse      : ", self.impasse_available(board_state) * self.impasse_value)
    # print("Transpose    : ", self.transpose_available(board_state) * self.transpose_value)
    # print("Total        : ", h_value3)

    # revert the turn change
    board_state.turn = ~board_state.turn

    return float(h_value3)

"""
  Game
  ----

  Runs different game modes and AI search algorithm if needed

    - selfplay: plays against itself, moves can also be randomly chosen,
    otherwise the best move is chosen

    - human_human: plays against another human, by printing each move to the
    console with its index, and asking for the index of the move to be played

    - human_AI: plays against the AI, by printing each move to the console with
    its index, and asking for the index of the move to be played

    - test_game: debugging environment

    - alphabeta_minimax: runs the alphabeta minimax algorithm

    - explore_leaves: initializes AI search environment with configs and runs the
    minimax algorithm and returns the best moves
"""
class Game:
  def __init__(self) -> None:
    self.board = Board()
    self.valuator = Valuator()
    self.move_number = 0

  def selfplay(self, random=False) -> None:
    while not self.board.is_game_over():
      self.move_number += 1
      print(f"turn {COLOR_NAMES[self.board.turn]} move {self.move_number}")
      
      moves = sorted(self.explore_leaves(self.board, self.valuator), key=lambda x: x[0], reverse=self.board.turn)
      if len(moves) == 0:
        print("no move")
        return
      
      print("Top 5 moves:")
      for m in moves[:5]:
        print("  ", m)

      # moves are chosen randomly from the top 5 moves, mostly used for testing
      if random:
        selected_move_index = np.random.randint(0,len(moves))
        selected_move = moves[selected_move_index][1]
      else:
        selected_move = moves[0][1]
      
      self.board.push(selected_move)
      print(f"Played move: {selected_move}")

      self.board.print_board()
      print("=" * 20)

  def human_human(self):
    self.board.print_board()
    while not self.board.is_game_over():
      self.move_number += 1
      print(f"Move {self.move_number} - {COLOR_NAMES[self.board.turn]}")

      evaluation_move, moves = sorted(self.explore_leaves(self.board, self.valuator), key=lambda x: x[0], reverse=self.board.turn)
      print(evaluation_move)
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

      # human play
      if self.board.turn:
        for i, (v, m) in enumerate(moves):
          print(f"{i:02} - {m} - {v}")

        # input move and parse
        selected_move_index = int(input("Write index move: "))
        selected_move = moves[selected_move_index][1]

      else:
        # AI plays the first best move in the list
        selected_move = moves[0][1]

      self.board.push(selected_move)
      print(f"Played move: {selected_move}")

      self.board.print_board()
      print("*" * 16)

  def test_game(self):
    v = Valuator()
    v.reset()
    print(v(self.board))
    print(self.board.stack)
    print(self.board.singles)
    print(self.board.turn)
    self.board.print_board()

    move = self.board.legal_moves[0]

    self.board.push(move)
    
    print(v(self.board))
    print(self.board.stack)
    print(self.board.singles)
    print(self.board.move_stack)
    print(self.board.turn)
    self.board.print_board()

    undo_move = self.board.pop()
    
    print(v(self.board))
    print(self.board.stack)
    print(self.board.singles)
    print(self.board.turn)
    self.board.print_board()

    return undo_move

  # AI

  def alphabeta_minimax(self, state: Board, valuator: Valuator, depth: int, a: int, b: int, pv=False):

    MAX_DEPTH = 3

    # depth of search is variable depending on the number or legal moves and pieces on the board
    legal_moves = state.legal_moves
    if 0 < len(legal_moves) < 3:
      MAX_DEPTH = 10
    elif 3 <= len(legal_moves) < 8:
      MAX_DEPTH = 7

    nr_pieces = count_bits(state.occupied_co[state.turn])
    if 0 < nr_pieces < 2:
      MAX_DEPTH = 10
    elif 2 <= nr_pieces < 5:
      MAX_DEPTH = 7

    assert MAX_DEPTH > 0, "Maximum depth must be greater than 0"

    # if depth is higher than MAX_DEPTH or game is over, return evaluation
    if depth >= MAX_DEPTH or state.side_removed_all():
      return self.valuator(state, reached_end=True)

    # get current player turn
    turn = state.turn
    
    # initialize max evaluation for each player
    if turn == WHITE:
      max_evaluation = -1000
    else:
      max_evaluation = 1000
    
    # principal variation search
    if pv:
      move_eval = []

    # evaluate all moves and save their value    
    move_ordering = []
    for move in state.legal_moves:
      state = state.push(move)
      state_value = self.valuator(state)
      move_ordering.append((state_value, move))
      state.pop()

    # sort moves by value, descending for white, ascending for black
    moves = sorted(move_ordering, key=lambda x: x[0], reverse=state.turn)

    # get only top 10 moves if search depth goes beyond 3
    if depth >= 3:
      moves = moves[:10]

    # index 1 is the move, index 0 is the value
    for move in [x[1] for x in moves]:
      # make the temporary move (push and pop) and evaluate the state
      state.push(move)
      tree_value = self.alphabeta_minimax(state, valuator, depth+1, a, b)
      state.pop()

      # backpropagate through the principal variation
      if pv:
        move_eval.append((tree_value, move))

      if turn == WHITE:
        max_evaluation = max(max_evaluation, tree_value)
        a = max(a, max_evaluation)
        if a >= b:
          break  # b cut-off
      else:
        max_evaluation = min(max_evaluation, tree_value)
        b = min(b, max_evaluation)
        if a >= b:
          break  # a cut-off
      
    if pv:
      return max_evaluation, move_eval
    else:
      return max_evaluation

  def explore_leaves(self, state: Board, valuator: Valuator):
    start = time.time()
    
    self.valuator.reset()

    print("thinking...")
    max_evaluation, move_evaluation = self.alphabeta_minimax(state, valuator, 0, a=-1000, b=1000, pv=True)
    
    search_time = time.time() - start
    print("Board evaluation:", max_evaluation)
    print(f"Explored {valuator.count-1} nodes in {search_time:.3f} seconds | {int(valuator.count/search_time)} nodes/sec")

    return move_evaluation


# ============== PLAY GAME ==============

game = Game()

start = time.monotonic_ns()
game.selfplay()
# game.human_AI()
end = time.monotonic_ns()
print(f"Time elapsed during the process: {(end - start)/10**6} ms")
