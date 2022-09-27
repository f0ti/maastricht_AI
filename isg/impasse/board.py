import numpy as np
from piece import Piece

START_POS_WHITE = {
  1: [28, 30, 25, 27],
  2: [0, 2, 5, 7]
}

START_POS_BLACK = {
  1: [1, 3, 4, 6],
  2: [24, 26, 29, 31]
}


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

  def get_possible_moves(self)

  def render_piece_map(self):
    piece_map = np.chararray(32)
    for p in self.pieces:
      piece_map[p.position] = p.symbol()

    self.piece_map = piece_map.reshape((-1, 4))
    print(self.piece_map)

  def list_pieces(self):
    for piece in self.pieces:
      print(piece)

  def render(self):
    print(self.piece_map)