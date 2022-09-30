from typing import Optional
from chess import SQUARE_NAMES, PieceType, Square, piece_symbol


class Move():
  def __init__(self, from_square: Square, to_square: Square, crown: Optional[PieceType] = None) -> None:
    self.from_square = from_square
    self.to_square = to_square
    self.crown = crown


  def uci(self) -> str:
    if self:
      return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square]
    elif self.crown:
      return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square] + piece_symbol(self.promotion)
    else:
      return "0000"


  def __repr__(self) -> str:
    return f"<Move>{self.uci()}"


  def __str__(self) -> str:
    return self.uci()
