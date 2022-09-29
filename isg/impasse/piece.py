PIECE_SYMBOLS = [None, "s", "d"]


class Piece:

  def __init__(self, piece_type=1, color=True, position=None) -> None:
    self.color = color
    self.piece_type = piece_type
    self.position = position
    self.removed = False

  def symbol(self) -> str:
    symbol = PIECE_SYMBOLS[self.piece_type]
    return symbol.upper() if self.color else symbol

  def row_position(self) -> int:
    if self.position:
      return int(self.position / 4) + 1

  def __repr__(self) -> str:
    return f"Piece: {self.symbol()}"

  def __str__(self) -> str:
    return f"{self.symbol()} - {self.position.lower()}"

  def bear_off(self) -> None:
    self.removed = True
    self.position = None
