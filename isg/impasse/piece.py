PIECE_SYMBOLS = [None, "s", "d"]


class Piece:

  def __init__(self) -> None:
    self.color = True
    self.piece_type = 1
    self.position = None
    self.removed = False

  def symbol(self) -> str:
    symbol = PIECE_SYMBOLS[self.piece_type]
    return symbol.upper() if self.color else symbol

  def row_position(self) -> int:
    return int(self.position / 4) + 1

  def __repr__(self) -> str:
    return f"Piece: {self.symbol()}"

  def __str__(self) -> str:
    return f"{self.symbol()} - {self.position} - {self.row_position()}"

  def bear_off(self) -> None:
    self.removed = True
    self.position = None
