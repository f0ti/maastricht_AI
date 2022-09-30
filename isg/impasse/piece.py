PIECE_SYMBOLS = [None, "s", "d"]


class Piece:

  def __init__(self, piece_type=1, color=True) -> None:
    self.color = color
    self.piece_type = piece_type
    self.removed = False

  def symbol(self) -> str:
    symbol = PIECE_SYMBOLS[self.piece_type]
    return symbol.upper() if self.color else symbol

  def __repr__(self) -> str:
    return f"Piece: {self.symbol()}"

  def __str__(self) -> str:
    return f"{self.symbol()}"
