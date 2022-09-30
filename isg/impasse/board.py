from baseboard import BaseBoard


class Board(BaseBoard):
  def __init__(self, board_fen: str) -> None:
    super().__init__(board_fen)

    self.move_stack = []

    if board_fen:
      self.set_board_fen(board_fen)
    else:
      self.reset_board()

  