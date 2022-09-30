from board import Board


class Game():
  def __init__(self) -> None:
    self.board = Board()

game = Game()
print(game.board.print_board())
