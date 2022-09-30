import time
from board import *
from move_generator import MoveGenerator
from utils import *


class Game():
  def __init__(self) -> None:
    self.board = Board()

game = Game()
print(game.board.print_board())

start = time.monotonic()

backward_moves = game.board.get_backward_moves(A5)

end = time.monotonic()
print(f"Time elapsed during the process: {(end - start)} ms")

# move_gen = MoveGenerator(game.board)

# render_mask(moves)
