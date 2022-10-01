import time
from board import Board
from utils import print_legal_moves

class Game:
  def __init__(self) -> None:
    self.board = Board()

game = Game()

start = time.monotonic_ns()

game.board.legal_moves

end = time.monotonic_ns()
print(f"Time elapsed during the process: {(end - start)} ns")
