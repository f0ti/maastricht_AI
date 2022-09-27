from board import Board

class Game:
  
  def __init__(self) -> None:
    self.board = Board()

game = Game()
game.board.render_piece_map()
