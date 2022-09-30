from typing import Iterator
from chess import Board

from move import Move


class MoveGenerator():
  def __init__(self, board: Board) -> None:
    self.board = board

  def count(self) -> int:
    return len(list(self))

  def __iter__(self) -> Iterator[Move]:
    return self.board.generate_legal_moves()
  
  def __contains__(self, move: Move) -> bool:
    return self.board.is_legal(move)
