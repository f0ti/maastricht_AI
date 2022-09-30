from typing import Iterator
from chess import Board


class MoveGenerator():
  def __init__(self, board: Board) -> None:
    self.board = board

  def count(self) -> int:
    return len(list(self))

  def __iter__(self) -> Iterator[Move]:
    