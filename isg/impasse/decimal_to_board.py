#!/usr/bin/env python3

import numpy as np
import sys

n = int(sys.argv[1])
binary = "{:064b}".format(n)
board = np.array(list(binary), dtype=int).reshape(-1, 8)

board = np.flip(board, 1)
print(binary)
print(board)
