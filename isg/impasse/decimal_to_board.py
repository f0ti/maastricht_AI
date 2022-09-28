#!/usr/bin/env python3

import numpy as np
import sys

n = int(sys.argv[1])
binary = '{:032b}'.format(n)
board = np.array(list(binary), dtype=int).reshape(-1, 4)

print(binary)
print(board)
