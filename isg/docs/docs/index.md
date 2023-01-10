# Impasse Game Engine

## Design

### Color

color is a boolean data type, either True for the White player or False for the Black player

### Piece

  - PieceType
    piece type is an integer data type, 1 for the single pieces,
    2 for the double (stacked) pieces
  
  - PieceSymbols, PieceNames
    since the engine is a console app, the peice symbols are represented by the "⛀", "⛁" and "⛂", "⛃" UNICODE characters respectively for the black and white and black pieces for each of the piece types (0 index of the list is skipped so to ease the element access using the PieceType (int) as indices)

### Square

square is an int data type for each of the 64 squares, indices order is from left to right, with the bottom left square as index 0 and top right square as index 63
https://www.chessprogramming.org/File:BBUniverse.jpg

### Bitboard

one bitboard is an int data type, in binary it represents in ones and zeros the occupied and free squares for the bitboard is represents there are needed only 4 bitboards (integers) to keep the state of the board at any time
  - the black squares bitboard
  - the white squares bitboard
  - the single squares bitboard
  - the double squares bitboard

bitwise conjuction (AND) of the above represents the board state