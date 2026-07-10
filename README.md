# corridors

A Quoridor-family console game with an alpha-beta solver.

## Rules

- 11 rows (0..10) × 9 cols (A..I). Rows 1..9 are the playable area; rows 0 and 10 are end zones.
- Each player picks a starting column in their own end zone. That cell becomes the opponent's goal.
- Pawns move one step in the 4 compass directions. When adjacent to the opponent's pawn, jump straight over; if blocked behind, side-jump perpendicularly.
- Each player has 9 walls. Walls span two cells and cannot fully cut off either player's path to their goal.
- Reach the opponent's starting cell to win.

## Play

```
PYTHONPATH=src py -m corridors            # interactive mode chooser
PYTHONPATH=src py -m corridors --hh       # human vs human
PYTHONPATH=src py -m corridors --p2-ai    # human (P1) vs AI (P2)
PYTHONPATH=src py -m corridors --p1-ai --p2-ai   # AI vs AI
```

In-game commands: `m E9`, `w D3H`, `hint`, `moves`, `board`, `undo`, `resign`, `quit`.

## Layout

- `src/corridors/game.py` — State, Board, walls, BFS, move generation
- `src/corridors/solver.py` — alpha-beta + PVS + TT + Zobrist + IDDFS
- `src/corridors/play.py` — console loop and board renderer
- `tests/test_game.py` — sanity tests

## Tests

```
py -m pytest -q
```
