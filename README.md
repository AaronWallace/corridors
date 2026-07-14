# Corridors

A Quoridor-family board game engine with an alpha-beta solver and neural network training pipeline.

## Quick start (Linux / WSL)

```bash
git clone https://github.com/AaronWallace/corridors.git
cd corridors
chmod +x setup.sh corridors.sh
./setup.sh
./corridors.sh
```

The setup script installs dependencies, detects NVIDIA GPUs and installs PyTorch+CUDA automatically, then verifies the install. It creates a venv when appropriate (skips it when running as root in a container).

On Windows, use the PowerShell launcher instead:

```powershell
.\corridors.ps1
```

## Web app

Start the local browser interface from the project directory:

```powershell
.\corridors-web.ps1
```

On Linux/macOS:

```bash
./corridors-web.sh
```

Then open `http://127.0.0.1:8765`. The server discovers and preloads available
checkpoints from `nn_checkpoints`, and supports human-vs-AI and AI-vs-AI games.

## Rules

- 11 rows (0..10) x 9 cols (A..I). Rows 1..9 are playable; rows 0 and 10 are end zones.
- Each player picks a starting column in their own end zone. That cell becomes the opponent's goal.
- End-zone rows have no lateral movement. A pawn's first move is straight onto the board, it cannot return to its own end zone, and it may enter the opponent's end zone only at the exact goal cell.
- Pawns move one step in the 4 compass directions. An adjacent opponent may be jumped only straight ahead when the landing cell is open; side-jumps are not allowed.
- Each player has 9 walls. Walls span two cells, cannot overlap or cross, and cannot fully cut off either player's path.
- A move or wall placement may not immobilize a pawn; every player must retain an available pawn move.
- Reach the opponent's starting cell to win.
- The third occurrence of an identical position is a draw. Automated games also retain a configurable maximum-ply draw as a safety limit.

## Features

- **Autoplay** — AI vs AI with live Rich TUI or headless mode for SSH
- **Multiprocess parallelism** — dynamic worker count, queue-based messaging
- **Neural network pipeline** — generate self-play data, train a value network (PyTorch + CUDA), round-robin Elo tournaments
- **Balanced AlphaZero exploration** — equal logits give pawn and wall action types equal aggregate prior/noise, avoiding branching-count bias
- **Alpha-beta solver** — PVS + aspiration windows + iterative deepening + transposition table (in-memory + SQLite) + Zobrist hashing + move ordering + selective extensions

## Layout

| Path | Description |
|------|-------------|
| `src/corridors/game.py` | State, Board, walls, BFS, move generation |
| `src/corridors/solver.py` | Alpha-beta + PVS + TT + Zobrist + IDDFS |
| `src/corridors/play.py` | Main menu, autoplay, Rich TUI, headless mode |
| `src/corridors/parallel.py` | Multiprocess game workers |
| `src/corridors/settings.py` | Persistent settings (corridors.json) |
| `src/corridors/nn/` | Neural network package |
| `src/corridors/nn/encoding.py` | State → 9-plane tensor encoding |
| `src/corridors/nn/datasets.py` | Shard NPZ dataset management |
| `src/corridors/nn/model.py` | ValueNet (6 res blocks, ~453k params) |
| `src/corridors/nn/train.py` | Training loop (AdamW + cosine LR) |
| `src/corridors/nn/agent.py` | NetworkAgent for tournament play |
| `src/corridors/nn/tournament.py` | Round-robin Elo tournaments |
| `src/corridors/nn/menu.py` | NN training submenu |
| `tests/test_game.py` | Game engine tests |

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```
