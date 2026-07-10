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

## Rules

- 11 rows (0..10) x 9 cols (A..I). Rows 1..9 are playable; rows 0 and 10 are end zones.
- Each player picks a starting column in their own end zone. That cell becomes the opponent's goal.
- Pawns move one step in the 4 compass directions. Jump over an adjacent opponent; side-jump if blocked behind.
- Each player has 9 walls. Walls span two cells and cannot fully cut off either player's path.
- Reach the opponent's starting cell to win.

## Features

- **Autoplay** — AI vs AI with live Rich TUI or headless mode for SSH
- **Multiprocess parallelism** — dynamic worker count, queue-based messaging
- **Neural network pipeline** — generate self-play data, train a value network (PyTorch + CUDA), round-robin Elo tournaments
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
