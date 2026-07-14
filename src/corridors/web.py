"""Local web server for playing Corridors in a browser."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import solver
from .game import (
    NCOLS, State, WALLS_PER_PLAYER, apply_move, is_threefold_repetition,
    legal_moves,
)
from .settings import DEFAULTS
from .nn.checkpoints import (
    checkpoint_elo,
    load_elo_ratings,
    ranked_checkpoint_paths,
    resolve_checkpoint_path,
)

STATIC_ROOT = Path(__file__).with_name("web_static")
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent.parent / "nn_checkpoints"


def _model_catalog() -> list[dict]:
    catalog = []
    ratings = load_elo_ratings(CHECKPOINT_ROOT)
    for weights in ranked_checkpoint_paths(CHECKPOINT_ROOT):
        meta = {}
        meta_file = weights.with_suffix(".meta.json")
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        catalog.append({
            "name": weights.stem,
            "architecture": meta.get("arch", "value"),
            "elo": checkpoint_elo(weights, ratings),
            "validationLoss": meta.get("val_loss"),
            "positions": meta.get("positions"),
            "dataset": meta.get("data_run") or meta.get("dataset"),
            "loaded": weights.stem in REGISTRY.models if "REGISTRY" in globals() else False,
            "loadError": REGISTRY.preload_errors.get(weights.stem) if "REGISTRY" in globals() else None,
        })
    return catalog


def _agent_spec(raw: dict | None) -> dict:
    raw = raw or {}
    kind = raw.get("kind", "classical")
    if kind == "model":
        name = str(raw.get("checkpoint", ""))
        if not resolve_checkpoint_path(CHECKPOINT_ROOT, name).exists():
            raise ValueError(f"checkpoint not found: {name}")
        return {"kind": "model", "checkpoint": name}
    return {
        "kind": "classical",
        "depth": max(1, min(8, int(raw.get("depth", 4)))),
        "timeLimit": max(0.0, min(600.0, float(raw.get("timeLimit", 0.5)))),
    }


class SolverRegistry:
    """Long-lived solver resources shared by every browser game."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        from . import settings
        configured = settings.load()
        tt_path = (str(configured.get("tt_path", ""))
                   if configured.get("use_persistent_tt", True) else "")
        self.tt = solver.TT(sqlite_path=tt_path or None)
        self.models = {}
        self.preload_errors = {}
        self._preload_models()

    def _preload_models(self) -> None:
        if not CHECKPOINT_ROOT.exists():
            return
        for path in ranked_checkpoint_paths(CHECKPOINT_ROOT):
            try:
                from .nn.agent import NetworkAgent
                self.models[path.stem] = NetworkAgent(path.stem, device="cpu", seed=0)
            except Exception as exc:
                self.preload_errors[path.stem] = str(exc)

    def move(self, spec: dict, state, board):
        with self.lock:
            started = time.monotonic()
            if spec["kind"] == "model":
                agent = self.models.get(spec["checkpoint"])
                if agent is None:
                    from .nn.agent import NetworkAgent
                    agent = NetworkAgent(spec["checkpoint"], device="cpu", seed=0)
                    self.models[spec["checkpoint"]] = agent
                move = agent.pick_move(state, board)
                label = spec["checkpoint"]
            else:
                limit = spec["timeLimit"] or None
                move, _score, _stats, _pv = solver.best_move(
                    state, board, max_depth=spec["depth"], time_limit=limit,
                    tiebreak_epsilon=0, tt=self.tt, verbose=False,
                    flush_on_exit=False,
                )
                label = "classical"
            return move, {"solver": label, "elapsed": time.monotonic() - started}


@dataclass
class WebGame:
    board: object
    state: State
    players: dict
    mode: str
    history: list = field(default_factory=list)
    state_history: list[State] = field(default_factory=list)
    max_plies: int = int(DEFAULTS["max_plies"])
    draw_reason: str | None = None
    last_info: dict | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


REGISTRY = SolverRegistry()
GAMES: dict[str, WebGame] = {}
GAMES_LOCK = threading.RLock()


def _move_json(move):
    kind, arg = move
    return {"kind": kind, "at": list(arg)}


def _game_over(game: WebGame) -> bool:
    return game.state.winner(game.board) is not None or game.draw_reason is not None


def _adjudicate_draw(game: WebGame) -> None:
    if game.state.winner(game.board) is not None:
        return
    if is_threefold_repetition(game.state_history):
        game.draw_reason = "threefold repetition"
    elif len(game.history) >= game.max_plies:
        game.draw_reason = "maximum plies"


def _game_json(game_id: str, game: WebGame) -> dict:
    state, board = game.state, game.board
    moves = [] if _game_over(game) else legal_moves(state, board)
    return {
        "id": game_id,
        "mode": game.mode,
        "turn": state.turn,
        "winner": state.winner(board),
        "gameOver": _game_over(game),
        "drawReason": game.draw_reason,
        "pawns": {"1": list(state.p1), "2": list(state.p2)},
        "goals": {"1": list(board.p1_goal), "2": list(board.p2_goal)},
        "walls": [list(w) for w in sorted(state.walls)],
        "wallsLeft": {"1": state.p1_walls_left, "2": state.p2_walls_left},
        "players": game.players,
        "legal": [_move_json(m) for m in moves],
        "history": game.history[-12:],
        "lastInfo": game.last_info,
    }


def _new_game(payload: dict) -> tuple[str, WebGame]:
    p1_col = int(payload.get("p1Col", 4)) % NCOLS
    p2_col = int(payload.get("p2Col", 4)) % NCOLS
    board, state = State.start(p1_col, p2_col, WALLS_PER_PLAYER)
    mode = payload.get("mode", "human-ai")
    if mode == "ai-ai":
        players = {"1": _agent_spec(payload.get("p1")), "2": _agent_spec(payload.get("p2"))}
    else:
        human_side = str(payload.get("humanSide", "1"))
        ai_side = "2" if human_side == "1" else "1"
        players = {human_side: {"kind": "human"}, ai_side: _agent_spec(payload.get("ai"))}
    game_id = uuid.uuid4().hex[:12]
    max_plies = max(1, int(payload.get("maxPlies", DEFAULTS["max_plies"])))
    game = WebGame(board, state, players, mode, state_history=[state],
                   max_plies=max_plies)
    with GAMES_LOCK:
        GAMES[game_id] = game
    return game_id, game


def _parse_move(payload: dict):
    kind = payload.get("kind")
    at = payload.get("at", [])
    if kind == "m" and len(at) == 2:
        return ("m", (int(at[0]), int(at[1])))
    if kind == "w" and len(at) == 3:
        return ("w", (int(at[0]), int(at[1]), str(at[2]).upper()))
    raise ValueError("invalid move")


class Handler(BaseHTTPRequestHandler):
    server_version = "CorridorsWeb/1.0"

    def log_message(self, fmt, *args):
        return

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            models = _model_catalog()
            return self._json(200, {"checkpoints": [m["name"] for m in models],
                                    "models": models,
                                    "loaded": [m["name"] for m in models
                                               if m["name"] in REGISTRY.models],
                                    "loadErrors": REGISTRY.preload_errors})
        if path.startswith("/api/games/"):
            game_id = path.split("/")[3]
            game = GAMES.get(game_id)
            return self._json(200, _game_json(game_id, game)) if game else self._json(404, {"error": "game not found"})
        asset = "index.html" if path in ("", "/") else path.lstrip("/")
        file = (STATIC_ROOT / asset).resolve()
        if STATIC_ROOT.resolve() not in file.parents or not file.is_file():
            self.send_error(404)
            return
        body = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            path, payload = urlparse(self.path).path, self._body()
            if path == "/api/games":
                game_id, game = _new_game(payload)
                return self._json(201, _game_json(game_id, game))
            parts = path.split("/")
            if len(parts) != 5 or parts[1:3] != ["api", "games"]:
                return self._json(404, {"error": "not found"})
            game_id, action = parts[3], parts[4]
            game = GAMES.get(game_id)
            if not game:
                return self._json(404, {"error": "game not found"})
            with game.lock:
                if _game_over(game):
                    raise ValueError("game is over")
                player = game.players[str(game.state.turn)]
                if action == "move":
                    if player["kind"] != "human":
                        raise ValueError("current player is controlled by AI")
                    move = _parse_move(payload)
                    if move not in legal_moves(game.state, game.board):
                        raise ValueError("illegal move")
                    game.last_info = {"solver": "human", "elapsed": 0}
                elif action == "ai-move":
                    if player["kind"] == "human":
                        raise ValueError("current player is human")
                    move, game.last_info = REGISTRY.move(player, game.state, game.board)
                else:
                    return self._json(404, {"error": "not found"})
                mover = game.state.turn
                game.state = apply_move(game.state, move)
                game.state_history.append(game.state)
                game.history.append({"player": mover, **_move_json(move), **game.last_info})
                _adjudicate_draw(game)
                return self._json(200, _game_json(game_id, game))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(500, {"error": str(exc)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Corridors web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Corridors web app: http://{args.host}:{args.port}")
    print(f"Preloaded {len(REGISTRY.models)} neural model(s). Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        REGISTRY.tt.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
