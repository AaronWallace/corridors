"""NetworkAgent: plays Corridors using a trained network.

Supports both architectures:
  - ValueNet (value-only): 1-ply lookahead, batch-evaluate all children
  - AZNet (policy+value): use policy head directly, no search needed

pick_move: short-circuit immediate wins; epsilon-band random tie-break.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

from ..game import Board, Move, State, apply_move, legal_moves
from .encoding import encode_state


def _load_model(checkpoint: str, device: str):
    """Load checkpoint, auto-detecting architecture from .meta.json."""
    from . import model as model_mod
    meta = model_mod.read_meta(checkpoint)
    if meta.get("arch") == "az":
        from . import az_net
        return az_net.load_checkpoint(checkpoint, device=device), "az"
    return model_mod.load_checkpoint(checkpoint, device=device), "value"


class NetworkAgent:
    def __init__(self, checkpoint: str, device: str = "cpu",
                 epsilon_band: float = 0.02, seed: Optional[int] = None) -> None:
        self.name = checkpoint
        self.device = device
        self.model, self.arch = _load_model(checkpoint, device=device)
        self.epsilon_band = epsilon_band
        self.rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self.rng.seed(seed)

    @torch.no_grad()
    def pick_move(self, state: State, board: Board) -> Move:
        moves = legal_moves(state, board)
        if not moves:
            raise RuntimeError("no legal moves")

        # Short-circuit immediate wins. Only a pawn move onto the mover's own
        # goal can win — wall moves never end the game, so don't apply them.
        goal = board.p1_goal if state.turn == 1 else board.p2_goal
        for mv in moves:
            if mv[0] == "m" and mv[1] == goal:
                return mv

        if self.arch == "az":
            return self._pick_az(state, board, moves)
        return self._pick_value(state, board, moves)

    def _pick_az(self, state: State, board: Board, moves: List[Move]) -> Move:
        """AZ net: use policy head directly."""
        from .actions import move_to_index
        tensor = encode_state(state, board)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        logits = policy_logits[0].cpu().numpy()

        # Mask illegal moves, softmax over legal
        indices = [move_to_index(m) for m in moves]
        legal_logits = logits[indices]
        legal_logits -= legal_logits.max()
        exp = np.exp(legal_logits)
        probs = exp / exp.sum()

        best = float(probs.max())
        band = [m for m, p in zip(moves, probs) if p >= best - self.epsilon_band]
        return self.rng.choice(band)

    def _pick_value(self, state: State, board: Board, moves: List[Move]) -> Move:
        """Value-only net: 1-ply lookahead over all children."""
        children = [apply_move(state, mv) for mv in moves]
        batch = np.stack([encode_state(c, board) for c in children])
        x = torch.from_numpy(batch).to(self.device)
        values = (-self.model(x)).cpu().numpy()

        best = float(values.max())
        band = [m for m, v in zip(moves, values) if v >= best - self.epsilon_band]
        return self.rng.choice(band)
