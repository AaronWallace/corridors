"""NetworkAgent: plays Corridors using a trained ValueNet (1-ply lookahead).

pick_move: enumerate legal moves; short-circuit immediate wins; batch-encode
all children; single forward pass; negate (child value is from the opponent's
perspective); epsilon-band random tie-break for variety.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

from ..game import Board, Move, State, apply_move, legal_moves
from . import model as model_mod
from .encoding import encode_state


class NetworkAgent:
    def __init__(self, checkpoint: str, device: str = "cpu",
                 epsilon_band: float = 0.02, seed: Optional[int] = None) -> None:
        self.name = checkpoint
        self.device = device
        self.model = model_mod.load_checkpoint(checkpoint, device=device)
        self.epsilon_band = epsilon_band
        self.rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self.rng.seed(seed)

    @torch.no_grad()
    def pick_move(self, state: State, board: Board) -> Move:
        moves = legal_moves(state, board)
        if not moves:
            raise RuntimeError("no legal moves")
        children: List[State] = []
        for mv in moves:
            child = apply_move(state, mv)
            if child.winner(board) is not None:
                return mv  # immediate win
            children.append(child)

        batch = np.stack([encode_state(c, board) for c in children])
        x = torch.from_numpy(batch).to(self.device)
        # Child values are from the opponent's (side-to-move of child) view; negate.
        values = (-self.model(x)).cpu().numpy()

        best = float(values.max())
        band = [m for m, v in zip(moves, values) if v >= best - self.epsilon_band]
        return self.rng.choice(band)
