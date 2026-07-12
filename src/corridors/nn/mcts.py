"""Monte Carlo Tree Search with neural network guidance (AlphaZero-style).

The tree is stored as parallel numpy arrays for cache efficiency.
Each node tracks visit count N, total value W, prior P, and children.

Key features:
  - Dirichlet noise at root for exploration
  - PUCT selection (c_puct = 1.5)
  - Virtual losses for safe batched inference (not used in single-threaded mode)
  - Temperature-based move selection
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..game import Board, Move, State, apply_move, legal_moves
from .actions import NUM_ACTIONS, legal_move_mask, move_to_index

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRAC = 0.25


class Node:
    __slots__ = (
        "state", "board", "turn", "is_terminal", "terminal_value",
        "children", "child_moves", "child_indices",
        "N", "W", "P", "n_total",
    )

    def __init__(self, state: State, board: Board) -> None:
        self.state = state
        self.board = board
        self.turn = state.turn
        self.is_terminal = False
        self.terminal_value = 0.0
        self.children: Optional[List[Optional[Node]]] = None
        self.child_moves: Optional[List[Move]] = None
        self.child_indices: Optional[List[int]] = None
        self.N: Optional[np.ndarray] = None  # visit counts per child
        self.W: Optional[np.ndarray] = None  # total value per child
        self.P: Optional[np.ndarray] = None  # prior probability per child
        self.n_total = 0

        w = state.winner(board)
        if w is not None:
            self.is_terminal = True
            self.terminal_value = 1.0 if w == state.turn else -1.0

    @property
    def expanded(self) -> bool:
        return self.children is not None

    def expand(self, policy_logits: np.ndarray) -> None:
        """Expand this node using the network's policy output."""
        moves, indices = legal_move_mask(self.state, self.board)
        if not moves:
            self.is_terminal = True
            self.terminal_value = -1.0  # no moves = loss
            return

        logits = policy_logits[indices]
        logits -= logits.max()
        exp = np.exp(logits)
        priors = exp / exp.sum()

        self.child_moves = moves
        self.child_indices = indices
        self.children = [None] * len(moves)
        self.N = np.zeros(len(moves), dtype=np.float32)
        self.W = np.zeros(len(moves), dtype=np.float32)
        self.P = priors.astype(np.float32)

    def add_dirichlet_noise(self, alpha: float = DIRICHLET_ALPHA,
                            frac: float = DIRICHLET_FRAC) -> None:
        if self.P is None:
            return
        noise = np.random.dirichlet([alpha] * len(self.P)).astype(np.float32)
        self.P = (1 - frac) * self.P + frac * noise

    def select_child(self) -> int:
        """PUCT selection — returns child index."""
        sqrt_total = math.sqrt(self.n_total + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(self.N > 0, self.W / self.N, 0.0)
        u = C_PUCT * self.P * sqrt_total / (1 + self.N)
        return int(np.argmax(q + u))

    def backup(self, child_idx: int, value: float) -> None:
        """Backup value from child's perspective (negated for parent)."""
        self.N[child_idx] += 1
        self.W[child_idx] += value
        self.n_total += 1


def run_mcts(
    root_state: State,
    board: Board,
    evaluate_fn,
    num_simulations: int = 200,
    temperature: float = 1.0,
    add_noise: bool = True,
    reuse_root: Optional["Node"] = None,
) -> Tuple[np.ndarray, float, Optional[Move], Optional["Node"]]:
    """Run MCTS from root_state.
    Returns (policy_target, root_value, selected_move, selected_child).

    evaluate_fn(state, board) -> (policy_logits [227], value scalar)
        Called for each leaf node that needs expansion.

    reuse_root: the previously-played move's child node (its subtree carries
        forward — see tree reuse). If it matches root_state and is already
        expanded, its accumulated visits/values are kept and we only run enough
        new simulations to top up to num_simulations. Pass the returned
        selected_child back in as reuse_root next move.

    policy_target: normalized visit counts over the 227-action space (training target).
    root_value: average value at root after all simulations.
    selected_move: the move selected according to temperature (None if terminal).
    selected_child: the node under selected_move, for reuse next move (None if terminal).
    """
    # Tree reuse: continue the carried-over subtree when it is this position.
    if (reuse_root is not None and reuse_root.expanded
            and not reuse_root.is_terminal and reuse_root.state == root_state):
        root = reuse_root
    else:
        root = Node(root_state, board)

    if root.is_terminal:
        pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
        return pi, root.terminal_value, None, None

    if not root.expanded:
        policy, value = evaluate_fn(root_state, board)
        root.expand(policy)
        if root.is_terminal:  # no legal moves
            pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
            return pi, root.terminal_value, None, None

    if add_noise:
        root.add_dirichlet_noise()

    # Top up to num_simulations total visits — reused visits already count.
    while root.n_total < num_simulations:
        node = root
        path: List[Tuple[Node, int]] = []

        # Selection — descend to a leaf
        while node.expanded and not node.is_terminal:
            ci = node.select_child()
            path.append((node, ci))

            child = node.children[ci]
            if child is None:
                child_state = apply_move(node.state, node.child_moves[ci])
                child = Node(child_state, board)
                node.children[ci] = child
            node = child

        # Evaluation
        if node.is_terminal:
            leaf_value = node.terminal_value
        else:
            p, v = evaluate_fn(node.state, node.board)
            node.expand(p)
            leaf_value = float(v)

        # Backup — alternate sign as we go up (each level is the opponent)
        value = -leaf_value
        for parent, ci in reversed(path):
            parent.backup(ci, value)
            value = -value

    # Build policy target from root visit counts
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for i, idx in enumerate(root.child_indices):
        pi[idx] = root.N[i]

    # Select move
    if temperature < 0.01:
        # Deterministic — pick most visited
        best = int(np.argmax(root.N))
    else:
        # Sample proportional to N^(1/temp)
        counts = root.N.copy()
        if temperature != 1.0:
            counts = counts ** (1.0 / temperature)
        s = counts.sum()
        if s <= 0:
            best = int(np.argmax(root.N))
        else:
            best = int(np.random.choice(len(root.child_moves), p=counts / s))

    selected_move = root.child_moves[best]

    # Materialize the chosen child so its subtree can be reused next move.
    selected_child = root.children[best]
    if selected_child is None:
        selected_child = Node(apply_move(root.state, root.child_moves[best]), board)
        root.children[best] = selected_child

    # Normalize pi to sum to 1
    total = pi.sum()
    if total > 0:
        pi /= total

    root_value = float(root.W.sum() / max(root.n_total, 1))
    return pi, root_value, selected_move, selected_child
