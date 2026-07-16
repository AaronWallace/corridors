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
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..game import Board, Move, State, apply_move
from .actions import NUM_ACTIONS, legal_move_mask

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRAC = 0.25
LEGAL_TARGET_EPSILON = 1e-8
# Cap for the optional per-game expansion cache (legal moves + priors per
# state). Entries share interned Move tuples, so ~3 KB each worst case.
EXPANSION_CACHE_MAX = 20_000

# state -> (moves, action-index array, prior array). See run_mcts's
# expansion_cache parameter.
ExpansionCache = Dict[State, Tuple[List[Move], np.ndarray, np.ndarray]]


def _action_type_groups(moves: Sequence[Move]) -> Tuple[np.ndarray, ...]:
    """Indices grouped as pawn moves and wall moves, omitting empty groups."""
    pawn: List[int] = []
    wall: List[int] = []
    for i, move in enumerate(moves):
        (pawn if move[0] == "m" else wall).append(i)
    return tuple(np.asarray(group, dtype=np.intp)
                 for group in (pawn, wall) if group)


def _balanced_priors(logits: np.ndarray, moves: Sequence[Move]) -> np.ndarray:
    """Softmax with equal aggregate mass per action type for equal logits."""
    adjusted = np.asarray(logits, dtype=np.float32).copy()
    for group in _action_type_groups(moves):
        adjusted[group] -= math.log(len(group))
    adjusted -= adjusted.max()
    exp = np.exp(adjusted)
    return exp / exp.sum()


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
        self.child_indices: Optional[np.ndarray] = None  # action indices (intp)
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

    def expand(self, policy_logits: np.ndarray,
               cache: Optional[ExpansionCache] = None) -> None:
        """Expand this node using the network's policy output.

        cache: optional per-game state -> (moves, indices, priors) memo.
        Legal-move generation is the most expensive part of a simulation, and
        transpositions re-reach the same state often, so hits skip it (and the
        prior computation) entirely. Only valid when evaluate_fn is
        deterministic per state within the game, which self-play guarantees.
        """
        cached = cache.get(self.state) if cache is not None else None
        if cached is not None:
            moves, indices, priors = cached
        else:
            moves, index_list = legal_move_mask(self.state, self.board)
            if not moves:
                self.is_terminal = True
                self.terminal_value = -1.0  # no moves = loss
                return
            indices = np.asarray(index_list, dtype=np.intp)
            priors = _balanced_priors(policy_logits[indices], moves)
            if cache is not None and len(cache) < EXPANSION_CACHE_MAX:
                cache[self.state] = (moves, indices, priors)

        self.child_moves = moves
        self.child_indices = indices
        self.children = [None] * len(moves)
        self.N = np.zeros(len(moves), dtype=np.float32)
        self.W = np.zeros(len(moves), dtype=np.float32)
        # Copy: Dirichlet noise and cache entries must never share storage.
        self.P = priors.astype(np.float32, copy=True)

    def add_dirichlet_noise(self, alpha: float = DIRICHLET_ALPHA,
                            frac: float = DIRICHLET_FRAC) -> None:
        if self.P is None:
            return
        groups = _action_type_groups(self.child_moves)
        noise = np.zeros_like(self.P)
        type_mass = 1.0 / len(groups)
        for group in groups:
            noise[group] = (
                np.random.dirichlet([alpha] * len(group)).astype(np.float32)
                * type_mass
            )
        self.P = (1 - frac) * self.P + frac * noise

    def select_child(self, c_puct: float = C_PUCT) -> int:
        """PUCT selection — returns child index."""
        n = self.N
        sqrt_total = math.sqrt(self.n_total + 1)
        # W is 0 wherever N is 0, so dividing by max(N, 1) gives exactly the
        # old where(N > 0, W/N, 0) without the (slow) errstate guard.
        q = self.W / np.maximum(n, 1.0)
        u = c_puct * self.P * sqrt_total / (1 + n)
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
    c_puct: float = C_PUCT,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    dirichlet_frac: float = DIRICHLET_FRAC,
    state_history: Optional[Sequence[State]] = None,
    remaining_plies: Optional[int] = None,
    expansion_cache: Optional[ExpansionCache] = None,
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

    state_history: actual game positions through ``root_state``. A simulation
        that produces a third occurrence of an exact position is scored as a
        draw. When omitted, the root is treated as its first occurrence.

    remaining_plies: optional number of moves before the game's maximum-ply
        draw. Simulations reaching that horizon are scored as draws, except
        when the final move wins the game.

    expansion_cache: optional per-game dict memoizing legal moves and priors
        by state (see Node.expand). Share one dict across all run_mcts calls
        of a game; do not share across games (goals differ) or across
        nondeterministic evaluate_fn implementations.

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

    history_counts = Counter(state_history or ())
    if not state_history or state_history[-1] != root_state:
        history_counts[root_state] += 1

    if not root.expanded:
        policy, value = evaluate_fn(root_state, board)
        root.expand(policy, expansion_cache)
        if root.is_terminal:  # no legal moves
            pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
            return pi, root.terminal_value, None, None

    if add_noise:
        root.add_dirichlet_noise(dirichlet_alpha, dirichlet_frac)

    # Top up to num_simulations total visits — reused visits already count.
    history_get = history_counts.get
    while root.n_total < num_simulations:
        node = root
        path: List[Tuple[Node, int]] = []
        path_counts: Dict[State, int] = {}
        adjudicated_draw = False

        # Selection — descend to a leaf
        while node.expanded and not node.is_terminal:
            ci = node.select_child(c_puct)
            path.append((node, ci))

            child = node.children[ci]
            if child is None:
                child_state = apply_move(node.state, node.child_moves[ci])
                child = Node(child_state, board)
                node.children[ci] = child
            node = child

            if not node.is_terminal:
                s = node.state
                occurrences = path_counts.get(s, 0) + 1
                path_counts[s] = occurrences
                if history_get(s, 0) + occurrences >= 3:
                    adjudicated_draw = True
                    break
                if remaining_plies is not None and len(path) >= remaining_plies:
                    adjudicated_draw = True
                    break

        # Evaluation
        if adjudicated_draw:
            leaf_value = 0.0
        elif node.is_terminal:
            leaf_value = node.terminal_value
        else:
            p, v = evaluate_fn(node.state, node.board)
            node.expand(p, expansion_cache)
            leaf_value = float(v)

        # Backup — alternate sign as we go up (each level is the opponent)
        value = -leaf_value
        for parent, ci in reversed(path):
            parent.backup(ci, value)
            value = -value

    # Build policy target from root visit counts. A negligible positive floor
    # records the exact legal-action mask in the policy target. Training uses
    # it to apply the same action-type normalization without storing a
    # separate 227-element mask.
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    pi[root.child_indices] = np.maximum(root.N, LEGAL_TARGET_EPSILON)

    # Select move
    if temperature < 0.01:
        # Deterministic — pick most visited
        best = int(np.argmax(root.N))
    else:
        # Sample proportional to N^(1/temp)
        counts = root.N.copy()
        if temperature != 1.0:
            counts = counts ** (1.0 / temperature)
            if not np.all(np.isfinite(counts)):
                # Large budgets at low temperature can overflow float32
                # (e.g. 10000^10). Renormalizing by the max first is exact
                # for the sampling distribution and cannot overflow.
                counts = (root.N.astype(np.float64) / root.N.max())
                counts **= (1.0 / temperature)
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
