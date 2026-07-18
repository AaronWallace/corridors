"""Breadth-first Recursive Monte-Carlo Tree Search (RMCTS).

This is an experimental implementation of Frankston and Howard's RMCTS
algorithm.  Unlike AlphaZero PUCT, the search tree is allocated from the
network prior, one breadth level at a time.  That makes every level one large
inference batch.  Values and optimized posterior policies are then computed
from the leaves back to the roots.

The public entry point searches a *batch* of Corridors positions.  Keeping the
batch boundary here is important: calling RMCTS once per position would throw
away the algorithm's main performance advantage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..game import Board, Move, State, apply_move
from .actions import NUM_ACTIONS, legal_move_mask
from .mcts import (
    C_PUCT,
    DIRICHLET_ALPHA,
    DIRICHLET_FRAC,
    LEGAL_TARGET_EPSILON,
    _action_type_groups,
    _balanced_priors,
)


BatchEvaluate = Callable[
    [Sequence[State], Sequence[Board]], tuple[np.ndarray, np.ndarray]
]


@dataclass
class RMCTSResult:
    """Search result for one root, plus accounting useful to benchmarks."""

    policy: np.ndarray
    value: float
    move: Optional[Move]
    simulations: int
    evaluations: int
    nodes: int
    max_depth: int


@dataclass
class _Node:
    state: State
    board: Board
    simulations: int
    parent: Optional["_Node"] = None
    parent_action: int = -1
    depth: int = 0
    terminal_value: Optional[float] = None
    network_value: float = 0.0
    moves: list[Move] = field(default_factory=list)
    indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.intp))
    prior: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32))
    children: list["_Node"] = field(default_factory=list)
    child_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32))
    posterior: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32))
    posterior_value: float = 0.0


def assign_simulations(budget: int, prior: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Systematically distribute ``budget`` integer simulations by ``prior``.

    This is Algorithm 3 in the RMCTS paper.  It uses one random offset, has an
    exact total, and each action's expected count is ``budget * prior[action]``.
    """
    p = np.asarray(prior, dtype=np.float64)
    if budget <= 0:
        return np.zeros(len(p), dtype=np.int32)
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        p = np.full(len(p), 1.0 / len(p), dtype=np.float64)
    else:
        p = p / total
    # Points are x, x+1, ..., x+budget-1 on a CDF scaled to [0, budget].
    points = rng.random() + np.arange(budget, dtype=np.float64)
    chosen = np.searchsorted(np.cumsum(p) * budget, points, side="right")
    np.minimum(chosen, len(p) - 1, out=chosen)
    return np.bincount(chosen, minlength=len(p)).astype(np.int32)


def optimized_posterior(q: np.ndarray, prior: np.ndarray, simulations: int,
                        c_puct: float = C_PUCT) -> np.ndarray:
    """Return the regularized posterior policy with equalized UCB values.

    For T explored simulations the solution is
    ``pi[a] = (c/sqrt(T) * prior[a]) / (lambda - Q[a])``.  Newton iteration
    finds the unique lambda above max(Q), matching Algorithm 4/reference code.
    """
    q64 = np.asarray(q, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    p /= p.sum()
    if len(p) == 1:
        return np.ones(1, dtype=np.float32)
    t = max(1, int(simulations))
    c0 = float(c_puct) / np.sqrt(t)
    q_max = float(q64.max())
    best = int(np.argmax(q64))
    delta = max(1e-12, c0 * float(p[best]))
    for _ in range(100):
        denom = q_max - q64 + delta
        terms = c0 * p / denom
        f = float(terms.sum() - 1.0)
        if f <= 1e-12:
            break
        fprime = float((-terms / denom).sum())
        new_delta = delta - f / fprime
        if not np.isfinite(new_delta) or new_delta <= delta:
            break
        delta = new_delta
    out = c0 * p / (q_max - q64 + delta)
    out /= out.sum()
    return out.astype(np.float32)


def _root_noise(prior: np.ndarray, moves: Sequence[Move], rng,
                alpha: float, frac: float) -> np.ndarray:
    noise = np.zeros_like(prior)
    groups = _action_type_groups(moves)
    type_mass = 1.0 / len(groups)
    for group in groups:
        noise[group] = rng.dirichlet(np.full(len(group), alpha)) * type_mass
    return ((1.0 - frac) * prior + frac * noise).astype(np.float32)


def _is_adjudicated_draw(node: _Node, histories: Sequence[Counter],
                         remaining_plies: Sequence[Optional[int]],
                         root_index: int) -> bool:
    horizon = remaining_plies[root_index]
    if horizon is not None and node.depth >= horizon:
        return True
    occurrences = histories[root_index].get(node.state, 0)
    cursor: Optional[_Node] = node
    # Root itself is already represented in the real-game history.
    while cursor is not None and cursor.parent is not None:
        if cursor.state == node.state:
            occurrences += 1
        cursor = cursor.parent
    return occurrences >= 3


def run_rmcts_batch(
    root_states: Sequence[State],
    boards: Sequence[Board],
    evaluate_batch: BatchEvaluate,
    num_simulations: int = 200,
    temperature: float = 1.0,
    add_noise: bool = True,
    c_puct: float = C_PUCT,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    dirichlet_frac: float = DIRICHLET_FRAC,
    state_histories: Optional[Sequence[Sequence[State]]] = None,
    remaining_plies: Optional[Sequence[Optional[int]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[RMCTSResult]:
    """Search multiple roots with breadth-first, batched RMCTS.

    ``num_simulations`` follows the RMCTS definition: evaluating the root costs
    one simulation, and its remaining budget is assigned to children by the
    prior.  Terminal nodes can represent multiple assigned simulations without
    requiring neural evaluation, so ``evaluations <= simulations``.
    """
    if len(root_states) != len(boards):
        raise ValueError("root_states and boards must have equal length")
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    count = len(root_states)
    if count == 0:
        return []
    rng = rng or np.random.default_rng()
    histories_raw = state_histories or [[state] for state in root_states]
    if len(histories_raw) != count:
        raise ValueError("state_histories must match the root batch")
    histories = [Counter(items) for items in histories_raw]
    horizons = list(remaining_plies or [None] * count)
    if len(horizons) != count:
        raise ValueError("remaining_plies must match the root batch")

    roots = [_Node(state, board, num_simulations)
             for state, board in zip(root_states, boards)]
    all_nodes: list[list[_Node]] = [[root] for root in roots]
    frontier: list[tuple[int, _Node]] = []
    evaluations = [0] * count

    for ri, root in enumerate(roots):
        winner = root.state.winner(root.board)
        if winner is not None:
            root.terminal_value = 1.0 if winner == root.state.turn else -1.0
        else:
            frontier.append((ri, root))

    while frontier:
        states = [node.state for _, node in frontier]
        level_boards = [node.board for _, node in frontier]
        logits_batch, values = evaluate_batch(states, level_boards)
        logits_batch = np.asarray(logits_batch)
        values = np.asarray(values).reshape(-1)
        if len(logits_batch) != len(frontier) or len(values) != len(frontier):
            raise ValueError("evaluate_batch returned the wrong batch length")

        next_frontier: list[tuple[int, _Node]] = []
        for batch_i, (ri, node) in enumerate(frontier):
            evaluations[ri] += 1
            node.network_value = float(values[batch_i])
            moves, index_list = legal_move_mask(node.state, node.board)
            if not moves:
                node.terminal_value = -1.0
                continue
            node.moves = moves
            node.indices = np.asarray(index_list, dtype=np.intp)
            node.prior = _balanced_priors(logits_batch[batch_i, node.indices], moves)
            if node.parent is None and add_noise:
                node.prior = _root_noise(
                    node.prior, moves, rng, dirichlet_alpha, dirichlet_frac)

            if node.simulations == 1:
                continue
            counts = assign_simulations(node.simulations - 1, node.prior, rng)
            node.child_counts = counts
            for action, assigned in enumerate(counts):
                if assigned <= 0:
                    continue
                child_state = apply_move(node.state, moves[action])
                child = _Node(
                    child_state, node.board, int(assigned), node, action,
                    node.depth + 1,
                )
                node.children.append(child)
                all_nodes[ri].append(child)
                winner = child_state.winner(node.board)
                if winner is not None:
                    child.terminal_value = (
                        1.0 if winner == child_state.turn else -1.0)
                elif _is_adjudicated_draw(child, histories, horizons, ri):
                    child.terminal_value = 0.0
                else:
                    next_frontier.append((ri, child))
        frontier = next_frontier

    # Children are always allocated after parents, so reverse allocation order
    # is a valid bottom-up traversal for every tree.
    for nodes in all_nodes:
        for node in reversed(nodes):
            if node.terminal_value is not None:
                node.posterior_value = node.terminal_value
                continue
            if node.simulations == 1 or not node.children:
                node.posterior_value = node.network_value
                continue

            explored = np.flatnonzero(node.child_counts)
            q = np.empty(len(explored), dtype=np.float32)
            child_by_action = {child.parent_action: child for child in node.children}
            for i, action in enumerate(explored):
                child = child_by_action[int(action)]
                child_value = child.posterior_value
                q[i] = (child_value if child.state.turn == node.state.turn
                        else -child_value)
            explored_prior = node.prior[explored].astype(np.float64)
            explored_prior /= explored_prior.sum()
            post = optimized_posterior(
                q, explored_prior, node.simulations - 1, c_puct)
            node.posterior = np.zeros(len(node.moves), dtype=np.float32)
            node.posterior[explored] = post
            searched_value = float(np.dot(post, q))
            node.posterior_value = (
                ((node.simulations - 1) * searched_value + node.network_value)
                / node.simulations
            )

    results: list[RMCTSResult] = []
    for ri, root in enumerate(roots):
        pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
        move: Optional[Move] = None
        if root.terminal_value is None and root.moves:
            local = root.posterior
            if not local.any():
                local = root.prior.copy()
            # Preserve the legal-action mask for the training loss.
            local = np.maximum(local, LEGAL_TARGET_EPSILON)
            local /= local.sum()
            pi[root.indices] = local
            if temperature < 0.01:
                chosen = int(np.argmax(local))
            else:
                weights = local.astype(np.float64) ** (1.0 / temperature)
                weights /= weights.sum()
                chosen = int(rng.choice(len(root.moves), p=weights))
            move = root.moves[chosen]
        results.append(RMCTSResult(
            policy=pi,
            value=float(root.posterior_value),
            move=move,
            simulations=num_simulations,
            evaluations=evaluations[ri],
            nodes=len(all_nodes[ri]),
            max_depth=max(node.depth for node in all_nodes[ri]),
        ))
    return results

