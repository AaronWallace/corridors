"""End-to-end RMCTS self-play throughput benchmark for Corridors."""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..game import NCOLS, WALLS_PER_PLAYER, Board, State, apply_move
from .encoding import encode_state
from .rmcts import run_rmcts_batch


@dataclass
class _Game:
    board: Board
    state: State
    history: list[State]
    ply: int = 0
    winner: Optional[int] = None
    done: bool = False


class TorchBatchEvaluator:
    """Checkpoint evaluator with explicit inference chunking and metrics."""

    def __init__(self, checkpoint: str, device: str, batch_size: int,
                 fp16: bool = False) -> None:
        import torch
        from .az_net import AZNet, load_checkpoint

        self.torch = torch
        self.device = device
        self.batch_size = max(1, batch_size)
        self.fp16 = fp16
        self.model = (load_checkpoint(checkpoint, device=device)
                      if checkpoint else AZNet().to(device).eval())
        if fp16:
            self.model = self.model.half()
        self.positions = 0
        self.batches = 0
        self.inference_s = 0.0

    def warmup(self, states, boards) -> None:
        if not states:
            return
        self(states[:min(len(states), self.batch_size)],
             boards[:min(len(boards), self.batch_size)])
        self.positions = self.batches = 0
        self.inference_s = 0.0

    def __call__(self, states, boards):
        if not states:
            return (np.empty((0, 227), dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        encoded = np.stack([encode_state(s, b) for s, b in zip(states, boards)])
        policies, values = [], []
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start:start + self.batch_size]
            t0 = time.monotonic()
            with self.torch.inference_mode():
                x = self.torch.from_numpy(chunk).to(self.device)
                if self.fp16:
                    x = x.half()
                policy, value = self.model(x)
                policies.append(policy.float().cpu().numpy())
                values.append(value.float().cpu().numpy())
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            self.inference_s += time.monotonic() - t0
            self.positions += len(chunk)
            self.batches += 1
        return np.concatenate(policies), np.concatenate(values)


def benchmark_rmcts(
    *,
    checkpoint: str = "",
    device: str = "cuda",
    games: int = 16,
    simulations: int = 200,
    max_plies: int = 80,
    inference_batch: int = 512,
    fp16: bool = False,
    seed: int = 731,
    on_ply: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Play a synchronized batch of RMCTS games and return throughput metrics."""
    rng = np.random.default_rng(seed)
    starts = random.Random(seed)
    active: list[_Game] = []
    for _ in range(games):
        board, state = State.start(
            starts.randrange(NCOLS), starts.randrange(NCOLS), WALLS_PER_PLAYER)
        active.append(_Game(board, state, [state]))

    evaluator = TorchBatchEvaluator(checkpoint, device, inference_batch, fp16)
    evaluator.warmup([g.state for g in active], [g.board for g in active])
    t0 = time.monotonic()
    searches = requested_sims = nodes = max_depth_seen = 0
    completed_plies = 0

    while active:
        roots = [g for g in active if not g.done]
        if not roots:
            break
        results = run_rmcts_batch(
            [g.state for g in roots], [g.board for g in roots], evaluator,
            num_simulations=simulations, temperature=1.0, add_noise=True,
            state_histories=[g.history for g in roots],
            remaining_plies=[max_plies - g.ply for g in roots], rng=rng,
        )
        searches += len(results)
        requested_sims += simulations * len(results)
        nodes += sum(result.nodes for result in results)
        max_depth_seen = max(max_depth_seen,
                             max((result.max_depth for result in results), default=0))
        for game, result in zip(roots, results):
            if result.move is None:
                game.winner = 2 if game.state.turn == 1 else 1
                game.done = True
                continue
            game.state = apply_move(game.state, result.move)
            game.history.append(game.state)
            game.ply += 1
            completed_plies += 1
            winner = game.state.winner(game.board)
            if winner is not None:
                game.winner = winner
                game.done = True
            elif game.ply >= max_plies or Counter(game.history)[game.state] >= 3:
                game.done = True
        active = [g for g in active if not g.done]
        if on_ply:
            elapsed = max(time.monotonic() - t0, 1e-9)
            on_ply({
                "active_games": len(active), "plies": completed_plies,
                "elapsed_s": elapsed, "simulations": requested_sims,
                "evaluations": evaluator.positions,
            })

    elapsed = max(time.monotonic() - t0, 1e-9)
    return {
        "algorithm": "rmcts",
        "checkpoint": checkpoint or "random initialization",
        "device": device,
        "fp16": fp16,
        "games": games,
        "searches": searches,
        "plies": completed_plies,
        "simulations": requested_sims,
        "evaluations": evaluator.positions,
        "nodes": nodes,
        "max_depth": max_depth_seen,
        "elapsed_s": elapsed,
        "games_per_s": games / elapsed,
        "plies_per_s": completed_plies / elapsed,
        "simulations_per_s": requested_sims / elapsed,
        "evaluations_per_s": evaluator.positions / elapsed,
        "inference_batches": evaluator.batches,
        "avg_inference_batch": (evaluator.positions / evaluator.batches
                                if evaluator.batches else 0.0),
        "inference_s": evaluator.inference_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--inference-batch", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    metrics = benchmark_rmcts(**vars(args))
    print(
        f"RMCTS {metrics['games']} games / {metrics['plies']} plies in "
        f"{metrics['elapsed_s']:.2f}s\n"
        f"  {metrics['simulations_per_s']:,.0f} sims/s  |  "
        f"{metrics['plies_per_s']:,.1f} plies/s  |  "
        f"{metrics['games_per_s']:.3f} games/s\n"
        f"  {metrics['evaluations_per_s']:,.0f} evals/s  |  "
        f"avg inference batch {metrics['avg_inference_batch']:.1f}  |  "
        f"max tree depth {metrics['max_depth']}"
    )


if __name__ == "__main__":
    main()

