"""Convert classical self-play shards into AlphaZero-shaped training data.

Classical autoplay generates (state, played_move, tt_score, outcome) tuples;
AZNet needs (state, policy_target [227], value_target [-1..+1]). This module
turns the former into the latter, so a strong classical solver can bootstrap
AZ training out of the cold-start value collapse (net can't distinguish
positions → wandering draws → no learning signal).

Each classical position becomes one AZ training row:
  - state:  copied as-is (both formats use the same encode_state layout)
  - policy: one-hot on the move the classical solver actually played
  - value:  the game's actual outcome from side-to-move perspective, in {-1,0,+1}

Only shards that recorded ``moves`` (added after this feature landed) are
convertible; older shards without moves are skipped with a warning.

Run:
    python -m corridors.nn.convert_classical <source_dataset> <az_run_name>

For example:
    python -m corridors.nn.convert_classical 10000g_d2_e15 classical_warmstart
    # writes nn_data/alphazero/classical_warmstart/shard_XXXX.npz files that
    # the AZ Train Network menu picks up like any other AZ run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .actions import NUM_ACTIONS
from .datasets import DATA_ROOT, _shard_files


AZ_ROOT = DATA_ROOT / "alphazero"


def _iter_source_shards(src: Path):
    """Yield each source shard's (states, moves, outcomes) arrays. Skips
    shards without a moves array (pre-feature) with a printed note."""
    shards = _shard_files(src)
    if not shards:
        raise FileNotFoundError(f"no shards in {src}")
    for path in shards:
        with np.load(path) as z:
            names = set(z.files)
            if "moves" not in names:
                print(f"  skipping {path.name}: no 'moves' array "
                      "(recorded by an older classical run)")
                continue
            if "tensors" not in names or "outcomes" not in names:
                print(f"  skipping {path.name}: missing tensors/outcomes")
                continue
            yield path.name, (z["tensors"], z["moves"], z["outcomes"])


def _write_az_shard(dst_dir: Path, shard_idx: int,
                    states: np.ndarray, policies: np.ndarray,
                    outcomes: np.ndarray) -> Path:
    """Match the AZ ShardWriter's file naming and NPZ layout so the AZ menus
    treat these shards identically to native self-play output."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    path = dst_dir / f"shard_{shard_idx:04d}.npz"
    tmp = dst_dir / f".tmp_{path.name}"
    np.savez_compressed(tmp, states=states, policies=policies, outcomes=outcomes)
    tmp.replace(path)
    return path


def convert(src_name: str, az_run_name: str) -> dict:
    """Convert a classical dataset into an AZ-shaped run.

    Reads shards from nn_data/<src_name>/, writes to nn_data/alphazero/<az_run_name>/.
    Returns a summary dict with counts and paths.
    """
    src = DATA_ROOT / src_name
    if not src.exists():
        raise FileNotFoundError(f"source dataset not found: {src}")
    dst = AZ_ROOT / az_run_name
    if dst.exists() and any(dst.glob("shard_*.npz")):
        raise FileExistsError(
            f"destination already has shards: {dst}. "
            "Delete it or pick a new az_run_name.")

    total_positions = 0
    total_skipped = 0
    shard_idx = 0
    t0 = time.monotonic()
    for src_name_, (tensors, moves, outcomes) in _iter_source_shards(src):
        # One-hot policy target: 1.0 at the played action, 0 elsewhere. This is
        # a "hard" policy target — not the MCTS visit distribution AZ normally
        # uses, but it teaches the policy head to prefer the classical solver's
        # choice, which is what we want for a warm start.
        n = int(tensors.shape[0])
        # Guard against any out-of-range indices (shouldn't happen, but be safe).
        valid = (moves >= 0) & (moves < NUM_ACTIONS)
        if not valid.all():
            skipped = int((~valid).sum())
            total_skipped += skipped
            tensors = tensors[valid]
            moves = moves[valid]
            outcomes = outcomes[valid]
            n = int(tensors.shape[0])
        if n == 0:
            continue
        policies = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        policies[np.arange(n), moves.astype(np.int64)] = 1.0
        # outcomes in the classical shard are int8 {-1, 0, +1}; AZ shards use
        # float32 with the same semantics (side-to-move value target).
        az_outcomes = outcomes.astype(np.float32)
        _write_az_shard(dst, shard_idx, tensors.astype(np.float32), policies,
                        az_outcomes)
        shard_idx += 1
        total_positions += n

    # Write a run.json so the AZ Train menu shows a sensible config for the run.
    run_meta = {
        "mode": "converted_classical",
        "source_dataset": src_name,
        "selfplay": {"num_games": 0, "notes": "converted from classical autoplay"},
        "positions": total_positions,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (dst / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return {
        "source": src_name,
        "destination": f"alphazero/{az_run_name}",
        "shards_written": shard_idx,
        "positions": total_positions,
        "skipped_invalid_moves": total_skipped,
        "elapsed_s": time.monotonic() - t0,
    }


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="classical dataset name (under nn_data/)")
    parser.add_argument("az_run_name",
                        help="destination run name (under nn_data/alphazero/)")
    args = parser.parse_args()
    result = convert(args.source, args.az_run_name)
    print("\nConversion complete:")
    for k, v in result.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    _cli()
