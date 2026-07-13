# Curated checkpoints

This directory contains checkpoints deliberately shared through Git.

Use **Neural network training → Copy checkpoint to shared best** to copy a
machine-local checkpoint here. Checkpoints stored directly in
`nn_checkpoints/`, along with `elo.json`, remain local to each system and are
ignored by Git.

At runtime, a local checkpoint takes precedence over a curated checkpoint with
the same name. If no local copy exists, the curated checkpoint is loaded
automatically.
