__version__ = "0.1.0"

# Keep the package import lightweight: game + solver are pure-Python and safe to
# import eagerly, but `play` is the Rich TUI. Importing it here would force rich
# (and, transitively, heavy deps) into every headless user — including spawned
# self-play workers. Import `corridors.play` explicitly where the UI is needed.
from . import game, solver  # noqa: F401
