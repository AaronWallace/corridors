"""Build glue for the optional Cython engine.

Project metadata lives in pyproject.toml; this file only wires up the
_engine extension. The extension is optional: if Cython or a C compiler is
missing, the build warns and continues, and corridors runs on the pure-Python
engine in game.py (identical behavior, much slower search).

Build in place (puts the .pyd/.so next to game.py, which the PYTHONPATH-based
launchers and pytest pick up):

    python setup.py build_ext --inplace
"""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Warn-and-continue when the compiled engine cannot be built."""

    _failed = ()

    def run(self):
        self._failed = set()
        try:
            super().run()
        except Exception as exc:  # e.g. no C compiler installed
            self._warn(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._failed.add(ext.name)
            self._warn(exc)

    def copy_extensions_to_source(self):
        # Skip the inplace-copy of anything that failed to compile; it has
        # already been warned about and re-raising here just repeats it.
        self.extensions = [e for e in self.extensions
                           if e.name not in self._failed]
        try:
            super().copy_extensions_to_source()
        except Exception as exc:  # DistutilsFileError on a locked .pyd
            # On Windows a loaded .pyd is locked: the compile succeeded but
            # the copy over the in-use file fails. The existing build stays.
            print("=" * 72)
            print("NOTE: compiled engine built, but the existing extension file")
            print(f"could not be replaced ({exc}).")
            print("It is likely loaded by a running Python process. The previous")
            print("build remains in place; re-run after closing corridors sessions")
            print("if you need the freshly built one.")
            print("=" * 72)

    @staticmethod
    def _warn(exc):
        print("=" * 72)
        print("WARNING: could not build the corridors._engine extension:")
        print(f"    {exc}")
        print("corridors will run on the pure-Python engine (correct but much")
        print("slower search). Install a C compiler and reinstall to fix.")
        print("=" * 72)


try:
    from Cython.Build import cythonize
    extensions = cythonize(
        [Extension("corridors._engine", ["src/corridors/_engine.pyx"])],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except ImportError:
    print("WARNING: Cython not installed; skipping corridors._engine build.")
    extensions = []

setup(ext_modules=extensions, cmdclass={"build_ext": OptionalBuildExt})
