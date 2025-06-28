"""Lightweight package initialization.

The original module eagerly imported most project submodules which pulled in
heavy optional dependencies. In minimal environments (like the tests used in
this kata) those imports fail before any functionality can run.  To avoid these
errors the package no longer imports other modules at import time.  Instead, the
consumer should import the required submodules directly, e.g.::

    from src import training
    training.train_models(...)

This keeps ``import src`` safe even when dependencies such as ``yaml`` or
``pandas`` are missing.
"""

__all__: list[str] = []


