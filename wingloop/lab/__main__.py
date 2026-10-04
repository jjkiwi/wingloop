"""``python -m wingloop.lab`` -- the same as the ``flylab`` command.

Useful on Windows, where pip's Scripts folder is often not on PATH.
"""

from .cli import main

main()
