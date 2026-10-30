"""Dronomucha: the drone flown by the fly's brain, put to work.

The lab (:mod:`wingloop.lab`) is where the fly and its drone are built and
measured; this package is where the drone is given jobs, and it can change
freely without touching the lab. A job is a :class:`~.zadanie.Zadanie` --
what is in its world, what it looks for, what its eye sees through -- in
:mod:`.zadania`. The drone, its eye, its brain and its navigator are the
lab's; :mod:`.swiat` is the one place a job reaches into the lab, and it
puts everything back after.

    python -m wingloop.dronomucha lista
    python -m wingloop.dronomucha szukaj ratunek_gory --obszary 12
    python -m wingloop.dronomucha film ratunek_morze --plik morze.mp4

docs/dronomucha.md has how to add a job.
"""

from . import zadania  # noqa: E402,F401 -- registers the jobs shipped
from .misja import film, obszar, pilot, szukaj
from .mozg import lekcje, mozg
from .swiat import oko, swiat
from .zadanie import ZADANIA, Zadanie, dodaj, zadanie

__all__ = [
    "ZADANIA",
    "Zadanie",
    "dodaj",
    "film",
    "lekcje",
    "mozg",
    "obszar",
    "oko",
    "pilot",
    "swiat",
    "szukaj",
    "zadanie",
]
