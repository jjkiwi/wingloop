"""The brain for a task: its lessons, kept on disk, and the brain taught on them."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wingloop.lab import autopilot

from .swiat import swiat
from .zadanie import Zadanie

#: Where the tasks' lessons are kept.
DATA = Path(__file__).with_name("data")


def lekcje(fly_eye, z: Zadanie, fresh: bool = False, progress=print, folder=DATA):
    """A task's lessons: ((features, kinds), (close-look features, kinds)).

    The tethered look at each kind, turning as the drone turns, and close
    looks flown, one object at a time -- the lab's two lessons
    (:func:`~wingloop.lab.navigate.tethered_training`,
    :func:`~wingloop.lab.autopilot.flown_close_looks`), in the task's world.
    Kept in ``folder`` and read back after; ``fresh`` flies them again.
    """
    from wingloop.lab.dronebrain import SCAN_RATE
    from wingloop.lab.navigate import tethered_training

    path = Path(folder) / f"{z.name}_lekcje.npz"
    tethered, close = z.lessons
    if path.exists() and not fresh:
        with np.load(path) as d:
            if len(d["K"]) == tethered and len(d["CK"]) == close:
                return (d["F"], d["K"]), (d["CF"], d["CK"])
    with swiat(z):
        kinds = tuple(z.kinds)
        t0 = time.time()
        if progress:
            progress(f"Dron ogląda na uwięzi każdy rodzaj obiektu ({z.pl})...")
        F, K = tethered_training(
            fly_eye, kinds, samples=tethered, seed=0, progress=progress, scan_rate=SCAN_RATE
        )
        if progress:
            progress("... i podlatuje do każdego, by obejrzeć go z bliska...")
        CF, CK = autopilot.flown_close_looks(
            fly_eye, kinds, close, seed=0, drone=True, progress=progress
        )
        if progress:
            progress(f"  {len(K)} + {len(CK)} widoków w {time.time() - t0:.0f} s -> {path}")
    np.savez_compressed(path, F=F, K=K, CF=CF, CK=CK)
    return (F, K), (CF, CK)


def mozg(fly_eye, z: Zadanie, fresh: bool = False, progress=print):
    """The brain taught to look for the task's target: (mushroom body, valence)."""
    (F, K), close = lekcje(fly_eye, z, fresh=fresh, progress=progress)
    with swiat(z):
        return autopilot.teach_targets(F, K, {z.target}, close=close)
