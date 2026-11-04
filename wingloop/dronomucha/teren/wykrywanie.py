"""Warm spots in a picture, measured in metres, and the fly's mushroom body judging them.

A warm spot is a patch warmer than the ground round it by :data:`PROG`
kelvin: the ground's own temperature is taken as the median of the picture
in blocks, so a whole sunlit slope is ground, and a boulder on it, or a
person, is a spot. Each spot is measured as the drone can measure it from
up there (:data:`CECHY`): how warm its warmest pixel is, C, and how much
warmer than the ground round it, K; how much ground it covers, m², how long
and how wide it is, m -- from its pixels and how much ground a pixel covers;
and that last, the pixel's size, so the brain knows how blurred the rest is.

These six numbers, coded as lines (each a row of neurons tuned to its own
value, as in :mod:`~wingloop.dronomucha.zmysly`), are what the fly's
mushroom body takes in; it learns with its dopamine rule which spots are
people (:func:`naucz`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How much warmer than the ground round it a spot must be, K.
PROG = 2.5
#: What is measured of a spot, and each measure's lines: (least, most, lines, log scale).
CECHY = ("temp", "nadwyzka", "pole", "dlugosc", "szerokosc", "piksel")
STROJENIE = {
    "temp": (-10.0, 40.0, 12, False),
    "nadwyzka": (1.0, 40.0, 10, True),
    "pole": (0.03, 20.0, 10, True),
    "dlugosc": (0.15, 8.0, 10, True),
    "szerokosc": (0.1, 4.0, 10, True),
    "piksel": (0.05, 3.0, 8, True),
}


@dataclass
class Plama:
    """A warm spot: where it is on the ground (x, y, z), its measures (:data:`CECHY`)."""

    gdzie: np.ndarray
    cechy: np.ndarray


def tlo(temp: np.ndarray, blok: int) -> np.ndarray:
    """The ground's own temperature under each pixel: the median of ``blok``-pixel blocks."""
    from scipy.ndimage import zoom

    h, w = temp.shape
    bh, bw = max(1, h // blok), max(1, w // blok)
    cut = temp[: bh * blok, : bw * blok] if bh * blok and bw * blok else temp
    med = np.median(cut.reshape(bh, blok, bw, blok), axis=(1, 3)) if blok > 1 else temp
    return zoom(med, (h / med.shape[0], w / med.shape[1]), order=1)[:h, :w]


def plamy(widok, prog: float = PROG) -> list:
    """Every warm spot in a picture (:class:`~.kamera.Widok`), measured."""
    from scipy import ndimage

    temp, gsd = widok.temp, widok.gsd
    g = float(np.median(gsd))
    blok = int(np.clip(round(6.0 / g), 2, 16))
    anomaly = temp - tlo(temp, blok)
    labels, n = ndimage.label(anomaly > prog)
    out = []
    for j, box in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels[box] == j
        a = anomaly[box][mask]
        pts = widok.ziemia[box][mask]
        area = float((gsd[box][mask] ** 2).sum())
        w = a / a.sum()
        centre = (pts * w[:, None]).sum(axis=0)
        if len(pts) > 1:
            cov = np.cov((pts[:, :2] - centre[:2]).T, aweights=w)
            ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0.0))
        else:
            ev = np.zeros(2)
        length = max(4.0 * ev[-1], g)
        width = max(4.0 * ev[0], g)
        k = int(np.argmax(a))
        out.append(
            Plama(
                centre,
                np.array([float(temp[box][mask][k]), float(a[k]), area, length, width, g]),
            )
        )
    return out


def kod(cechy, gain: int = 4) -> np.ndarray:
    """Measures (n, :data:`CECHY`) as lines: a bump of activity round each value."""
    r = np.atleast_2d(np.asarray(cechy, dtype=float))
    rows = []
    for j, name in enumerate(CECHY):
        lo, hi, n, log = STROJENIE[name]
        f = np.log if log else (lambda v: v)
        centres = np.linspace(f(lo), f(hi), n)
        width = centres[1] - centres[0]
        v = f(np.clip(r[:, j], lo, hi))
        rows.append(np.exp(-0.5 * ((v[:, None] - centres[None]) / width) ** 2))
    lines = np.repeat(np.concatenate(rows, axis=1), gain, axis=1)
    return lines[0] if np.ndim(cechy) == 1 else lines


class Mozg:
    """The fly's mushroom body judging warm spots: +1 a person, -1 not."""

    def __init__(self, mb, gain: int = 4):
        self.mb, self.gain = mb, gain

    def ocena(self, plama: Plama) -> float:
        return float(self.mb.valence(kod(plama.cechy, self.gain)))


def naucz(cechy, ludzie, gain: int = 4, seed: int = 0) -> Mozg:
    """Teach the fly's mushroom body on measured spots: dopamine for the people's, punishment
    for the rest's -- as a prediction error, three passes (docs/uczenie_maszynowe.md)."""
    from wingloop.lab.learning import MushroomBody
    from wingloop.lab.navigate import teach

    X = kod(cechy, gain)
    kinds = np.where(np.asarray(ludzie, bool), "czlowiek", "inne")
    mb = MushroomBody(X.shape[1], seed=seed, on_off=True, sparseness=0.03)
    teach(mb, X, kinds, {"czlowiek": 1.0, "inne": -1.0}, aim=0.3, epochs=3, seed=seed)
    return Mozg(mb, gain)
