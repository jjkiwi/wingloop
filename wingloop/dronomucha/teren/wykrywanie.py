"""Warm spots in a picture, measured in metres, and the fly's mushroom body judging them.

A warm spot is a patch warmer than the ground round it by :data:`PROG`
kelvin: the ground's own temperature is taken as the median of the picture
in blocks, so a whole sunlit slope is ground, and a boulder on it, or a
person, is a spot. Each spot is measured as the drone can measure it from
up there (:data:`CECHY`): how much warmer than the air its warmest pixel is,
K -- the drone carries a thermometer; a frosty morning's 5 C and a warm
day's are not the same thing -- and how much warmer than the ground round
it, K; how much ground it covers, m², how long
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
    "temp": (-5.0, 40.0, 12, False),
    "nadwyzka": (1.0, 40.0, 10, True),
    "pole": (0.03, 20.0, 10, True),
    "dlugosc": (0.15, 8.0, 10, True),
    "szerokosc": (0.1, 4.0, 10, True),
    "piksel": (0.05, 3.0, 8, True),
}


@dataclass
class Plama:
    """A warm spot: where it is on the ground (x, y, z), its measures (:data:`CECHY`), and
    how warm its warmest pixel is, C."""

    gdzie: np.ndarray
    cechy: np.ndarray
    temp: float = 0.0


def tlo(temp: np.ndarray, blok: int) -> np.ndarray:
    """The ground's own temperature under each pixel: the median of ``blok``-pixel blocks."""
    from scipy.ndimage import zoom

    h, w = temp.shape
    bh, bw = max(1, h // blok), max(1, w // blok)
    cut = temp[: bh * blok, : bw * blok] if bh * blok and bw * blok else temp
    med = np.median(cut.reshape(bh, blok, bw, blok), axis=(1, 3)) if blok > 1 else temp
    return zoom(med, (h / med.shape[0], w / med.shape[1]), order=1)[:h, :w]


#: How much ground the ground's own temperature is taken over, m: more than a boulder.
BLOK = 6.0


def plamy(widok, prog: float = PROG, powietrze: float = 0.0) -> list:
    """Every warm spot in a picture (:class:`~.kamera.Widok`), measured; ``powietrze``: the
    air's temperature, C, which its warmth is measured from."""
    from scipy import ndimage

    temp, gsd = widok.temp, widok.gsd
    g = float(np.median(gsd))
    # As big from low down, where it is many pixels, as from high up: with it
    # held to 16 pixels, a boulder looked at from 15 m was its own ground.
    blok = int(np.clip(round(BLOK / g), 2, max(2, min(temp.shape) // 2)))
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
                np.array(
                    [float(temp[box][mask][k]) - powietrze, float(a[k]), area, length, width, g]
                ),
                float(temp[box][mask][k]),
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


#: Spots seen with a pixel under this, m -- from below about 70 m with the T256s -- are
#: what the second compartment (:attr:`Mozg.z_bliska`) learns on.
BLISKO = 0.2


class Mozg:
    """The fly's mushroom body judging warm spots: +1 a person, -1 not.

    ``z_bliska``, when there is one, is a second compartment taught only on spots seen
    from close (:data:`BLISKO`): the look that confirms a candidate is judged by it. Close
    up a person is plainly bigger than a fox or a chamois, and a brain taught on every
    height -- and on hot days, when a person is barely warmer than the ground -- weighs
    that less than one taught on the close looks alone.
    """

    def __init__(self, mb, gain: int = 4, z_bliska: "Mozg | None" = None):
        self.mb, self.gain, self.z_bliska = mb, gain, z_bliska

    def ocena(self, plama: Plama) -> float:
        return float(self.mb.valence(kod(plama.cechy, self.gain)))


def naucz(cechy, ludzie, gain: int = 4, seed: int = 0, blisko: float | None = BLISKO) -> Mozg:
    """Teach the fly's mushroom body on measured spots: dopamine for the people's, punishment
    for the rest's -- as a prediction error, three passes (docs/uczenie_maszynowe.md). With
    ``blisko``, a second compartment is taught on the spots seen with a pixel under it
    (:attr:`Mozg.z_bliska`), if there are people among them."""
    from wingloop.lab.learning import MushroomBody
    from wingloop.lab.navigate import teach

    cechy, ludzie = np.asarray(cechy, dtype=float), np.asarray(ludzie, bool)
    X = kod(cechy, gain)
    kinds = np.where(ludzie, "czlowiek", "inne")
    mb = MushroomBody(X.shape[1], seed=seed, on_off=True, sparseness=0.03)
    teach(mb, X, kinds, {"czlowiek": 1.0, "inne": -1.0}, aim=0.3, epochs=3, seed=seed)
    close = None
    if blisko is not None and len(cechy):
        near = cechy[:, 5] < blisko
        if ludzie[near].any() and (~ludzie[near]).any():
            close = naucz(cechy[near], ludzie[near], gain, seed=seed + 1, blisko=None)
    return Mozg(mb, gain, close)
