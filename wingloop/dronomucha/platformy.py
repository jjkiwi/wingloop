"""The two drones the work goes on with: a quadcopter and a foam plane.

The fly's brain, the senses, the search map, the memory of objects and the
operator are the same for both; what differs is the body that carries them.
A rescue search in the mountains is about a square kilometre flown at
30-60 m with the camera looking down, then a closer look at what was warm
(docs/platformy.md):

- the **quadcopter** (:data:`QUAD`) hovers: it can stop over a warm spot and
  come down to look -- the development platform, and the one for the closer
  look; 10-15 minutes in the air;
- the **plane** (:data:`SAMOLOT`) cannot stop, but flies 30-45 minutes at
  twice the speed: it is what covers the square kilometre, and looks closer
  by circling lower.

Every measure here is in real metres -- unlike the lab's arena, at 1:50.
:func:`pokrycie` is how much ground a body with a camera searches in one
flight: the arithmetic a search plan starts from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Kamera:
    """A thermal camera looking down: pixels across and along, field of view across (degrees),
    frames a second, noise (K), and roughly what it costs (zloty, autumn 2026)."""

    nazwa: str
    px: tuple
    fov: float
    hz: float
    netd: float
    cena: int

    def piksel(self, wysokosc: float) -> float:
        """How much ground one pixel covers straight below, m, from ``wysokosc`` m."""
        return 2 * wysokosc * np.tan(np.deg2rad(self.fov) / 2) / self.px[0]

    def pas(self, wysokosc: float) -> float:
        """How wide a strip of ground the picture covers, m."""
        return self.piksel(wysokosc) * self.px[0]


#: The cameras considered (docs/platformy.md has where they come from). The
#: 256 x 192 USB cameras give their temperatures per pixel on Linux, by the
#: open programs people wrote for them; check before buying.
KAMERY = {
    "mlx90640_110": Kamera("MLX90640 110°", (32, 24), 110.0, 8.0, 0.25, 250),
    "mlx90640_55": Kamera("MLX90640 55°", (32, 24), 55.0, 8.0, 0.25, 250),
    "t256s": Kamera("Sipeed T256s", (256, 192), 56.0, 25.0, 0.05, 900),
    "p2pro": Kamera("InfiRay P2 Pro", (256, 192), 56.0, 25.0, 0.04, 1100),
    "lepton35": Kamera("Lepton 3.5", (160, 120), 57.0, 9.0, 0.05, 1200),
    "384": Kamera("moduł 384×288", (384, 288), 50.0, 25.0, 0.04, 2000),
}


@dataclass(frozen=True)
class Platforma:
    """A body to carry the brain: whether it hovers, how fast it searches (m/s), how
    slow it can fly (m/s, 0 for a hover), minutes in the air, the height it
    searches from (m), the tightest circle it flies (m), and what it can lift
    besides itself (g)."""

    nazwa: str
    pl: str
    zawis: bool
    predkosc: float
    najwolniej: float
    lot_min: float
    wysokosc: float
    krag: float
    udzwig: float

    def zapas(self) -> float:
        """Minutes of searching: a fifth of the flight kept back to get home."""
        return 0.8 * self.lot_min


QUAD = Platforma(
    "quad",
    "quadrocopter",
    zawis=True,
    predkosc=8.0,
    najwolniej=0.0,
    lot_min=13.0,
    wysokosc=40.0,
    krag=0.0,
    udzwig=500.0,
)
SAMOLOT = Platforma(
    "samolot",
    "samolot piankowy",
    zawis=False,
    predkosc=15.0,
    najwolniej=10.0,
    lot_min=40.0,
    wysokosc=60.0,
    krag=40.0,
    udzwig=300.0,
)
PLATFORMY = {"quad": QUAD, "samolot": SAMOLOT}

#: How much of a person's strip a pass sees well enough to find them: the
#: edges of a picture look at the ground slantwise and from further. And how
#: many pixels across a person must fill to be found at all.
UZYTECZNY_PAS, PIKSELE_NA_CZLOWIEKA = 0.7, 2.0


def pokrycie(platforma: Platforma, kamera: Kamera, wysokosc: float | None = None) -> dict:
    """What one flight searches: the pixel, the strip, km² searched, whether a person is seen.

    A person lying down is about 0.5 m by 1.7 m; across, the camera must
    give them :data:`PIKSELE_NA_CZLOWIEKA` pixels to be told from a warm
    stone (``widac``). The strip searched is :data:`UZYTECZNY_PAS` of the
    picture's, flown at the platform's speed for :meth:`Platforma.zapas`
    minutes.
    """
    h = platforma.wysokosc if wysokosc is None else wysokosc
    piksel = kamera.piksel(h)
    pas = kamera.pas(h) * UZYTECZNY_PAS
    km2 = pas * platforma.predkosc * platforma.zapas() * 60 / 1e6
    return dict(
        wysokosc=h,
        piksel=piksel,
        pas=pas,
        km2=km2,
        widac=0.5 / piksel >= PIKSELE_NA_CZLOWIEKA / 2 and 1.7 / piksel >= PIKSELE_NA_CZLOWIEKA,
    )


def najwyzej(kamera: Kamera) -> float:
    """The highest a camera can search from and still give a person enough pixels, m."""
    # Across their shorter side, half a metre must be PIKSELE_NA_CZLOWIEKA / 2 pixels.
    piksel = 0.5 / (PIKSELE_NA_CZLOWIEKA / 2)
    return piksel * kamera.px[0] / (2 * np.tan(np.deg2rad(kamera.fov) / 2))
