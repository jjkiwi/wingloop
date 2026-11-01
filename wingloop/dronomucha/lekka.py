"""The light drone: no fly's eye -- a cheap thermal camera's pixels instead.

The fly's optic lobe (flyvis, :mod:`wingloop.lab.vision`) is the costliest
part of the drone's brain: 21-31 ms a look on a server's core, fifty looks a
second, PyTorch to run it. Measured on the rescue jobs' lessons, the drone's
other senses -- the thermal camera's kelvin, the rangefinder's distance and
size, five numbers (:mod:`.zmysly`) -- tell a person from a rock, a chamois,
a seal or debris as well as the eye's 325 features or better, and the fly's
mushroom body learns them alone (docs/lekka.md). So the light drone keeps
the fly's mushroom body and drops its eye.

What it looks through is a cheap thermal sensor, as it would be built:

- :data:`MLX90640`: a 32 x 24 thermopile array, 110 x 75 degrees, about
  8 frames a second, a quarter of a kelvin of noise -- a few tens of zloty
  from China, on the Raspberry Pi's I2C. 768 pixels: about as many as the
  fly's eye has columns (721);
- :data:`LEPTON`: a 160 x 120 microbolometer, 57 x 44 degrees, 9 frames a
  second -- the upgrade, a few times dearer.

The sensor sits on the same one-axis tilt as the rangefinder: level, tilted
down ``tilt`` degrees, turned with the drone, which looks around no faster
than the sensor's pictures let it (:attr:`Czujnik.skan`). :class:`LekkieOko` is the
lab's :class:`~wingloop.lab.watch.FlysEye` with the optic lobe taken out
and these pixels put in its place (:class:`Piksele`): the navigator finds,
fixates and flies to what is warm in them exactly as it did to what was dark
in the fly's eye, and the mushroom body judges the senses' five numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wingloop.lab import watch


@dataclass(frozen=True)
class Czujnik:
    """A thermal sensor: pixels across and up, field of view across and up (degrees),
    frames a second, noise (K), and how far down it is tilted (degrees).

    Tilted 25 degrees down: at 10, a person lying ahead left the bottom of a
    75-degree view as the drone came in, and it lost them or arrived at the
    rock spire behind -- measured on three mountain areas, 1 person of 6
    found against 4 at 25.
    """

    nazwa: str
    size: tuple
    fov: tuple
    hz: float
    netd: float
    tilt: float = 25.0

    @property
    def skan(self) -> float:
        """How fast the drone may turn looking around, deg/s: :data:`NA_KLATKE` a picture.

        Turning faster, what the eye fixates jumps further between pictures
        than the window it is fixated in: at 270 deg/s and 8 pictures a
        second, 34 degrees.
        """
        return NA_KLATKE * self.hz


#: Degrees the drone turns between two pictures, at most, looking around.
NA_KLATKE = 15.0

MLX90640 = Czujnik("MLX90640", (32, 24), (110.0, 75.0), 8.0, 0.25)
LEPTON = Czujnik("Lepton 3.5", (160, 120), (57.0, 44.0), 9.0, 0.05)
CZUJNIKI = {"mlx90640": MLX90640, "lepton": LEPTON}

#: The fly's eye's field, square degrees: 721 columns 4 degrees apart. A
#: fraction of the view means the same angular size to the navigator in any
#: eye when it is a fraction of this.
POLE_OKA = 10_000.0


class Piksele:
    """What stands in for the optic lobe: the thermal sensor's pixels and nothing more.

    The same face to the navigator as :class:`~wingloop.lab.vision.FlyEye`:
    ``photoreceptors`` (the picture, warm dark), each pixel's ``column_x``
    and ``column_y`` (+1 the edge of the view, + left, + up), :meth:`dark`.
    No features for the brain (:meth:`fovea_shape` is empty): the light
    drone's brain judges its senses.
    """

    def __init__(self, czujnik: Czujnik):
        w, h = czujnik.size
        fw, fh = czujnik.fov
        half = fw / 2
        az = np.linspace(half, -half, w)
        el = np.linspace(fh / 2, -fh / 2, h) - czujnik.tilt
        AZ, EL = np.meshgrid(az, el)
        self.azimuth, self.elevation = AZ, EL  # degrees, the body level
        self.column_x = (AZ / half).ravel()
        self.column_y = (EL / half).ravel()
        self.pixel = (fw / w) * (fh / h)  # square degrees
        self.reset()

    def reset(self) -> None:
        self.photoreceptors = np.full(self.column_x.size, 0.5, dtype=np.float32)

    def step(self, image: np.ndarray, frames: int = 1) -> np.ndarray:
        self.photoreceptors = np.asarray(image, dtype=np.float32).ravel()
        return self.photoreceptors

    def fovea_shape(self, activity) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def dark(self, threshold: float = 0.2, within: float = 1.0, centre: float = 0.0):
        """How much of the view is warm, and where: (share of :data:`POLE_OKA`, bearing)."""
        hot = self.photoreceptors < threshold
        if within < 1.0:
            hot &= np.abs(self.column_x - centre) < within
        if not hot.any():
            return 0.0, 0.0
        return float(hot.sum() * self.pixel / POLE_OKA), float(self.column_x[hot].mean())


class LekkieOko(watch.FlysEye):
    """The fly's eye with its optic lobe taken out: a cheap thermal sensor's pixels.

    It shows a new picture ``czujnik.hz`` times a second and holds it between;
    the noise is drawn once a picture. :meth:`thermal_view` is the same
    sensor, in kelvin: the senses (:mod:`.zmysly`) read it.
    """

    def __init__(self, czujnik: Czujnik = MLX90640, every_ms: float = 20.0, items=()):
        from wingloop.lab.arena import FLOOR, eye_things
        from wingloop.lab.scene import Camera, Thing, World

        things = []
        for x, y, rgb in watch.COLUMNS if watch.SCENERY["columns"] else ():
            r, g, b = (float(v) for v in rgb.split())
            luminance = 0.3 * r + 0.55 * g + 0.15 * b
            things.append(Thing("bar", [x, y, FLOOR], 3.0, height=60.0, brightness=luminance))
        self.background = list(things)
        self.world = World(watch._warm(things + eye_things(items)), ground_height=FLOOR)
        self.czujnik = czujnik
        self.camera = Camera(size=czujnik.size[0], fov=czujnik.fov[0])
        self.eye = Piksele(czujnik)
        self.every_ms = every_ms
        self.image = np.full(czujnik.size[::-1], 0.5, dtype=np.float32)
        self.activity = None
        self.pose = None
        self.nir = None
        self.thermal = True
        self._moving = self._moving_things(items)
        az, el = np.deg2rad(self.eye.azimuth), np.deg2rad(self.eye.elevation)
        self.promienie_termo = np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
        )
        self._sight = self.promienie_termo.reshape(-1, 3).T
        self.elevation = self.eye.elevation.ravel()
        self.rng = np.random.default_rng(0)
        self._clock = np.inf  # ms since the last picture
        self._k = None  # the last picture, kelvin, and the pose it is held for
        self._k_for = None

    def _picture(self, position, R) -> np.ndarray:
        from wingloop.lab.scene import thermal_along

        d = self.promienie_termo @ watch._level(R).T
        k = thermal_along(
            self.world, position, d, clutter=watch.IR_CLUTTER, sky=-watch.THERMAL_SPAN
        )
        return k + self.czujnik.netd * self.rng.standard_normal(k.shape)

    def look_from(self, position, R) -> None:
        """A look: a new picture if the sensor has one by now, else the last one held."""
        self._clock += self.every_ms
        self.pose = (np.asarray(position, dtype=float), np.asarray(R, dtype=float))
        if self._k is None or self._clock >= 1000.0 / self.czujnik.hz:
            self._k, self._clock = self._picture(*self.pose), 0.0
        self._k_for = self.pose
        k = self._k
        span = float(np.clip(k.max(), watch.THERMAL_SPAN_MIN, watch.THERMAL_SPAN))
        self.image = np.clip(0.8 - 0.78 * k / span, 0.02, 0.95).astype(np.float32)
        self.activity = self.eye.step(self.image)

    def thermal_view(self, rng=None) -> np.ndarray:
        """The sensor's picture, kelvin above the ground: the one held, or one taken here."""
        if self.pose is None:
            return np.zeros(self.czujnik.size[::-1])
        if self._k is None or self._k_for is not self.pose:
            self._k, self._k_for = self._picture(*self.pose), self.pose
        return self._k

    def set_items(self, items) -> None:
        super().set_items(items)
        self._k = None


def lekkie_oko(z, czujnik: Czujnik = MLX90640) -> LekkieOko:
    """The light drone's eye for a task."""
    from wingloop.lab import autopilot

    from .swiat import swiat

    with swiat(z):
        return LekkieOko(czujnik, every_ms=autopilot.EYE_MS[True])
