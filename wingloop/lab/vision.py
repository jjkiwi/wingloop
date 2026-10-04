"""The fly's optic lobe, stepped frame by frame in closed loop.

The network is ``flyvis``'s pretrained connectome-constrained model
(Lappalainen et al., *Nature* 2024): 65 cell types repeated over the 721
columns of one eye, 45,669 neurons whose connectivity and signs come from the
connectome and whose few free parameters were trained on optic-flow
estimation. Nothing here retrains it. This module only feeds it the images a
moving animal would see, keeps its state between frames, and pools its
activity into the handful of signals a controller or a classifier can use.

Hexal positions come from the eye renderer itself (``BoxEye.receptor_centers``)
so "left" and "up" mean what the image means, rather than a guess at the hex
axis convention.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Cell types pooled into the feature vector. Every one of the 65 is used;
#: these are the ones the controllers read directly.
MOTION_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
#: Retinotopic contrast detectors: ON (Mi1, Tm3) and OFF (Tm1, Tm2, Tm9).
CONTRAST_TYPES = ("Mi1", "Tm3", "Tm1", "Tm2", "Tm9")


def _write_h5(path, val) -> None:
    """datamate's HDF5 writer, without the step that fails on Windows.

    datamate 1.0.0 opens the file for writing, finds it empty, and then
    deletes it *while its own handle is still open*. Linux allows that;
    Windows refuses ("the process cannot access the file because it is being
    used by another process"), so flyvis could never build its connectome
    cache there and the first optic-lobe load always failed. The original
    opened with mode "w", which truncates, so its update-in-place branch could
    never succeed anyway: removing any old file first and writing fresh is
    the same result with no open handle in the way.
    """
    import h5py

    val = np.asarray(val)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path.rmdir()
    elif path.exists():
        path.unlink()
    with h5py.File(path, libver="latest", mode="w") as f:
        f["data"] = val
        f.swmr_mode = True


def _prepare_flyvis_cache() -> None:
    """Use the safe writer, and drop cache builds a crash left half-written."""
    import datamate.directory
    import datamate.io
    import flyvis

    datamate.io._write_h5 = _write_h5
    datamate.directory._write_h5 = _write_h5
    cache = Path(flyvis.root_dir) / "connectome"
    for build in cache.glob("ConnectomeFromAvgFilters_*"):
        meta = build / "_meta.yaml"
        if not meta.exists() or "status: done" not in meta.read_text():
            shutil.rmtree(build, ignore_errors=True)


@lru_cache(maxsize=2)
def _load(model: str):
    _prepare_flyvis_cache()
    import flyvis
    from flyvis import NetworkView

    view = NetworkView(flyvis.results_dir / model)
    net = view.init_network()
    net.eval()
    return net


@dataclass
class FlyEye:
    """One compound eye and its optic lobe.

    ``model`` names a pretrained network inside ``flyvis.results_dir``; the
    default is the best model of the published ensemble.
    """

    model: str = "flow/0000/000"
    dt: float = 0.01
    extent: int = 15
    kernel_size: int = 13
    _state: object = field(default=None, repr=False)

    def __post_init__(self):
        import torch
        from flyvis.datasets.rendering import BoxEye
        from flyvis.utils.hex_utils import get_hex_coords

        self._torch = torch
        self.net = _load(self.model)
        self.eye = BoxEye(extent=self.extent, kernel_size=self.kernel_size)
        nodes = self.net.connectome.nodes
        self.types = np.array(
            [t.decode() if isinstance(t, bytes) else str(t) for t in nodes.type[:]]
        )
        u, v = np.asarray(nodes.u[:]), np.asarray(nodes.v[:])
        hu, hv = get_hex_coords(self.extent)
        index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(hu, hv, strict=True))}
        centres = self.eye.receptor_centers.cpu().numpy().astype(float)  # (y, x) px
        span = np.abs(centres).max(axis=0)
        hexal = np.array([index[(int(a), int(b))] for a, b in zip(u, v, strict=True)])
        # Normalised position of every neuron's column: x in [-1, 1], + is the
        # image's left (the animal's left); y in [-1, 1], + is up.
        self.x = -centres[hexal, 1] / span[1]
        self.y = -centres[hexal, 0] / span[0]
        # The same positions for the 721 photoreceptor columns themselves.
        self.column_x = -centres[:, 1] / span[1]
        self.column_y = -centres[:, 0] / span[0]
        self.photoreceptors = np.full(len(centres), 0.5, dtype=np.float32)
        self.cell_types = tuple(sorted(set(self.types)))
        self._by_type = {t: np.flatnonzero(self.types == t) for t in self.cell_types}
        self.reset()

    @property
    def n_neurons(self) -> int:
        return len(self.types)

    def reset(self) -> None:
        """Back to the steady state the network reaches under grey light."""
        with self._torch.no_grad():
            self._state = self.net.steady_state(1.0, self.dt, 1)
        self.baseline = None

    def _hex(self, images: np.ndarray):
        t = self._torch.tensor(np.asarray(images, dtype=np.float32))
        if t.ndim == 2:
            t = t[None]
        # Upsample past the eye's minimum frame plus its kernel. Below that,
        # BoxEye resizes to the minimum and then zero-pads, and the border
        # columns' kernels reach into the padding: a uniform grey image read
        # 0.27 instead of 0.5 on 34 border columns, which is a dark object at
        # the edge of the eye that is not there.
        side = int(self.eye.min_frame_size.max()) + self.kernel_size
        if min(t.shape[-2:]) < side:
            t = self._torch.nn.functional.interpolate(
                t[:, None], size=(side, side), mode="bilinear", align_corners=False
            )[:, 0]
        hexed = self.eye(t[None])  # (1, frames, 1, 721)
        self.photoreceptors = hexed[0, -1, 0].cpu().numpy()
        return hexed

    def step(self, image: np.ndarray, frames: int = 1) -> np.ndarray:
        """Show one image for ``frames`` integration steps; return the activity.

        Returns the activity of every neuron after the last step, shape
        (n_neurons,).
        """
        movie = np.repeat(np.asarray(image, np.float32)[None], frames, axis=0)
        with self._torch.no_grad():
            states = self.net.simulate(
                self._hex(movie), self.dt, initial_state=self._state, as_states=True
            )
        self._state = states[-1]
        activity = states[-1].nodes.activity[0].cpu().numpy()
        if self.baseline is None:
            self.baseline = activity.copy()
        return activity

    def run(self, movie: np.ndarray) -> np.ndarray:
        """A whole movie at once, from the current state. Shape (frames, n_neurons)."""
        with self._torch.no_grad():
            states = self.net.simulate(
                self._hex(movie), self.dt, initial_state=self._state, as_states=True
            )
        self._state = states[-1]
        return np.stack([s.nodes.activity[0].cpu().numpy() for s in states])

    # ------------------------------------------------------------- readouts

    def type_activity(self, activity: np.ndarray, cell_type: str) -> np.ndarray:
        return activity[self._by_type[cell_type]]

    def features(self, activity: np.ndarray) -> np.ndarray:
        """Pool every cell type over five regions of the eye.

        For each of the 65 types: mean over the whole eye, the left and right
        halves, and the upper and lower halves -- 325 numbers that keep which
        cells are active and roughly where, and drop the exact column.
        """
        out = []
        for t in self.cell_types:
            idx = self._by_type[t]
            a, x, y = activity[idx], self.x[idx], self.y[idx]
            out += [
                a.mean(),
                a[x > 0].mean(),
                a[x < 0].mean(),
                a[y > 0].mean(),
                a[y < 0].mean(),
            ]
        return np.asarray(out, dtype=np.float32)

    def fovea(self, activity: np.ndarray, radius: float = 0.25) -> np.ndarray:
        """Every cell type's mean over the middle of the eye, minus its whole-eye mean.

        What a fixating animal has in front of it, relative to everything else:
        65 numbers. A fixated object sits here, and subtracting the whole-eye
        mean takes out most of what the sky, the ground and the distance do to
        every column at once -- which is what made the pooled features a poor
        code for learning from a handful of flights.
        """
        out = []
        for t in self.cell_types:
            idx = self._by_type[t]
            a, x, y = activity[idx], self.x[idx], self.y[idx]
            centre = (np.abs(x) < radius) & (np.abs(y) < radius)
            out.append(a[centre].mean() - a.mean())
        return np.asarray(out, dtype=np.float32)

    def salience(self, activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where something stands out: per-column deviation of the contrast cells.

        Returns (weights, x, y) over the 721 columns of the first contrast
        type's layout; the weights are the summed absolute deviation of all
        contrast types from the grey baseline.
        """
        if self.baseline is None:
            self.baseline = activity.copy()
        dev = np.abs(activity - self.baseline)
        idx0 = self._by_type[CONTRAST_TYPES[0]]
        w = np.zeros(len(idx0))
        for t in CONTRAST_TYPES:
            w += dev[self._by_type[t]]
        return w, self.x[idx0], self.y[idx0]

    def bearing(self, activity: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
        """Horizontal position of the most salient thing, and how much there is.

        Returns (bearing in [-1, 1], + left; size as the fraction of columns
        above ``threshold`` of the peak). Size growing over time is looming.
        """
        w, x, _ = self.salience(activity)
        if w.max() <= 1e-6:
            return 0.0, 0.0
        hot = w > threshold * w.max()
        return float(np.average(x[hot], weights=w[hot])), float(hot.mean())

    def dark(self, threshold: float = 0.2) -> tuple[float, float]:
        """How much of the view is dark, and where: (fraction, bearing).

        Read off the photoreceptor layer, not the network: the optic lobe
        model has no looming detectors (LC4 and LPLC2 sit downstream of what
        flyvis covers), so the angular size of a dark object is taken where
        the eye takes it in. Its growth over time is looming.
        """
        hot = self.photoreceptors < threshold
        if not hot.any():
            return 0.0, 0.0
        return float(hot.mean()), float(self.column_x[hot].mean())

    def motion(self, activity: np.ndarray) -> float:
        """Net horizontal motion signal, + for front-to-back on the left eye half.

        T4a/T5a prefer front-to-back and T4b/T5b back-to-front (horizontal
        directions), so their difference is the horizontal optic-flow signal
        the fly uses for its optomotor response.
        """
        a = sum(self.type_activity(activity, t).mean() for t in ("T4a", "T5a"))
        b = sum(self.type_activity(activity, t).mean() for t in ("T4b", "T5b"))
        return float(a - b)
