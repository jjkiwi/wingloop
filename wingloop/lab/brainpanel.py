"""The fly's nervous system in flight, drawn live next to the 3-D view.

What is on the panel, from the eye inwards:

- **what the fly sees** -- the arena rendered from the fly's own position and
  attitude through a 120 degree lens, and the same image as its 721
  photoreceptor columns receive it;
- **its optic lobe** -- the pretrained connectome-constrained model from
  ``flyvis`` stepping on those images in real (simulated) time: the T4 and T5
  motion detectors and the ON and OFF contrast cells, one dot per column;
- **halteres** -- the body rotation rates the haltere loop feels;
- **descending and motor neurons** -- the throttle drive on DNa08/DNp31 and
  the power and steering motor neurons, read off the connectome's measured
  response curves at the current command;
- **wing commands** -- the three stroke knobs the controller is turning: the
  stroke bias that pitches, the left-right amplitude difference that rolls,
  and the rotation-phase difference that yaws.

The optic lobe is the only part with tens of thousands of neurons actually
running; the descending and motor rows are a rate model's readout, and the
haltere and wing rows are the controller's own signals. The labels say which.
"""

from __future__ import annotations

from collections import deque

import numpy as np

#: How many seconds of simulated flight the traces show.
HISTORY = 0.4

BARS = [
    ("haltere: pitch rate", "halteres"),
    ("haltere: roll rate", "halteres"),
    ("haltere: yaw rate", "halteres"),
    ("DNa08/DNp31 drive", "descending"),
    ("power MNs (DLMn, DVMn)", "motor"),
    ("steering MNs", "motor"),
    ("wing: stroke bias", "wing"),
    ("wing: amplitude L-R", "wing"),
    ("wing: rotation phase L-R", "wing"),
]
#: Two-line tick labels, so neighbouring bars do not write over each other.
LABELS = {
    "haltere: pitch rate": "haltere\npitch rate",
    "haltere: roll rate": "haltere\nroll rate",
    "haltere: yaw rate": "haltere\nyaw rate",
    "DNa08/DNp31 drive": "DNa08/DNp31\ndrive",
    "power MNs (DLMn, DVMn)": "power MNs\nDLMn, DVMn",
    "steering MNs": "steering\nMNs",
    "wing: stroke bias": "wing: stroke\nbias (pitch)",
    "wing: amplitude L-R": "wing: ampl.\nL-R (roll)",
    "wing: rotation phase L-R": "wing: phase\nL-R (yaw)",
}
COLOURS = {
    "halteres": "#8f5bd6",
    "descending": "#2a6f97",
    "motor": "#c1121f",
    "wing": "#2b9348",
}
MAPS = (("T4a", "T4b"), ("T5a", "T5b"), ("Mi1",), ("Tm9",))
MAP_TITLES = (
    "T4 ON motion (a-b)",
    "T5 OFF motion (a-b)",
    "Mi1 (ON contrast)",
    "Tm9 (OFF contrast)",
)


def signals(state: dict, knobs: dict | None, rates) -> dict:
    """The bar values, each scaled to roughly [-1, 1]."""
    n = state["neurons"]
    k = knobs or {}
    from ..body.control import MAX_PHASE, TRIM_BIAS

    return {
        "haltere: pitch rate": float(np.clip(rates[1] / 2.0, -1, 1)),
        "haltere: roll rate": float(np.clip(rates[0] / 2.0, -1, 1)),
        "haltere: yaw rate": float(np.clip(rates[2] / 2.0, -1, 1)),
        "DNa08/DNp31 drive": n["DNa08/DNp31 drive"],
        "power MNs (DLMn, DVMn)": n["power MNs (DLMn, DVMn)"] / 0.4,
        "steering MNs": n["steering MNs"] / 0.02,
        "wing: stroke bias": float(
            np.clip((k.get("bias", TRIM_BIAS) - TRIM_BIAS) / 0.2, -1, 1)
        ),
        "wing: amplitude L-R": float(np.clip(k.get("asymmetry", 0.0) / 0.1, -1, 1)),
        "wing: rotation phase L-R": float(
            np.clip(k.get("phase_asymmetry", 0.0) / MAX_PHASE, -1, 1)
        ),
    }


class BrainPanel:
    def __init__(self, eye=None, live: bool = True):
        import matplotlib

        if not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.eye, self.live = plt, eye, live
        if live:
            plt.ion()
        self.fig = plt.figure(figsize=(11, 6.2))
        if live and self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("flylab: the fly's brain in flight")
        grid = self.fig.add_gridspec(
            3, 6, height_ratios=[1.0, 1.0, 0.75], hspace=0.55, wspace=0.35
        )
        self.ax_view = self.fig.add_subplot(grid[0, 0])
        self.ax_photo = self.fig.add_subplot(grid[0, 1])
        self.ax_maps = [self.fig.add_subplot(grid[0, 2 + i]) for i in range(4)]
        self.ax_bars = self.fig.add_subplot(grid[1, :])
        self.ax_trace = self.fig.add_subplot(grid[2, :])
        self.view_img = self.ax_view.imshow(np.zeros((64, 64)), cmap="gray", vmin=0, vmax=1)
        self.ax_view.set_title("what it sees", fontsize=9)
        self.photo = self.maps = None
        if eye is not None:
            self.photo = self.ax_photo.scatter(
                eye.column_x * -1,
                eye.column_y,
                c=np.zeros(721),
                s=6,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            self.ax_photo.set_title("photoreceptors (721)", fontsize=9)
            self.maps = []
            for ax, types, title in zip(self.ax_maps, MAPS, MAP_TITLES, strict=True):
                idx = eye._by_type[types[0]]
                sc = ax.scatter(
                    -eye.x[idx],
                    eye.y[idx],
                    c=np.zeros(len(idx)),
                    s=6,
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                )
                ax.set_title(title, fontsize=9)
                self.maps.append((sc, types))
        else:
            for ax in [self.ax_photo, *self.ax_maps]:
                ax.text(
                    0.5, 0.5, "optic lobe off\n(press B)", ha="center", va="center", fontsize=8
                )
        for ax in [self.ax_view, self.ax_photo, *self.ax_maps]:
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal")
        names = [b for b, _ in BARS]
        self.bars = self.ax_bars.bar(
            range(len(BARS)), np.zeros(len(BARS)), color=[COLOURS[g] for _, g in BARS]
        )
        self.ax_bars.set_xticks(range(len(BARS)), [LABELS.get(n, n) for n in names], fontsize=8)
        self.ax_bars.set_ylim(-1.05, 1.05)
        self.ax_bars.axhline(0, color="#999", lw=0.8)
        self.ax_bars.set_title(
            "halteres  |  descending neurons  |  motor neurons  |  wing commands", fontsize=9
        )
        self.history = {k: deque() for k in ("t", "drive", "power", "yaw", "speed")}
        self.lines = {
            "drive": self.ax_trace.plot(
                [], [], color=COLOURS["descending"], label="DNa08/DNp31 drive"
            )[0],
            "power": self.ax_trace.plot(
                [], [], color=COLOURS["motor"], label="power MNs / 0.4"
            )[0],
            "yaw": self.ax_trace.plot(
                [], [], color=COLOURS["wing"], label="rotation phase L-R"
            )[0],
            "speed": self.ax_trace.plot([], [], color="#555", label="speed / 400 mm/s")[0],
        }
        self.ax_trace.set_ylim(-1.05, 1.05)
        self.ax_trace.legend(
            loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, ncol=4, frameon=False
        )
        self.ax_trace.set_xlabel("simulated time (ms)", fontsize=8)
        self.title = self.fig.suptitle("", fontsize=10)
        if live:
            self.plt.show(block=False)

    def update(self, state, knobs, rates, image=None, activity=None) -> None:
        values = signals(state, knobs, rates)
        for bar, (name, _) in zip(self.bars, BARS, strict=True):
            bar.set_height(values[name])
        if image is not None:
            self.view_img.set_data(image)
        if activity is not None and self.eye is not None and self.maps is not None:
            self.photo.set_array(self.eye.photoreceptors)
            base = self.eye.baseline if self.eye.baseline is not None else activity
            for sc, types in self.maps:
                if len(types) == 2:
                    a = self.eye.type_activity(activity, types[0]) - self.eye.type_activity(
                        activity, types[1]
                    )
                else:
                    a = self.eye.type_activity(activity, types[0]) - self.eye.type_activity(
                        base, types[0]
                    )
                scale = max(np.abs(a).max(), 1e-3)
                sc.set_array(np.clip(a / scale, -1, 1))
        h = self.history
        t = state["t"] * 1000
        for key, value in (
            ("t", t),
            ("drive", values["DNa08/DNp31 drive"]),
            ("power", values["power MNs (DLMn, DVMn)"]),
            ("yaw", values["wing: rotation phase L-R"]),
            ("speed", min(state["speed"] / 400.0, 1.0)),
        ):
            h[key].append(value)
        while h["t"] and h["t"][0] < t - HISTORY * 1000:
            for q in h.values():
                q.popleft()
        for key, line in self.lines.items():
            line.set_data(h["t"], h[key])
        if h["t"]:
            self.ax_trace.set_xlim(h["t"][0], max(h["t"][-1], h["t"][0] + 1))
        self.title.set_text(
            f"t = {t:5.0f} ms   speed {state['speed']:4.0f} mm/s   "
            f"height {state['z']:5.1f} mm   heading {state['heading']:+5.0f} deg"
        )
        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def image(self) -> np.ndarray:
        """The panel as an RGB array, for video."""
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def is_open(self) -> bool:
        return (not self.live) or self.plt.fignum_exists(self.fig.number)
