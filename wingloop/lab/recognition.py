"""Recognising objects from the fly's optic-lobe activity.

The optic lobe model was trained on optic flow, not on objects, and it stops
where the fly's object-selective neurons (the lobula columnar LC types) begin.
So recognition here is a *readout*: a linear classifier trained on the pooled
activity of the 65 cell types while the animal moves past an object. Linear on
purpose -- a deep classifier on top would recognise objects from any input,
and the question worth asking is what the fly's own cells already make
linearly separable.

Two controls ship with it, because a classifier always produces a number:

- **shuffled labels** -- the same pipeline on permuted classes, which must sit
  at chance;
- **photoreceptors only** -- the same pooling and the same classifier on the
  eye's input layer, skipping the network. Whatever the optic lobe adds is the
  difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import Camera, Thing, World, render

CLASSES = ("sphere", "bar", "box")


def random_scene(rng: np.random.Generator, kind: str) -> tuple[World, np.ndarray]:
    """One object somewhere ahead, and the observer's forward velocity."""
    distance = rng.uniform(3.0, 7.0)
    lateral = rng.uniform(-2.0, 2.0)
    size = rng.uniform(0.35, 0.8)
    brightness = rng.uniform(0.02, 0.25)
    if kind == "bar":
        thing = Thing(
            "bar",
            [distance, lateral, 0.0],
            size * 0.5,
            height=rng.uniform(2.5, 4.0),
            brightness=brightness,
        )
    else:
        thing = Thing(
            kind, [distance, lateral, rng.uniform(1.0, 2.0)], size, brightness=brightness
        )
    return World([thing]), np.array([rng.uniform(0.5, 2.0), rng.uniform(-0.3, 0.3), 0.0])


def observe(
    eye, world: World, velocity, frames: int = 8, dt: float = 0.02, camera: Camera | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fly past the scene for a few frames; return (network features, input features).

    Both are averaged over the second half of the clip, after the onset
    transient.
    """
    eye.reset()
    pos = np.array([0.0, 0.0, 1.5])
    net, raw = [], []
    for k in range(frames):
        act = eye.step(render(world, pos, (0.0, 0.0, 0.0), camera), frames=2)
        pos = pos + np.asarray(velocity) * dt
        if k >= frames // 2:
            net.append(eye.features(act))
            raw.append(pool_input(eye))
    return np.mean(net, axis=0), np.mean(raw, axis=0)


def pool_input(eye) -> np.ndarray:
    """The photoreceptor image pooled the way :meth:`FlyEye.features` pools a type."""
    a, x, y = eye.photoreceptors, eye.column_x, eye.column_y
    pools = [a.mean(), a[x > 0].mean(), a[x < 0].mean(), a[y > 0].mean(), a[y < 0].mean()]
    # Plus a coarse 6x6 grid, so the baseline has spatial detail to work with
    # and cannot lose merely for being given less.
    grid = []
    for gx in np.linspace(-1, 1, 7)[:-1]:
        for gy in np.linspace(-1, 1, 7)[:-1]:
            m = (x >= gx) & (x < gx + 1 / 3) & (y >= gy) & (y < gy + 1 / 3)
            grid.append(a[m].mean() if m.any() else 0.5)
    return np.asarray(pools + grid, dtype=np.float32)


def make_dataset(eye, n_per_class: int = 100, seed: int = 0, frames: int = 8):
    """Render, fly and record: returns (network X, input X, labels)."""
    rng = np.random.default_rng(seed)
    xs, raws, ys = [], [], []
    for _ in range(n_per_class):
        for label, kind in enumerate(CLASSES):
            world, velocity = random_scene(rng, kind)
            net, raw = observe(eye, world, velocity, frames=frames)
            xs.append(net)
            raws.append(raw)
            ys.append(label)
    return np.asarray(xs), np.asarray(raws), np.asarray(ys)


@dataclass
class ObjectRecognizer:
    """Standardise, then multinomial logistic regression."""

    classes: tuple = CLASSES
    C: float = 1.0
    model: object = field(default=None, repr=False)

    def fit(self, X, y) -> ObjectRecognizer:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.C, max_iter=2000)
        )
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.atleast_2d(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(X))

    def name(self, x) -> str:
        return self.classes[int(self.predict(x)[0])]

    def score(self, X, y) -> float:
        return float((self.predict(X) == y).mean())


def evaluate(X, X_input, y, seed: int = 0, folds: int = 5) -> dict:
    """Cross-validated accuracy for the network, the input baseline, and chance."""
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(y)
    out = {"network": [], "photoreceptors": [], "shuffled": []}
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        out["network"].append(
            ObjectRecognizer().fit(X[train], y[train]).score(X[test], y[test])
        )
        out["photoreceptors"].append(
            ObjectRecognizer().fit(X_input[train], y[train]).score(X_input[test], y[test])
        )
        out["shuffled"].append(
            ObjectRecognizer().fit(X[train], shuffled[train]).score(X[test], shuffled[test])
        )
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()} | {
        "chance": 1.0 / len(set(y.tolist()))
    }
