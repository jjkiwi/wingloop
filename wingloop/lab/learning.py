"""Learning what an object is worth: a mushroom body on the visual features.

The rule is the fly's, and the one ``flyloop.brain.dopamine`` uses: dopamine
arriving while a Kenyon cell is active **depresses** that cell's synapse onto a
mushroom-body output neuron. Reward-signalling dopamine (the PAM cluster)
depresses the synapses onto an *avoidance* output; punishment (PPL1) depresses
those onto an *approach* output. What the animal then does with an object is
the difference between the two outputs it still drives.

The architecture follows the fly's numbers in shape, not in identity:

- **Kenyon cells** -- 2,000, each summing a handful of randomly chosen inputs,
  as each real KC samples about seven projection neurons;
- **APL inhibition** -- only the top 5% of KCs stay active, the sparse code
  that keeps two stimuli from sharing their memories;
- **two output neurons** and **two dopamine signals**.

The inputs are the pooled optic-lobe features from :mod:`.vision`, standing in
for the visual projection neurons that reach the mushroom body's accessory
calyx. That substitution is the one liberty taken here, and it is stated: the
KC sampling is random, not the measured visual-PN to KC wiring.

**What it learns, measured** on 360 recorded flights past one object each
(five random splits, 240 to train and 120 to test, one pass of training):

==========================  ===============  ==================
pairing                     correct choices  unpaired control
==========================  ===============  ==================
bar rewarded, sphere        92-95%           38-64%
punished
sphere rewarded, box        60-67%           37-53%
punished
==========================  ===============  ==================

The second row is not the learning rule failing. Given ideal object channels
the same rule reaches valences of +0.94 and -0.98; given the optic lobe, the
sphere and the box are what the *eye* confuses -- the linear recogniser in
:mod:`.recognition` calls a box a sphere 40% of the time and a sphere a box
26%, while it gets the bar right 98%. The mushroom body can only value what the eye
tells apart, which is the real animal's constraint too.

**Learning in the drone, from its own flights.** Trained on the pooled
features of 23 fixations made while hovering and turning, the same rule
learned its training set perfectly and new flights at chance (leave-one-out,
50%): each fixation's code was dominated by the sky, the ground and the
distance, so no two flights of the same object shared Kenyon cells. A
fixating animal has the object in the middle of its eye, and with the input
taken from there -- every cell type's central mean minus its whole-eye mean,
:meth:`~.vision.FlyEye.fovea` -- 20 training flights give **80% correct
choices on 20 new ones**, against 39-47% for unpaired dopamine. A linear
readout of the same 65 numbers transfers perfectly, so the gap is the price
of a random sparse expansion trained on twenty examples.

One measured trap: with a high learning rate, every confidently misrecognised
sample depresses the synapses its Kenyon cells use for *both* outputs, and
with enough of them both outputs reach zero and every valence reads 0. On a
separable pair it does not matter; on a confusable one it is why the
valences are small rather than wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MushroomBody:
    n_inputs: int
    n_kc: int = 2000
    claws: int = 7
    sparseness: float = 0.05
    learning_rate: float = 0.6
    seed: int = 0
    mean: np.ndarray = field(default=None, repr=False)
    scale: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.wiring = np.zeros((self.n_kc, self.n_inputs), dtype=np.float32)
        for k in range(self.n_kc):
            self.wiring[k, rng.choice(self.n_inputs, self.claws, replace=False)] = 1.0
        self.reset()

    def reset(self) -> None:
        """Forget everything: every KC drives both outputs equally."""
        self.approach = np.ones(self.n_kc, dtype=np.float32)
        self.avoid = np.ones(self.n_kc, dtype=np.float32)

    def calibrate(self, X: np.ndarray) -> None:
        """Input normalisation from unlabelled experience (no reward involved)."""
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0) + 1e-6

    def kenyon(self, x: np.ndarray) -> np.ndarray:
        """The sparse KC code for one input: a 0/1 vector with 5% ones."""
        z = (x - self.mean) / self.scale if self.mean is not None else x
        drive = self.wiring @ np.abs(z)  # rectified: KCs are excited by deviations
        k = max(1, int(round(self.sparseness * self.n_kc)))
        code = np.zeros(self.n_kc, dtype=np.float32)
        code[np.argpartition(drive, -k)[-k:]] = 1.0
        return code

    def outputs(self, x: np.ndarray) -> tuple[float, float]:
        code = self.kenyon(x)
        k = code.sum()
        return float(code @ self.approach / k), float(code @ self.avoid / k)

    def valence(self, x: np.ndarray) -> float:
        """+1 strongly approach, -1 strongly avoid, 0 indifferent."""
        approach, avoid = self.outputs(x)
        return float(np.clip(approach - avoid, -1.0, 1.0))

    def learn(self, x: np.ndarray, dopamine: float) -> None:
        """Pair this stimulus with reward (``dopamine`` > 0) or punishment (< 0)."""
        code = self.kenyon(x)
        if dopamine > 0:  # PAM: depress the active KCs' avoidance synapses
            self.avoid *= 1.0 - self.learning_rate * dopamine * code
        elif dopamine < 0:  # PPL1: depress their approach synapses
            self.approach *= 1.0 + self.learning_rate * dopamine * code


def train(mb: MushroomBody, X, y, reward: dict, epochs: int = 1, seed: int = 0) -> None:
    """Show each sample with the dopamine its class earns, in random order."""
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            mb.learn(X[i], reward.get(int(y[i]), 0.0))


def learning_index(mb: MushroomBody, X, y) -> dict:
    """Mean valence per class on stimuli the animal has not been trained on."""
    return {int(c): float(np.mean([mb.valence(x) for x in X[y == c]])) for c in np.unique(y)}


def save(mb: MushroomBody, path) -> None:
    """Everything the animal has learned, and the wiring it learned it on."""
    np.savez_compressed(
        path,
        wiring=mb.wiring,
        approach=mb.approach,
        avoid=mb.avoid,
        mean=mb.mean,
        scale=mb.scale,
        params=np.array(
            [mb.n_inputs, mb.n_kc, mb.claws, mb.sparseness, mb.learning_rate, mb.seed]
        ),
    )


def load(path) -> MushroomBody:
    d = np.load(path)
    n_inputs, n_kc, claws, sparseness, lr, seed = d["params"]
    mb = MushroomBody(
        int(n_inputs), int(n_kc), int(claws), float(sparseness), float(lr), int(seed)
    )
    mb.wiring, mb.approach, mb.avoid = d["wiring"], d["approach"], d["avoid"]
    mb.mean, mb.scale = d["mean"], d["scale"]
    return mb
