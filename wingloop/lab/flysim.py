"""The fly itself in flight, with the neurons that fly it on the record.

This is ``wingloop.body`` assembled into one call: NeuroMechFly with its legs
tucked and its wings driven by a muscle oscillator, blade-element
aerodynamics, the haltere loop on attitude, the throttle loop on height
through the connectome's power channel, and -- when a target is given -- the
visual steering loop through DNbe001.

What "brain activity" means here is stated rather than implied. The neurons
recorded are the ones the connectome measurements in ``wingloop.brain``
connect to the wings, read off the stored response curves at each step:

- the **throttle command** -- the drive on DNa08 and DNp31, the descending
  pair that reaches the power motor neurons selectively;
- the **power motor neurons** (DLMn, DVMn) -- their mean activation at that
  command, from the connectome's measured command-to-activation curve;
- the **steering motor neurons** -- their activation at the same command;
- **DNbe001's left-right difference** -- the steering signal, from its
  measured tuning to the bearing of a visual object.

These are a rate model's steady-state answers to the current command, not a
spiking simulation of the whole brain running alongside the body.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "tests"


@lru_cache(maxsize=1)
def _models():
    from ..aero.wing import wing_from_mesh
    from ..body.hinge import add_free_base, neuromechfly_model, tuck_legs

    d = Path(tempfile.mkdtemp(prefix="flylab-"))
    rigid, _ = tuck_legs(neuromechfly_model(), d / "rigid.xml", keep="__none__")
    free = add_free_base(rigid, d / "free.xml", dofs="free")
    wing = wing_from_mesh(
        np.load(FIXTURES / "rwing_vertices.npy"), n=20, span_axis=2, chord_axis=1
    )
    return free, wing


@dataclass
class Brain:
    """The connectome-measured response curves the flight neurons are read from."""

    command: np.ndarray
    activation: np.ndarray
    steering: np.ndarray
    bearings: np.ndarray
    dnbe001: np.ndarray

    @classmethod
    def stored(cls) -> Brain:
        power = dict(np.load(FIXTURES / "power_command.npz"))
        tuning = dict(np.load(FIXTURES / "dnbe001_tuning.npz"))
        return cls(
            command=power["command"],
            activation=power["activation"],
            steering=power["steering"],
            bearings=tuning["bearings"],
            dnbe001=tuning["command"],
        )

    def read(self, throttle: float, bearing: float | None) -> dict:
        c = float(np.clip(throttle, self.command[0], self.command[-1]))
        out = {
            "DNa08/DNp31 drive": c,
            "power MNs (DLMn, DVMn)": float(np.interp(c, self.command, self.activation)),
            "steering MNs": float(np.interp(c, self.command, self.steering)),
            "DNbe001 L-R": 0.0,
        }
        if bearing is not None:
            out["DNbe001 L-R"] = float(np.interp(bearing, self.bearings, self.dnbe001))
        return out


def fly(
    seconds: float = 0.5,
    height: float = 0.0,
    target: tuple | None = None,
    drive: float = 0.0,
    record_every: int = 50,
) -> dict:
    """Fly the physical fly; return its trajectory and its flight neurons.

    ``height`` is the altitude setpoint for the throttle loop, mm. ``target``
    is an (x, y) point in mm to steer toward, or None to hold heading.
    Recorded every ``record_every`` physics steps (1 ms at the default).
    """
    from ..body.control import HaltereController, SteeringController, Throttle
    from ..body.flight import FlightBody
    from ..body.power import PowerOscillator, PowerStroke, aerodynamic_load
    from ..brain.readout import FlightReadout, PowerReadout

    free, wing = _models()
    brain = Brain.stored()
    readout = PowerReadout(command=brain.command, activation=brain.activation)
    oscillator = PowerOscillator(drive=drive, load=aerodynamic_load(wing))
    oscillator.angle = np.deg2rad(1.0)
    body = FlightBody(free, wing, timestep=2e-5)
    common = dict(
        stroke=PowerStroke(oscillator),
        throttle=Throttle(readout=readout, oscillator=oscillator, target=height),
    )
    if target is not None:
        steering = FlightReadout(bearings=brain.bearings, command=brain.dnbe001)
        controller = SteeringController(readout=steering, target=tuple(target), **common)
    else:
        controller = HaltereController(**common)

    log = {k: [] for k in ("t", "x", "y", "z", "pitch", "roll", "heading", "neurons")}

    original = controller.knobs
    step = {"n": 0}

    def knobs(b):
        k = original(b)
        step["n"] += 1
        if step["n"] % record_every == 0:
            bearing = getattr(controller, "bearing", None) if target is not None else None
            log["neurons"].append(brain.read(controller.throttle.last or 0.0, bearing))
        return k

    controller.knobs = knobs
    trace = controller.fly(body, seconds)
    n = len(trace["t"])
    back = np.flatnonzero(np.diff(trace["t"]) < 0)
    n = int(back[0]) + 1 if len(back) else n
    idx = np.arange(record_every - 1, n, record_every)[: len(log["neurons"])]
    for k in ("t", "x", "y", "z", "pitch", "roll", "heading"):
        log[k] = [float(v) for v in np.asarray(trace[k])[idx]]
    log["neurons"] = log["neurons"][: len(idx)]
    log["diverged_at"] = controller.diverged_at
    return log
