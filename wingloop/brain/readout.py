"""The connectome's steering command, for a body that flies.

`flyloop` steers a walking fly on ``DNa02_R - DNa02_L``. That readout does not
transfer: DNa02 supplies **0.21%** of the descending drive onto the wing
steering muscles. Flight runs on a different set, and which member of it
carries steering had to be measured rather than assumed.

Two things were measured, and they divide the labour cleanly.

**DNg02 is the actuator and it is blind.** It is the largest single descending
input to the wing steering muscles -- 9.07%, more than double the next -- and
about 29 cells across seven subtypes, which is what population-coded amplitude
control looks like from the wiring. It receives **1.48%** of its input from
visual neurons and 0.14% from the LC/LPLC projection types, and its
right-minus-left difference is **exactly zero at every bearing from -90 to
+90**. Nothing about where an object is reaches it.

**DNbe001 is the one that sees.** 20.4% visual input, 10.5% from LC/LPLC, the
second largest descending drive onto the wing muscles at 4.06%, and the
steepest bearing tuning of any candidate: a range of 0.175 across the visual
field against DNa10's 0.134 and DNge107's 0.042, with the sign the right way
round -- an object to the right drives the right side harder.

So the command path this module implements is **vision -> DNbe001 -> wing
amplitude asymmetry -> roll**, and the last arrow is the measured 38.2 of roll
torque per unit of asymmetry from :mod:`wingloop.body.control`.

**The curve is precomputed, and that is not a shortcut.** One pass through the
rate model costs about 0.16 s while a wingbeat is 4.6 ms, so a per-step readout
would be three hundred times slower than the animal. It is also the wrong
model of the animal: a fly's visual system does not resolve individual
wingbeats. Sampling the open-loop tuning once and interpolating is the same
thing `flyloop` calls the controller's open-loop shape, used here as a lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Descending neurons whose bearing tuning was compared. The winner is first.
CANDIDATES = ("DNbe001", "DNa10", "DNp09", "DNge107", "DNa08", "DNg02")

#: The type this reads steering from, and the one it does not.
STEERING_TYPE = "DNbe001"
AMPLITUDE_TYPE = "DNg02"

#: Bearings sampled when building the curve, degrees, positive to the right.
DEFAULT_BEARINGS = tuple(float(b) for b in range(-90, 91, 15))

_HINT = (
    "The flight readout needs flyloop for the connectome and the rate model:\n"
    "    pip install -e /path/to/flyloop"
)


def tuning(
    connectome,
    *,
    types=CANDIDATES,
    bearings=DEFAULT_BEARINGS,
    hops: int = 5,
    half_width: float = 5.0,
):
    """Right-minus-left difference for each type, against object bearing.

    The mirror-pair quantity `flyloop` established: taking one side alone
    confounds the stimulus with a wiring lean, and the difference cancels it.
    """
    try:
        from flyloop.brain.rate import population_index, rate_brain, steady_state
        from flyloop.vision.hexproject import HexWorldView
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_HINT) from exc

    view = HexWorldView(connectome)
    record = {}
    for side in ("L", "R"):
        record.update(population_index(connectome, tuple(types), side=side))
    brain = rate_brain(connectome, view.sensory, num_layers=hops)

    out = {"bearing": np.asarray(bearings, dtype=float)}
    columns = {name: [] for name in types}
    for bearing in bearings:
        peak = brain.run(
            steady_state(view.pattern([(bearing, half_width)]), hops), record=record
        ).peak()
        for name in types:
            columns[name].append(
                float(peak.get(f"{name}_R", 0.0) - peak.get(f"{name}_L", 0.0))
            )
    out.update({k: np.asarray(v) for k, v in columns.items()})
    return out


@dataclass
class FlightReadout:
    """Bearing in, steering command out, through the connectome.

    ``gain`` converts the descending difference into the wing amplitude
    asymmetry the body takes. At 1.0 the raw difference is used, which spans
    about +-0.09 across the visual field and is therefore already in a usable
    range -- the measured roll authority is 38.2 per unit, so the full field is
    worth about 3.4 of roll torque against a pitch disturbance of 3.5. That is
    a coincidence of scale, not a calibration, and it is why the gain exists.
    """

    bearings: np.ndarray
    command: np.ndarray
    gain: float = 1.0
    source: str = STEERING_TYPE
    extra: dict = field(default_factory=dict)

    @classmethod
    def measure(cls, connectome, *, gain: float = 1.0, source: str = STEERING_TYPE, **kw):
        curve = tuning(connectome, **kw)
        if source not in curve:
            raise KeyError(f"{source!r} was not among the types measured")
        return cls(
            bearings=curve["bearing"],
            command=curve[source],
            gain=gain,
            source=source,
            extra={k: v for k, v in curve.items() if k not in ("bearing", source)},
        )

    def asymmetry(self, bearing: float) -> float:
        """Wing amplitude asymmetry for an object at this bearing, degrees.

        Beyond the sampled field the command is held at its edge value rather
        than extrapolated: the tuning is not linear out there and inventing a
        slope would put the strongest commands where the measurements stop.
        """
        return float(
            self.gain * np.interp(bearing, self.bearings, self.command)
        )

    @property
    def span(self) -> float:
        """How much command the whole visual field is worth."""
        return float(self.command.max() - self.command.min())


#: Motor neuron types of the flight power muscles: the dorsal longitudinals
#: that drive the downstroke and the dorsoventrals that drive the upstroke.
POWER_TYPES = ("DLMn a, b", "DLMn c-f", "DVMn 1a-c", "DVMn 2a, b", "DVMn 3a, b")

#: Descending neurons used as the flight command, chosen by measurement.
#:
#: Of everything that reaches the power motor neurons, these two are the
#: power-selective ones: DNp31 supplies 20.2% of their descending input
#: against 1.9% of the steering muscles', and DNa08 12.6% against 3.5%.
#: DNg02 is larger still on the power muscles at 18.4% but drives the steering
#: muscles just as hard at 9.1%, so it is not a throttle.
#:
#: Driven together they raise the power motor neurons to 0.374 while leaving
#: the steering motor neurons at 0.019 -- a twentyfold separation, which is
#: what makes this a throttle channel rather than a second steering one.
COMMAND_TYPES = ("DNa08", "DNp31")

#: The whole wing steering apparatus, one motor neuron per side, as MaleCNS
#: names it: 32 neurons.
#:
#: **All of it, deliberately.** An earlier version of this recorded four types
#: and reported the command as 82x selective. Against the full pool it is 20x.
#: Both numbers are real; only the second is about the steering apparatus, and
#: a selectivity measured against the muscles a command happens to miss is not
#: a measurement of selectivity.
STEERING_MN_TYPES = (
    "b1 MN", "b2 MN", "b3 MN",
    "i1 MN", "i2 MN",
    "iii1 MN", "iii3 MN",
    "hg1 MN", "hg2 MN", "hg3 MN", "hg4 MN",
    "ps1 MN", "ps2 MN",
    "tp1 MN", "tp2 MN", "tpn MN",
)

#: Oscillator drive per unit of power motor neuron activity.
#:
#: **A calibration, not a measurement.** The rate model returns activations in
#: arbitrary units and :class:`~wingloop.body.power.PowerOscillator` wants a
#: dimensionless gain, so something has to bridge them. This is set so a full
#: command produces the drive that gives a hovering-sized stroke, and it is
#: the same kind of free parameter as the coupling gain in `flyloop`: every
#: result computed through it has to be read with it in view.
DRIVE_PER_ACTIVATION = 8.0


def power_response(connectome, *, levels=(0.0, 0.25, 0.5, 0.75, 1.0), hops: int = 4):
    """Power motor neuron activity against flight command level.

    Injects at :data:`COMMAND_TYPES` and propagates, which is the command
    formulation: the brain decides to fly, the command neurons fire, the power
    muscles follow. The steering motor neurons are recorded alongside so the
    separation between the two channels stays visible rather than assumed --
    and so it can be checked against a family that does not separate them, as
    :class:`PowerReadout` tabulates.
    """
    try:
        import numpy as _np
        from flyloop.brain.rate import rate_brain, steady_state
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_HINT) from exc

    types = connectome.neurons["type"].astype(str)
    command = _np.flatnonzero(types.isin(COMMAND_TYPES).to_numpy())
    if len(command) == 0:
        raise KeyError(f"connectome has none of {COMMAND_TYPES}")
    record = {
        "power": _np.flatnonzero(types.isin(POWER_TYPES).to_numpy()),
        "steering": _np.flatnonzero(types.isin(STEERING_MN_TYPES).to_numpy()),
    }
    if len(record["power"]) == 0:
        raise KeyError("connectome has no power muscle motor neurons")

    brain = rate_brain(connectome, command, num_layers=hops)
    out = {"command": _np.asarray(levels, dtype=float), "power": [], "steering": []}
    for level in levels:
        drive = _np.full(len(command), float(level), dtype=_np.float32)
        res = brain.run(steady_state(drive, hops), record=record)
        for key in ("power", "steering"):
            out[key].append(float(res.populations[key].max()))
    out["power"] = _np.asarray(out["power"])
    out["steering"] = _np.asarray(out["steering"])
    return out


@dataclass
class PowerReadout:
    """Flight command in, oscillator drive out, through the power muscles.

    The counterpart of :class:`FlightReadout`: a scalar throttle, with no
    bearing in it. What makes that the right shape is **selectivity of
    drive**, measured by pushing each descending family forward and recording
    both motor pools (peak activation over four hops, against all 32 steering
    motor neurons -- see :data:`STEERING_MN_TYPES` for why all of them):

    ========  ==========  ============  ======  =========
    family    power MNs   steering MNs  ratio   visual in
    ========  ==========  ============  ======  =========
    DNg02        0.4366       0.0410     10.6x      1.48%
    DNa08        0.2676       0.0126     21.3x      0.71%
    DNp31        0.1403       0.0071     19.7x     29.51%
    DNg110       0.0521       0.0052     10.0x      0.64%
    DNbe001      0.0319       0.0151      2.1x     20.40%
    DNa02        0.0000       0.0004      0.0x      2.82%
    ========  ==========  ============  ======  =========

    Three regimes, and the dissociation is what the claim rests on: families
    that move the power muscles an order of magnitude harder than the steering
    ones, one that moves both about alike (DNbe001, 2.1x -- the steering
    command, and no kind of throttle), and one that moves steering only
    (DNa02). :data:`COMMAND_TYPES` takes DNa08 and DNp31, the two most
    selective: 0.374 against 0.0185, **20x**.

    Note where DNg02 sits. It is the largest drive by some way and the least
    selective of the three, which is the same verdict the input shares gave
    when the command was chosen -- 18.4% of the power muscles' descending
    input but 9.1% of the steering muscles'. It belongs to amplitude control,
    which is a different job from opening a throttle.

    **And one claim measurement took away.** The power motor neurons receive
    0.00% of their input directly from visual neurons -- but so do the
    steering motor neurons, exactly 0.00%. Direct blindness is a property of
    wing motor neurons in general and separates nothing, so it is not a reason
    for anything, and the throttle is not blind a synapse up either: DNp31,
    half of this command, draws **29.51%** of its input from visual neurons,
    more than the 20.40% of DNbe001, which is the steering command. Vision can
    open this throttle. For a fly flying *toward* something that is a feature,
    but it was not designed in and it is not what the scalar command models.
    """

    command: np.ndarray
    activation: np.ndarray
    gain: float = DRIVE_PER_ACTIVATION
    steering: np.ndarray | None = None

    @classmethod
    def measure(cls, connectome, *, gain: float = DRIVE_PER_ACTIVATION, **kw):
        curve = power_response(connectome, **kw)
        return cls(
            command=curve["command"],
            activation=curve["power"],
            gain=gain,
            steering=curve["steering"],
        )

    def drive(self, command: float) -> float:
        """Oscillator drive for a flight command in [0, 1]."""
        return float(
            self.gain * np.interp(command, self.command, self.activation)
        )

    @property
    def separation(self) -> float:
        """How much more this command moves the power muscles than the steering
        ones, at full command. Large is what makes it a throttle."""
        if self.steering is None or self.steering[-1] == 0:
            return float("inf")
        return float(self.activation[-1] / self.steering[-1])
