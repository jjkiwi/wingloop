# What is verified, and what is relayed

Inherited from `flyloop`, where a claim survived fifteen runs because it was
never a number. The rule: anything stated as fact carries a citation that
somebody checked, or it sits under **relayed** until somebody does.

## Verified

Checked against the literature on 2026-09-18.

### Aerodynamic coefficients

The quasi-steady model developed for *Drosophila melanogaster* from
dynamically-scaled robotic wing experiments:

- **Translational lift**, `lift_coefficient`:
  `CL(a) = 0.225 + 1.58 sin(2.13a - 7.2deg)`
- **Translational drag**, `drag_coefficient`:
  `CD(a) = 1.92 - 1.55 cos(2.04a - 9.82deg)`
- **Kramer rotational coefficient `Cr = 1.55`**, for rotation about the
  quarter-chord (`x0_hat = 0.25`), which is the biologically realistic axis.

The code writes the phase offsets in radians (0.1256 and 0.1714); those are
7.2 and 9.82 degrees, and `test_published_coefficients_are_unchanged` pins both
forms so a later edit cannot drift them silently. The lift coefficient peaks at
45.6 degrees, which is the behaviour these fits exist to capture.

### Descending neuron roles

- **DNg02 sets wingbeat amplitude.** About 15 pairs, working by population
  coding: the more DNg02 neurons active, the larger the amplitude, giving a
  smooth and wide dynamic range rather than a switch.
- **DNbe001 is a multisensory integration hub.** Broad sensory input, including
  antennal mechanosensory, with outputs reaching wing, haltere **and** front leg
  (T1) neuropils at once -- which implicates it in flight-to-walking
  coordination and reflex control.
- **DNa08 is a higher-order command population**, sexually dimorphic in
  morphology, assigned a direct role in initiating or sustaining flight,
  projecting from the anterior-dorsal brain to the ventral nerve cord.

## Relayed -- needs checking against the papers

- **Wingbeat frequency 218 Hz** and **stroke amplitude ~75 degrees** for
  hovering *D. melanogaster*, used by `stroke_average_lift`.
- **Drosophila flight power muscles are asynchronous and stretch-activated**:
  not driven one action potential to one contraction, but producing extra
  force a delay after being stretched, with the dorsal longitudinal and
  dorsoventral groups coupled through thorax deformation so each stretches the
  other. Neural input sets how hard, the resonance sets how fast. The whole of
  `wingloop/body/power.py` rests on this and it is stated from memory.
- **Circulation reaches most of its steady value within a couple of chord
  lengths of travel** -- `wake.RISE_CHORDS`. The direction and timescale of
  the Wagner effect; the true Wagner function is a sum of exponentials and
  this is one, which is enough to test whether wake memory matters here and
  not enough to predict forces from.
- **Reynolds number ~150** at this scale, and the attached leading-edge vortex
  that follows from it.

## Measured here, from the data

Not citations: numbers this project computed from MaleCNS v1.0 and from
NeuroMechFly's own model files.

### The connectome agrees with the physiology, independently

Grouping descending neurons by family and summing their input onto the 32 wing
steering motor neurons:

| family | share of all descending drive onto wing steering MNs |
| --- | ---: |
| **DNg02** | **9.07%** |
| DNbe001 | 4.06% |
| DNge107 | 3.88% |
| DNa08 | 3.53% |
| DNa10 | 3.20% |
| DNp63 | 3.18% |
| DNp49 | 3.16% |

DNg02 is the largest single descending input to the wing steering muscles, by
more than a factor of two over the next one. MaleCNS carries **29 DNg02 neurons
across seven subtypes (a-g)**, against the literature's ~15 pairs -- and a
population of that size distributed over seven subtypes is what population-coded
amplitude control looks like from the wiring side.

Neither figure was derived from the other. The physiology says DNg02 sets
amplitude by population coding; the wiring, read on its own, says DNg02 is the
biggest population aimed at those muscles. DNbe001 and DNa08 are single pairs
apiece, which is what command-type neurons look like, and they sit second and
fourth.

### The power channel, and the two channels apart

The flight power muscles are in MaleCNS as 24 motor neurons across five types
(DLMn a,b and c-f; DVMn 1a-c, 2a,b, 3a,b). Driving each descending family
forward and recording both motor pools -- all 24 power motor neurons and all
32 steering ones -- separates them cleanly:

| family | power MNs | steering MNs | ratio | visual input |
| --- | ---: | ---: | ---: | ---: |
| DNg02 | 0.4366 | 0.0410 | 10.6x | 1.48% |
| **DNa08** | 0.2676 | 0.0126 | **21.3x** | 0.71% |
| **DNp31** | 0.1403 | 0.0071 | **19.7x** | 29.51% |
| DNg110 | 0.0521 | 0.0052 | 10.0x | 0.64% |
| DNbe001 | 0.0319 | 0.0151 | 2.1x | 20.40% |
| DNa02 | 0.0000 | 0.0004 | 0.0x | 2.82% |

Three regimes: power-selective by an order of magnitude, shared (DNbe001, the
steering command), and steering-only (DNa02, which `flyloop` steers walking
with and which reaches the power muscles not at all). `COMMAND_TYPES` takes
DNa08 and DNp31 together -- 0.374 against 0.0185, **20x** -- and that is the
signal `PowerReadout` turns into oscillator drive.

DNg02 is the largest drive here *and* the least selective of the three, the
same verdict its input shares gave: 18.4% of the power muscles' descending
input against 9.1% of the steering muscles'. It is an amplitude control, not a
throttle, which is what the physiology above says it is.

**Two corrections this measurement forced, recorded because the wrong version
was written down first.**

1. *The separation is 20x, not 82x.* The first pass recorded four steering
   types instead of all sixteen and reported 82x, with DNa08 and DNg02 coming
   out infinitely selective. Against the whole apparatus neither is: the
   ordering survives, the magnitude does not. A selectivity measured against
   the muscles a command happens to miss is not a measurement of selectivity.
2. *The throttle is not blind.* The power motor neurons take **0.00%** of
   their input directly from visual neurons -- but so do the steering motor
   neurons, exactly 0.00%, so direct blindness is a property of wing motor
   neurons in general and separates nothing. And a synapse up the command is
   the more visual of the two: **DNp31 29.51%** against DNbe001's 20.40%.
   Vision can open this throttle. The scalar command in `PowerReadout` does
   not use that, and the class says so rather than claiming an anatomy it
   does not have.

The method was checked against itself: the same visual-share computation
returns 20.40% for DNbe001 and 29.51% for DNp31, so a 0.00% on the motor
neurons is the data and not a broken query.

### The whole stack, on a rail

Command in the brain to altitude in a body, with nothing written down between
them except the one activation-to-drive calibration. Two hundred milliseconds
on a vertical rail:

| command | stroke amplitude | height at 200 ms |
| ---: | ---: | ---: |
| 0.25 | 24.6 deg | -169 mm |
| 0.50 | 48.7 deg | -93 mm |
| 0.75 | 63.5 deg | +1.8 mm |
| 1.00 | 71.2 deg | +100 mm |

The hover point lands near three-quarter command. That is a consequence of the
connectome curve and the calibration rather than something aimed at, and it is
why the calibration was left where it is. Free flight at full command holds
attitude for 227.5 ms and climbs 151 mm.

### The altitude loop's two constants

Both swept on the rail rather than assumed, by fitting the vertical velocity
once the muscle has settled:

| flight command | stroke amplitude | vertical acceleration |
| ---: | ---: | ---: |
| 0.50 | 49.9 deg | -3956 mm/s^2 |
| 0.60 | 57.8 deg | -1995 |
| 0.70 | 65.2 deg | +89 |
| 0.80 | 72.0 deg | +2186 |
| 0.90 | 78.2 deg | +4267 |
| 1.00 | 84.1 deg | +6394 |

**21038 mm/s^2 per unit command**, linear across the range, crossing zero at a
**hover command of 0.695**. That is not the same as the break-even on a 200 ms
rail run, which is nearer 0.75: that one starts from rest and spends its first
wingbeats falling while the muscle spins up, so it has height to make back.

### The rest

- MaleCNS contains the VNC: 12,967 `vnc_intrinsic`, 699 `vnc_motor`.
- The **complete wing steering apparatus** is present, one motor neuron per
  side: b1, b2, b3, i1, i2, iii1, iii3, hg1-hg4, ps1, ps2, tp1, tp2, tpn.
  Power muscles too: DLMn a,b and c-f; DVMn 1a-c, 2a,b, 3a,b.
- **Descending neurons supply 7.59% of input to the wing steering motor
  neurons on average, 28.16% at most** -- fourteen times the 0.53% MBON->DNa02
  coupling that `flyloop` spends its runs on.
- **The walking steering readout does not transfer.** DNa02, which carries
  every steering result in `flyloop`, supplies **0.21%** of the descending
  drive onto wing motor neurons. DNa01 supplies 0.00%, DNp09 0.15%, DNa03
  0.08%.
- **NeuroMechFly cannot fly and is not close.** Its built model has 48
  actuators and **none of them is a wing or haltere**; the wings are rigid
  geoms welded to the thorax with `contype=0`, so they do not even collide.
  `density`, `viscosity` and `wind` are all 0 -- there is no air in the
  simulation. This is why the aerodynamics had to be written rather than
  configured.
- **Unit system**, read off the built model rather than assumed: length in
  millimetres (wing mesh spans 2.40), mass in grams (total body 1.027e-3),
  gravity 9810 mm/s^2.
