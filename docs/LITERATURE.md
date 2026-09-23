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
attitude for 338 ms and climbs 239 mm (re-measured after the sensing change
below; it was 227.5 ms and 151 mm through the first-order filter).

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

### How the loop senses, and the spike that nearly hid it

Flight time against sensor-filter lag, first-order filter at bandwidth 40:

| tau (ms) | 3.0 | 4.0 | 4.5 | 4.8 | 5.0 | 5.2 | 5.5 | 6.0 | 7.0 | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| holds 30 deg | 146 | 193 | 214 | 249 | **353** | **379** | 256 | 236 | 221 | 204 | 187 |

The peak was a spike, not the shape of the curve. A 1% change in stroke
amplitude moved it: best at 4.8 ms for 74.25 degrees, 5.2 for 75.00, 5.5 for
75.75.

**The spike was the missing yaw damping.** Re-measured once the wings are told
the body rotates, the curve is smooth and the same shape at every amplitude:

| tau (ms) | 3.0 | 4.8 | 5.0 | 5.2 | 6.0 | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| holds, amplitude 74.25 | 514 | 993 | 997 | 999 | **1002** | 928 | 619 |
| holds, amplitude 75.75 | 444 | 891 | 899 | 899 | **903** | 864 | 595 |

One broad optimum at 5-6 ms. The chaotic landscape that made a single setting
look twice as good as its neighbours was an animal with no passive yaw
damping at all.

Sensed pitch-rate power, by band, in free flight: **75.4% at 200-240 Hz** (the
wingbeat), 17.8% above 500 (harmonics, peak at 654 = 3x), and under 4%
anywhere below 200. The disturbance is the stroke, almost entirely.

Flight against loop bandwidth, at three stroke amplitudes (median):

| rad/s | 30 | 40 | 50 | 55 | 60 | 70 | 80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first-order, 6 ms | 160 | 236 | 209 | 183 | 163 | 146 | 134 |
| one-wingbeat boxcar | 162 | 237 | 277 | 295 | **320** | 260 | 154 |

A 229-sample boxcar at this timestep is 56 dB down at 218 Hz against the
first-order filter's 26, with a group delay of 2.29 ms against 5.

### The pitch tether, and where its pin goes

The centre of mass sits **1.07 mm above the model origin and 0.30 mm behind
it**. A pitch tether pinned at the origin therefore converts the net
aerodynamic force into a torque about the pin that no trim can cancel, and the
tethered animal spins continuously at 80-107 Hz at every gain and every filter.
Pinned through the centre of mass it does not.

Even correct, the tether holds only 30-90 ms where the free animal holds 350:
constraining the translation removes something that stabilises the free
flight, so a tether is the harder preparation here, not the cleaner one.

### Yaw: the axis with no loop on it

Inertia about the three body axes, from the model's mass matrix: **roll
0.001502, pitch 0.002014, yaw 0.000591**. Yaw is the light one by a factor of
three, which is why it ran away first.

Stroke-averaged torque about the centre of mass, by knob, at 75 degrees of
amplitude:

| knob | roll | pitch | yaw |
| --- | ---: | ---: | ---: |
| amplitude asymmetry 0.2 | 7.63 | -1.21 | 0.0000 |
| symmetric phase 0.2 | 0.15 | 0.04 | 0.0004 |
| phase asymmetry 0.3 | 0.51 | 0.03 | -0.164 |

**YAW_PER_PHASE = -0.5666 per radian**, odd in the knob and linear to 3.6% of
full scale from -0.45 to +0.45. The rotation-phase asymmetry is the only knob
in this model that yaws the animal; the amplitude asymmetry makes exactly
none. It is not clean in the other direction -- the same knob makes 0.58 to
1.50 of roll, mostly even in the knob so it does not cancel between sides --
but against a roll authority of 38.2 per unit of amplitude asymmetry the roll
loop answers it with 0.04 of a knob that saturates at 0.45.

Closing the yaw loop took free flight from 320 ms to 1004 ms as first
measured -- but most of that gap was an animal missing its passive yaw
damping and carrying a standing roll offset. With both corrected, the
uncontrolled animal flies 1020-1140 ms and the loop is worth **1.15 to 1.29**
on top, at three stroke amplitudes.

| stroke amplitude | 74.25 | 75.00 | 75.75 |
| --- | ---: | ---: | ---: |
| yaw loop off | 1140 | 1073 | 1020 |
| yaw loop on | 1473 | 1230 | 1250 |

The remaining failure is still the heading, which departs twenty to fifty
milliseconds before the attitude does.

### Flapping counter-torque

Yaw torque against an imposed spin, at zero steering knob, measured before and
after the wings were told about the body's rotation:

| imposed spin | 250 | 500 | 1000 | 2000 deg/s |
| --- | ---: | ---: | ---: | ---: |
| before | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| after | -0.159 | -0.318 | -0.637 | -1.273 |

**-0.0365 per rad/s**, linear to 2%, which gives yaw a time constant of
`YAW_INERTIA / c = 16 ms`. Tens of milliseconds is the right order for a fly.

There is no damping coefficient in the model: the torque follows from adding
`omega x r` at the wing's radius of gyration to the air it meets. The wings
had only ever been given the body's translational velocity.

Yaw authority of the phase knob across its *whole* range, which is wider than
the earlier table:

| phase asymmetry (deg) | 10 | 20 | 26 | 35 | 45 | 55 | 70 | 90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yaw torque | -0.09 | -0.20 | -0.27 | -0.37 | -0.47 | -0.50 | -0.39 | 0.00 |

It peaks at 55 degrees and collapses to zero by 90, where the wing is flipping
mid-stroke rather than at the reversals. `MAX_PHASE` at 45 degrees is
therefore close to the best the knob can do, not past it.

### Sideslip makes yaw

| condition | roll | pitch | yaw |
| --- | ---: | ---: | ---: |
| level, still | 0.604 | 0.009 | 0.0000 |
| rolled 5 degrees | 0.604 | 0.009 | 0.0008 |
| climbing 1500 mm/s | 0.604 | 0.009 | 0.0000 |
| rolled 5 and climbing 1500 | 0.849 | -0.004 | **0.2362** |
| pitched -10 and climbing 1500 | 0.623 | -0.769 | 0.0976 |

Roll alone does nothing and climbing alone does nothing; together they make a
standing yaw torque large enough to consume half the yaw knob's authority. A
rolled animal moving through air has sideslip, and sideslip yaws it.

### Yaw loop tuning, and two things that did not work

Flight duration (attitude inside 30 degrees), median over three stroke
amplitudes, with counter-torque and the roll integral in place.

Yaw bandwidth on its own, against the shared 60 rad/s:

| rad/s | 20 | 25 | 30 | 40 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| knob saturates at | 107.9 deg | 69.0 | 47.9 | 27.0 | 12.0 |
| median flight | 1206 ms | 1154 | 1090 | 1042 | **1218** |

Non-monotonic, and the shared 60 is already best. At 60 the loop is bang-bang
-- 12 degrees of heading error saturates the knob -- and lowering the gain to
stop that makes the heading drift instead (142 degrees of wander at bandwidth
20 against 106 at 60) without buying any flight.

Roll integral, the gain that removes the sideslip yaw torque:

| gain | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| median flight | 1069 | 1103 | 1154 | **1218** | 1360 | 1146 |

1.5 is taken rather than 2.0 because it improves all three amplitudes and has
the better worst case (1097 against 1044); 2.0 has the better median and one
amplitude barely above baseline. The spread between amplitudes is comparable
to the effect throughout, so this is worth about 15%.

### Stroke kinematics, re-measured with yaw damping

Flight duration against stroke shape, three amplitudes, with flapping
counter-torque in the force model:

| amplitude | 74.25 | 75.00 | 75.75 |
| --- | ---: | ---: | ---: |
| plain sinusoid | 1473 ms | 1230 | 1250 |
| sharpness 0.9 | 2467 ms | 2123 | 1879 |
| ratio | 1.68 | 1.73 | 1.50 |

The sharper sweep cuts the within-stroke torque swing by 20% and now flies
50-73% longer for it. Every earlier measurement of this came out negative,
and the penalty shrank each time the sensing improved -- nine-fold at 20 ms of
filter lag, 1.5 at 5 ms, 1.25 under the stroke boxcar -- before changing sign
once the wings were told the body rotates.

`sharpness` remains off by default, so every other number in these documents
is still the sinusoid.

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
