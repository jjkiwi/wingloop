# wingloop

A fly that flies, driven by the connectome that flies it.

This is the expensive half of a question `flyloop` could only answer for
walking. That project puts a connectome readout into NeuroMechFly and gets a
fly that walks toward things. Asked what it would take to make it fly, the
measurement came back in two halves that could not have been more different.

**The brain half is ready and better connected than the walking one.** MaleCNS
contains the ventral nerve cord, so the entire flight motor apparatus is in the
data: the twelve wing steering muscles per side, the power muscles that drive
the oscillation, the halteres. Descending neurons supply **7.6% of the input to
the wing steering motor neurons on average and 28% at most** -- fourteen times
the 0.53% coupling `flyloop` fights with.

**The body half does not exist.** NeuroMechFly's wings are decoration: the
built model has 48 actuators and not one of them moves a wing, the wing geoms
are welded to the thorax and excluded from collision, and the simulation
contains no air at all -- `density`, `viscosity` and `wind` are zero. Nothing
is missing a parameter. The physics has to be written.

**And the readout does not carry over.** DNa02, the neuron behind every
steering result in `flyloop`, supplies 0.21% of the descending drive onto wing
motor neurons. Flight runs on a different set of descending neurons entirely.
So this is a new project rather than a branch.

## What exists so far

Quasi-steady flapping-wing aerodynamics, which is the part nothing else
provides. Three terms, because a flapping wing makes most of its lift from
effects a steady flow does not have:

- **translational** lift and drag, with coefficients that peak near 45 degrees
  instead of stalling near 15 -- the attached leading-edge vortex at Re ~ 150
- **rotational**, the Kramer force, which appears only when the wing rotates
  while translating and is therefore what makes rotation *timing* steer a fly
- **added mass**, the air dragged along with an accelerating wing, which at
  this scale is not negligible

```python
from wingloop.aero.wing import wing_from_mesh
from wingloop.aero.blade_element import hover_check

wing = wing_from_mesh(vertices, n=20, span_axis=2, chord_axis=1)
hover_check(wing, body_mass=1.0265e-3)
# {'lift': 13.7, 'weight': 10.1, 'ratio': 1.36}
```

The wing is measured off NeuroMechFly's own mesh, not assumed: 2.40 mm long,
1.96 mm^2, mean chord 0.816 mm, which is a *Drosophila* wing.

### The test that matters

`test_two_wings_hold_the_animal_up`. A wrong air density, a wrong length unit
or a wrong mass unit all return perfectly plausible floats, and nothing else in
a physics model notices. A real fly hovers, so lift within a small factor of
body weight is the evidence that the unit system is coherent.

It has already earned its place: the first air-density constant written here
was wrong by a factor of a million, and this is the test that fails on it --
checked by putting the bad value back.

It says the model is dimensionally sane. It does **not** say the coefficients
are right; those are relayed and marked as such in `docs/LITERATURE.md`.

## The wings move, and the forces reach the body

`add_wing_hinges` rewrites the MJCF so each wing hangs off three joints --
stroke, deviation and rotation -- and `FlightBody` reads those joints every
step, computes the blade-element forces and writes them in through
`mj_applyFT` at the centre of pressure.

Driven with a harmonic stroke at 218 Hz, the wings lift **1.358 times body
weight**, against 1.364 from the analytic model with no simulator involved.
Two independent paths to the same number is the evidence that the frames and
sign conventions in the body layer do not quietly undo the physics. Side
forces cancel between the mirrored wings to within 1% of body weight.

Three things went wrong on the way there, and each is now a test:

- **The span is local y, not local z.** The wing mesh is longest along its own
  axis 2, but the wing body carries a 90 degree rotation and the mesh sits
  inside it. With the hinge axes as first written, the rotation joint pitched
  the wing about a vertical axis. It still flapped; it produced -0.001 of body
  weight.
- **The coefficient fits are signed.** They run from 0 to 90 degrees, and the
  upstroke always presents a negative pitch angle, where `CL` returns -1.31
  instead of +1.31 -- the wing pushing the animal into the ground for half of
  every cycle.
- **An instantaneous wing flip is an infinite force.** The rotational term
  scales with the rotation rate, so a `sign()` in the stroke profile is not
  merely unrealistic. MuJoCo answered with "Nan, Inf or huge value in QACC".

The wings are driven **kinematically**, not through the position servo. Drag
on one wing peaks near four times body weight, and a servo stiff enough to win
that fight needs a timestep nobody wants; when it loses, the joint rate
disagrees in sign with the command, the drag term flips with it and the
simulation diverges. Prescribed kinematics is the standard arrangement for
validating a blade-element model, and it is honest about what is tested:
whether these kinematics produce these forces in this body, not whether a
muscle could drive them.

## It flies, and then it tips over

`tuck_legs` welds everything but the root, `add_free_base` gives that root
either all six degrees of freedom or a single vertical slide, and the blade
elements now see the animal's own speed through the air as well as their own
sweep.

**On a vertical rail it climbs at the rate the force balance predicts**: 3523
mm/s^2 observed against 3514 predicted, 0.3% apart, 11 mm of altitude in 80 ms.
A rail is a real preparation rather than a dodge -- it asks whether the animal
makes enough force to climb without also asking it to balance, and those turn
out to have different answers.

**Free in six degrees of freedom it lifts off and tumbles.** It gains height,
pitches 32 degrees within 13 ms and is spinning at thousands of rad/s by 80 ms.
That is not a failure of the model: the wing hinge sits about a quarter of a
millimetre ahead of the centre of mass, vertical force there is a nose-down
torque, and nothing in an open-loop stroke opposes it. A fly is passively
unstable in pitch. This is what halteres are for, and what the controller is
for.

### The wings left the physics

Prescribed kinematics and dynamic wing joints cannot both be true. A wing swept
at 1792 rad/s carries Coriolis and centrifugal terms that the mass matrix
couples straight into the body, so the body responds to accelerations the next
command erases -- measured on the rail as -13804 mm/s^2 where the force balance
said +4280. Lightening the wings to break the coupling only makes their own
joints singular; every scaling tried diverged within one step.

So the simulated model has no wing joints at all. The wing pose is composed
from the commanded angles and **checked against MuJoCo's own kinematics on a
hinged model, where it agrees to 3e-16**. What the simulator carries is a rigid
body with a root joint, and the two wings' forces reach it as a single wrench
about the centre of mass.

The moment arm in that wrench is the steering mechanism, and it needed a test
that could tell. A symmetric stroke cancels the lateral offsets, so applying
the force at the hinge instead of the centre of pressure gives the same answer
and nothing notices -- a sabotage check caught the first version of the test
passing either way. Beating one wing at 60% rolls the animal at 7.4 with the
real arm and 1.6 with the hinge, and the threshold now sits between them.

## Closing the loop makes it fly

`HaltereController` reads body attitude and angular rate -- the functional
stand-in for what halteres measure -- and works the two knobs the animal has:
a symmetric shift of the mean stroke angle for pitch, an amplitude difference
between the sides for roll. Both authorities are measured, not assumed:
**-18.9 of pitch torque per radian of bias** and **38.2 of roll per unit of
asymmetry**, with the trim point at -10.7 degrees.

| | open loop | loop closed |
| --- | --- | --- |
| pitch at 40 ms | **+84.7 deg** | inside 13 deg |
| height at 40 ms | falling | climbing |
| attitude over the first 100 ms | gone | pitch < 13 deg, roll < 19 deg |
| after 300 ms | tumbling | **+85 mm of altitude** |

Twenty-two wingbeats of attitude hold against a fly that is past 45 degrees
within nine. It is not yet indefinite: past about 100 ms the attitude degrades
and by 300 ms it has swung through 60-80 degrees, still airborne and still
climbing. Moving the bandwidth and the filter changes that a little and does
not fix it, which points at the stroke rather than the loop -- within one beat
the torque about the centre of mass swings between -17 and +20 while its cycle
mean is near zero, against an authority of 8 in pitch and 17 in roll.

### The torque was being read in the wrong frame

`xfrc_applied` acts at the root body's inertial point, and on this model that
point is the world origin: `FlyBody` is a massless wrapper while the animal's
mass sits a millimetre away. The wrench applied there is correct physics, but
the moment it *reports* is about the origin -- **+0.66 in pitch where the
moment about the centre of mass is -3.49**, different magnitude and opposite
sign. Trimmed on the first number, the controller pushed the wrong way and the
animal went over faster with the loop closed than without it.
`wrench_about_com` is the fix and a test pins both frames.

### Lift belongs to the animal's stroke plane, not to the world

Held vertical regardless of attitude, a pitched fly still gets its whole weight
straight up, and the controller is steering something that cannot be steered.
Tying the normal to the body's own vertical is what makes attitude couple back
into the forces -- and it is why the open-loop fly now loses height as well as
tipping over.

## The connectome steers it

Vision reaches a descending neuron, its right-minus-left difference becomes a
wing amplitude asymmetry, and the asymmetry rolls the animal. Over 100 ms:

| object | command | sideways |
| ---: | ---: | --- |
| 60 deg left | -0.079 | **28 mm to the left** |
| 30 deg left | -0.051 | 16 mm to the left |
| no bearing information | 0.000 | -9.9 mm (the stabiliser's own drift) |
| 30 deg right | +0.058 | 32 mm to the right |
| 60 deg right | +0.096 | 34 mm to the right |

The fly ends up displaced toward the side the object is on, which is fixation
-- the same behaviour the walking version shows. The control is what makes it a
measurement rather than a demo: the identical machinery run with a **flat**
curve gives -9.851 mm at every bearing, exactly what the stabiliser does with
no readout at all. All of the steering comes from the connectome.

### Which neuron, and why not the obvious one

`flyloop` steers on DNa02. That does not transfer -- DNa02 supplies 0.21% of
the descending drive onto the wing steering muscles -- so the candidate had to
be measured.

| | drive onto wing muscles | visual input | bearing tuning |
| --- | ---: | ---: | ---: |
| **DNg02** | **9.07%** | 1.48% | **0.000000** |
| **DNbe001** | 4.06% | 20.4% | **0.186** |
| DNa10 | 3.20% | 29.8% | 0.134 |
| DNge107 | 3.88% | 20.3% | 0.042, inverted |

**DNg02 is the actuator and it is blind.** It is the largest single descending
input to the wing steering muscles by more than a factor of two, about 29 cells
across seven subtypes -- population-coded amplitude control, exactly as the
physiology says -- and its right-minus-left difference is *exactly zero at
every bearing from -90 to +90*. Nothing about where an object is reaches it.

**DNbe001 is the one that sees**, with the steepest tuning of any candidate and
the sign a fixating fly needs. So the command path is vision → DNbe001 → wing
amplitude asymmetry → roll, and the last arrow is the measured 38.2 of roll
torque per unit of asymmetry.

The curve is sampled once and interpolated. That is not a shortcut: one pass
through the rate model costs 0.16 s against a 4.6 ms wingbeat, and a fly's
visual system does not resolve individual wingbeats either.

## Closing the bearing loop, and the two things it exposed

`SteeringController` now takes a `target` and recomputes the bearing from the
animal's own heading every step. The sign convention is not the obvious one and
is tested on its own: this fly faces +x and its **left is +y**, because the
right wing's span points to -y.

**The uncontrolled fly yaws left on its own.** With no steering command at all
the heading wanders more than 10 degrees in 90 ms, and always the same way.
Read as an absolute bearing that alone looks like fixation for objects on the
left and avoidance for objects on the right -- the same confound the
mirror-pair design exists to catch. So every steering number here is the
difference between the command on and the command off, with the drift
cancelled.

**Amplitude asymmetry produces adverse yaw.** Commanding a right turn yaws the
animal *left* for the first 50 ms -- by 46 degrees relative to the no-command
run -- and only reverses once the bank has developed. The wing told to beat
harder carries more drag, and the drag turns the animal the wrong way before
the tilted lift vector turns it the right way.

## Rotation phase is the knob that works

Shifting *when* the wings flip, rather than how far they sweep, acts through
the rotational force term instead of through drag. They are different controls,
not two strengths of the same one:

| knob | torque about the centre of mass |
| --- | --- |
| amplitude asymmetry, +-0.2 | **roll** +-7.8, almost no yaw |
| rotation-phase asymmetry, +-30 deg | **yaw** +-0.32, almost no roll |

And that changes everything about the steering:

| bearings pulled toward straight ahead | 40 ms | 60 ms | 90 ms |
| --- | --- | --- | --- |
| amplitude | 0 of 6 | 0 of 6 | 5 of 6 |
| **rotation phase** | **6 of 6** | **6 of 6** | **6 of 6** |

Phase yaws the animal the right way from 10 ms and never reverses; amplitude
spends 50 ms going the wrong way first. Over 100 ms an object 45 degrees to the
left turns the fly 122 degrees left of where it would otherwise have gone, and
one 45 degrees to the right turns it 113 degrees right, against a flat-curve
control that lands on the same heading whatever the bearing.

This is what real flies do, and the model now says why they do it: amplitude
pays the drag penalty first, rotation timing does not. `steer_mode="amplitude"`
is kept so the comparison stays runnable.

## A hypothesis of ours, refuted

The reasoning was that the within-stroke torque swing is what ends the flight
-- it runs from -28 to +27 about a near-zero mean, against a control authority
of 8 in pitch -- and that a more realistic stroke would shrink it. So
`harmonic_stroke` grew two knobs: `sharpness`, bending the sweep from a
sinusoid toward a triangle, and `deviation`, the out-of-plane motion at twice
the wingbeat that makes a real wingtip trace a figure-of-eight.

The first half of the reasoning is right. The second is wrong, and in the
opposite direction:

| stroke | torque swing | holds attitude until | altitude at 300 ms |
| --- | ---: | ---: | ---: |
| **sinusoid** | 55.2 | **187 ms** | **+79 mm** |
| sharpness 0.6 | 51.1 | 152 ms | +39 mm |
| sharpness 0.9 | 44.1 | 20 ms | -70 mm |
| deviation 15 deg | 48.8 | 60 ms | -61 mm |

Two confounds were ruled out rather than argued away. The **trim** was
re-measured for every stroke shape and moves by less than half a degree
(-10.67 to -11.13). The **lost lift** was restored by raising the amplitude to
79.4 degrees, which recovers the force to the last decimal and the attitude
hold not at all: 19 ms against 20.

So the swing is not what limits the flight, and neither is the standing roll
offset -- the controller never saturates, commanding 0.03 against a limit of
0.45, and adding an integral term to close the offset makes the flight
slightly *shorter*, 179 ms against 187.

## What actually limited it: phase margin

The next suspect was wake memory. A quasi-steady model applies the
steady-state force at every instant, while real circulation takes a couple of
chord lengths to build and carries across a reversal -- so `wingloop.aero.wake`
adds a lag on circulation, in the travelled-distance domain.

It changes the aerodynamics almost not at all: mean lift identical to two
decimals, torque swing 4% different and in the wrong direction. **It changes
the flight enormously** -- 300 ms down to 104. Same forces, added delay, and
the delay sits inside the attitude feedback loop.

That is what identified the limit. Sweeping the sensor filter against loop
bandwidth:

| filter lag | 3 ms | **5 ms** | 10 ms | 20 ms | 30 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| controlled flight | 146 ms | **353 ms** | 204 ms | 187 ms | 145 ms |

At every bandwidth tried, more lag is worse -- and it cannot go to zero,
because within a stroke the torque swings between -28 and +27 about a
near-zero mean and an unfiltered loop chases the beat. The filter trades
stroke noise against phase margin, and 5 ms is where that trade sits.

**With it there, the whole 300 ms run stays inside 11 degrees of pitch and 17
of roll, climbing 148 mm.** The fly flies.

And it corrects the refutation above. Most of the stroke-kinematics penalty
was the filter: measured at 20 ms the sharp stroke held 20 ms against 187, a
factor of nine; at 5 ms it is 171 against 250, a factor of 1.5. The first
measurement had the sign right and the size wrong by six times.

Two bugs were written on the way, and both are now tests. Lagging the *force*
rather than the circulation holds the mid-stroke peak through the reversal --
a peak-hold, not a memory -- and inflated mean lift 8.6-fold. And unpacking
`lift, drag, path = wake.update(...)` shadowed `path`, the unit vector the
force is applied along, a few lines above: a vector times a scalar, broadcast
across three axes, reported as 364 of lift where the wing was making 27.

## The stroke is produced, not written down

Everything above handed the wings a stroke and asked what forces it made. That
validates a force model and assumes the answer to the question the animal
poses, because the frequency, amplitude and waveform were all written down.

*Drosophila* power muscles are **asynchronous**: stretch-activated rather than
driven one spike to one contraction, with the downstroke and upstroke groups
coupled through the thorax so each stretches the other. The nervous system
sets how hard they pull; the resonance sets how fast. `PowerOscillator` is
that, as a resonant second-order system with a delayed stretch-activation
term and the aerodynamic load it works against.

| neural drive | amplitude | frequency |
| ---: | ---: | ---: |
| 0.0 | **0.0 deg** (it stops) | -- |
| 0.5 | 15.7 deg | 217.9 Hz |
| 1.0 | 32.9 deg | 217.3 Hz |
| 2.0 | 62.3 deg | 216.1 Hz |
| 3.0 | 86.2 deg | 215.3 Hz |

**Amplitude rises fivefold; frequency moves 1.2%.** That is the asynchronous
signature, and it is what the model predicts rather than what it was told: a
synchronous muscle would do the opposite. The stiffness *is* calibrated -- it
was chosen to put the resonance at the observed 218 Hz -- so that number is an
input, and `stiffness_for` exists so it cannot be mistaken for a derivation.

Driven this way the fly flies: attitude held past 200 ms and **181 mm of
altitude**, against 148 for the prescribed sine, because the muscle settles on
a larger stroke than the sine was told to make. Rotation is still commanded,
which is the division the animal has -- power muscles asynchronous, steering
muscles synchronous.

The first version had no aerodynamic load, and ran to amplitudes of 2125
degrees. A real wing is damped overwhelmingly by the air it is pushing: the
oscillation grows until the muscle's power equals the air's, and without the
air there is nothing to stop it.

## The drive comes from the connectome too

The oscillator above still took its drive as a number someone typed. The
muscles it stands for have motor neurons in this connectome -- 24 of them,
DLMn and DVMn -- so the number can be measured instead.

Pushing each descending family forward and recording *both* motor pools, all
24 power motor neurons and all 32 steering ones, sorts them into three kinds:

| family | power MNs | steering MNs | ratio | visual input |
| --- | ---: | ---: | ---: | ---: |
| DNg02 | 0.4366 | 0.0410 | 10.6x | 1.48% |
| **DNa08** | 0.2676 | 0.0126 | **21.3x** | 0.71% |
| **DNp31** | 0.1403 | 0.0071 | **19.7x** | 29.51% |
| DNbe001 | 0.0319 | 0.0151 | 2.1x | 20.40% |
| DNa02 | 0.0000 | 0.0004 | 0.0x | 2.82% |

Power-selective by an order of magnitude, shared, and steering-only. DNbe001 --
the neuron this project steers with -- moves both pools alike and is no kind of
throttle; DNa02, which carries every steering result in `flyloop`, does not
reach the power muscles at all. `PowerReadout` takes DNa08 and DNp31: **0.374
against 0.0185, a twentyfold separation**, turned into oscillator drive by one
calibration that the class labels as a calibration.

So the whole path runs from a command in the brain to altitude in a body. On a
vertical rail, 200 ms:

| command | stroke amplitude | height at 200 ms |
| ---: | ---: | ---: |
| 0.25 | 24.6 deg | -169 mm |
| 0.50 | 48.7 deg | -93 mm |
| 0.75 | 63.5 deg | **+1.8 mm** |
| 1.00 | 71.2 deg | +100 mm |

**The fly hovers at three-quarter throttle** -- a consequence of the connectome
curve and the calibration, not a target. At full command in free flight it
holds attitude for 227.5 ms and climbs 151 mm.

### Two claims this cost

Writing it up produced two statements that measurement then took back, and
both are in the code where they were wrong rather than deleted.

**The separation is 20x, not 82x.** The first pass recorded four steering
types instead of all sixteen, and reported 82x with DNa08 and DNg02 coming out
*infinitely* selective. Against the whole steering apparatus neither is. The
ordering survived; the number did not. A selectivity measured against the
muscles a command happens to miss is not a measurement of selectivity.

**The throttle is not blind.** It was nearly written up as deliberately blind:
the power motor neurons take 0.00% of their input directly from visual
neurons. They do -- and so do the steering motor neurons, exactly 0.00%, so
the fact separates nothing. Worse for the story, a synapse up this command is
the *more* visual of the two: DNp31 at 29.51% against DNbe001's 20.40%. Vision
can open this throttle. The scalar command does not use that, and now says so.

## Closing the throttle loop

The command above was still a number someone chose, held for the whole flight.
So the animal could stay upright while it sank -- and does: **even at the
measured hover command it loses 64 mm over 600 ms**, because it spends the
first wingbeats falling while the muscle spins up and a constant never makes
that back. A tenth below and it drops 412 mm; a tenth above and it climbs 324.

`Throttle` closes it, through the channel the connectome supplies -- height
error to flight command, command to power motor neuron activity, activity to
drive, drive to stroke amplitude. The law is the attitude loop's shape and its
gains come from the same place, a measured authority rather than a knob:

```
command = hover - (w^2 (z - z*) + 2 zeta w zdot) / CLIMB_PER_COMMAND
```

Both constants are swept, not assumed: **21038 mm/s^2 of vertical acceleration
per unit command**, linear from 0.5 to 1.0, crossing zero at a **hover command
of 0.695**. Bandwidth is 20 rad/s -- half the attitude loop's, because an
animal cannot chase a height faster than it can hold the attitude it climbs on
-- and critically damped, because an altitude loop that overshoots downward has
a floor to hit.

| | held 0.60 | held 0.695 | held 0.80 | **closed** |
| --- | ---: | ---: | ---: | ---: |
| height after 600 ms | -412 mm | -64 mm | +324 mm | **-0.03 mm** |

Asked for 30 mm from a standing start it arrives in 219 ms and overshoots by
0.03. It dips 2.2 mm while the muscle starts and never again.

### What it leans on, and what it does not buy

There is no integral term, so the loop is exactly as accurate as the hover
trim: a trim wrong by `d` settles `d * 21038 / 20^2` from the target. At 0.65
against a true 0.695 that predicts -2.37 mm and the body gives **-2.40**; at
0.75 it predicts +2.89 and gives **+2.86**. That is asserted as a number in the
tests, so the constant cannot rot quietly behind a loop that still looks like
it works.

And it does not buy flight time. Free, with both loops closed, attitude holds
216.7 ms against 227.5 at a held full command -- the limit is still phase
margin in the attitude loop, as it has been since that was identified, and
closing a loop on height was never going to move it. What changes is that at
200 ms the animal is at 41.7 mm on its way to a commanded 50, instead of 95 on
its way to wherever.

One more claim measurement took back. The filter on the sensed climb rate was
written up as *not optional* -- by analogy with the attitude loop, where it is
the single most important number. Swept, it is nearly optional at the bottom:
unfiltered the loop still holds a tenth of a millimetre and puts 1.3% of stroke
ripple into the command. The end that bites is the top, at 50 ms, where lag
eats the phase margin and the error jumps to 0.91 mm. Why heave tolerates what
pitch does not is not established; the obvious guess -- that the body's mass
integrates the ripple away -- is wrong, since heave velocity ripples *more*
against its own mean than pitch rate does, 3.5 against 1.7.

## What does not exist yet


## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Nothing here needs MuJoCo or a connectome; the wing mesh is checked in as
vertices so the geometry test runs anywhere.
