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

## flylab: the whole animal in one program

`wingloop.lab` puts the pieces together behind one command, and borrows every
piece someone else has already built properly:

| part | what it is | from |
| --- | --- | --- |
| flight | NeuroMechFly with blade-element wings, haltere, throttle and steering loops | this repository |
| brain | the descending and motor neurons the MaleCNS connectome wires to the wings | MaleCNS v1.0 ([Berg et al. 2025](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2), [neuPrint](https://male-cns.janelia.org)), via `connectome-interpreter` |
| vision | the pretrained, connectome-constrained optic lobe: 45,669 neurons, 65 cell types, 721 columns | [`flyvis`](https://github.com/TuragaLab/flyvis), Lappalainen et al., *Nature* 2024 |
| learning | Kenyon cells and the dopamine rule that depresses their output synapses | the fly's rule, as used in `flyloop` |
| drone | a Crazyflie 2.x quadrotor flown on the fly's three control laws | parameters from `gym-pybullet-drones` |

```bash
pip install -e ".[body,lab]"
flyvis download-pretrained --skip_large_files         # once: the optic lobe weights
flylab fly --seconds 0.5 --height 5 --target 50 50   # the fly, and its flight neurons
flylab see                                           # object recognition, with controls
flylab teach --reward bar=1 sphere=-1                # training flights with dopamine
flylab drone --memory memory.npz --object bar@120:5 sphere@-100:5
flylab demo                                          # all of it, one HTML report
```

**What each part does, measured:**

- **Recognition.** A linear readout of the optic lobe tells a sphere, a bar
  and a box apart in 77% of flights past them, against 59% from the
  photoreceptors alone and 35% with the labels shuffled (chance 33%). The bar
  is recognised 98% of the time; the sphere and the box are what this eye
  confuses -- a box is called a sphere 40% of the time.
- **Learning.** Dopamine paired with a bar and punishment paired with a
  sphere teach the mushroom body to choose correctly in 92-95% of new
  flights, against 38-64% for unpaired dopamine. In the drone, from its own
  hovering fixations, it reaches 80% on new flights after twenty training
  flights -- once its input is taken from the middle of the eye, where a
  fixated object is.
- **The autonomous mission.** Trained on forty such flights with the bar
  rewarded and the sphere punished, the drone turns a full circle, fixates
  each object, values the bar at +0.27 to +0.34 and the sphere at -0.19 to
  -0.26, and flies to the bar -- stopping 1.9 m short of it, never closer to
  the sphere than where it started, in both object layouts tried. With the
  dopamine reversed, the same drone in the same world flies to the sphere.
- **The drone** uses the fly's attitude law (gain = inertia times bandwidth
  squared over authority, critically damped), the throttle law for height,
  visual steering on the bearing the optic lobe reports, and the mushroom
  body to decide what to approach and what to avoid.

**What this is not.** The brain recorded in flight is the connectome's
steady-state readout of the flight neurons, not a spiking whole-brain model
running alongside the body. The optic lobe stops where the fly's
object-selective neurons begin, so recognition is a trained readout, not a
model of the lobula. The mushroom body's Kenyon cells sample their inputs at
random rather than through the measured visual-projection wiring. And the
drone reads its speed from its own state, as a drone with a flow sensor
would, rather than from the optic flow its eye computes.

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

## A hypothesis of ours, refuted -- and then reinstated

*(Read this section as history. The measurement it reports stood for most of
this project's life and was wrong the whole time; `sharpness` is 0.9 by
default now. What went wrong and how it was found is further down, under
**the oldest refutation in this README turns out to be wrong**.)*

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

That is what identified the limit, and the direction has survived everything
since: inside this loop, delay costs flight. It cannot go to zero either --
within a stroke the torque swings between -28 and +27 about a near-zero mean,
and an unfiltered loop chases the beat.

**The size of the prize, though, was wrong, and the way it was wrong is worth
keeping.** The sweep that chose the filter reported 353 ms at 5 ms of lag
against 187 at 20, and that 353 went into this README as what the filter was
worth. Swept finely, it was a spike:

| tau (ms) | 4.0 | 4.5 | 4.8 | **5.0** | **5.2** | 5.5 | 6.0 | 7.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| controlled flight | 193 | 214 | 249 | **353** | **379** | 256 | 236 | 221 |

Change the stroke amplitude by 1% -- which has nothing to do with the sensing
-- and the peak moved to a different time constant: 4.8 ms at an amplitude of
74.25 degrees, 5.2 at 75.00, 5.5 at 75.75. There was always a spike near
350-380 ms somewhere and where it landed was a coincidence.

**And the spike turned out to be a missing physical term.** Much further down
this README the wings are finally told that the body rotates. Re-measured with
that in, the same sweep is smooth, and the same shape at every amplitude:

| tau (ms) | 3.0 | 4.8 | 5.0 | 5.2 | **6.0** | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| controlled flight | 514 | 993 | 997 | 999 | **1002** | 928 | 619 |

One broad hump with its optimum at 5-6 ms, which is where the default had been
sitting since the beginning: **chosen for a bad reason and right anyway.** The
chaotic landscape that made a single setting look twice as good as its
neighbours was the animal having no yaw damping to fall back on.

## Spending the phase margin on gain instead of lag

A first-order filter rejects the stroke beat by lagging everything, and the
lag is inside the loop. That is the whole trade. But the disturbance is not
broadband: **75% of the power in the sensed pitch rate is at the wingbeat**,
with most of the rest at its harmonics. Something that rejects one frequency
and its harmonics specifically should cost far less phase.

A running mean over exactly one wingbeat does that. Its nulls are exact, at
the stroke frequency and every harmonic -- 56 dB down at 218 Hz for the
229-sample window this timestep gives, against 26 dB for the filter it
replaces -- and its group delay is half a period, **2.29 ms against 5**.

At first it looked worse: 237 ms against 353. That comparison was against the
spike. Against the trend the two are level, and the difference shows up where
it should, in the bandwidth the loop can afford. Three stroke amplitudes, so a
coincidence would show:

| bandwidth (rad/s) | 30 | 40 | 50 | 55 | **60** | 70 | 80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first-order, 6 ms | 160 | **236** | 209 | 183 | 163 | 146 | 134 |
| stroke boxcar | 162 | 237 | 277 | 295 | **320** | 260 | 154 |

Identical where the old default sat, and separating above it. The first-order
filter peaks at 40 rad/s and falls away; the boxcar keeps climbing to 60.
**236 ms to 320, and the usable bandwidth from 40 to 60** -- the phase margin,
spent on loop gain instead of on lag. Both are now the defaults.

(This one survived the yaw-damping correction, in direction if not in size.
Re-measured with the wings feeling the body's rotation: at bandwidth 40 the
two are level, 1030 ms against 986, and at 60 the boxcar is ahead 1073 to 775.
A factor of 1.38 where this section measured 1.36.)

It shows in the attitude, not just in how long the flight lasts. The 300 ms
run that used to stay inside 11 degrees of pitch and 17 of roll while climbing
148 mm now stays inside **3.2 degrees of pitch and 9.9 of roll, climbing
154 mm**. Every flight number quoted before this section was measured through
the first-order filter at bandwidth 40; the ones after it were re-measured.

And this corrects the refutation above a second time. The stroke-kinematics
penalty has shrunk every time the sensing improved, always for the same
reason: at 20 ms of lag the sharp stroke held 20 ms against 187, a factor of
nine; at 5 ms, 171 against 250, a factor of 1.5; under the boxcar at bandwidth
60, 254 against 320, a factor of 1.25. The sign has survived all three and the
size has not, so what the tests assert now is the sign.

### A rig that was wrong, and what it cost to find out

The clean way to do all of this would be to measure the loop's phase margin
directly rather than infer it from flight times, which needs a preparation
that pitches and nothing else. `add_free_base(dofs="pitch")` is that tether,
and the first version of it pinned the animal at the model origin -- **1.07 mm
below the centre of mass and 0.30 ahead of it**. The net aerodynamic force
then no longer accelerates the animal; it torques it about the pin, with an
arm no trim can reach, and the tethered fly simply spins: a continuous
rotation at 80-107 Hz, at every filter setting and every gain. That looks
exactly like a control failure and is a rig failure.

Pinned through the centre of mass it stops spinning -- and still held only
30-90 ms where the free animal held 350. That was written up here as a tether
being a *harder* problem than free flight, as if constraining the translation
removed something stabilising. **It removed nothing: the tether had no
haltere.** The angular-rate reading handled a free joint and returned zeros
for anything else, so on the tether the loop ran on its proportional term
alone and oscillated to +-78 degrees inside 300 ms. Reading the hinge rate,
the same controller on the same rig stays inside four degrees, and the tether
became the clean preparation it was built to be.

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

Driven this way the fly flies: attitude held for the whole 300 ms inside 4.1
degrees of pitch and 10.8 of roll, and **240 mm of altitude**, against 154 for
the prescribed sine, because the muscle settles on a larger stroke than the
sine was told to make. Rotation is still commanded,
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
holds attitude for 338 ms and climbs 239 mm.

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
276 ms against 338 at a held full command -- the limit is still phase margin
in the attitude loop, and closing a loop on height was never going to move it.
What changes is where the animal ends up: at 300 ms it is at 45 mm, on its way
to the 50 it was asked for, instead of 239 on its way to wherever.

One more claim measurement took back. The filter on the sensed climb rate was
written up as *not optional* -- by analogy with the attitude loop, where it is
the single most important number. Swept, it is nearly optional at the bottom:
unfiltered the loop still holds a tenth of a millimetre and puts 1.3% of stroke
ripple into the command. The end that bites is the top, at 50 ms, where lag
eats the phase margin and the error jumps to 0.91 mm. Why heave tolerates what
pitch does not is not established; the obvious guess -- that the body's mass
integrates the ripple away -- is wrong, since heave velocity ripples *more*
against its own mean than pitch rate does, 3.5 against 1.7.

## The third loop, which was the one that mattered

Yaw was left uncontrolled through all of the above, on the reasoning that a
symmetric stroke does not produce much of it. That is true and it is beside
the point. **Yaw is the light axis** -- inertia 0.000591 against 0.002014 in
pitch and 0.001502 in roll -- so the little it gets is plenty. Measured on an
ordinary flight, the heading reached +40 degrees by 100 ms and -86 by 300, at
up to 1587 deg/s, while pitch and roll stayed inside ten. *(Wrong axis: that was the inertia about the frame origin. About the
centre of mass roll is the light one, and yaw is two thirds of pitch. See
"Two frame errors".)*

The knob is the left-right rotation-phase asymmetry, and it is the only one
that yaws this animal at all:

| knob | roll | pitch | yaw |
| --- | ---: | ---: | ---: |
| amplitude asymmetry 0.2 | 7.63 | -1.21 | **0.0000** |
| symmetric phase 0.2 | 0.15 | 0.04 | 0.0004 |
| **phase asymmetry 0.3** | 0.51 | 0.03 | **-0.164** |

0.5666 per radian, odd in the knob and linear to 3.6% of full scale. It is the
same division the steering work found from the other end -- rotation phase
yaws, amplitude rolls -- and the gains come from it and the yaw inertia the
same way the pitch and roll gains do.

**Closing it took the flight from 320 ms to 1004**, with the heading inside
ten degrees at 200, 400, 600 and 800 ms -- and this section originally said it
tripled the flight and that everything above it had been second-order beside
it.

**That factor was mostly two other things, and the next section is about
finding them.** The wings had never been told the body rotates, so the
uncontrolled animal had no yaw damping to fall back on; and a standing roll
offset was making yaw through sideslip that nothing was answering. With both
fixed, the *uncontrolled* animal already flies 1020-1140 ms and the yaw loop
is worth **1.15 to 1.29** on top -- at every stroke amplitude, and nothing
like a tripling. A loop measured against a broken plant flatters itself.

With yaw held the sensing and bandwidth choices also matter much less than
they did, which is what the tests for them now say by switching the yaw loop
off explicitly.

### Two things it broke, and one it found

**Steering had to change.** With the yaw loop holding a heading, adding a
steering command to the phase knob no longer turns the animal -- the
stabiliser simply undoes it, measured at -14.6 degrees by 90 ms and back to
-1.3 by 250. So the command moves the *setpoint* instead: bearing error into
turn rate, which is the fixation law, and the yaw loop flies it. The setpoint
is not allowed to run more than a bounded lead ahead of the animal, because
without that it ramps at the commanded rate whatever the body does, saturates
the phase knob, and the roll that knob cross-couples takes the flight down at
138 ms. Turns are now physiological rather than a spin: a left object turns
+2.4 degrees in 100 ms and a right one -11.4, against a -0.9 baseline, where
the old code reported over 100 and that number was the spin.

**And the phase knob meant two different things.** `PowerStroke` added
`phase_asymmetry` to a saturating velocity proxy where `harmonic_stroke`
rotates it inside the cosine -- the same name for a different physical
quantity, **opposite in sign and thirty times larger**. Nothing had noticed,
because nothing before the yaw loop read the knob's sign. A loop with gains
measured on one generator was positive feedback on the other: the
muscle-driven flight went from 338 ms to 54. `PowerStroke` now recovers the
oscillator's cycle phase properly, from the envelope
`sqrt(phi^2 + (phi_dot/w)^2)`, and both generators agree in sign and to within
a factor of two in size.

The flight no longer ends in a tumble. It ends in a **flat spin**: past about
950 ms the heading departs while pitch and roll are still inside thirty, and
the phase knob saturates at 45 degrees, which is only 12 degrees of heading
error. That is the next thing.

*(A measuring note. The 9000 degrees of spin this first appeared to show was
an artifact: MuJoCo resets its clock when it diverges, so post-divergence
samples carry small times and pass a `t <= held` mask while the recorded
heading keeps accumulating. The trace has to be cut by index, at the point
time runs backwards. This project has now been bitten by that clock reset
twice.)*

## The flat spin, and the term that was missing

The yaw loop held a heading for a second and then lost it to a flat spin --
heading running away while pitch and roll were still inside thirty degrees.
The obvious suspect was authority: the phase knob saturates at 45 degrees,
which the loop reaches at only 12 degrees of heading error.

It was not that. Swept past saturation, the knob's yaw authority is still
climbing at 45 degrees and peaks at 55, so the limit sits near the best the
knob can do:

| phase asymmetry | 10 | 20 | 26 | 35 | **45** | 55 | 70 | 90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yaw torque | -0.09 | -0.20 | -0.27 | -0.37 | **-0.47** | -0.50 | -0.39 | 0.00 |

And 45 degrees of knob is 43142 deg/s^2, which would arrest the 1246 deg/s
spin in 29 milliseconds. Authority was never the problem.

**The problem was that the animal had no yaw damping whatever.** Imposing a
spin and reading the torque:

| imposed spin | 250 | 500 | 1000 | 2000 deg/s |
| --- | ---: | ---: | ---: | ---: |
| yaw torque, before | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yaw torque, after | -0.159 | -0.318 | -0.637 | -1.273 |

The wings were being told the body's *translational* velocity and nothing
else. A real wing sits out at the radius of gyration, so when the body rotates
that point moves at `omega x r` as well: one wing advances into the air, the
other retreats, the forces stop balancing and the difference opposes the spin.
That is **flapping counter-torque**, and it is the dominant passive damping on
insect yaw. There is no damping coefficient in the fix -- the torque falls out
of putting the wing's own position into the oncoming air.

It is linear in the spin rate to 2%, giving yaw a time constant of
`I/c = 16 ms`. Yaw was the only axis where the omission was fatal, because it
is the only one with no restoring term of its own: pitch and roll have gravity
and the stroke plane, and yaw had nothing at all.

Open loop it takes the flight from 320 ms to 431. With the yaw loop closed it
was worth much less -- 1004 to 1069 -- which said the spin was not what ended
the flight any more, and sent the search somewhere else.

### What the loop was really fighting

Through the whole stable phase the yaw knob sat at a **standing -25 degrees**,
over half its authority, just to hold the heading. Any transient then
saturated it. Tracking that back:

| condition | roll torque | pitch | yaw |
| --- | ---: | ---: | ---: |
| level, still | 0.604 | 0.009 | 0.0000 |
| rolled 5 degrees | 0.604 | 0.009 | 0.0008 |
| climbing 1500 mm/s | 0.604 | 0.009 | 0.0000 |
| **rolled 5 and climbing** | 0.849 | -0.004 | **0.2362** |

Neither alone does anything; together they make a large standing yaw torque.
A rolled animal moving through the air has sideslip, and sideslip yaws it.

And this animal flies permanently rolled, because the roll loop is
proportional-derivative against a constant disturbance -- the pitch trim
cross-couples +0.6 into roll -- so it settles several degrees off level and
stays there. That offset was known and documented as harmless: *closing it
changes nothing, 179 ms against 187*. True when it was written, when a tumble
ended the flight at 187 ms and a few degrees of roll had nothing to do with
it. It stopped being true when the yaw loop pushed flight past a second, and
nobody re-measured it.

Turning the roll integral on takes mean roll from 7.94 degrees to 0.17 and the
standing yaw deflection from -22.5 degrees to +4.3. Medians over three stroke
amplitudes: **1069 ms at gain 0, 1154 at 1.0, 1218 at 1.5**, 1360 at 2.0,
1146 at 3.0. Gain 1.5 is taken rather than 2.0 because it improves all three
amplitudes where 2.0 leaves one at 1044, barely above baseline.

### Two more attempts: one refuted, one confirmed backwards

With the standing load gone the loop stops holding a bias and starts
**limit-cycling** instead: the heading swings plus or minus thirteen degrees
while the knob bangs between its limits, saturated for whole stretches. That
is because twelve degrees of heading error already asks for the entire knob --
yaw carries a third of pitch's inertia, so the shared bandwidth of 60 rad/s
puts the gain very high. *(The inertia ratio was the frame error; the saturation is real, and is
why yaw now has its own bandwidth.)*

Giving yaw its own, lower bandwidth widens the linear range exactly as the
arithmetic says, and buys nothing:

| yaw bandwidth | 20 | 25 | 30 | 40 | **60** |
| --- | ---: | ---: | ---: | ---: | ---: |
| knob saturates at | 107.9 deg | 69.0 | 47.9 | 27.0 | 12.0 |
| median flight | 1206 ms | 1154 | 1090 | 1042 | **1218** |

Non-monotonic, and the shared 60 is already the best of them. Below it the
loop stops saturating and starts *drifting* -- the heading wanders to 142
degrees at bandwidth 20 against 106 at 60 -- and flies no longer for it.
Saturation is real and is not what limits this.

The other attempt went the other way. The yaw knob drags roll with it, and
lopsidedly: +1.51 of roll torque at +45 degrees against +0.33 at -45, so the
even part does not cancel between the sides. In the roll loop's units that is
0.039 of amplitude asymmetry against a limit of 0.45, one tenth of the
authority available, and cancelling it looked pointless. It is not: feeding
the measured curve forward helps at every amplitude tried. How much is beyond
this to say -- 1218 to 1250 on the median with the accurate curve, and 1218 to
1377 with an earlier version whose fit was wrong at small deflections. **Both
are positive at all three amplitudes and they disagree by more than the
effect**, so the sign is the claim and the size is not.

(That wrong fit is worth its own line. An unconstrained quadratic through the
measured points predicted +0.14 of roll torque at zero deflection, where there
is none by construction. Forcing it through zero left 12.9% of full scale, a
cubic 13.2%, and only a quartic reached 2.9% -- at which point the polynomial
is carrying the shape rather than describing it. The curve is now stored as
the measurements and interpolated.)

### And the oldest refutation in this README turns out to be wrong

Near the top there is a section called *A hypothesis of ours, refuted*: the
reasoning that a sharper, more realistic sweep would cut the within-stroke
torque swing and buy flight, and the measurement that it cut the swing by 20%
and made the flight nine times worse.

That penalty shrank every time the loop's sensing improved -- a factor of nine
at 20 ms of filter lag, 1.5 at 5 ms, 1.25 under the stroke boxcar -- and each
time this README recorded the shrinking and kept the sign. With the missing
yaw damping restored the sign goes too:

| stroke amplitude | 74.25 | 75.00 | 75.75 |
| --- | ---: | ---: | ---: |
| plain sinusoid | 1473 ms | 1230 | 1250 |
| **sharpness 0.9** | **2467 ms** | **2123** | **1879** |
| ratio | 1.68 | 1.73 | 1.50 |

**The sharper stroke flies half again to three-quarters longer.** The original
hypothesis was right and every measurement against it was taken on an animal
that could not bank the reduced swing, because what was ending its flight was
not the swing.

**`sharpness` is now 0.9 by default**, and the sinusoid is the special case.
That is not free, and the cost is the honest part: every number in this
repository above this line was measured on the sinusoid, so each one is a
measurement of an animal flying a stroke it no longer flies. The tests that
compare stroke shapes now ask for the sinusoid explicitly.

What it costs in the air is lift -- 12.19 against 13.66, from 1.36 of body
weight down to 1.21 -- and that shows up as climb: the 300 ms run that rose
148 mm on the sinusoid rises **65 mm** now. The stroke buys attitude with
lift. The amplitude that would
restore it exactly is 79.41 degrees, and that is deliberately *not* taken: the
table above was measured at 75 degrees, and raising the amplitude is a
separate change with its own measurements owing. The pitch trim moves with the
shape and barely: the torque about the centre of mass vanishes at -10.95
degrees of bias against -10.67 for the sinusoid, so `TRIM_BIAS` is a quarter
of a degree out and stays.

### What the new default moved

Turning `sharpness` on re-opened almost every tuning question in this file,
which is the price of it and worth listing rather than burying:

| | sinusoid | sharpness 0.9 |
| --- | ---: | ---: |
| flight, yaw loop closed | 1230 ms | **2123 ms** |
| climb over 300 ms | 148 mm | 65 mm |
| cycle-mean lift | 13.66 | 12.19 |
| best first-order filter lag | 5-6 ms | **3.5 ms** |
| boxcar against that filter, bandwidth 40 | 237 / 236 | **1162 / 366** |
| bearing loop needs | 150 ms | 250 ms |

The filter optimum moving is the interesting one. It is **not** another spike:
the first one was discredited because a 1% change in stroke amplitude moved
it, and this peak sits at 3.5 ms for every amplitude tried, moving only with
`sharpness` itself -- a stroke parameter, which is a dependence a real
optimum should have. A sharper sweep carries its torque differently and wants
less filtering, so the lag costs more; 20 ms of it is now worth 21 ms of
flight.

The boxcar's advantage grew the same way and for the same reason. A sharper
sweep puts more of its disturbance at the wingbeat and its harmonics, which is
exactly what a one-period boxcar nulls and a first-order filter can only
smear.

### The amplitude that buys the lift back

The sharper stroke gave away lift, and `79.41 degrees` is the amplitude that
solves for getting it back: cycle-mean 13.663 against the sinusoid's 13.664,
1.358 of body weight. The climb returns with it -- **148.2 mm over 300 ms**,
against 76.6 at 75 degrees, and 148 for the sinusoid this whole chain started
from.

It costs attitude, and the cost is not small. With each amplitude flown on its
own measured authorities:

| bandwidth | 75.00 deg | 79.41 deg |
| --- | ---: | ---: |
| 45 | 1526 ms / 88.1 mm | 960 ms / 151.9 mm |
| 55 | 1921 / 81.7 | 1193 / 149.1 |
| **60** | **2011 / 76.6** | **1244 / 148.2** |
| 70 | 2620 / 68.2 | 1523 / 130.5 |

A wider sweep makes a wider within-stroke torque swing, and the loop has more
to answer. The choice between these two amplitudes is a straight choice
between climbing and staying upright.

### A calibration error, and what it was hiding

Every control authority in `wingloop.body.control` is a property of the stroke
being flown, and the commit that turned `sharpness` on changed the stroke
while re-measuring only the lift and the trim. At the same 75 degrees, purely
from sharpening the sweep:

| constant | sinusoid (stored) | sharpness 0.9 | error |
| --- | ---: | ---: | ---: |
| `PITCH_PER_BIAS` | -18.85 | -16.04 | -15% |
| `ROLL_PER_ASYMMETRY` | 38.2 | 32.51 | -15% |
| `YAW_PER_PHASE` | -0.5666 | -0.8419 | **+49%** |

Gains are `inertia * bandwidth^2 / authority`, so overstating the authority
understates the gain: **the loop was quietly running below its nominal
bandwidth**, and it flew *longer* that way -- 2123 ms against 2011 with the
gains correct. Which means "bandwidth 60" was never bandwidth 60, and the
sweep that picked it was measuring a loop that was not the one described.

All of them are now measured at the default stroke, and the tests that check
them read the stroke off the controller instead of writing 75 degrees down
again -- which is why they passed through the error in the first place.

### The muscle-driven stroke flies on the wrong authorities, and should

The connectome stack -- throttle, oscillator, `PowerStroke` -- flies a
different stroke from the one every authority above was measured on: it
settles at 75.6 degrees with the sinusoid's shape, and its own static
authorities differ by +21% in pitch, +19% in roll and -22% in yaw. That is the
same class of error the previous section fixed, so it was measured the same
way. Deriving the muscle path's gains from its own numbers **flies shorter**:
635 ms against 721 with the stored feedforward, 644 with its own curve, over
three drive levels.

That is the second time. Correcting the harmonic path's authorities two
sections up also flew shorter -- 2011 ms against 2123 -- and both times the
error that happened to lower the pitch and roll gains was the one that flew
longer. The reading this repository adopts is that a cycle-averaged authority
measured at trim with the body held still is a slightly wrong plant for a
loop whose body is rotating, translating and re-settling its stroke, and that
`inertia * bandwidth^2 / authority` is the approximation that is off rather
than the stroke it was measured on. One authority set serves both generators,
the errors are written down, and the fix -- identifying the authority in
closed loop -- is a different project.

### The authorities, identified in flight

Twice now a "corrected" static authority set has flown shorter than a stale
one, and both times the reading was that a cycle-averaged torque measured
with the body held still is a slightly wrong plant for a loop in flight. So
the plant was identified in flight: a small sinusoid on each knob during a
real flight, and the ratio of torque about the centre of mass to the total
knob at that frequency.

**At the loop's own crossover this cannot work** -- the loop drives its
command to cancel the injection, and the ratio is two near-zero numbers.
Measured: 0.05-0.24 of static with scrambled phases. It is kept as a
negative result: at crossover the quantity to read is the loop gain, not the
plant, and the tether below reads it.

Well above crossover, where the loop cannot answer, the plant is excited
directly:

| knob | 100 rad/s | 150 | 200 | **300** | static |
| --- | ---: | ---: | ---: | ---: | ---: |
| pitch, per rad of bias | -8.97 | -11.67 | -11.21 | **-17.42** | -17.40 |
| roll, per unit asymmetry | 19.9 | 25.4 | 28.7 | **35.7** | 34.3 |

Pitch and roll come back with the phase of a pure gain and **recover the
static value exactly at 300 rad/s**. So the static numbers are right as the
knob-to-torque gain; what they miss is whatever reduces the effective
authority to 50-85% at 100-200 rad/s. Two candidates were measured and both
are wrong: pitch and roll have essentially no aerodynamic damping (time
constants near two seconds), and torque against body angle at rest is
exactly zero, because tilting a still body rotates the whole picture rigidly.
What remains has to exist only in flight -- the relative wind of a 1500 mm/s
climb is the candidate -- and quantifying it needs the identification
extended to record velocity. (It was the climb; see below. The damping
numbers here were also wrong, for a reason that comes up below too.)

Yaw is the exception on damping: 16 ms, the loop's own timescale, and the
gain formula derives the rate gain as though the plant had none. Putting the
damping into the derivation halves the yaw rate gain, and flying that is
**4% shorter at all three amplitudes** -- 1371/1281/1205 ms against
1461/1341/1223. Not shipped. That makes three correct derivations this
session that flew worse than the numbers they were correcting, and the common
factor is a loop that spends much of every flight with a knob saturated,
where the linear design those derivations assume does not describe it. Yaw
also does not identify cleanly at any frequency tried, and that is left open.

### The bandwidth, re-swept, and kept

*(Every number in this section is in units that turned out to be wrong: the
gains came from inertias about the wrong point, and "60" was three different
loops. See the next section.)*

With the authorities right the loop had never actually run at 60 rad/s, so
the sweep that chose it was measuring something else. Re-swept at three
amplitudes one percent apart (flight in ms / worst excursion in the first
second):

| rad/s | 78.62 deg | 79.41 deg | 80.20 deg | median |
| --- | ---: | ---: | ---: | ---: |
| **60** | 1461 / 13 | 1341 / 14 | 1223 / 13 | 1341 |
| 80 | 1839 / 15 | 1073 / 19 | 1591 / 13 | 1591 |
| 90 | 1688 / 13 | 1848 / 9 | 1022 / 28 | 1688 |
| 100 | 2092 / 13 | 1759 / 24 | 1349 / 23 | 1759 |
| 110 | 145 / 30 | 1086 / 20 | 509 / 30 | 509 |
| 120 | 186 / 30 | 520 / 30 | 107 / 30 | 186 |

The median rises all the way to 100, and 100 beats 60 at every one of the
three amplitudes -- which is the test every other default in this file was
moved on. **It is not moved.** The cliff is at 110, where two of three
amplitudes collapse, so 100 sits inside 10% of it; and its excursions are
twice those at 60, so it flies longer by riding the limit rather than by
being calm. Sixty is the only row whose three amplitudes agree inside 240 ms
-- everything above it spreads by 750-830 inside a 1% amplitude change, which
is the same chaotic terrain that produced the filter-lag spike this README had
to withdraw. A default that a 6% stroke change could push over a cliff is not
a default.

### Where it ended, before the frame errors

**1069 ms to about 1250, and the failure mode is unchanged.** The heading
still departs before the attitude does, by twenty to fifty milliseconds, in
every configuration tried. The flat spin was delayed by finding a missing
physical term and a standing disturbance; it was not eliminated, and two of
the three fixes are worth less than the spread between stroke amplitudes.

What would settle it is not more tuning. At this point the knobs interact,
and the differences between settings are smaller than the differences between
stroke amplitudes -- which is exactly the situation that produced the
filter-lag spike this README already had to withdraw. The next real step is a
measurement that does not go through flight duration at all.

## Two frame errors, and the loop read off the rig

The measurement that does not go through flight duration was the pitch
tether, and it had been written off. Fixing it (it had no haltere: see "A rig
that was wrong") made it the cleanest preparation in the project, and the
first thing it measured was the loop gain -- inject at the plant input, read
``-C/(C+D)`` -- which came out **2.67 times the design at every frequency**.
A constant factor at every frequency is not dynamics. It is a number.

### The inertia was about the wrong point

The gains are inertia times bandwidth squared over authority, and the inertia
had been read off the free joint's block of the mass matrix. That block is
about the joint's own point, the body's frame origin -- which sits 1.1 mm from
the centre of mass, the same 1.1 mm that made the first torque readings push
the wrong way. The torque had been moved to the centre of mass; the inertia it
is divided by had not.

| axis | about the origin (used) | about the centre of mass | ratio |
| --- | ---: | ---: | ---: |
| pitch | 0.002014 | **0.000753** | 2.67 |
| roll | 0.001502 | **0.000335** | 4.48 |
| yaw | 0.000591 | **0.000496** | 1.19 |

Two independent readings agree on the right column: the whole-body inertia
summed about the centre of mass, and the tether's own mass matrix, whose hinge
passes through it. And the ratio is exactly the parallel-axis term.

So "bandwidth 60" was three loops, none of them at 60 and none critically
damped: **pitch at 98 rad/s and damping ratio 1.6, roll at 127 and 2.1, yaw
at 65 and 1.1**. The cliff at "110" was pitch at 180 and roll at 233. And the
README's claim that yaw is the light axis was wrong: about the centre of mass
roll is, by a third.

### And so was the wing's air speed

The same pair of points, once more. A wing's air speed is the body's linear
velocity plus omega cross the arm to the wing, and a free joint's linear
velocity is the *origin's*. The arm was taken from the centre of mass. That
drops omega cross (centre of mass minus origin) from both wings -- a uniform
wind of 1.07 mm/s per rad/s of pitch that is not there -- and the fore-aft
authority turns it into **+0.0032 of pitch torque per rad/s: an anti-damping
three times the pitch damping the model then reported**. Checked now against
the simulator's own kinematics, point by point. Corrected, the aerodynamic
damping time constants are 180 ms in pitch, 120 in roll and 13 in yaw, not
"near two seconds, a hundred times slower than the loop". Still thirty times
slower than the attitude loop, so the earlier conclusion survives; the number
it was drawn from did not.

### The loop, read off the tether

With both fixed, the tether measures the loop that was designed:

| rad/s | measured gain | design | measured phase | design |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 5.636 | 5.625 | -145.7 | -145.5 |
| 160 | 2.196 | 2.186 | -137.5 | -137.6 |
| 250 | 1.280 | 1.271 | -140.3 | -140.6 |
| 329 | 0.917 | 0.909 | -146.6 | -146.9 |

Within 1% in magnitude and 0.3 degrees in phase, at every frequency. The
crossover is near 306 rad/s and **the phase margin is 35 degrees**, read off
rather than inferred: the PD alone would have 76 there, and the one-wingbeat
boxcar's half-period delay spends 40 of them. That is the first number in this
README about the loop's stability that did not come from how long something
flew.

### What the in-flight deficit was: the climb

The closed-loop identification above found pitch and roll authority at
50-85% of static between 100 and 200 rad/s, and left it open. It is the
climb. Tilt a still animal and nothing changes; tilt a climbing one and the
climb acquires a component along the stroke path, one half-stroke meets
faster air than the other, and the torque follows the tilt:

| climb, mm/s | pitch torque per rad of pitch | roll torque per rad of roll |
| ---: | ---: | ---: |
| 0 | 0.000 | 0.000 |
| 1000 | +2.82 | +1.88 |
| 2000 | +5.64 | +3.76 |

Linear in the climb, and destabilising. This stroke climbs at 2.2 m/s, where
that is 6.1 per radian in pitch against an inertia of 0.000753: **an unstable
pole at 90 rad/s** that the loop has to out-stiffen before it does anything
else. Inside the loop the stiffness divides the apparent authority by
``1 + K/(I w^2)``, and that predicts the deficit: 0.55 at 100 rad/s against
0.47-0.50 measured, 0.83 at 200 against 0.84-0.87, 0.92 at 300 against 0.98.
Hiding the climb from the wings, and nothing else, brings the pitch authority
at 100 rad/s back to 1.08. On the tether, which cannot climb, it is 1.02, 0.97
and 1.03 at 100, 150 and 200.

It also explains which loops fly. With the right inertia and a critically
damped loop at a true 60 rad/s, the climbing animal loses attitude in 21 ms:
the loop is softer than the climb. With the altitude loop closed instead --
the muscle-driven stroke, throttle through the connectome channel, holding
zero -- there is no climb, and the same animal holds **the whole three
seconds** at every bandwidth tried, 60 included: within 16 degrees of level
at 60 and within 4 at 140, heading inside ten throughout. Hovering was never the hard problem. Climbing
at two metres a second on a stroke that makes 1.36 body weights of lift was.

### The bandwidth, chosen again

Critically damped on every axis, on the corrected physics, three amplitudes
one percent apart, flight in ms over three seconds of climbing:

| rad/s | 78.62 deg | 79.41 deg | 80.20 deg | median |
| --- | ---: | ---: | ---: | ---: |
| old gains | 1391 | 1266 | 1139 | 1266 |
| 100 | 922 | 878 | 817 | 878 |
| 120 | 1364 | 1073 | 1109 | 1109 |
| 140 | 1559 | 1913 | 1394 | 1559 |
| 160 | 2031 | 1647 | 1617 | 1647 |
| 180 | 1698 | 1910 | 1702 | 1702 |
| 200 | 2155 | 735 | 1117 | 1117 |

The shape is familiar -- a best row with a cliff just above it -- and so is
the reason. At 160 the yaw gain is 12.95 and the phase knob saturates at 3.5
degrees of heading error: the yaw loop chatters between its limits, drags
roll and pitch with it, and steering reverses (a left object gave -3.7
degrees of heading, a right one +13.1). Give yaw its own bandwidth and the
cliff goes away:

| pitch and roll, with yaw at 65 | 78.62 deg | 79.41 deg | 80.20 deg | median |
| --- | ---: | ---: | ---: | ---: |
| 120 | 1213 | 1130 | 1057 | 1130 |
| 140 | 1477 | 1367 | 1276 | 1367 |
| **160** | **1536** | **1441** | **1332** | **1441** |
| 180 | 1563 | 1455 | 1362 | 1455 |
| 200 | 1573 | 1458 | 1367 | 1458 |

A plateau from 140 up with nothing falling off it. **160 in pitch and roll,
65 in yaw**: 65 because it is the yaw loop the project has flown all along
(the old gains ran yaw at 65, and these are within 2% of them), which keeps
every steering result valid; 160 because it sits in the middle of the plateau
and beats the old gains at every amplitude. Yaw at 90 flies about 10% longer
again and was not taken, for the same reason 100 was not taken over 60 last
time: the gain is where the knob starts saturating inside the errors steering
produces. Amplitude steering changes with the faster roll loop -- the bank
arrives sooner and the adverse-yaw phase shortens, +1.98 degrees at 30 ms and
gone by 50 -- and that test was re-measured rather than loosened blindly.

### Where it ends now

The failure mode is still the heading. Every climbing flight in the tables
above ends with it 90-170 degrees off; attitude goes second. But the question
has changed shape. A hovering animal does not fail at all inside three
seconds, and the climbing one fails because of a destabilising stiffness that
grows with climb speed -- which is a property of flying straight up at 2 m/s,
not of the controller.

## What does not exist yet


## Running the tests

```bash
pip install -e ".[dev,body]"
pytest -q             # 115 tests, about fifteen minutes on one core
pytest -q -m sweep    # the 9 measurement sweeps, about half an hour
pytest -q -n 4        # with pytest-xdist: about five minutes on four cores
```

The aerodynamics and the connectome readout run anywhere -- the wing mesh is
checked in as vertices and the tuning curves as fixtures -- and everything
that puts the wings on a body skips itself without MuJoCo.

**The sweeps are deselected by default, on purpose.** Nine tests re-measure
tables that already live in this README and in `docs/LITERATURE.md`: the
filter-lag curve at three amplitudes, the sensing-against-bandwidth grid, the
stroke-shape comparison, and the multi-window steering and yaw-hold
comparisons. They are the discriminators that have caught every false optimum
here, and they are worth running whenever the stroke or the loop changes. They
are not worth 27 of the suite's 37 minutes on every edit, and a suite that
grows slower every time the animal flies longer -- which is what a
time-to-failure metric does -- was on its way to being one nobody ran.
