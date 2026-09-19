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

## What does not exist yet

- The power-muscle oscillator that would drive the stroke instead of imposing it
- **Real stroke kinematics.** A pure harmonic sweep with a tanh flip and no
  deviation is what produces the within-stroke torque swings the controller
  cannot answer. Real strokes put their rotation at the reversals and trace a
  figure-of-eight; both should shrink those swings. This is the next thing to
  measure.
- The descending readout, from the flight DNs rather than DNa02 -- and the
  arithmetic for it is now in hand on both sides: DNg02 sets amplitude by
  population coding over about 29 cells, and an amplitude asymmetry of 0.2
  produces 7.6 of roll torque.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Nothing here needs MuJoCo or a connectome; the wing mesh is checked in as
vertices so the geometry test runs anywhere.
