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

## What does not exist yet

- Wing hinge joints and actuators added to the MJCF, and the forces applied
  into MuJoCo each step
- The power-muscle oscillator that produces the stroke
- The descending readout, from the flight DNs rather than DNa02
- Haltere feedback, which is how the animal stabilises
- Free flight: the force model currently sees only the wing's own sweep, not
  the body's velocity through the air, so it describes hovering or a tether

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Nothing here needs MuJoCo or a connectome; the wing mesh is checked in as
vertices so the geometry test runs anywhere.
