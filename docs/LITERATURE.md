# What is verified, and what is relayed

Inherited from `flyloop`, where a claim survived fifteen runs because it was
never a number. The rule: anything stated as fact carries a citation that
somebody checked, or it sits under **relayed** until somebody does.

## Verified

Nothing yet. This project is two days old.

## Relayed — needs checking against the papers

- **Translational force coefficients.** `lift_coefficient` and
  `drag_coefficient` in `wingloop/aero/blade_element.py` use the standard
  empirical fits for *Drosophila* wings (the Dickinson/Sane line of work,
  dynamically-scaled robotic wing). The functional forms and every constant in
  them are relayed from memory and **not checked against the source**. They
  produce a hover ratio of 1.08-1.36, which says they are not wildly wrong; it
  does not say they are right.
- **Kramer rotational coefficient 1.55** for rotation about the quarter-chord.
- **Wingbeat frequency 218 Hz** and **stroke amplitude ~75 degrees** for
  hovering *D. melanogaster*.
- **Reynolds number ~150** at this scale, and the attached leading-edge vortex
  that follows from it.
- **DNg02 controls wingbeat amplitude.** Appears in the top descending inputs
  to the wing steering muscles (measured, see below) but its published role is
  relayed.

## Measured here, from the data

These are not citations. They are numbers this project computed from MaleCNS
v1.0 and from NeuroMechFly's own model files, and each has a test.

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
  0.08%. Flight is driven by a different set: DNbe001 (4.1%), DNge107 (3.9%),
  DNa08 (3.5%), DNa10 (3.2%), DNp63 and DNp49 (3.2% each), DNg02_a and
  DNg02_b (~5.2% together).
- **NeuroMechFly cannot fly and is not close.** Its built model has 48
  actuators and **none of them is a wing or haltere**; the wings are rigid
  geoms welded to the thorax with `contype=0`, so they do not even collide.
  `density`, `viscosity` and `wind` are all 0 -- there is no air in the
  simulation. This is why the aerodynamics had to be written rather than
  configured.
- **Unit system**, read off the built model rather than assumed: length in
  millimetres (wing mesh spans 2.40), mass in grams (total body 1.027e-3),
  gravity 9810 mm/s^2.
