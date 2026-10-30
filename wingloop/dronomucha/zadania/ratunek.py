"""Search and rescue: a living person, found by heat, in the mountains and at sea.

A climber on a snowfield or a castaway in the sea is warm: a clothed body is
some 15-30 K warmer than snow at -5 C, a face and life jacket afloat some
8-18 K warmer than a 12 C sea. That is what a thermal camera is for, and the
drone's eye looks through one, "black hot" -- the warm dark, as the fly's eye
looks for what is dark (:data:`~wingloop.lab.watch.THERMAL_SPAN`), its gain
set to the warmest thing in view.

Warm is not enough. In the mountains a dark rock in the sun can be as warm
as a person, and a chamois is warm too; at sea a seal is about a person's
size and warm, and floating debris warms in the sun. Telling the person from
those is the brain's lesson.

The arena's scale is 1:50: a person lying down is 34 x 10 x 6 mm, 1.7 m by
half a metre; the drone flies some 17 mm -- under a metre -- up, its eye
looking ahead, as a fly's does. A rescue drone searches from 50-120 m with
its camera looking down; docs/ratownictwo.md has what that leaves out.
"""

from __future__ import annotations

from wingloop.lab.arena import FLOOR

from .. import wyglad
from ..zadanie import Zadanie, dodaj, kwadrat

GORY = dodaj(
    Zadanie(
        name="ratunek_gory",
        pl="góry: człowiek na śniegu",
        target="czlowiek",
        kinds={
            # Lying down, in a jacket: 10-25 C at its surface on -5 C snow.
            "czlowiek": dict(
                pl="człowiek",
                shape="box",
                size=(17.0, 5.0, 3.0),
                z=FLOOR + 3.0,
                heat=(15.0, 30.0),
                rgb=(0.85, 0.2, 0.08),
                parts=wyglad.czlowiek_lezacy(),
            ),
            # A dark rock: as cold as the air in shade, 30 K warmer in the sun.
            "kamien": dict(
                pl="kamień",
                shape="sphere",
                size=7.0,
                z=FLOOR + 4.0,
                heat=(0.0, 30.0),
                rgb=(0.25, 0.24, 0.22),
                parts=wyglad.kamien(),
            ),
            # Standing, in its winter coat: 3-13 C at the fur.
            "kozica": dict(
                pl="kozica",
                shape="box",
                size=(10.0, 3.0, 7.0),
                z=FLOOR + 7.0,
                heat=(8.0, 18.0),
                rgb=(0.42, 0.3, 0.18),
                parts=wyglad.kozica(),
            ),
        },
        scenery=dict(
            floor=("0.93 0.94 0.97", "0.86 0.88 0.93"),
            columns=True,  # rock spires
            **wyglad.scena_gory(),
        ),
        column_heat=2.0,
        clutter=0.5,
        span=(6.0, 20.0),
        pattern=kwadrat(30.0, 8),
        ground_c=-5.0,
        about="Alpinista leżący na śniegu; mylące: nagrzane kamienie, kozice.",
    )
)

MORZE = dodaj(
    Zadanie(
        name="ratunek_morze",
        pl="morze: rozbitek w wodzie",
        target="rozbitek",
        kinds={
            # Afloat on the back in a life jacket, 1.7 m long, in a 12 C sea:
            # face and jacket above the water. (A head alone, half a metre
            # across, was lost to the eye beyond 60 mm -- 3 m -- from a
            # metre up.)
            "rozbitek": dict(
                pl="rozbitek",
                shape="box",
                size=(17.0, 5.0, 1.5),
                z=FLOOR + 1.0,
                heat=(8.0, 18.0),
                rgb=(0.95, 0.45, 0.05),
                parts=wyglad.rozbitek(),
            ),
            # A seal, 1.4 m, wet fur: a few kelvin over the water, more basking.
            "foka": dict(
                pl="foka",
                shape="box",
                size=(14.0, 4.0, 2.0),
                z=FLOOR + 1.0,
                heat=(1.0, 8.0),
                rgb=(0.35, 0.35, 0.38),
                parts=wyglad.foka(),
            ),
            # Something afloat, a metre across, warmed a little by the sun.
            "smieci": dict(
                pl="śmieci",
                shape="box",
                size=(10.0, 8.0, 1.0),
                z=FLOOR + 0.5,
                heat=(0.0, 4.0),
                rgb=(0.85, 0.85, 0.8),
                parts=wyglad.smieci(),
            ),
        },
        scenery=dict(
            floor=("0.1 0.27 0.42", "0.13 0.32 0.48"),
            columns=False,
            **wyglad.scena_morze(),
        ),
        column_heat=0.0,
        clutter=0.3,
        span=(3.0, 10.0),
        pattern=kwadrat(30.0, 8),
        # Flat on the water, a person never fills 7% of the eye, as the
        # drone's arrival needs: at its nearest, 18 mm off, 5.7%; then it
        # passes under. It has arrived at 4.5%.
        nawigacja=dict(arrive_size=0.045),
        ground_c=12.0,
        about="Rozbitek w kamizelce na wodzie; mylące: foki, dryfujące śmieci.",
    )
)
