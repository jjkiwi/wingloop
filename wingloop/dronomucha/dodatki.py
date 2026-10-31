"""The functions people add to the fly's brain, each one that can be put in or left out.

The core is the fly's -- its optic lobe and its mushroom body -- and the
drone flies on it alone with :data:`MUCHA`. Each add-on is measured against
that, on the same search areas (docs/hybryda.md):

``zmysly``
    the thermal camera's kelvin and the rangefinder's distance and size,
    taken into the mushroom body next to the eye (:mod:`.zmysly`); the
    number is how much of the calyx they get (:func:`~.zmysly.kod`'s
    ``gain``), 0: none;
``kora``
    a cortex's second opinion on each view, and how much it counts
    (:class:`~.hybryda.Kora`, :func:`~.hybryda.laczona`), 0: none;
``mapa``
    where to search next from a map of probabilities (:class:`~.mapa.Mapa`'s
    settings), instead of the task's fixed pattern; None: the pattern;
``pamiec``
    remember objects, not places: where each thing it has been to is now,
    moving or not (:mod:`.pamiec`);
``operator``
    ask a person when it is not sure (:mod:`.operator`'s settings); None:
    decide alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Dodatki:
    """Which add-ons the drone flies with; all off is the fly's brain alone."""

    zmysly: int = 0
    kora: float = 0.0
    mapa: dict | None = None
    pamiec: bool = False
    operator: dict | None = None

    @property
    def mozg(self) -> bool:
        """Does the brain itself change -- senses or a cortex?"""
        return bool(self.zmysly or self.kora)


#: The fly's brain alone.
MUCHA = Dodatki()

#: Everything that flies without a person: the senses in the calyx at four
#: times their lines, the cortex at 0.3 -- the best measured on the lessons
#: (docs/hybryda.md) -- the search map and the memory of objects.
HYBRYDA = Dodatki(zmysly=4, kora=0.3, mapa={}, pamiec=True)

#: Named sets, for the command line.
ZESTAWY: dict[str, Dodatki] = {
    "mucha": MUCHA,
    "hybryda": HYBRYDA,
    "hybryda_z_operatorem": replace(HYBRYDA, operator={}),
}
