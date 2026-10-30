"""An operator's order, in Polish words: which job, and with what.

"Szukaj rozbitka na morzu, pytaj mnie, gdy nie jesteś pewien" -- the job is
the one whose target or world the words name, and the add-ons
(:mod:`.dodatki`) the ones they ask for. Words, not a language model: each
add-on and each job is known by the stems of the words that name it, so
the drone does what it was told or says it did not understand; it never
guesses. A speech recogniser in front of it makes it a voice order.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from .dodatki import ZESTAWY, Dodatki
from .zadanie import ZADANIA, zadanie

#: Stems that name each add-on.
SLOWA = {
    "operator": ("pytaj", "zapytaj", "pytan", "niepew", "nie jestes pew", "upewn"),
    "mapa": ("map", "dokladn", "systematycz", "prawdopodob"),
    "pamiec": ("dryf", "rusza", "porusz", "plyn", "pamiet"),
    "zmysly": ("czujnik", "dalmierz", "temperatur", "zmysl"),
    "kora": ("kora", "kore", "kory", "druga opini", "klasyfik"),
}
#: And each job, besides its target's and its kinds' names.
MIEJSCA = {
    "ratunek_morze": ("morz", "wod", "rozbit", "plyw", "toni"),
    "ratunek_gory": ("gor", "gorach", "snieg", "lawin", "alpin", "wspinacz", "turyst"),
}


def _plain(text: str) -> str:
    """Lower case, without Polish letters' marks: "Góry" and "gory" alike."""
    text = text.lower().replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def rozkaz(text: str, base: Dodatki | None = None):
    """(the job, the add-ons, what was understood) from an order's words.

    ``base``: the add-ons to start from; "wszystko" or "pełn..." asks for
    the full hybrid (:data:`~.dodatki.ZESTAWY` ``"hybryda"``). Raises
    ValueError when no job, or more than one, is named.
    """
    zadanie("ratunek_gory")  # the jobs shipped
    words = _plain(text)
    named = []
    for name, z in ZADANIA.items():
        stems = (*MIEJSCA.get(name, ()), _plain(z.kinds[z.target]["pl"])[:5])
        if any(re.search(r"\b" + re.escape(s), words) for s in stems):
            named.append(name)
    if len(named) != 1:
        known = ", ".join(sorted(ZADANIA))
        raise ValueError(f"Nie wiem, o które zadanie chodzi ({named}); znane: {known}")
    d = base or ZESTAWY["mucha"]
    if re.search(r"\b(wszystk|peln)", words) and "hybryda" in ZESTAWY:
        d = ZESTAWY["hybryda"]
    understood = []
    for part, stems in SLOWA.items():
        if not any(re.search(r"\b" + re.escape(s), words) for s in stems):
            continue
        understood.append(part)
        default = {"operator": {}, "mapa": {}, "pamiec": True, "zmysly": 4, "kora": 0.3}[part]
        if not getattr(d, part):
            d = replace(d, **{part: default})
    return ZADANIA[named[0]], d, understood
