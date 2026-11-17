"""A 3-D simulator of the search, made to look like a computer game.

The search itself is :mod:`~wingloop.dronomucha.teren`'s: the valley, the
people and animals on it, the platforms flying the search and the fly's
brain judging what the thermal camera sees. :mod:`.eksport` writes all of
it -- the ground, what covers it and how warm it is, every object's parts,
every platform's way, step by step, and what became of each candidate --
into one file; ``web/`` draws it in a browser with three.js: a sky and a
low sun, shadows, haze, dwarf pine and scree, the tarn, the three drones,
the thermal camera's picture, and a game's head-up display. ``nagraj.js``
records it, frame by frame, to a film (docs/sym3d.md).
"""
