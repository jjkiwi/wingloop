"""A search in real metres: a Tatra valley, a thermal camera looking down, two platforms.

The lab's arena is at 1:50 -- a drone under a metre up, a few metres round
-- and a search in the mountains is a square kilometre flown at 30-120 m.
This package is that search:

- :mod:`.teren`: the ground -- a glacial valley with a cirque, a tarn, a
  trail and ridges, snow, rock, scree and dwarf pine, and how warm each is
  in the sun;
- :mod:`.obiekty`: what is on it -- people lying, sitting and walking,
  chamois, sun-warmed boulders -- each a model of parts for the eye and a
  map of warm parts for the thermal camera;
- :mod:`.kamera`: the thermal camera looking down, a pixel's ground mixed
  with what lies in it, and the laser rangefinder;
- :mod:`.lot`: the quadcopter and the plane flying a search, lane by lane,
  and coming lower to look again at what was warm;
- :mod:`.wykrywanie`: warm spots found, measured in metres, judged by the
  fly's mushroom body;
- :mod:`.film`: the plane from behind, with the camera's view and the
  rangefinder's beam drawn in.
"""
