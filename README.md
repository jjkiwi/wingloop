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

**On Windows, the short way:** double-click `flylab.bat` in this folder. The
first run creates `.venv`, installs everything and downloads the optic-lobe
weights (10-20 minutes); every run then checks the install, runs the demo and
opens the report. It needs Python 3.12 installed (`winget install
Python.Python.3.12`).

**On Windows, by hand**, in PowerShell. Python must be 3.10-3.12: flyvis and flygym
do not install on 3.13 or newer. Install Python 3.12 from python.org (or
`winget install Python.Python.3.12`), then:

```powershell
git clone https://github.com/jjkiwi/wingloop
cd wingloop
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -e ".[body,lab]"
python -m flyvis_cli.download_pretrained_models --skip_large_files
python -m wingloop.lab check         # says what is missing, if anything
python -m wingloop.lab demo          # about ten minutes; writes flylab-run\flylab.html
```

`python -m wingloop.lab` is the same program as `flylab`, and works when
pip's Scripts folder is not on PATH.

**Watching it fly.** `flylab watch` (on Windows: double-click `latanie.bat`)
opens a 3-D window with the fly flapping its way over a chequered floor past
coloured columns, and you fly it from the keyboard: arrows for speed and
turns, W/S for height, space to pause, C to free the camera. A page in the
browser shows the nervous system as it flies, refreshed a few times a second
from a server on this computer only: what the fly sees, its photoreceptors,
the pretrained optic lobe running on that view (T4 and T5 motion detectors,
Mi1 and Tm9 contrast cells, one dot per column), the haltere signals, the
descending and motor neurons, and the three wing commands. B switches the
optic lobe off, which roughly halves the work; shadows are off unless asked
for with `--pretty`.
`flylab video` flies a scripted tour (hover, forward, turn, climb, turn back)
and saves it as an MP4 with the same readout burned in.

**Flying to targets on its own.** `flylab cel kula` (on Windows: `cel.bat`,
which asks what to fly to) teaches the fly a target and then lets it fly
there with nobody at the controls. Three kinds of dark object stand in the
arena -- a ball (`kula`), a flat slab (`płytka`) and a pole (`słupek`), round,
horizontal and vertical like the shapes of the classic fly experiments --
two of each, spread around the start.

1. *Teaching, on the tether*, as it is done with real flies in a flight
   simulator: the eye is shown every kind from many sides and distances, the
   way a flying fly meets them -- the panorama turning past at the speed the
   fly turns, then held in the middle of the eye while it fixates -- and the
   mushroom body gets dopamine for the target kind and punishment for the
   others. The looking is done once and shipped (`--od-nowa` repeats it on
   your computer, a few minutes); the teaching happens on every run and
   takes a fraction of a second, so any target is one command away.
2. *The mission*, in the physics: the fly turns on the spot, fixates each
   dark object it sees, and asks its mushroom body what it is worth. It flies
   at the best one, tracking it in the middle of its eye, brakes as it
   looms, and remembers the place by path integration. Then it looks again.
   When nothing it likes is in sight -- from one target the others can be too
   far to make out -- it flies home and looks from there, and when nothing
   is in sight from home either, it is done.
3. *Learning in flight*: where it lands, the world answers -- reward on a
   target, the sugar of the real experiment, punishment on anything else --
   and the dopamine goes onto the Kenyon cells of the very view it chose on.
   A mistake makes the next object that looks like it less attractive; a
   find makes the next one more.

The brain page gains a section for it: the arena from above with the fly's
track, what it has learned, each object it judged on its last look and what
it is flying to now. `flylab video --cel kula` records a whole mission;
`flylab cel kula --bez-okna` flies it without a window and prints the result.

Measured, nine missions -- three arenas, each kind the target once, two of
each kind in every arena, the shipped tethered views -- flying without help
and never losing the air:

| target | arena 0 | arena 1 | arena 2 |
| --- | --- | --- | --- |
| ball | 2 of 2, 2 mistakes | 2 of 2, 2 mistakes | 2 of 2, 2 mistakes |
| slab | 2 of 2, 1 mistake | 2 of 2 | 1 of 2 |
| pole | 2 of 2 | 2 of 2, 1 mistake | 1 of 2 |

16 of the 18 targets found, in 5 to 25 seconds of flight, and 8 landings on
something else -- 6 of them by the fly sent to the balls. The ball is the
eye's hard case: on 53 fixations the physical fly made in nine other arenas,
the mushroom body taught on the tether calls the ball right 83% of the time
against 96% for the slab and 91% for the pole, because a ball looks like a
small dark blob from every side and so, a little, does everything else far
enough away. Those numbers took three fixes to reach, each found by flying:
teaching on views shaped like a flying fly's fixation rather than a still
one (in flight every object had been valued below zero), giving the Kenyon
cells ON and OFF inputs instead of ``|z|`` (87% to 96% on new tethered
views), and turning the panorama in from further away so the eye arrives
at the object in the state a scanning eye is in (the ball from 81% to 89%
on the same 45 views). Learning in flight is on in every run; in these
missions the mistakes were on different kinds each time, so the one
punishment per kind came too late to change a choice -- what it changes
is the second visit to a kind, and there was none.

**The same brain flying a drone.** `flylab cel kula --dron` (or answer
`dron` in `cel.bat`) hands the fly's eye, mushroom body and navigator,
unchanged, to the Crazyflie quadrotor of `flylab drone`, flying on the fly's
control laws in the fly's arena made 50 times bigger -- a room about eight
metres across, objects 0.3-0.8 m, flying 0.85 m up. Scaled evenly, a scene
looks exactly the same from a camera scaled with it, so the drone's camera
sees, pixel for pixel, what the fly's eye would, and what the fly learned on
the tether serves the drone without being learned again. Only the body
differs: the navigator's pitch becomes a speed for the drone's speed loop,
it looks around at 90 degrees a second rather than the fly's 240 (its yaw
comes from motor drag and lags), it brakes until it has actually stopped
before recording where the object is (it has more momentum than a fly, and
recording the place early sent it back to objects it had reached), and the
brain page shows its gyroscope and motors where the fly's shows its
haltere-to-wing pathway. On the same three arenas as the fly, each target
once, it found every target -- 2 of 2 three times -- with 1-2 landings on
something else per mission, in 42-60 s of flight; the simulation runs about
nine times slower than real time, the optic lobe taking most of it.

**The fast mode.** `--szybko` (the default in `latanie.bat` and `cel.bat`,
which ask) replaces the full simulation with a reduced-order fly fitted to
it, and steps the optic lobe at the 20 ms it was trained at rather than 5.
The full simulation was flown through pitch, heading and height steps and
its answers turned out simple: forward speed 30 + 22.3 mm/s per degree of
pitch with a 0.12 s lag; heading a critically damped loop at 28 rad/s
hovering, 17 at 290 mm/s; height one at 20 rad/s, the throttle loop's own.
`wingloop/lab/fastfly.py` is those numbers, and a test replays the recorded
full-physics flights through it (speed within 15-35 mm/s RMS, heading within
4-8 degrees, 20 on a 135-degree reversal where the full fly overshoots).
The wings are drawn beating from the stroke generator; the flight does not
come from them. The tethered lesson is shipped for both eye rates -- at 20
ms it is better, 91-100% on new views, and takes 91 s to compute instead of
489.

| | full physics | fast mode |
| --- | --- | --- |
| a mission, computed | 6-21 min for 5-25 s of flight | 18-32 s for 12-20 s |
| targets found, 9 missions | 16 of 18, 8 mistakes | 16 of 18, 9 mistakes |
| the drone, targets found | 6 of 6 (3 missions) | 9 of 12, 7 mistakes (6 missions) |

The drone does a little worse in the fast mode: its valences come out a
shade lower than the fly's, enough that once both targets were judged just
below zero and it never set off. It turns at 90 deg/s while the tethered
lesson turns the panorama at the fly's 240, and at the coarser eye step that
difference shows. (The fast fly also got `patience`: a mission ends after
two empty looks from home, not one; the full-physics table above was
measured before it.) Without a graphics card the 3-D window costs more than
the flying -- here, in software, the fast mode runs 7-9 times slower than
real time with the window and 1.5 times without; the window draws at most
30 frames a second for that reason, and the fly is never let fly faster than
a fly.

**Practice: fewer mistakes.** `flylab trening kula --misje 15` (or
`trening.bat`) sends the fly -- or, with `--dron`, the drone -- on practice
missions in new arenas, in the fast mode. Its brain carries over from one
mission to the next, so every landing teaches it: reward on a target,
punishment on anything else, onto the Kenyon cells of the view it chose on.
Those are the views it really has in flight, which the tethered lesson only
approximates, and the punished ones are exactly its mistakes. What it learned
is saved (`pamiec_kula_20ms.npz` and the like, per target, body and eye
rate), `flylab cel` loads it by itself, and another practice session builds
on it; `--od-nowa` goes back to the tether.

Measured on six arenas it never practised in, before and after fifteen
practice missions:

| target | tethered lesson only | after practice |
| --- | --- | --- |
| ball | 12 of 12 targets, 14 mistakes | 10 of 12, 6 mistakes |
| slab | 11 of 12, 6 mistakes | 12 of 12, 2 mistakes |
| pole | 9 of 12, no mistakes | 11 of 12, 4 mistakes |
| all | 32 of 36, 20 mistakes | 33 of 36, 12 mistakes |

The drone, the same way (its own practice, its own memory):

| target | tethered lesson only | after practice |
| --- | --- | --- |
| ball | 9 of 12 targets, 8 mistakes | 10 of 12, 4 mistakes |
| slab | 12 of 12, 8 mistakes | 12 of 12, 2 mistakes |
| pole | 8 of 12, 1 mistake | 12 of 12, 6 mistakes |
| all | 29 of 36, 17 mistakes | 34 of 36, 12 mistakes |

Practice helps both: fewer mistakes on the ball and the slab, and on the
pole, where the tethered lesson left them too cautious to find their
targets, more targets -- bought with some mistakes, since a reward makes
what looks like a pole more attractive and a few other things look a little
like one. Six test arenas are few, and single missions vary by a target or
two either way; the totals are the numbers to read.

Two things came with it. Practice makes the fly choosier, and a choosier
fly sometimes liked nothing at all and ended its mission on the spot with
both targets in view; so when it has landed nowhere and likes nothing it is
now *curious* -- it tries the most promising object once and learns from
what is there. And the dose: half the dopamine per landing kept more
targets on the ball (11 of 12) but cut the mistakes far less (9-10 against
3-5), so a landing gets the tether's full dose.

**Becoming an expert.** Practice alone halved the mistakes. To find out
where the rest came from, the drone and the fly were sent on explorer
missions -- liking everything, landing everywhere -- and every view they
had was kept with what the object really was: 38 arenas, about 1,600 views.
That record said the eye is right about 90% of the time on an object 40-60
mm away and about 50% beyond 80 -- so most mistakes were judgements made
from too far. Five teaching methods came out of it, each measured before it
went in:

1. *A close look before landing* (`Navigator.inspect`): the fly stops where
   the object fills a fiftieth of its eye -- 30-60 mm -- looks again, and
   lands only if it still likes it.
2. *A close-look lesson on the tether*: taught only on scanning views, the
   mushroom body turned down nearly every close look at a ball (13% right):
   a stopped eye after nose-down flight is nothing like a turning one. The
   tether now also replays the whole approach -- the turn, a half second of
   nose-down flight at the fast fly's measured speed, the brake, the hover
   with its 10-30 mm/s of drift -- and shows the object close up; judged on
   new close looks, 93-100% right with 0-3% false alarms.
3. *Landing to learn*: in practice the fly lands after every close look,
   liked or not. A close look it rejects brings no dopamine, so without
   this, practice could never correct a close look that wrongly turns a
   target down -- which on the tether lesson alone is most of them for the
   ball.
4. *Lessons on the way in, and on the far view* (`trace_lessons`): dopamine
   at the landing also reaches the Kenyon cells of the views seen on the
   approach, at half strength -- as it does in the fly, where KC-to-output
   synapses stay eligible for seconds -- and the view from afar that the
   choice was made on. On recorded flight views the way-in lesson beat a
   lesson on the chosen view alone (the ball 71% right with 2% false alarms
   against 48% and 3%). Without the far view the drone, which liked an
   object from afar and turned it down close up, went back to the same
   object up to sixteen times: nothing ever punished what drew it there.
   Replaying all experience in a balanced "sleep" lesson was tried and did
   worse: it pulled the brain back toward the tether's bias.
5. *Each body taught the way it flies*: the drone learned the fly's lesson
   -- the panorama turning at 240 deg/s, the fly's approach -- and in its
   first practice it reached 74% of its targets where the fly reached 92%.
   It turns at 90 deg/s, flies in at its own 1.5 m/s on its own speed loop,
   tilting up to 20 degrees, and has to stop before it looks. Its tether
   lesson now replays exactly that; on the views the drone really had in
   flight it called 68% of the balls within 80 mm balls, against 42% on the
   fly's lesson. A drone mission may also run 90 s rather than 60: slower
   to turn and to stop, it had run out of time in a third of its missions.

A novice -- the tether lesson only -- does not take the close look: on its
own it turned down most targets (7 of 24 balls). A brain that has practised
does. `flylab egzamin` (and the end of `flylab trening`) sits it in arenas it
never practised in, with no learning, and grades it: *ekspert* at 90% of
targets or more and at most 0.2 mistakes a mission, *zaawansowany* at 75%
and 0.5.

The fly, twelve exam arenas per target:

| target | novice | novice with close looks | after 20 practice missions |
| --- | --- | --- | --- |
| ball | 21 of 24, 28 mistakes | 7 of 24, 8 | 20 of 24, 0 |
| slab | 23 of 24, 9 | 21 of 24, 0 | 23 of 24, 5 |
| pole | 19 of 24, 1 | 14 of 24, 1 | 23 of 24, 0 |
| all | 63 of 72, 38 mistakes | 42 of 72, 9 | **66 of 72 (92%), 5 (0.14 a mission)** |

The mistakes went from 38 to 5 for about the same number of targets. Then
the practice was made longer -- 50 missions in new arenas, examined after
25 and after 50:

| target | after 25 | after 50 |
| --- | --- | --- |
| ball | 17 of 24, 0 mistakes | 20 of 24, 0 |
| slab | 22 of 24, 2 (*ekspert*) | 22 of 24, 3 |
| pole | 22 of 24, 1 (*ekspert*) | 21 of 24, 0 |
| all | 61 of 72 (85%), 3 | 63 of 72 (88%), 3 |

A second fly, practising in other arenas, did the same: 60 of 72 with 3
mistakes after 25, 62 with 3 after 50 (the slab 22 of 24 with one mistake
and the pole 22 with none -- *ekspert* both); a third fly on the ball, 18 of
24 with one mistake after 25 and after 50. Past 25 missions the fly learns
little more: its mistakes were already down to one landing in every ten
exam missions, and what keeps it from *ekspert* on the ball is a ball it
does not find -- in four or five missions of twelve one of the two balls is
never landed on -- not a wrong landing. Practice cannot teach what never
lands, so that is where the next lesson has to come from.

The drone, twelve exam arenas per target:

| target | novice with close looks | 20 missions, the fly's lesson | 25 missions, its own lesson | 50 missions |
| --- | --- | --- | --- | --- |
| ball | 4 of 24, 10 mistakes | 14 of 24, 8 | 18 of 24, 1 | 15 of 24, 2 |
| slab | 17 of 24, 0 | 20 of 24, 4 | 18 of 24, 1 | 18 of 24, 1 |
| pole | 12 of 24, 3 | 19 of 24, 2 | 18 of 24, 1 | 18 of 24, 0 |
| all | 33 of 72, 13 mistakes | 53 of 72, 14 | **54 of 72 (75%), 3 (0.08 a mission)** | 51 of 72, 3 |

Taught the way it flies and with the far view punished, the drone makes a
fifth of the mistakes it made on the fly's lesson and almost never lands
on the wrong thing -- but it finds three targets in four, and 25 more
missions did not change that. Like the fly's ball, what is left is caution:
in half its exam missions it reaches one target and never the
other.

**Why the last targets were missed.** The exams were flown again with every
decision written down -- what each look around saw and valued, where it
went, what the close look said, why each approach ended -- and two of the
reasons turned out not to be judgement at all:

- *The drone gave up a few centimetres short.* An approach is given up
  after 4 s, and the clock ran from the choice, through the stop and the
  close look. In the first five arenas of the drone's ball exam, eight of
  its nine give-ups came right after a close look it *liked*; it came back,
  judged the ball again from too close -- valence 0 -- turned it down, and
  never went back. All four balls it missed there went that way. A liked close look
  now starts the clock again. The same thing had cost it lessons in
  practice: a close look it liked ended without a landing, so without
  dopamine.
- *Missions ended with time left.* Both bodies called a mission done with
  half its time to go and a target valued just below zero from afar --
  -0.012 to 0 -- that it never looked at from where its eye is reliable.
  A practised brain now goes and takes a close look at the most promising
  object it has not been to before it calls a mission done
  (`look_before_leaving`). Since the object was not liked from afar, the
  close look has to like it clearly to overrule that (`sure`, 0.02): on the
  drone, close looks at what it had turned down from afar came out 0.000-0.018
  on what it should not land on, and without the margin it landed on nine
  things in twelve slab missions.

Twelve exam arenas per target, targets found of 24 and mistakes (for the
fly, each of its practised brains):

| | after 50 missions | with both | and 15 more missions with both |
| --- | --- | --- | --- |
| fly, ball | 20, 0 / 18, 2 / 18, 1 | 21, 0 / 20, 2 / 21, 1 | 19, 1 |
| fly, slab | 22, 3 / 22, 1 | 22, 3 / 22, 1 (*ekspert*) | 21, 0 |
| fly, pole | 21, 0 / 22, 0 | **23, 0 / 24, 0** (*ekspert*) | |
| drone, ball | 15, 2 | 21, 5 | **21, 1** |
| drone, slab | 18, 1 | 16, 2 | 18, 2 |
| drone, pole | 18, 0 | 19, 0 | 17, 1 |

The fly finds 66 of 72 where it found 62-63, with the same 3 mistakes, and
is an expert on the pole. The drone finds the ball it used to miss -- and
practising with the fix, now that what it likes close up ends in a landing
and a lesson, it stopped landing on the slabs its ball brain liked a little
(5 mistakes to 1). On the slab and the pole it did not gain. There, what is
left is the drone's close look itself: on the tether its brains tell a
target's close view from the rest by a wide margin (0.12-0.18 against
below -0.05), in flight the slab or pole it should land on and the things
it should not both come out within a few hundredths of zero.

**The close look, taught in flight.** The tether's close-look lesson
replays an approach -- the turn, the run-in, the brake, the hover -- and a
brain taught on it judges the replay perfectly. Judged on 98 close looks the
drone really took, it liked 27% of the slabs and 48% of the poles it
should have; the fly's liked 15% of its balls. Whatever the replay left out
of the body, it mattered. So the close look is now taught the way it is
taken: the body itself -- the fast fly, or the drone -- with its own
navigator, in a small arena with one object 55-95 mm away and two others
off to the sides, turns to it, flies at it, stops and looks, and that view,
with what the object really is, is the lesson (`flown_close_looks`, 300
close looks per body and eye rate, shipped). On close looks it was not
taught on, 94-100% of the targets were liked and 0-5% of the rest. It is
taught at a tenth of the tether's dose: at the full dose it pulled the far
view of the ball down -- the fly liked 23% of the balls it saw while
looking around, against 84%.

Writing down how each close look was taken then showed where the drone's
remaining wrong verdicts came from: every one had been taken while it was
still turning, 22-116 deg/s, with the object 5-42 degrees off the middle
of its eye -- it had started its approach close by, already near enough to
look -- and every one taken facing the object and still was right. Three
things in the navigator followed, each measured on the arenas it failed:

- *The close look waits* until the object is near the middle of the eye
  and the body has stopped turning (`centred`, `still`): for the drone 5
  degrees and 20 deg/s; a hovering fly keeps turning 25-35 deg/s after its
  drifting target, judges well doing so, and drifts into the object if it
  waits, so for the fly 15 degrees and 60 deg/s.
- *It turns to its target before it tracks anything.* Standing by one
  object, the drone locked onto that one at the edge of its eye instead of
  the one it had chosen, and went back to it a dozen times in a mission.
- *Two objects close together are two places.* A place is remembered short
  of the object, by 20-35 mm for a slab, so two slabs 50 mm apart were
  remembered as one and the second never counted; a place it reaches now
  counts as an old one only if the old one is ahead of it.

And curiosity takes a close look too, instead of landing to find out.

Practice then took the drone's last wrong landings away -- the slab brain
had liked a few balls close up -- and in 212 close looks after 25 practice
missions it landed on nothing it should not have. A brain that sure can
afford to look more, so the drone, and only the drone, also goes back to
where it saw something it never looked at (`go_back`) -- an object at the
edge of what the eye makes out is seen on one look around and not the
next -- and with nothing left, flies 40 mm out into the widest stretch of
the panorama it has seen nothing in, more than 90 degrees of it, and looks
around there (`explore`): a ball 78 mm away is too small for its eye from
home. For the fly, whose close looks are less sure, the extra looks cost
more wrong landings than they found targets.

Last, one more for the drone. Its practised close looks were never wrong
from 40-55 mm, and the targets it still missed had been turned down by a
hair -- -0.005, -0.04 -- from 62-74 mm: a close look that turns an object
down by less than 0.06 before it fills 4% of the eye is taken again from
there (`unsure`).

With all of it a novice -- the tether's scanning views and the flown close
looks, no practice -- takes close looks; the old novice found 63 of 72 with
38 mistakes (the fly) and 33 with 13 (the drone, with close looks). In the
twelve exam arenas, targets found of 24 and mistakes:

| | old, after 50 missions | new novice | new, practised |
| --- | --- | --- | --- |
| fly, ball | 20, 0 | 21, 0 | 22, 0 (25 missions) |
| fly, slab | 22, 3 | **24, 0** | |
| fly, pole | 21, 0 | **24, 0** | |
| drone, ball | 15, 2 | 23, 0 | **24, 0** (50 missions) |
| drone, slab | 18, 1 | 24, 4 | **24, 0** |
| drone, pole | 18, 0 | 24, 0 | **24, 0** |

The drone's novice column was flown before `go_back`, `explore` and
`unsure`; with them and 25 practice missions it found 70 of 72, after 50
all 72, with no mistake. Those twelve arenas are the ones every failure
above was found on, so the last brain also sat twelve arenas no one had
looked at (100-111): **72 of 72 again, no mistake** -- the drone is an
expert on all three targets, 144 targets in 72 missions. Those three
brains are shipped (`wingloop/lab/data/pamiec_*_dron_20ms.npz`): `flylab cel
kula --dron --szybko` flies with them unless there is practice of your own,
and `--od-nowa` starts again from the lessons. The fly, a novice
there, found 68 of 72 with 2 mistakes; on the exam arenas what it missed
was a ball its close look judged just below zero from 18-27 mm, where it
ends up after braking.

**Camouflaged targets.** `--kamuflaz 0.7` paints every object that much in
the floor's checker (`Item.camo`, 0-1); at 1 it is painted like the floor
beneath it, tile for tile. The expert drone, which sees what is dark, is
blind to it, and it is not wrong about anything -- it sees nothing. On six
new arenas, targets found of 12 per target, no mistake anywhere:

| camouflage | ball | slab | pole |
| --- | --- | --- | --- |
| 0 | 12 | 12 | 12 |
| 0.25 | 5 | 11 | 12 |
| 0.5 | 0 | 0 | 6 |
| 1 | 0 | 0 | 0 |

What a camouflaged object still does is break the skyline: whatever of it
stands above the horizon blots out the sky, and no floor ever does. So the
brain for camouflaged targets also sees what is darker than the sky more
than 3 degrees above the horizon (`FlysEye.standing`, `Navigator.skyline`),
the horizon taken from the body's sense of its own attitude, as a fly's is
from its halteres and ocelli -- tilted 20 degrees and rolled 10, it sees no
floor. A fully camouflaged slab is then 2.6-3.6% of the eye, a pole 3%, a
ball 0.4-0.7%, where the fixation threshold is 0.3%. But the landscape's
columns break the skyline too, and some stand among the objects, so they
become a kind of their own in the lessons -- a non-target, punished, and a
mistake when landed on (`LANDMARK`). The camouflage brain has its own
lessons (`--kamuflaz` on `trening` and `egzamin`): every object painted a
random 0-1, the columns shown and flown to, and practice in arenas painted
0-1, a different amount each mission.

On the same six new arenas, targets found of 12 and wrong landings (columns
included), for the expert that sees what is dark, the camouflage novice, and
the camouflage brain after 25 practice missions:

| | half camouflage: expert / novice / practised | full: expert / novice / practised |
| --- | --- | --- |
| ball | 0 / 7, 0 / 2, 0 | 0 / 1, 3 / 1, 0 |
| slab | 0 / 9, 8 / 9, 1 | 0 / 7, 4 / 1, 0 |
| pole | 6 / 9, 3 / **10, 0** | 0 / 6, 3 / 3, 1 |

It sees them now; what it cannot do as well is tell them apart. On 60 flown
close looks at full camouflage the best any threshold could do was 77-91%
right, the ball worst: painted like the floor, an object is known by the
shape of what of it breaks the skyline alone, and the targets still rank
above the rest -- a slab at -0.10 against -0.21 to -0.35 -- but below zero.
Practice trades targets for fewer wrong landings: camouflaged objects look alike, and the
punishments for the many non-targets it lands on spread onto the targets.
At full dose it turned down nearly everything (0-2 of 12, no mistake); at
0.35 of a reward's dopamine (`CAMO_PUNISH`) it finds 9-10 of 12 slabs and
poles at half camouflage with at most one wrong landing. The ball, and full
camouflage, are not solved, and no camouflage brain is shipped.

**Looking from three sides did not pay.** If one close look is right 77-91%
of the time, three from different sides should be right more often, as
long as their mistakes differ. The drone can now fly sideways
(`DroneFlight.strafe`) and go round an object at the distance it is,
turning to keep it in the middle of the eye, and turning ahead of how fast
it wheels round; how far round it has gone is how far it has turned, so no
distance is needed (`Navigator.looks`, `around`). Going round to a place
worked out from a guessed distance failed: one look in two, it came round
to a column behind the object, or to nothing. Measured on 80 objects at
each camouflage, each looked at from 0, 45 and 90 degrees round:

- The looks from the side were judged worse than the first: the ball's
  second look told it apart at AUC 0.48-0.58, against 0.83-0.93 for the
  first. Their mistakes were mostly not the first look's (correlation -0.4
  to 0.7, near none for the ball), but the brain had been taught on looks
  from straight ahead only.
- Taught on the looks from the side too (the close-look lesson flown with
  all three), the ball's side looks rose to 0.68-0.76. But the mean of three
  told targets apart no better than the first look alone: from 0.03 worse
  to 0.03 better, kind for kind.
- In a mission three looks take three times as long, about 10 s an object.
  With the same brain on the same six fully camouflaged arenas, the drone
  decided on 4 objects a mission instead of 7, and found 4 balls and 6 poles
  of 12 instead of 8 and 9, with wrong landings in about the same share.

So the drone looks once (`CAMO_LOOKS`). What limits it is less how often it
looks than where it draws the line: up close, its targets are valued below
zero. Taught the three-look lesson and given a landing bar taken from it --
where that lesson's targets and the rest are told apart best -- it found 8 of 12
fully camouflaged balls, where the novice found 1, but with 11 wrong
landings in 6 missions instead of 3. That is a trade, not a cure, and it was
not kept.

**Peering, judging against the rest, and a bigger lesson.** Three more
tries, each measured on looks or arenas the brain had not been taught on:

- *Peering* (`Navigator.peer`): take the close look flying sideways,
  turning to keep the object in the middle, as a locust sways before it
  jumps. What is near moves across the eye unlike what is far behind it,
  and the 325 numbers a close look is made of include the optic lobe's
  motion cells (T4, T5). Taught on 300 peering looks and judged on 120 new
  ones, the pole was told apart better -- AUC 0.97 against 0.93 at full
  camouflage, 0.98 against 0.94 at half -- the slab and the ball no better
  (0.90 and 0.86 against 0.93 and 0.85). In 0.4 s the drone moves only 3 mm
  sideways; peering twice as fast for twice as long (16 mm) helped the pole a
  little more (0.98) and made the ball worse (0.77). Left off.
- *Judging against the rest* (`Navigator.relative`): land on a close look
  that is liked well above the ones turned down so far in the mission, and
  before calling the mission done, go back to the best one turned down if
  it stands out. Replayed over every close look of 18 practice missions at
  full camouflage, it was a trade at every setting: each extra target cost
  one or two wrong landings (balls 2 found and 2 wrong became 4 and 4; poles
  3 and 1 became 5 and 5). In a mission the target is often not the
  best-looking thing there: columns compete with the poles, slabs with the
  balls. Left off.
- *A bigger close-look lesson* (`CAMO_CLOSE_LOOKS`, `CAMO_CLOSE_DOSE`): this
  one helped. The camouflaged objects' differences are subtle, and more of
  them are learned from more looks. On 120 new flown looks, AUC with the
  lesson of 300 and of 900 (taught at 0.2 of the dopamine):

  | | full camouflage: 300 / 900 | half: 300 / 900 |
  | --- | --- | --- |
  | ball | 0.86 / **0.93** | 0.91 / **0.94** |
  | slab | 0.94 / **0.96** | 0.97 / **0.98** |
  | pole | 0.94 / **0.96** | 0.95 / **0.97** |

  More targets are also liked outright, the bar at zero: 83% of the balls
  instead of 68%, 91% of the poles instead of 83%. The lesson is now 900
  looks, flown in three parts of 300 (seeds 0, 1, 2), the old lesson its
  first part.

On the six new arenas (100-105), targets found of 12 and wrong landings
(columns included), for the novice taught the lesson of 300, the novice
taught the lesson of 900, and the latter after 25 practice missions:

| | half camouflage: 300 / 900 / practised | full: 300 / 900 / practised |
| --- | --- | --- |
| ball | 7, 0 / **9, 0** / 6, 0 | 1, 3 / **9, 7** / 1, 0 |
| slab | 9, 8 / 9, 5 / 6, 1 | 7, 4 / 7, 5 / 1, 0 |
| pole | 9, 3 / 8, 5 / **10, 1** | 6, 3 / **10, 5** / 6, 5 |

The bigger lesson finds more -- nine fully camouflaged balls of twelve where
the novice found one -- and still lands wrongly about once a mission at
full camouflage. Practice again buys fewer wrong landings with fewer targets
found. Camouflage is better, not solved.

Twice as big again did not pay. A lesson of 1800 looks (seeds 0-5) still
told new looks apart a little better -- AUC 0.94 for the ball at full
camouflage against 0.93, 0.98 for the slab against 0.96, 0.97 for the pole
against 0.96 -- but in missions it made no difference that showed: on twelve
new fully camouflaged arenas (100-111) the drone found 54 targets of 72 with
1800 and 53 with 900, and landed wrongly 29 times against 23. The lesson
stays at 900.

**Machine-learning techniques, tried on the camouflage brain**
(`docs/uczenie_maszynowe.md`, in Polish, has the survey and every number).
They were chosen on 1500 flown close looks kept for the purpose and checked
once on 240 others:

- *Classical learners as a yardstick.* SVM, random forest and gradient
  boosting told targets apart at AUC 0.98-0.999 on the same 325 numbers,
  where the mushroom body managed 0.95-0.98. There was more in the eye than
  the brain took out.
- *What helped the mushroom body:* dopamine as a prediction error
  (Rescorla-Wagner: `teach(..., aim=0.3, epochs=3)`, after Bennett,
  Philippides & Nowotny 2021), a sparser Kenyon-cell code (3% active rather
  than 5%), and the close looks at twice the dose. On the test set, at full
  camouflage, AUC 0.98 for the slab instead of 0.96 and 0.99 for the pole
  instead of 0.96. More Kenyon cells, more claws, bagged ensembles and
  mirrored views gave little or nothing.
- *Dataset shift.* That brain was worse in missions: 48 targets of 72 at full
  camouflage and 36 wrong landings, against 53 and 23. In a mission half of
  what the drone looks at closely is a landscape column, which the lessons --
  one object at a time -- hardly show it. So it flew 100 practice missions
  (arenas 4000-4099), turning every close look down to see everything, and
  the 712 looks it took, labelled from where things are, became a third
  lesson (`mission_looks`; dataset aggregation). On looks from missions it had
  not been taught on, the ball was told apart at AUC 0.93 instead of 0.91, the
  slab 0.92 instead of 0.90, the pole 0.98 instead of 0.96, and the columns it
  liked fell from 14-29% to 9-22%.

On the new arenas, targets found and wrong landings:

| | before | prediction error, lessons only | **prediction error + mission looks** |
| --- | --- | --- | --- |
| full camouflage, 12 arenas (72 targets) | 53, 23 | 48, 36 | **37, 8** |
| half camouflage, 6 arenas (36 targets) | 26, 10 | 27, 16 | **25, 5** |

At half camouflage it finds as many with half the wrong landings; at full
camouflage it is far more careful -- 0.22 wrong landings a mission instead of
0.64, at the price of 37 targets instead of 53. Of its landings 82% are on a
target, against 70%. It is the camouflage brain now.

A second round of mission looks did not pay. Flown by this brain in 100 more
practice arenas (4100-4199), 719 looks told targets apart only a little
better (AUC +0.002-0.007), and in missions on the same new arenas, with the
aims raised to give the targets back, it found 65 and landed wrongly 15 times
where this brain found 62 and landed wrongly 13: noise. Left out.

**Faster missions** (`dronebrain.PACE`). Two thirds of the drone's mission
went in looking around: at 90 deg/s a look around takes four seconds, and
after its last target -- found 30 s in, on average -- it looked on for as
long again before calling the mission done. Measured in 18 practice arenas
per target, with the expert brains: looking around at 135, 180, 225, 270 and
360 deg/s took the missions from 76 s to 62, 51, 43, 40 and 38 s (no faster
past 270: the yaw loop cannot follow), and one empty look before calling it
done instead of two saved another 3-7 s; flying faster saved 2 s more and
was left out. Every target was found and nothing wrongly landed on, except
one slab missed at 180 deg/s with two empty looks. The drone now looks
around at 270 deg/s and stops after one empty look. The lessons are still
flown at the old pace, the pace they were recorded at. On the new arenas:

| | before: found, wrong, mean mission | now |
| --- | --- | --- |
| no camouflage, 12 arenas (72 targets) | 72, 0, 73 s | **72, 0, 39 s** |
| full camouflage, 12 arenas (72 targets) | 37, 8, 87 s | **39, 7, 44 s** |
| half camouflage, 6 arenas (36 targets) | 25, 5, 88 s | **27, 5, 50 s** |

Missions take half the time, as accurate as before. Checked again on twelve
new arenas, tuned on by nothing (200-211): all 72 targets, no wrong landing, 40.6 s
a mission; at full camouflage 33 of 72, 6 wrong landings, 49 s. The fly looks
around at 240 deg/s already; one empty look instead of two (`FLY_PACE`) took
its missions on the same new arenas from 25.7 s to 22.8, 69 targets of 72
found either way, 4 wrong landings against 5.

**The drone's own camera: depth, not a sharper picture**
(`docs/kamera_drona.md`, in Polish). The brain sees through a fly's eye --
721 photoreceptors, 4 degrees apart, brightness only -- but a drone's camera
need not be a fly's. What was weighed:

- *More pixels, or a zoom.* The optic lobe takes 721 inputs, so more pixels
  can only be spent by spreading them over less of the world -- a zoom for
  the close look. Every lesson (900 close looks, 712 mission looks) was
  flown at 120 degrees and would be flown again, and a finer grid of the
  eye's middle (3x3, 5x5 instead of five regions) had told targets apart no
  better: under camouflage a sharper picture is a sharper picture of the
  floor's checks. Left for later.
- *Colour.* The camouflage here is brightness only, so a colour camera would
  undo it for nothing -- an artefact of the simulation, since real
  camouflage matches colour too. Left out.
- *Thermal infrared.* The simulation has no temperatures; any result would be
  whatever temperatures were written in. Left out.
- *A time-of-flight depth camera* (active infrared, the VL53L5CX / Arducam
  ToF class): it measures how far each ray goes, so how tall and how wide
  the thing in front is. Paint does not change that. Chosen.

`scene.depth` casts the camera's rays against the same solids the picture
is rendered from; `FlysEye.measure` is the depth camera -- 70 degrees, 24x24
rays, 80 mm of reach (4 m at the drone's scale), 1% noise -- and returns the
distance, height and width of what is solid nearest the middle of its view.
At a close look in a camouflage mission the drone measures what it is
looking at, turns down whatever is not its target's size however much the
brain likes it, and lands on what is its target's size if the brain likes
it by more than -0.05 (`Navigator.sizes`, `size_bar`). The sizes
(`autopilot.TARGET_SIZES`) were learned from 271 measured close looks in
practice missions (arenas 6000-6039): the ball 21-26 mm tall and 8-16 mm
wide, the slab as tall and 22-34 mm wide, the pole 35-42 mm tall and 3-11 mm
wide; the columns, taller than the camera sees, fit none. Judged on those
looks, the brain alone would have landed on 25 non-targets for 62 targets,
the brain and the size on 4 for 58.

In missions on the new arenas 200-211, targets found, wrong landings, mean
mission:

| | full camouflage (72 targets) | half camouflage (72 targets) |
| --- | --- | --- |
| no depth camera | 33, 6, 49 s | 50, 4, 46 s |
| turning down the wrong size | 29, 0, 42 s | 40, 0, 41 s |
| ... and landing on the right size from -0.05 | 40, 0, 42 s | 41, 0, 40 s |
| ... and two empty looks before it is done | **43, 0, 46 s** | **44, 0, 43 s** |
| a far measure that does not fit is not trusted | 45, 3, 52 s | 48, 3, 50 s |
| seeing by depth as well as by eye | 21, 0, 26 s | -- |

The drone now flies the fourth row: 87 targets of 144 against 83, and no
wrong landing in 72 missions against 10. Measured from far -- 50-75 mm --
the depth camera is not always on the object: the drone faces it only
roughly, and measures the column behind it (55 mm tall, 3 wide), which is
what costs the slabs at half camouflage. Trusting only near measures gives
those targets back, and three wrong landings with them (`size_near`); going
closer to measure again found no more and made missions slower (60 s).
Seeing by depth (`depth_sight`) made the drone call missions done too soon.
Both are left off. Of four ways to cut the object out of the depth picture,
the one used -- what joins the middle within 15 mm of its depth -- told
targets from the rest best.

What you watch is the validated physics, not an animation: the rigid
NeuroMechFly with blade-element wings is simulated, and a second copy of the
fly with its wing joints left in is posed from it every frame. Forward flight
is the hover loop with a pitch setpoint -- tilting the stroke plane tilts the
lift -- and it is stable: 8 degrees nose-down cruises at 216 mm/s and 15
degrees at 376 mm/s, roll inside 4 degrees, height held. It runs 20-40 times
slower than real time on a laptop, and says so.

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

## The halteres read the room

Teaching the fly to fly to things on its own (see flylab, above) needed it to
turn on the spot, and it could not: **every sustained turn ended the flight
once the heading passed about 90 degrees** -- fast or slow, saccadic or
smooth, hovering or flying forward, between 85 and 120 degrees every time.

`angular_rate`, the haltere signal, returned the body's rotation rate in the
room's axes. The pitch and roll angles were (correctly) measured about the
animal's own axes. The two agree only while the animal points along the
room's x axis, which is how every flight in this README starts. Turned 90
degrees, the pitch rate was fed to the roll loop and the roll rate to the
pitch loop -- a derivative term on the wrong axis, which is positive
feedback -- and the animal fell. The rates are now rotated into the heading
frame: yaw about the vertical, roll and pitch about the animal's own level
axes. A haltere senses in the body's frame, so this is also simply what the
organ reports.

| turn command | before | after |
| --- | ---: | ---: |
| 30 degrees every 0.4 s, hovering | fell at 1.06 s, heading 117 | 3.1 s, heading 240, roll under 7 |
| 45 degrees every 0.25 s, hovering | -- | 3.1 s, heading 542 |
| 120 degrees/s, hovering | fell at 1.05 s | 3.1 s, heading 354, roll under 3 |
| 90 degrees/s at 8 degrees nose-down | fell at 1.14 s | 3.1 s, heading 261, roll under 3 |

"Where it ends now", above, says the climbing flights fail with the heading
90-170 degrees off and attitude second. That was this. The default
controller's climbing flight, 1441 ms in the bandwidth table, now holds the
whole three seconds with its worst swing at 10.9 degrees. The tables above
were measured with the error in place and are kept as the record of what was
measured; they rank settings against each other under the same error.

The measurement sweeps (`pytest -m sweep`) were re-run on the fixed loop,
and five of the nine had to change, because flight time stopped telling
the settings apart -- most of these flights now last as long as they are
asked to. They compare the attitude error now, and what they found:

| sweep | before the fix | after |
| --- | --- | --- |
| sharp stroke against the sinusoid | 1441 against 1178 ms | both 3000 ms; RMS error 5.9 against 7.7 degrees: **still better** |
| stored authorities against the muscle stroke's own | stored ahead by 3% | both 2000 ms; RMS 7.9-8.2 against 10.5-10.8: **still better, by more** |
| one-wingbeat boxcar against a 6 ms low-pass | boxcar ahead at 40 and 60 rad/s | level at 40 (about 1 s each, ended by the climb); at 60 both 1.6 s, worst swing 7.6 against 10.5: **still better at 60** |
| the low-pass filter's optimum | a peak at 3 ms at all three amplitudes | **gone**: 0.93-1.06 s at every tau from 1.5 to 8 ms -- the climb ends these flights, and the peak was the frame error's |
| the yaw loop | worth 1.15-1.29 in flight time | **refuted on this rig**: both fly 2.5 s; climbing with no throttle the yaw knob saturates, the animal turns 1467 degrees with the loop and 902 without, and the loop costs attitude (RMS 5.2 against 1.7) |

The yaw loop does its job where it is meant to -- in level flight on the
throttle it turns the fly to any commanded heading, which is what flying to
a target needs -- but the free climbing flight this README's tables use
outruns it.

One test changed what it asserts. The wake-memory test recorded the
circulation lag ending the flight at 618 ms against a full second without
it. With the rates fixed both flights last the second, and the lag shows up
as what it is -- lost margin, the worst attitude swing going from 10.9 to
19.8 degrees -- and that is what the test now pins.

## What does not exist yet


## Running the tests

```bash
pip install -e ".[dev,body]"
pytest -q             # 174 tests, about forty minutes on one core
pytest -q -m sweep    # the 9 measurement sweeps, about half an hour
pytest -q -n 4        # with pytest-xdist: about ten minutes on four cores
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
