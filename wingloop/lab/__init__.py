"""flylab: the fly's flight, brain, vision, learning and a drone built on them.

The pieces other people have built properly are imported rather than
rewritten:

- **vision** -- ``flyvis`` (Lappalainen et al., *Nature* 2024): the pretrained,
  connectome-constrained model of the fly's optic lobe, 45,669 neurons in 65
  cell types over 721 columns.
- **body and flight** -- this repository's ``wingloop.body``: NeuroMechFly in
  MuJoCo with blade-element aerodynamics and the haltere and throttle loops.
- **brain** -- the MaleCNS connectome readouts in ``wingloop.brain``.
- **learning** -- the mushroom-body rule ``flyloop`` uses: dopamine depresses
  the Kenyon-cell to output-neuron synapses that were active.

What is new here is the glue, the object-recognition readout, and a
quadrotor that flies on the fly's control laws.
"""
