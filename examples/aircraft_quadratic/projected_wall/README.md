# CAD-projected nose surface

The full aircraft wall is stored in this directory. The companion export includes a dense CAD-projected nose preview.

aircraft_wall_nose_projected_Q2.vtu contains the full aircraft wall with its
original nine-node quad connectivity. Corner nodes, midside nodes and face
centers in the nose and forebody were all projected directly to CAD through
x = 6.2 m. The update tapers to the previous wall by x = 10 m. The rest of the
aircraft retains the wing-root trial geometry. Coordinates are in metres.

At z = 5.0292 m and |y| <= 0.95 m, the largest sampled streamwise differences
from the CAD section are 47.0 mm for the previous trial, 16.2 mm for the
corrected quadratic interpolation and 2.1 mm for the dense projection.
These figures compare the exported sections with the CAD tessellation.
The maximum Q2 wall-node movement is 36.8 mm. The modified quadratic
patches had positive sampled orientation and no detected crossings in a
6 by 6 per-face intersection screen against the full wall. The dense patch
has no nonmanifold edges and passed a sampled orientation comparison.

These are surface files. The corrected coordinates have not been propagated
into a NekRS volume mesh or a restart field.
