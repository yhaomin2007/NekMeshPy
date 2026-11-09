# Wing-root CFL mesh trial

This directory contains the exact coordinate patch for the 651,360-hex
wing-root trial, together with its NekRS v26 case files. The patch applies to
the accepted quadratic case packaged as `aircraft_quadratic_nekrs_v26.zip`.
It modifies the first six radial layers. The RE2 header, boundary records,
element ordering and quadratic displacement sidecar match that baseline.

From this directory, with the baseline archive already extracted:

```bash
python3 apply_mesh_patch.py /path/to/quadratic/aircraft.re2 ./aircraft.re2
cp /path/to/quadratic/aircraft_q2.bin .
sha256sum -c SHA256SUMS
```

The reconstruction checks both the baseline and the result by SHA-256.
Use `aircraft_smoke.par` for a NekRS startup check.

All 8,352 modified hexes passed whole-element Jacobian positivity checks.
A 6 by 6 wall tessellation found no crossings involving the modified patches.
Minimum axis-adjacent GLL spacing at degree 7 is 35.4 µm. Wall quads 1762 and
6946 have local normal rotations up to 92° relative to the baseline at matched
parametric points. The largest added wall displacement is 3.75 mm, with no
additional CAD projection. The case is intended for comparative CFL testing.
