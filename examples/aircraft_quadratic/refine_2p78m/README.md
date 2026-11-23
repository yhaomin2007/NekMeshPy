# Refine existing curved aircraft mesh (no CAD)

This code reproduces the 2026-11-23 aircraft mesh refinement from the existing `aircraft.re2`, `aircraft.usr`, and `aircraft_q2.bin`. Only scripts and documentation are tracked; generated mesh binaries and visualizations are intentionally excluded.

## Geometry and target

- Input: **651,360 hexes**, 10,856 wall quads, 60 wall-normal layers.
- Tangential subdivision: every parent wall quad -> **2×2** children.
- Radial layer splitting: parent layer 0 -> 5 graded layers, original remaining layers preserved.
- Output: **2,779,136 conformal hexes**, 43,424 wall quads, 64 radial layers.
- The original curved wall is reconstructed using the RE2 + USR + Q2 correction mapping; the code **does not** project to CAD.
- Graded first-parent-layer knots: `[0, .018, .065, .19, .46, 1]`.
- Target Reynolds number: 1.5e6; GLL polynomial order: N=7.

## Dependencies and paths

Python 3 with NumPy is required. `export_surface.py` additionally requires Python VTK.

**The uploaded, verified generation scripts currently contain absolute working paths.** Before running in a different environment, edit the `SRC`, `OUT`, or `P` constants at the top of each script:

- Input path: `/mnt/data` holding the three input geometry files.
- Output path: `/mnt/data/aircraft_refined_regenerated`.
- The independent parent mapping check also references input files under `/mnt/data`.

Run from an environment with adequate memory and free disk space (the refined RE2 alone is ~561 MB):

```bash
python build_refined.py
python validate.py
python check_conformity.py
python check_parent_mapping.py
python export_surface.py
```

The output files are `aircraft.re2`, `aircraft_q2.bin`, `aircraft.usr`, `validation.json`, `conformity.json`, `reconstruction.json`, and `aircraft_wall_quadratic.vtu`.

## Previous validation from the delivered run

| Metric | Value |
|---|---:|
| Evaluated Jacobian sample points | 32,133,760 |
| Nonpositive sampled determinants | 0 |
| Minimum sampled scaled Jacobian | 0.01118836 |
| Unique wall edges / paired incidences | 86,848 / 2 per edge |
| Quadratic edge-midpoint mismatch | 0 |
| Radial interface-corner mismatch | 0 |
| 4,000 parent-mapping checks, maximum geometric error | 5.02e-13 m |
| Estimated first GLL y+ (median / P95 / P99 / maximum) | 0.436 / 0.965 / 1.402 / 4.989 |

**Limitations:** these checks are sampled, not full whole-cell Bernstein certifications; some inherited cells are significantly skewed. The wall-unit values use an approximate turbulent flat-plate skin-friction coefficient, not simulated local shear. This mesh has not yet been runtime validated in NekRS on Aurora.

The RE2 stores linear corner geometry and boundary records; the quadratic geometry is reconstructed in `usrdat2`. The generated `aircraft.re2`, `aircraft.usr`, and `aircraft_q2.bin` must be deployed together.

## NekRS/Aurora smoke test

1. Make a backup of the working original case, and deploy the three generated geometry files together into a new run directory.
2. Use the already-working `.par` and `.udf` files, ensuring the generated `aircraft.usr` is compiled/loaded.
3. Confirm 2,779,136 elements and 521,088 Q2-curved cells at startup.
4. Check geometry validity, wall/far-field BC IDs, solver setup, and minimum stable timestep with a conservative CFL.
5. Plot the wall VTU over the fluid fields in VisIt/ParaView. Evaluate actual wall shear and wall-parallel GLL spacings before production wall-resolved LES.

**No mesh binaries are committed to this repository.** The code was copied from the regenerated, audited delivery package.
