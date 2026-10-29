# Aft-fuselage directional refinement

The user's review identified overly stretched axial cells at the aft fuselage. Radial-layer smoothing alone cannot fix insufficient axial resolution. This revision subdivides the long surface-edge directions by 2 or 4, propagates opposite-edge counts through the connected quad mesh, and applies the same subdivisions through all 60 radial volume layers. No hanging faces are introduced.

## Changes and measured result

- Hexes: 529,920 -> 651,360 (+22.92%). Points: 538,874 -> 662,338. Wall quads: 8,832 -> 10,856.
- Targets: 90 aft-body quads with centre x>60 m and edge aspect ratio>8. Conformal propagation affects 920 parent wall quads, including attached tail sheets.
- Target-region edge aspect ratio (quad-count statistics): median 11.219 -> 2.809; p95 30.012 -> 5.644; maximum 40.694 -> 10.173. The target region has 90 parent quads and 332 children.
- All aft-body quads with centre x>58 m: median 4.220 -> 2.679; p95 16.987 -> 5.745; maximum 40.694 -> 10.173. Counts are 424 before and 864 after.
- The original coordinates are bitwise retained. Every child is the exact tensor restriction of its accepted parent trilinear volume map; all child corner restriction errors measured zero in the saved checkpoint. The underlying bilinear surface geometry is unchanged, including approved junctions and tips.
- New surface nodes evaluate the existing bilinear surface. They are not separately projected to CAD. The previously accepted linear-CAD approximation is retained; no additional geometry relaxation is introduced.

## Validation

All 651,360 hexes pass analytic trilinear Jacobian sampling at GLL N=3,5,7. Minimum normalized determinants are 0.0121763, 0.0108292 and 0.0103541, respectively, unchanged from the accepted parent checkpoint. No nonpositive sampled elements were found.

The mesh has one connected component and is conformal/watertight: zero hanging points, open edges or nonmanifold faces. Minimum within-element N=7 GLL pair spacing remains 40.5085 micrometres, including non-axis pairs. This is not a global all-element point-cloud nearest-pair search.

Nonadjacent-overlap evidence is inherited from the accepted parent screen (7,424,305 AABB candidates, 52 polyhedral flags checked with bidirectional 9x9x9 trilinear samples, zero strictly interior samples). Exact tensor subdivision partitions each parent map and does not introduce geometric overlap. Parent sampling limitations remain; this is not an exhaustive exact intersection certificate.

RE2 corner coordinates were read back and match the selected mesh exactly. No flow solver run has been performed. Thin trailing-edge sheets outside the target region remain anisotropic; this revision does not claim to remove all mesh anisotropy.

## Contents and reproduction

This source-only repository checkpoint excludes mesh inputs and generated outputs. Copy the matching `inputs/` and, for checkpoint review, `output/` directories from the separately supplied aircraft_tail_refined_package archive.

`inputs/volume_parent.npz` is the accepted 529,920-hex input; `inputs/parent_intersection_audit.json` binds its overlap screen to the same coordinate hash. `output/tail_refined.npz` contains the refined mesh, original-to-child mapping and surface parameter restrictions. JSON reports and the before/after image are included.

Install requirements.txt and run `OPENBLAS_NUM_THREADS=1 python rebuild.py` from this directory. Install this repository with `pip install -e .`; the scripts import the repository package. Rebuild uses no CAD solver and needs no old workspace.

The separately supplied aircraft_tail_refined.vtu includes parent-cell IDs and quality fields. aircraft_tail_refined.re2.gz is linear corner geometry with wall ID 1 and farfield ID 2; W/O are export placeholders and require the intended flow boundary conditions.

Run `python export_wall.py` for the wall-only VTU with parent surface quad IDs, volume-node IDs and edge aspect ratios.
