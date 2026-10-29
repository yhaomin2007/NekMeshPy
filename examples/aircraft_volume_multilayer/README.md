# Aircraft multilayer volume deformation — source checkpoint

This stage deforms the existing 529,920 linear hexes to match the approved projected surface. It preserves wall coordinates and connectivity, adjusts the first 30 radial layers, and fixes interface 30 and all outer interfaces. Quadratic geometry remains future work.

## Algorithm and validation

`harmonic.py` distributes displacement through 30 layers. `optimize.py` untangles shared-node hexes with explicit radial-thickness and derivative-length constraints; a second pass checks interior GLL samples. `scan.py` evaluates every element at GLL N=3,5,7. `gll_spacing.py` checks actual within-element GLL nearest-pair spacing, including non-axis neighbours. Topology, nonadjacent-overlap screening, Jacobian crosschecks, geometry-change records, section views, VTU export and RE2 corner round-trip verification are included.

The delivered checkpoint passed the finite geometry checks: minimum scaled Jacobians 0.01218 / 0.01083 / 0.01035 at N=3 / 5 / 7, minimum within-element N=7 spacing 40.5 micrometres, no hanging points and no detected intersections. No additional surface deviation was introduced; the approved surface retains its previously documented linear-CAD deviations. No flow solver run has been performed. Finite sampling is not a proof of global injectivity between samples.

## Inputs and execution

This public directory contains source code only. CAD/mesh data, generated outputs and checkpoint files are supplied separately in the aircraft_volume_multilayer package.

1. Install NekMeshPy from this repository: `pip install -e .` from the repository root.
2. Install additional dependencies: `pip install -r examples/aircraft_volume_multilayer/requirements.txt`.
3. Copy the package's `inputs/` directory here. It contains `volume_input.npz` (matching volume coordinates/connectivity/radial mapping) and `approved_surface.npz` (approved surface and wall correspondence). Do not substitute an earlier far-field mesh.
4. Run `OPENBLAS_NUM_THREADS=1 python rebuild.py` from this directory.

To inspect the supplied checkpoint without rerunning optimization, also copy its `output/` directory here, then run `python verify_checkpoint.py`. Use `python export_preview.py` for VTU and `python export_re2.py` for RE2. Dependencies resolve to this repository's NekMeshPy package; the vendored copy from the standalone package is not needed here.

RE2 wall/farfield boundary IDs are 1/2. W/O codes are export placeholders; configure the intended flow conditions. The minimum spacing is smaller than the historical volume baseline and should be accounted for when choosing a timestep.
