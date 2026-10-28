# Full-aircraft all-hex far field: global redistribution revision

This revision redistributes the complete outer angular grid and its connecting volume, addressing both the narrow fan above the tails and squeezed cells in their surrounding neighborhoods. It preserves every original near-body point and hex, the full-aircraft topology, element count, and nominal 750 m outer radius. Coordinates are metres.

## What changed

The previous sphere parameterization compressed the lifting-surface patch rings into narrow angular sectors. Those sectors were then retained throughout the radial far field. Earlier volume optimization mainly enforced positive sampled Jacobians, leaving abrupt changes in the direction of some transition trajectories.

The new sphere distribution minimizes global edge-length and face-area variation, within-cell edge imbalance, and corner skew, while retaining an orientation barrier. The 24-layer connecting volume is subsequently optimized with sampled orientation/skew constraints and a penalty on second differences along its shared-node trajectories. Inner interface nodes are fixed; all patches continue to share their interface nodes. The terminal sphere stays at 50 m; 30 outer radial layers then expand to 750 m with an outer radial step ratio of 1.18. That ratio does not describe every optimized transition spacing.

The delivered transition_geometry.npz is the selected balanced checkpoint after global angular redistribution, untangling, trajectory smoothing, and a final quality-recovery stage. It is the authoritative geometry for rebuilding this revision.

## Measured comparison

| Metric | Prior tail revision | This global revision |
|---|---:|---:|
| Whole-mesh minimum scaled Jacobian, order 3 | 0.086566 | 0.086566 |
| Transition minimum, order 3 | 0.112038 | 0.097398 |
| Tail transition minimum, order 3 | 0.119964 | 0.110984 |
| Maximum turn between adjacent tail macro trajectory segments | 25.35 degrees | 20.91 degrees |
| Maximum adjacent segment-length ratio on those trajectories | 1.319 | 1.219 |
| Outer-sphere edge-length 95th/5th percentile ratio | 5.202 | 3.835 |
| Outer-sphere face-area 95th/5th percentile ratio | 5.926 | 2.214 |
| Maximum edge ratio within an outer-sphere macro face | 10.367 | 6.668 |
| Transition edge-aspect-ratio maximum | 192.4 | 95.5 |
| Outer-grid edge-aspect-ratio median / 95th percentile | 3.769 / 8.223 | 3.121 / 7.061 |

Tail denotes H_* and V_* patches and their J_H_root / J_V_root interfaces. Trajectory metrics use shared macro nodes across the 24 transition layers; they are not a global fine-cell stretch bound. Percentile ratios measure distribution uniformity and are lower when more uniform. Scaled Jacobian here is the normalized determinant used by NekMeshPy, distinct from the min(J)/max(J) quantity printed by Nek solvers. Raw values and precise definitions are in output/redistribution_comparison.json.

## Mesh and checks

- 326,400 hexes, 331,962 points, 6,800 logical blocks (4 x 4 x 3 elements each).
- 32,640 original near-body elements, 130,560 transition elements, 163,200 outer elements.
- All 5,440 near-body/extension interface faces match exactly.
- One connected, watertight, conformal component; no nonmanifold faces or hanging nodes detected.
- GLL sampling at polynomial orders 3, 5 and 7 found no nonpositive scaled Jacobians. This is a finite sample check, not a certificate between all sampled points.
- The macro polyhedral overlap screen flagged seven pairs. Tests of every pair of their actual fine-element children found zero polyhedral overlaps. The macro corner approximation differs from the actual refined faces; flags remain recorded. This diagnostic is not an exhaustive fine-element intersection proof.
- Exported RE2 corners match the source element coordinates exactly; boundary records were read back and checked. No coordinate-tolerance welding is used for that comparison.
- The conservative CAD-to-faceted-far-boundary clearance bound is 713.015 m, or 101.782 reference chords. The nominal sphere centre is (34, 0, 6) m.

See output/checks.json and the two pair-audit JSON files for the complete records. No Nek solver run has been performed.

## Viewing

Open output/aircraft_farfield.vtu in ParaView and use Slice or Clip. Cell fields include region (1 near body, 2 transition, 3 outer), block_id, extension_layer and scaled_jacobian_order3. The PNG previews are cuts of that actual volume mesh. output/tail_extrusion_comparison.png compares it with the prior version.

After running `python examples/restore_large_files.py` from the repository
root, decompress the solver mesh before use:

```bash
gzip -dk output/aircraft_farfield.re2.gz
```

The reconstructed `output/aircraft_farfield.re2` contains linear corner
geometry. Boundary ID 1 is aircraft_wall (W); ID 2 is farfield (O as an export
placeholder). Appropriate external-flow boundary conditions must still be
configured; this is not a runnable Nek case.

## Rebuild and verification

From the NekMeshPy repository root, install the package and this example's
additional dependencies:

```bash
pip install -e .
pip install -r examples/aircraft_farfield/requirements.txt
cd examples/aircraft_farfield
python build_mesh.py
python verify_mesh.py
python visualize_mesh.py
python visualize_mesh.py --top
```

These commands use the saved transition and need no JAX. The full intermediate output/aircraft_farfield.npz is generated by build_mesh.py and is omitted from this download to keep it compact. Rebuild first before re-running verify_mesh.py. compare_tail.py accepts --baseline PATH_TO_PREVIOUS_VTU; the previous mesh is not duplicated in this package.

Optional optimization sources are included. `redistribute_global.py` creates the globally balanced angular checkpoint; with JAX CPU installed, the connecting volume can then be optimized in successive quality/smoothing stages. The delivered checkpoint already contains the selected result, so these steps are needed only for experimentation.

```bash
python redistribute_global.py
python optimize_transition.py --init previous_transition.npz --sphere sphere_global.npz --out stage1.npz --grid 3 --quality .12 --smooth .00002 --iterations 1800
python optimize_transition.py --init stage1.npz --sphere sphere_global.npz --out stage2.npz --grid 5 --quality .14 --smooth .00002 --iterations 1000
python optimize_transition.py --init stage2.npz --sphere sphere_global.npz --out stage3.npz --grid 5 --quality .13 --smooth .001 --iterations 1000
python optimize_transition.py --init stage3.npz --sphere sphere_global.npz --out candidate.npz --grid 5 --quality .12 --smooth .0002 --iterations 1000
```

To try such a new transition, use `build_mesh.py --bridge candidate.npz`. Before running `verify_mesh.py`, save the same selected geometry as `transition_geometry.npz`, because verification reconstructs its macro maps from that file. Every changed geometry needs the full checks repeated. The supplied `sphere_global.npz` records the selected angular redistribution; re-optimizing the sphere is not required for rebuilding.

## Domain choice and remaining work

FARFIELD_REFERENCES.md explains the HL-CRM 100-reference-chord criterion and the larger DPW-VII criterion of more than 100 semispans. This 750 m sphere follows the former and does not satisfy the latter. A domain sensitivity calculation is still required for the intended flow.

The near-body wall remains the inherited idealized approximation, with the prior CAD-distance audit reporting a maximum discrepancy of about 1.47 m. This revision does not finish CAD projection, high-order curved geometry, boundary-layer or wake resolution, or flow conditions. It improves the existing radial/block topology; it does not add a dedicated streamwise wake-block system. This remains a mesh-generation prototype for review, rather than a validated production CFD mesh.
