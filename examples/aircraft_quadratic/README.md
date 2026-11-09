# Aircraft mesh with quadratic wall geometry

This candidate uses the accepted 651,360-element aircraft connectivity. It adds
shared quadratic wall edge midpoints and one quadratic node per wall face.
The CAD atlas supplies the initial face association on the positive aircraft
side. The selected trimmed B-rep face is evaluated in native coordinates and
the targets are reflected for the opposite side. A quality limiter scales
individual shared node displacements where the full displacement would leave
too little Jacobian margin, compress the local GLL spacing, or raise the
uniform-flow CFL estimate past the allowed margin. Original wall corners retain their prior positions
and CAD errors. The new wall VTU gives the fraction of each CAD correction
applied and the residual distance to the selected CAD target in metres. Its
original wall vertices have cad_target_available = 0 and carry no new CAD error
measurement. A small group of wing patches retains its linear geometry
where the initial curving intersected nearby opposing patches.

The wall displacement decreases through eight layers and vanishes at their
outer interface. The volume map is quadratic in both directions along each
wall quad and linear in the extrusion coordinate. The outer work redistributes
the narrow tangential bands through the interior shell, preserving connectivity
and the existing outer boundary coordinates and inlet/outlet IDs.

## Run the NekRS v26 case

Run from the extracted case directory with your usual MPI/GPU launcher.
For example:

```bash
mpirun -np <ranks> nekrs --setup aircraft_smoke.par
```

The smoke case uses polynomial degree 3. The main case, aircraft.par, uses
degree 7. Both load aircraft.usr. The RE2 contains corner coordinates, while
usrdat2 reads aircraft_q2.bin and evaluates the quadratic correction on the
solver grid. Keep the RE2, USR and binary file together. The binary contains
little-endian float64 displacements for the five higher-order points in each
wall quad, in wall element order. The solver coordinates and corrections are
already nondimensionalized by 7.00532 m. The standalone wall VTU uses metres.

Start from the supplied uniform freestream initialization, or explicitly
interpolate a prior solution onto the new geometry before using it as initial
data. An old checkpoint indexed by the same element IDs belongs to the old
geometry.

## Validation and comparison

- validation.json gives the Jacobian and CAD correction results.
- layer_metrics.csv contains matched degree-7 GLL metrics by extrusion layer.
  Spacing is the minimum distance between adjacent GLL points along a reference
  axis. The CFL proxy is computed with uniform nondimensional velocity
  (1,0,0), local inverse Jacobians, and the NekRS GLL spacing stencil.
  Actual solver CFL depends on the evolved velocity and chosen time step.
- The Bernstein check bounds the Jacobian determinant on each reference
  element, with recursive subdivision of ambiguous coefficients and a
  floating-point safety margin.
- The section comparison visualizes the linear corner grids in the same
  physical plane. The VTU shows the nine-node curved aircraft wall.
- The case files were checked against the available v26 source. A NekRS
  startup and a run-to-run CFL comparison still need to be performed on the
  target system.

## Reproduce

Place the accepted tail-refinement package, the surface-tip review package
(with the aircraft B-rep and CAD atlas), and the original NekRS case in a
recovery directory. From the working directory, use the supplied mesh_upgrade
scripts. The Python workflow needs NumPy, SciPy, CadQuery, VTK and Matplotlib.

```bash
export OPENBLAS_NUM_THREADS=1
python mesh_upgrade/project_bounded.py recovery
python mesh_upgrade/repair_targets.py recovery
python mesh_upgrade/outer_coherent.py recovery
python mesh_upgrade/build.py recovery
python mesh_upgrade/build.py recovery --intersection-guard
python mesh_upgrade/export.py recovery --guarded
python mesh_upgrade/plot_sections.py recovery
python mesh_upgrade/plot_metrics.py
python mesh_upgrade/audit_wall.py
```
