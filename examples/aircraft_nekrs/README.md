# Aircraft mesh test: NekRS v26.0

Incompressible, no-slip implicit LES using HPFRT regularization, **without a wall
model**. This is an initial mesh/solver robustness test, not a validated
wall-resolved LES or CRM high-lift benchmark. The aircraft is our simplified
full-aircraft CAD, not the tutorial's deployed slat/flap/nacelle configuration.

## Scales and condition

NASA tutorial: https://www.nas.nasa.gov/LAVA/unst_docs/tutorials/crm_highlift/crm_highlift/

Borrow its reference chord `Lref = 7.00532 m`, `Re = 5.9e6`, and nominal
`M = 0.2`, `Tref = 288.15 K`. Use **alpha = beta = 0 degrees**, rather than
the tutorial's near-maximum-lift alpha = 19.7 degrees. The metre-valued accepted
CAD mesh is scaled directly; we do not shrink it to the 5.2% tunnel scale.
The borrowed chord is a chosen normalization, not a measured mean aerodynamic
chord of our simplified aircraft. No CRM reference area is used for forces.

With ideal-air `gamma = 1.4` and `R = 287.05 J/(kg K)`:

- `Uref = 0.2 sqrt(gamma R Tref) = 68.05846 m/s` (approximately).
- `tref = Lref/Uref = 0.10293 s` (approximately).
- `x* = x/Lref`, `u* = u/Uref`, `t* = t Uref/Lref`.
- `p* = (p - pinfinity)/(rho_ref Uref^2)`; density scale is arbitrary for this
  constant-density incompressible calculation.
- `rho* = 1`, `mu* = nu* = 1/Re = 1.694915254237288e-7`.
- Dimensional equivalent `nu = Uref Lref/Re`, approximately `8.0809e-5 m2/s`.

Mach and temperature only define a dimensional interpretation here; neither is
an input to the incompressible pressure/velocity equations. There is no energy
or scalar equation. The geometry is linear (trilinear hex maps), even when the
solution polynomial degree is 7. No further CAD curvature/projection is applied.

## Boundary IDs on the new RE2

| ID | Faces | Velocity boundary | Pressure treatment |
|---|---:|---|---|
| 1 | 10856 | `zeroDirichlet`: no-slip, impermeable aircraft | solver-derived pressure Neumann |
| 2 | 4434 | `udfDirichlet`: freestream `(r(t*),0,0)` | solver-derived pressure Neumann |
| 3 | 6422 | `zeroNeumann`: open outflow | solver-derived prescribed pressure, `p*=0` |

Flow is in **+x**, from the aircraft nose toward its tail. The sphere is not a
wall and no symmetry condition is applied anywhere to this full-aircraft case.
Pressure conditions are inferred by v26 from the velocity map; do not add a
separate pressure `boundaryTypeMap`.

Each original outer bilinear face is classified using the outward area-vector
x component over the *whole face*. Only faces strictly positive everywhere
are outlets. Faces containing any inflow, a tangent corner, or mixed signs
are prescribed freestream. There are 470 mixed-sign faces included in ID 2.
This is a facewise approximation of the spherical hemisphere split; it needs
no element cuts or hanging interfaces and does not change geometry.
The derivative cross product is affine in the face parameters, so its four
corner values bound its x component throughout each bilinear face.
The classification is for zero incidence only; regenerate it for a changed
freestream direction rather than just editing the inlet vector.

The outlet is a standard open pressure outlet, not a characteristic condition
or a backflow-stabilized model. Monitor actual outlet reverse flow in later,
longer simulations. The sphere is very remote from the aircraft.

## Prepare the mesh

From the repository root, install the package (`python3 -m pip install -e .`).
Use `output/tail_refined.npz` from the accepted tail-refinement package:
https://chatgpt.com/library?file_id=libfile_6a5a95a330f881919628c44ebb6d931d&account_id=personal

```bash
python3 examples/aircraft_nekrs/prepare_mesh.py /path/to/output/tail_refined.npz
```

This writes `examples/aircraft_nekrs/aircraft.re2` and a mesh report. It verifies
the approved coordinate SHA, all 60 radial layers' hex connectivity/order,
outward farfield orientation, exact scaled RE2 corner roundtrip, and binary
boundary records. It leaves all 651360 elements and connectivity intact.
Do **not** run with the earlier dimensional RE2 (one combined farfield ID).
No runtime coordinate scaling is performed: scale exactly once in this exporter.

## Run

Use a built NekRS **v26.0** environment. From this directory:

```bash
mpirun -np <ranks> nekrs --setup aircraft_smoke.par
mpirun -np <ranks> nekrs --setup aircraft.par
```

Choose ranks/backends for your machine through your usual launcher/environment.
`aircraft_smoke.par` uses degree 3, cubature degree 5, and stops at 100 steps.
It is only a startup/connectivity/BC smoke test; it may stop before the inlet
ramp completes and is not an LES resolution study. The main degree-7 case uses
cubature degree 11 and stops at `t*=0.1` (about 0.0103 physical seconds).
This is also a short startup test, not statistically developed flow. Extend
`endTime` after reviewing startup; do not interpret its coefficients as converged.
Both runs start independently from rest by default, not from each other's files.

A cubic smoothstep inlet ramp reaches unit speed at `t*=0.01`, reducing the
impulsive wall/inlet startup. `dt` starts at `1e-7`, targets CFL 0.5, and is capped
at `1e-4`; the solver's measured CFL, not this cap, determines the usable step.
Keep the conservative startup values until actual CFL and convergence are known.
At degree 7 the audited minimum within-element GLL spacing is about
`5.7825e-6` nondimensional (40.51 micrometres dimensional), so tiny elements can
make the time step much smaller than the cap. No small-cell filtering is used.

The UDF uses the v26 `nrs->fluid`, `deviceMemory`, `bcData::uxFluid` and
`isField("fluid velocity")` APIs. It initializes velocity and pressure to zero
only for a cold start. It logs integrated outward boundary fluxes every 50 steps:
wall flux should be zero, inlet negative, outlet positive after startup, and
their sum should approach the incompressible mass-balance tolerance. Also inspect
native CFL, divergence, linear iteration counts/residuals, finite solution fields,
and root/tip/tail flow slices. Reject growing numerical oscillations or persistent
mass imbalance before lengthening the run.

HPFRT is numerical subgrid dissipation (one highest mode, scaling coefficient 1),
not an explicit eddy-viscosity model. No synthetic inlet turbulence or trips are
included. No wall resolution/y+ or aerodynamic accuracy is claimed at Re=5.9e6.
The 651360-element degree-7 case has about 333.5 million element-local velocity
points per component; use an appropriately sized parallel allocation.

## Validation scope and primary API references

Checked against NekRS tag `v26.0`, commit
`96b3cf9e5bacede16568826c04a21bc0fe50dc7d`:

- `examples/turbPipe/turbPipe.par` and `.udf`: v26 sections, HPFRT,
  target-CFL syntax, boundary callback, initialization, surface-flux API.
- `src/platform/par/parseGeneral.hpp`, `parseMesh.hpp`, `parseFluid.hpp`:
  accepted general/mesh/velocity/pressure entries.
- `src/core/bdry/bdryBase.cpp`: `zeroNeumann` is the open-outflow type.
- https://github.com/Nek5000/nekRS/tree/v26.0

`validate_inputs.py /path/to/nekRS-v26.0` checks input keys/selected values against
the v26 source and boundary maps against the actual exported binary RE2.
This source/static check is not a compiled UDF or MPI run. NekRS is not installed
in the preparation environment, so compilation and solver startup remain to be
verified on the target installation. Uniform coordinate scaling preserves the
previously audited Jacobian signs; no new mesh deformation has been introduced.
