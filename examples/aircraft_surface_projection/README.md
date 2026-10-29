# Approved aircraft linear-surface projection: source checkpoint

The corrected projected surface was approved after VTU review on 2026-10-28. Volume deformation remains pending. This source-only checkpoint contains the CAD projection, tip-cap and trailing-edge repairs, validation, visualization and VTU export scripts. It does not include CAD, mesh inputs, generated geometry, or validation data.

## Reproduce

Install requirements.txt. Obtain the matching approved surface checkpoint separately and place its reference.brep and layout_macro.npz beside these scripts, plus the supplied output/surface_input.npz, output/base_surface.npz, output/cad_atlas.npz and output/body_centerline.npz under output/. These are the original checkpoint inputs, not the earlier far-field volume mesh.

Run `python rebuild.py` from this directory. The script remaps caps and adjoining strips, extends trailing-edge correspondence across the proper CAD segments, redistributes affected chordwise rows, resolves trimmed-face boundary tolerances, validates the surface, renders close-ups, and exports VTP and VTU. It keeps approved junction nodes fixed and preserves quad connectivity.

The scripts are specific to this CAD face numbering and layout. Surface orientation and intersections use finite sampling; they are not volume Jacobian checks or an exact global injectivity proof. The final reports record local linear-face deviations and whether the rebuilt coordinates match the approved checkpoint. Changed geometry requires validation and review before volume work.

No valid volume mesh conforming to the approved surface has yet been delivered. The earlier aircraft_farfield example remains a historical volume baseline.
