# Aircraft far-field extent: references and proposed baseline

Reference chord for the full-scale HL-CRM: 275.8 in × 0.0254 m/in = **7.00532 m**. Thus 100 reference chords is **700.532 m**. This scale is consistent with the approximately 59 m span of the supplied CAD; this is the CRM-HL configuration, not a verified Boeing 737 geometry.

## Directly relevant reference

Carolyn D. Woeber, Erick J. S. Gantt, Nicholas J. Wyman, *Mesh Generation for the NASA High Lift Common Research Model (HL-CRM)*, AIAA 2017-0363, DOI 10.2514/6.2017-0363. Section IV.A, printed page 4, specifies the far-field surface at 100 reference chords from the body and gives the HL-CRM reference chord as 275.8 in.

NASA-hosted paper: https://commonresearchmodel.larc.nasa.gov/wp-content/uploads/sites/7/2018/01/AIAA-2017-0363.pdf

## Corroborating and stricter guidance

- AIAA Fourth Drag Prediction Workshop, *Gridding Guidelines*, 3 November 2008, item 2(g): approximately 100 reference chords for all grid levels. https://www.aiaa-dpw.org/Workshop4/gridding_guidelines_4.html
- AIAA Seventh Drag Prediction Workshop, *Baseline RANS Grid Family Plan*, Revision B, 2022, slide 4: farther than 100 semispans. https://www.aiaa-dpw.org/Workshop7/DPW-VII_BaselineGridFamilyPlanRevB.pdf

The latter is substantially larger: for this supplied CAD's semispan of about 29.46 m, 100 semispans is approximately 2,946 m. A 750 m radius domain does **not** meet that stricter DPW-VII criterion. The two guidelines use different length scales and should not be treated as equivalent.

## Baseline choice

For the present CRM-HL mesh-generation exercise, use the directly applicable HL-CRM study's 100-chord distance as a baseline, with a nominal spherical boundary radius of **750 m**, centred at **(34, 0, 6) m**. The radius includes a margin for the aircraft's extent about that centre and for linear faceting of the sphere. The actual minimum boundary-radius/geometry-clearance bound must be verified on the generated mesh; a centre-based radius alone is not a body-to-boundary distance.

This is an engineering starting point, not a claim of domain independence. Once the intended Reynolds number, Mach number, angle of attack, and boundary treatment are specified, compare solutions on at least this domain and a larger domain, using lift/drag, surface pressure, and wake quantities of interest. If following the stricter DPW-VII guidance, choose a radius above approximately 3,000 m and check the actual surface clearance.

The near-body CAD fit and wall resolution of the existing prototype remain unfinished. Choosing a large far field does not resolve those limitations. The outer boundary tag in a mesh export does not by itself specify appropriate Nek inlet, outlet, freestream, or characteristic conditions. The reference studies' viscous-layer growth recommendations should also not be claimed as achieved by a coarse far-field blocking prototype.
