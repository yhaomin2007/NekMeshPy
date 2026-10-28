#!/usr/bin/env python3
"""Template-based pipe-network mesher using NekMeshPy, not a general CAD mesher.

Run: PYTHONPATH=/path/to/NekMeshPy python build_network.py network.json
The existing equal-bore tee template is extended by automatic compatible lofts.
Dimensions, bend placement, layer counts and interface joins come from network.json.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from nekmeshpy import hexmesh, quadmesh, writer, reader
from nekmeshpy.core.fields import geometric_spacing
from nekmeshpy.hexmesh import Seam
import junction_template as tee


def rotation(axis, angle):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)


def build(cfg, out):
    if cfg['units'] != 'm':
        raise ValueError('This demonstration expects metres.')
    m = cfg['mesh']
    for k, value in cfg.items():
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError(f'{k} must be positive')
    if m['half_ring_elements'] % 4 or m['half_ring_elements'] < 4:
        raise ValueError('half_ring_elements must be a positive multiple of 4')
    if m['geometry_order'] < 1 or m['radial_layers'] < 1 or m['target_axial_length'] <= 0:
        raise ValueError('Invalid mesh resolution')
    R = cfg['junction_diameter']/2
    L, H = cfg['junction_main_half_length'], cfg['junction_branch_height']
    if min(L, H) <= 2*R:
        raise ValueError('Tee legs must be longer than their diameter in this template')
    for side in ('main', 'branch'):
        if cfg[f'{side}_bend_radius'] <= cfg[f'{side}_outlet_diameter']:
            raise ValueError('Use a bend radius greater than the pipe diameter')
    target = m['target_axial_length']
    layers = lambda length: max(2, math.ceil(length/target))
    tee.R, tee.L, tee.H = R, L, H
    tee.N_HALF = m['half_ring_elements']
    tee.M = 2*tee.N_HALF
    tee.ORDER = m['geometry_order']
    tee.CENTER_SCALE = m['center_scale']
    tee.RADIAL = geometric_spacing(m['radial_layers'], m['radial_ratio'])
    a, b, c = tee.arc_main_lower(), tee.arc_collar(-1), tee.arc_collar(1)
    rings = [tee.join_arcs(a, b), tee.join_arcs(a, c), tee.join_arcs(b, c)]
    openings = [tee.opening_main(-L), tee.opening_main(L), tee.opening_branch(H)]
    profiles = [tee.leg_slices(op, ring, layers(length)+1)
                for op, ring, length in zip(openings, rings, (L, L, H))]
    terminal_names = ['inlet', 'main_connect', 'branch_connect']
    seam_names = [('tee_lr', 'tee_lb'), ('tee_lr', 'tee_rb'), ('tee_lb', 'tee_rb')]
    blocks = [hexmesh.loft(p, first_tag=end,
                          last_tag=tee.cap_tags(p[-1], *seams))
              for p, end, seams in zip(profiles, terminal_names, seam_names)]
    joins = [Seam(0, 'tee_lr', 1, 'tee_lr'), Seam(0, 'tee_lb', 2, 'tee_lb'),
             Seam(1, 'tee_rb', 2, 'tee_rb')]
    plan = []
    for side, section, center, direction, bend_axis, block_id in [
        ('main', profiles[1][0], np.array([L, 0., 0.]), np.array([1., 0., 0.]),
         np.array([0., 0., 1.]), 1),
        ('branch', profiles[2][0], np.array([0., 0., H]), np.array([0., 0., 1.]),
         np.array([1., 0., 0.]), 2),
    ]:
        # Preserve each tee opening's topology. Only coordinates change, so all
        # downstream interfaces inherit identical element counts and node order.
        length = cfg[f'{side}_transition_length']
        ratio = cfg[f'{side}_outlet_diameter']/cfg['junction_diameter']
        def transition(t, s=section, C=center.copy(), d=direction.copy(),
                       ell=length, rat=ratio):
            smooth = t*t*(3-2*t)  # zero slope at both ends
            scale = 1 + (rat-1)*smooth
            return quadmesh.transform(s, np.eye(3)*scale, C*(1-scale)+d*ell*t)
        first, end = f'{side}_connect', f'{side}_bend_connect'
        j = len(blocks)
        blocks.append(hexmesh.loft_fn(transition, np.linspace(0, 1, layers(length)+1),
                                     first_tag=first, last_tag=end))
        joins.append(Seam(block_id, first, j, first))
        plan.append(dict(name=f'{side}_transition', layers=layers(length),
                         diameter_in=2*R, diameter_out=2*R*ratio))
        section = transition(1.)
        center = center + direction*length
        radius = cfg[f'{side}_bend_radius']
        inward = np.cross(bend_axis, direction)
        pivot = center + radius*inward
        def elbow(t, s=section, C=center.copy(), O=pivot.copy(), axis=bend_axis.copy()):
            Q = rotation(axis, t*np.pi/2)
            return quadmesh.transform(s, Q, O-Q@O)
        k = len(blocks)
        bend_layers = layers(np.pi*radius/2)
        blocks.append(hexmesh.loft_fn(elbow, np.linspace(0, 1, bend_layers+1),
                                     first_tag=end, last_tag=f'{side}_tail_connect'))
        joins.append(Seam(j, end, k, end))
        plan.append(dict(name=f'{side}_90deg_bend', radius=radius, layers=bend_layers))
        section = elbow(1.)
        Q = rotation(bend_axis, np.pi/2)
        center = pivot+Q@(center-pivot)
        direction = Q@direction
        length = cfg[f'{side}_tail_length']
        tail_id = len(blocks)
        blocks.append(hexmesh.extrude(section, length, layers(length), axis=direction,
                                      first_tag=f'{side}_tail_connect',
                                      last_tag=f'{side}_outlet'))
        joins.append(Seam(k, f'{side}_tail_connect', tail_id, f'{side}_tail_connect'))
        plan.append(dict(name=f'{side}_tail', layers=layers(length),
                         outlet_center=(center+length*direction).tolist(),
                         outlet_normal=direction.tolist()))
    mesh = hexmesh.attach(blocks, joins)
    from features import add_features
    mesh, feature_info = add_features(mesh, cfg, out)
    topology = hexmesh.topology_report(mesh)
    tags = hexmesh.tag_report(mesh)
    scan = hexmesh.order_scan(mesh, orders=m['quality_sampling_orders'])
    linear = hexmesh.linear_order_scan(mesh, orders=m['quality_sampling_orders'])
    stats = hexmesh.quality_summary(mesh)
    assert topology.watertight and topology.conformal and topology.n_components == 1
    assert topology.n_nonmanifold_faces == 0
    assert tags.n_untagged_boundary == tags.n_tagged_interior == 0
    assert scan.clean and linear.clean
    assert not scan.skipped and not linear.skipped
    bc = {'wall': ('W  ', 1), 'inlet': ('v  ', 2),
          'main_outlet': ('O  ', 3), 'branch_outlet': ('O  ', 4), 'wall_bump': ('W  ', 5),
          'baffle': ('W  ', 6)}
    assert set(mesh.face_group_tags) == set(bc)
    out.mkdir(parents=True, exist_ok=True)
    writer.to_re2(mesh, str(out/'pipe_network.re2'), bc=bc)
    writer.to_fld(mesh, str(out/'pipe_network0.f00001'))
    writer.to_vtu(mesh, str(out/'pipe_network.vtu'), bc=bc)
    # Independently reload exported corner connectivity and boundary groups.
    reread = reader.read_re2(str(out/'pipe_network.re2'))
    assert reread.n_hexes == mesh.n_hexes
    assert hexmesh.is_watertight(reread) and hexmesh.is_conforming(reread)
    checks = {'n_hexes': mesh.n_hexes, 'geometry_order': mesh.order,
              'corner_points': mesh.n_points, 'n_blocks': len(blocks), 'features': feature_info,
              'topology': topology._asdict(), 'boundary_tags': tags._asdict(),
              'quality': stats._asdict(), 'curved_order_scan': scan._asdict(),
              'linear_re2_order_scan': linear._asdict(),
              'boundary_conditions': bc, 're2_roundtrip_passed': True,
              'overlap_check': 'not run; no global curved-element intersection certificate',
              'solver_run': 'not run', 'plan': plan}
    (out/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    (out/'quality_report.txt').write_text(hexmesh.report(mesh)+'\n')
    print(json.dumps(checks, indent=2))
    return mesh


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?', default=str(Path(__file__).with_name('network.json')))
    p.add_argument('--output', default=str(Path(__file__).with_name('output')))
    args = p.parse_args()
    build(json.loads(Path(args.input).read_text()), Path(args.output))
