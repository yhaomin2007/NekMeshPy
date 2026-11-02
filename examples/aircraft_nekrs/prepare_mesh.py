#!/usr/bin/env python3
"""Scale the accepted tail-refined checkpoint and split its spherical farfield."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekmeshpy.io.writer import _write_re2
from nekmeshpy.io.reader import _read_re2_raw

LREF = 7.00532
EXPECTED_SHA = '59e8b7f76d6068cb55e21df70f8965d23a11f20509922a5e4a093f99d153ba63'


def prepare(source, output):
    d = np.load(source)
    p, h, layers, q = (d[k] for k in ('points', 'hexes', 'layer_nodes', 'wall_quads'))
    if hashlib.sha256(p.tobytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError('Input is not the approved 651,360-element tail-refined checkpoint')
    nw, nr = len(q), len(layers) - 1
    assert len(h) == nw * nr == 651360
    # Verify the radial tensor ordering before using Nek faces 5 and 6.
    for k in range(nr):
        assert np.array_equal(h[k*nw:(k+1)*nw], np.column_stack((layers[k,q], layers[k+1,q])))
    outer = p[layers[-1,q]]
    # Bilinear-face oriented area vector is affine in (u,v). Its x-component
    # extrema occur at corners. Determine outward orientation using radial edges.
    norms = []
    for u, v in ((-1,-1),(1,-1),(1,1),(-1,1)):
        du = ((1-v)*(outer[:,1]-outer[:,0]) + (1+v)*(outer[:,2]-outer[:,3])) / 4
        dv = ((1-u)*(outer[:,3]-outer[:,0]) + (1+u)*(outer[:,2]-outer[:,1])) / 4
        norms.append(np.cross(du, dv))
    norms = np.stack(norms, axis=1)
    radial = outer.mean(axis=1) - p[layers[-2,q]].mean(axis=1)
    assert np.all(np.einsum('ni,ni->n', norms.mean(axis=1), radial) > 0)
    nx = norms[:,:,0]
    tol = 1e-12 * np.linalg.norm(norms, axis=2).max()
    # Any face with an inflowing portion is prescribed freestream. Tangent
    # and mixed faces go to inlet, leaving strictly outflow-only outlet faces.
    outlet = nx.min(axis=1) > tol
    mixed = (nx.min(axis=1) < -tol) & (nx.max(axis=1) > tol)
    assert outlet.any() and (~outlet).any()
    scaled = p / LREF
    rows = [(i,5,'wall','W  ',1,-1,0) for i in range(nw)]
    rows += [((nr-1)*nw+i,6,'outlet' if outlet[i] else 'inlet',
              'O  ' if outlet[i] else 'v  ',3 if outlet[i] else 2,-1,0) for i in range(nw)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_re2(str(output), scaled[h], len(h), [rows], np.arange(len(h)))
    raw, count, _ = _read_re2_raw(str(output),3,'nondimensional roundtrip')
    assert count == len(h) and np.array_equal(raw,scaled[h])
    # Independently read binary boundary records, not just writer inputs.
    with output.open('rb') as f:
        f.seek(84 + len(h)*25*8)
        assert np.fromfile(f,'<f8',1)[0] == 0 # no curved sides
        nbc = int(np.fromfile(f,'<f8',1)[0])
        records = np.fromfile(f,'<f8',nbc*8).reshape(-1,8)
    assert nbc == 2*nw and len(set(map(tuple,records[:,:2]))) == nbc
    ids, counts = np.unique(records[:,6].astype(int),return_counts=True)
    assert dict(zip(ids,counts)) == {1:nw,2:int((~outlet).sum()),3:int(outlet.sum())}
    report = dict(hexes=len(h), points=len(p), reference_length_m=LREF,
                  coordinate_operation='x_star=x_m/7.00532; no translation or deformation',
                  exact_scaled_corner_roundtrip=True, geometry_order=1,
                  boundary_faces={str(i):int(c) for i,c in zip(ids,counts)},
                  mixed_farfield_faces_prescribed_inlet=int(mixed.sum()),
                  outlet_rule='positive outward area-vector x component over entire bilinear face',
                  boundary_ids={'1':'no-slip aircraft','2':'freestream inlet / tangent belt','3':'pressure outlet'},
                  source_points_sha256=EXPECTED_SHA,
                  re2_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.mesh_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('checkpoint',type=Path)
    a.add_argument('--output',default=str(Path(__file__).resolve().parent/'aircraft.re2'))
    args=a.parse_args()
    prepare(args.checkpoint,args.output)
