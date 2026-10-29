from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(O/'tail_refined.npz');p=np.load(R/'inputs/volume_parent.npz');assert np.array_equal(d['points'][:len(p['points'])],p['points']);sha=hashlib.sha256(d['points'].tobytes()).hexdigest()
for name in ['refinement_summary.json','volume_quality.json','volume_topology.json','actual_GLL_spacing.json','stretch_comparison.json','geometry_restriction_validation.json','RE2_export_validation.json']:assert json.loads((O/name).read_text())['points_sha256']==sha,name
assert all(x['nonpositive']==0 for x in json.loads((O/'volume_quality.json').read_text())['scans'].values());r=json.loads((O/'geometry_restriction_validation.json').read_text());assert r['maximum_child_corner_restriction_error_m']<1e-10 and r['parent_inverse_samples_inside']==0
print('Coordinate/report hashes, preserved parent nodes and positive GLL checks verified.')
