"""Tie the final reports and file manifest to the current surface checkpoint."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;O=R/'output'
s=np.load(O/'surface_tips_projected.npz');sha=hashlib.sha256(np.ascontiguousarray(s['points']).tobytes()).hexdigest()
v=json.loads((O/'tip_CAD_validation.json').read_text());t=json.loads((O/'topology_and_baseline_checks.json').read_text());i=json.loads((O/'surface_intersections.json').read_text())
assert sha==v['points_sha256']==t['points_sha256']==i['points_sha256']
assert not v['nonpositive_surface_quad_ids'] and not v['nodes_outside_trimmed_CAD_faces']
assert not i['crossing_quad_pairs'] and not i['coplanar_overlap_quad_pairs']
assert t['junction_nodes_unchanged'] and t['connectivity_unchanged'] and t['unmodified_nodes_unchanged']
assert not t['open_or_nonmanifold_edges'] and not t['inconsistent_directed_edges']
summary={'status':'validated linear surface; volume deformation pending','points_sha256':sha,'checked_nodes':v['checked_nodes'],'checked_quads':v['checked_quads'],'topology':t,'CAD_validation_report':'tip_CAD_validation.json','intersections_report':'surface_intersections.json','approved_checkpoint_sha256':'65815981c601278073556e24ef74c42ef2dfa687f0c6bd7d19dbb1921595cf03','matches_approved_checkpoint':sha=='65815981c601278073556e24ef74c42ef2dfa687f0c6bd7d19dbb1921595cf03'}
(O/'tip_projection_summary.json').write_text(json.dumps(summary,indent=2))
files=[p for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='sha256_manifest.json']
(R/'sha256_manifest.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('Reports match current geometry; approved checkpoint:',summary['matches_approved_checkpoint'])
