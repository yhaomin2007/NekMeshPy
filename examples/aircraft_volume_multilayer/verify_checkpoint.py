from pathlib import Path
import hashlib,json,numpy as np
R=Path(__file__).resolve().parent;O=R/'output';P=np.load(O/'gll_constrained.npz')['points'];d=np.load(R/'inputs/volume_input.npz');s=np.load(R/'inputs/approved_surface.npz');sha=hashlib.sha256(P.tobytes()).hexdigest();assert np.array_equal(P[d['layer_nodes'][0]],s['points']);assert np.array_equal(P[d['layer_nodes'][30:]],d['points'][d['layer_nodes'][30:]])
for filename in ['volume_quality.json','volume_topology.json','actual_GLL_spacing.json','geometry_and_layer_changes.json','independent_jacobian_crosscheck.json','RE2_export_validation.json','intersection_audit.json','final_summary.json']:assert json.loads((O/filename).read_text())['points_sha256']==sha
q=json.loads((O/'volume_quality.json').read_text());assert all(r['nonpositive']==0 for r in q['scans'].values());assert json.loads((O/'intersection_audit.json').read_text())['strictly_inside_samples']==0
print('Checkpoint hashes, preserved boundaries and final reports verified.')
