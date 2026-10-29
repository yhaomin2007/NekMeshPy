from pathlib import Path
import json,csv,hashlib,numpy as np
R=Path(__file__).resolve().parent;O=R/'output';r=np.load(O/'tail_refined.npz');d=np.load(R/'inputs/volume_parent.npz');old=np.load(R/'inputs/volume_parent.npz')['points'];plan=np.load(O/'selected_plan.npz');P=r['points'];Q=r['wall_quads'];L=r['layer_nodes'];sourceQ=d['wall_quads'];sourceL=d['layer_nodes'];parent=r['parent_quad'];v0=old[sourceL[0,sourceQ]];v1=P[L[0,Q]]
def metric(v):
 length=np.linalg.norm(v-np.roll(v,-1,axis=1),axis=2);return length.max(1)/length.min(1)
a0=metric(v0);a1=metric(v1);regions={}
for key,ids in [('targeted_stretched_aft_quads',plan['target_parent_quads']),('aft_body',plan['aft_parent_quads'])]:
 m=np.isin(parent,ids);regions[key]={'before_quads':len(ids),'after_quads':int(sum(m)),'before_aspect_median_p95_max':np.quantile(a0[ids],[.5,.95,1]).tolist(),'after_aspect_median_p95_max':np.quantile(a1[m],[.5,.95,1]).tolist()}
report={'points_sha256':hashlib.sha256(P.tobytes()).hexdigest(),'regions':regions,'geometry':'Original nodes retained bitwise; new surface and volume nodes evaluate the accepted bilinear/trilinear maps. No CAD relaxation, projection or smoothing.','scope':'Edge-length aspect ratio; thin trailing-edge sheets elsewhere are not targeted.'};(O/'stretch_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
# All child corners must agree with the source restriction; multilinearity then fixes their entire map.
H=r['hexes'];parentH=r['parent_hex'];uv=r['sub_uv'];qn=len(Q);maxerr=0.
for first in range(0,len(H),2048):
 ids=np.arange(first,min(first+2048,len(H)));pq=ids%qn;corners=old[d['hexes'][parentH[ids]]];u=uv[pq,:,0];v=uv[pq,:,1];w=np.stack([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v],axis=2);expected=np.concatenate([np.einsum('eni,eij->enj',w,corners[:,:4]),np.einsum('eni,eij->enj',w,corners[:,4:])],axis=1);maxerr=max(maxerr,float(np.max(np.linalg.norm(expected-P[H[ids]],axis=2))))
assert maxerr<1e-10;assert np.array_equal(P[:len(old)],old)
parentaudit=json.loads((R/'inputs/parent_intersection_audit.json').read_text());assert parentaudit['points_sha256']==hashlib.sha256(old.tobytes()).hexdigest() and parentaudit['strictly_inside_samples']==0
rpt={'points_sha256':report['points_sha256'],'maximum_child_corner_restriction_error_m':maxerr,'original_nodes_bitwise_preserved':True,'parent_points_sha256':parentaudit['points_sha256'],'intersection_status':'Inherited from screened parent geometry by exact tensor subdivision. This does not introduce geometric overlap; finite parent screening limitations remain.','parent_AABB_candidates':parentaudit['AABB_candidate_pairs'],'parent_inverse_samples_inside':0};(O/'geometry_restriction_validation.json').write_text(json.dumps(rpt,indent=2));print('maximum restriction residual',maxerr)
