from pathlib import Path
import sys,time,json,hashlib,numpy as np
from quality import linear_scan,linear_spacing
R=Path(__file__).resolve().parent;O=R/'output';sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
d=np.load(O/'tail_refined.npz');old=np.load(R/'inputs/volume_parent.npz')['points'];P=d['points'];H=d['hexes'];L=d['layer_nodes'];Q=d['wall_quads'];start=time.time();report={'points_sha256':hashlib.sha256(P.tobytes()).hexdigest(),'hexes':len(H),'nodes':len(P),'original_nodes_exactly_preserved':bool(np.array_equal(P[:len(old)],old)),'scans':{}};arrays={}
for n in [3,5,7]:
 q=linear_scan(P,H,n);report['scans'][str(n)]={'minimum_scaled_Jacobian':float(q.min()),'nonpositive':int(sum(q<=0)),'worst_element':int(q.argmin()),'bad_ids':np.flatnonzero(q<=0).tolist()};arrays['SJ_N'+str(n)]=q;print('N',n,report['scans'][str(n)],'seconds',round(time.time()-start),flush=True)
sp=linear_spacing(P,H,7);arrays['axis_adjacent_GLL_spacing_N7_m']=sp;report['axis_adjacent_GLL_spacing_N7']={'minimum_m':float(sp.min()),'worst_element':int(sp.argmin()),'p01_m':float(np.quantile(sp,.01))};report['refinement_geometry']={'original_node_displacement_max_m':float(np.linalg.norm(P[:len(old)]-old,axis=1).max()),'added_nodes':len(P)-len(old)};(O/'volume_quality.json').write_text(json.dumps(report,indent=2));np.savez_compressed(O/'volume_quality_fields.npz',**arrays);print(json.dumps({k:v for k,v in report.items() if k!='displacements'},indent=2),flush=True)
