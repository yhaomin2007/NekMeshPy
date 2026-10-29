"""Conformal directional subdivision; exact restriction of accepted trilinear maps."""
from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parent;O=R/'output';source=R;d=np.load(R/'inputs/volume_parent.npz');P=np.load(R/'inputs/volume_parent.npz')['points'];L=d['layer_nodes'];Q=d['wall_quads'];H=d['hexes'];plan=np.load(O/'selected_plan.npz');shape=plan['subdivisions'];nw=len(L[0]);keys={};spec=[]
def vertex(q,u,v):
 w=np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v]);nz=np.flatnonzero(w>0)
 if len(nz)==1:return int(q[nz[0]])
 key=tuple(sorted((int(q[i]),float(w[i])) for i in nz))
 if key not in keys:keys[key]=nw+len(spec);spec.append((q.copy(),w))
 return keys[key]
qnew=[];parent=[];uv=[]
for k,q in enumerate(Q):
 us=np.linspace(0,1,int(shape[k,0])+1);vs=np.linspace(0,1,int(shape[k,1])+1)
 for j in range(len(vs)-1):
  for i in range(len(us)-1):
   coords=[(us[i],vs[j]),(us[i+1],vs[j]),(us[i+1],vs[j+1]),(us[i],vs[j+1])];qnew.append([vertex(q,u,v) for u,v in coords]);parent.append(k);uv.append(coords)
Qn=np.array(qnew);parent=np.array(parent);uv=np.array(uv);Ln=np.empty((61,nw+len(spec)),int);Ln[:,:nw]=L;extra=[]
qq=np.array([q for q,w in spec]);w=np.array([w for q,w in spec])
for k in range(61):
 Ln[k,nw:]=np.arange(len(spec))+len(P)+k*len(spec);extra.append(np.einsum('ni,nij->nj',w,P[L[k,qq]]))
Pn=np.concatenate([P,*extra]);Hn=np.concatenate([np.c_[Ln[k,Qn],Ln[k+1,Qn]] for k in range(60)]);parenthex=np.concatenate([parent+k*len(Q) for k in range(60)]);assert np.array_equal(Pn[:len(P)],P)
np.savez_compressed(O/'tail_refined.npz',points=Pn,hexes=Hn,layer_nodes=Ln,wall_quads=Qn,family=d['family'][parent],parent_quad=parent,parent_hex=parenthex,sub_uv=uv,original_node_count=len(P));report={'source_points_sha256':hashlib.sha256(P.tobytes()).hexdigest(),'points_sha256':hashlib.sha256(Pn.tobytes()).hexdigest(),'hexes':len(Hn),'points':len(Pn),'wall_quads':len(Qn),'new_surface_nodes':len(spec),'original_nodes_bitwise_preserved':True,'method':'Opposite-edge subdivision counts propagated through shared surface edges; same subdivisions on all 60 radial layers. Children are exact restrictions of the original bilinear surface and trilinear volume maps. No node projection or smoothing.'};(O/'refinement_summary.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
