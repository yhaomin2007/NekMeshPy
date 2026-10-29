"""Exact within-element nearest GLL pair, with a conservative exclusion bound."""
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from quality import S
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from nekmeshpy.core.interp import gll_nodes
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'));P=np.load(O/(sys.argv[1] if len(sys.argv)>1 else 'gll_constrained.npz'))['points'];H=d['hexes'];f=np.load(O/'volume_quality_fields.npz');axis=f['axis_adjacent_GLL_spacing_N7_m'];best=float(axis.min());worst=int(axis.argmin());n=7;g=2*gll_nodes(n)-1;dg=np.diff(g).min();U=np.array(np.meshgrid(g,g,g,indexing='ij')).reshape(3,-1).T;W=np.prod(1+U[:,None,:]*S[None,:,:],axis=2)/8;M=np.c_[S,S[:,0]*S[:,1],S[:,0]*S[:,2],S[:,1]*S[:,2],S.prod(1)]/8;candidates=[];start=time.time();lower=[]
for k in range(0,len(H),2048):
 corners=P[H[k:k+2048]];C=np.einsum('enk,nm->emk',corners-corners[:,:1],M);J=C[:,:3].transpose(0,2,1);inv=np.linalg.inv(J);mixed=np.einsum('eij,emj->emi',inv,C[:,3:]);norm=np.linalg.norm(mixed,axis=-1);v=np.stack([norm[:,0]+norm[:,1]+norm[:,3],norm[:,0]+norm[:,2]+norm[:,3],norm[:,1]+norm[:,2]+norm[:,3]],axis=1);beta=np.linalg.norm(v,axis=1);small=np.linalg.svd(J,compute_uv=False)[:,-1];bound=np.maximum(0,1-beta)*small*dg;lower.append(bound);candidates.extend((np.flatnonzero(bound<best*(1+1e-10))+k).tolist())
lower=np.concatenate(lower);print('exact spacing candidates',len(candidates),'of',len(H),'axis upper bound',best,flush=True);last=time.time();pair=None;checked=0
for cell in candidates:
 if lower[cell]>best:continue
 xyz=W@P[H[cell]];dist,nn=cKDTree(xyz).query(xyz,k=2);m=float(dist[:,1].min());checked+=1
 if m<best:best=m;worst=cell;row=int(dist[:,1].argmin());pair=[row,int(nn[row,1])]
 if time.time()-last>25:print('spacing checked',checked,'best m',best,'seconds',round(time.time()-start),flush=True);last=time.time()
r={'points_sha256':hashlib.sha256(P.tobytes()).hexdigest(),'GLL_polynomial_order':n,'points_per_element':len(U),'minimum_within_element_GLL_pair_spacing_m':best,'element':worst,'pair_local_indices':pair,'axis_adjacent_upper_bound_m':float(axis.min()),'cells_with_exact_kdtree_search':checked,'total_hexes':len(H),'scope':'Closest distinct GLL pair within each trilinear element, including non-axis neighbours. Elements excluded only using a conservative centre-Jacobian perturbation singular-value bound; not a global all-element point cloud nearest-pair test.','bound':'For J=Jc(I+E), bound ||E||2 by the Frobenius norm of column-wise coefficient norm sums on [-1,1]^3. Then mapped pair separation >= sigma_min(Jc)*(1-beta)*minimum parameter point separation when beta<1.'};(O/'actual_GLL_spacing.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
