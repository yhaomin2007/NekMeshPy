"""All changed-cell AABB pairs, then finite polyhedral/trilinear diagnostics."""
from pathlib import Path
import argparse,json,time,hashlib,multiprocessing as mp
import numpy as np,vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
import sys
sys.path.insert(0,str(Path(str(Path(__file__).resolve().parents[2]))))
from nekmeshpy.core import topology
R=Path(__file__).resolve().parent;O=R/'output';B=R/'baseline'
p=argparse.ArgumentParser();p.add_argument('--fit',default='gll_constrained.npz');a=p.parse_args();sol=np.load(O/a.fit);base=np.load(Path(str(Path(__file__).resolve().parent/'inputs/volume_input.npz')));P=sol['points'];H=base['hexes'];moved=np.linalg.norm(P-base['points'],axis=1)>1e-12;touched=np.any(moved[H],axis=1);changed_ids=np.flatnonzero(touched);changed=len(changed_ids);start=time.time()
pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(P,deep=True));ca=vtk.vtkCellArray();ca.SetCells(len(H),numpy_to_vtkIdTypeArray(np.c_[np.full(len(H),8),H].ravel().astype(np.int64),deep=True));g=vtk.vtkUnstructuredGrid();g.SetPoints(pts);g.SetCells(vtk.VTK_HEXAHEDRON,ca);loc=vtk.vtkStaticCellLocator();loc.SetDataSet(g);loc.BuildLocator()
lo=P[H].min(1);hi=P[H].max(1);centroid,radius,vertex,normal=topology._hex_geometry(P,H);samples=topology._hex_sample_points(P,H);tol=1e-8*np.linalg.norm(np.ptp(P,axis=0));flags=[];candidate_count=0;ii=[];jj=[]
def process():
 global ii,jj,candidate_count
 if not ii:return
 i=np.concatenate(ii);j=np.concatenate(jj);candidate_count+=len(i);ii=[];jj=[]
 for k in range(0,len(i),2500):
  ia=i[k:k+2500];ja=j[k:k+2500];bad=topology._points_inside(samples[ia],vertex[ja],normal[ja],tol)|topology._points_inside(samples[ja],vertex[ia],normal[ia],tol)|(np.linalg.norm(centroid[ia]-centroid[ja],axis=1)<tol)
  flags.extend(np.c_[ia[bad],ja[bad]].tolist())
bins=np.floor(np.log2(np.maximum(radius,1e-12))).astype(int)
for bucket in np.unique(bins):
 targets=np.flatnonzero(bins==bucket);tree=cKDTree(centroid[targets]);maxr=radius[targets].max()
 for first in range(0,changed,512):
  ids=changed_ids[first:first+512];hits=tree.query_ball_point(centroid[ids],radius[ids]+maxr,workers=2);counts=np.array([len(h) for h in hits]);i=np.repeat(ids,counts)
  if not len(i):continue
  j=targets[np.concatenate(hits).astype(int)];mask=((j>i)|~touched[j])&(np.linalg.norm(centroid[i]-centroid[j],axis=1)<radius[i]+radius[j]);i=i[mask];j=j[mask]
  mask=np.all(hi[j]>lo[i]+1e-10,axis=1)&np.all(hi[i]>lo[j]+1e-10,axis=1);i=i[mask];j=j[mask]
  if len(i):ii.append(i);jj.append(j);process()
 print('intersection radius bin',int(bucket),'candidates',candidate_count,'flags',len(flags),'seconds',round(time.time()-start),flush=True)
process();print('polyhedral flags',len(flags),flush=True)
(O/'polyhedral_screen.json').write_text(json.dumps({'pairs':flags,'points_sha256':hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest()},indent=2))
S=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
def basis(u):return np.prod(np.where(S[None,:,:],u[:,None,:],1-u[:,None,:]),axis=2)
def jac(u,v):
 d=[]
 for axis in range(3):
  fac=np.where(S[None,:,:],u[:,None,:],1-u[:,None,:]);fac[:,:,axis]=2*S[None,:,axis]-1;d.append(np.prod(fac,axis=2)@v)
 return np.stack(d,axis=2)
uvw=np.array(np.meshgrid(*([np.linspace(0,1,9)]*3),indexing='ij')).reshape(3,-1).T;findings=[]
def inverse_check(pair):
 a,b=pair;result=[]
 for source,target in [(a,b),(b,a)]:
  A=P[H[source]];B=P[H[target]];offset=B.mean(0);scale=np.linalg.norm(np.ptp(B,axis=0));B=(B-offset)/scale;Y=(basis(uvw)@A-offset)/scale;u=np.full_like(Y,.5)
  for it in range(40):
   residual=basis(u)@B-Y;J=jac(u,B)
   try:step=np.linalg.solve(J,residual[:,:,None])[:,:,0]
   except np.linalg.LinAlgError:step=np.einsum('nij,nj->ni',np.linalg.pinv(J,rcond=1e-12),residual)
   u-=np.clip(step,-1,1);u=np.clip(u,-3,4)
  err=np.linalg.norm(basis(u)@B-Y,axis=1);inside=np.all((u>1e-6)&(u<1-1e-6),axis=1)&(err<1e-9)
  result.append({'source_cell':source,'target_cell':target,'strictly_inside_samples':int(sum(inside))})
 return result
with mp.get_context('fork').Pool(4) as pool:
 for k,result in enumerate(pool.imap(inverse_check,flags,chunksize=4)):
  findings.extend(result)
  if k%100==0:print('inverse trilinear',k,len(flags),'seconds',round(time.time()-start),flush=True)
report={'changed_cells_screened':changed,'total_cells':len(H),'AABB_candidate_pairs':candidate_count,'polyhedral_flagged_pairs':flags,'inverse_trilinear_9x9x9_findings':findings,'strictly_inside_samples':sum(f['strictly_inside_samples'] for f in findings),'tolerance_m':float(tol),'scope':'All AABB-overlapping pairs involving the modified prefix of cells are screened. Polyhedral flags are checked with 9x9x9 true trilinear samples in both directions. Finite diagnostics, not a proof of no intersections. Unchanged outer-outer pairs retain their previous audit.','seconds':time.time()-start}
report['scope']='All AABB-overlapping pairs involving any cell with moved vertices are screened, including lower-index unchanged targets. Polyhedral flags are checked using 9x9x9 true trilinear samples in both directions. Finite diagnostics, not a proof of no intersections. Unchanged cells restrict the previously audited parent trilinear maps.'
report['points_sha256']=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest();report['connectivity_sha256']=hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest()
(O/'intersection_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['polyhedral_flagged_pairs','inverse_trilinear_9x9x9_findings']},indent=2));assert report['strictly_inside_samples']==0
