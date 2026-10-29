"""Finite surface self-intersection screen for bilinear root quads versus the full wall."""
from pathlib import Path
import numpy as np,json,time,hashlib
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(O/'surface_tips_projected.npz');P=d['points'];Q=d['quads'];mask=np.zeros(len(P),bool);mask[d['modified_nodes']]=True;root=np.any(mask[Q],axis=1);triangles=[];parents=[]
for i,q in enumerate(Q):
 n=3 if root[i] else 1;u=np.linspace(0,1,n+1);C=P[q];grid=np.array([[(1-a)*(1-b)*C[0]+a*(1-b)*C[1]+a*b*C[2]+(1-a)*b*C[3] for a in u] for b in u])
 for j in range(n):
  for k in range(n):
   a,b,c,e=grid[j,k],grid[j,k+1],grid[j+1,k+1],grid[j+1,k];triangles.extend([[a,b,c],[a,c,e]]);parents.extend([i,i])
A=np.array(triangles);parent=np.array(parents);source=np.flatnonzero(root[parent]);ctr=A.mean(1);rad=np.linalg.norm(A-ctr[:,None],axis=2).max(1);lo=A.min(1);hi=A.max(1);bins=np.floor(np.log2(np.maximum(rad,1e-12))).astype(int);normal=np.cross(A[:,1]-A[:,0],A[:,2]-A[:,0]);norm=np.linalg.norm(normal,axis=1);pairs=set();coplanar=set();candidates=0;start=time.time()
def crossings(a,b):
 n=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);v0=b[:,1]-b[:,0];v1=b[:,2]-b[:,0];d00=(v0*v0).sum(1);d01=(v0*v1).sum(1);d11=(v1*v1).sum(1);det=d00*d11-d01*d01;bad=np.zeros(len(a),bool)
 for k in range(3):
  edge=a[:,(k+1)%3]-a[:,k];den=(edge*n).sum(1);t=np.divide(((b[:,0]-a[:,k])*n).sum(1),den,out=np.full(len(a),np.inf),where=abs(den)>1e-24);point=a[:,k]+np.clip(t,-10,10)[:,None]*edge;vv=point-b[:,0];d20=(vv*v0).sum(1);d21=(vv*v1).sum(1);u=np.divide(d11*d20-d01*d21,det,out=np.full(len(a),-1.),where=det>1e-30);v=np.divide(d00*d21-d01*d20,det,out=np.full(len(a),-1.),where=det>1e-30);bad|=(t>1e-7)&(t<1-1e-7)&(u>1e-7)&(v>1e-7)&(u+v<1-1e-7)
 return bad
def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def coplanar_area(a,b,n):
 keep=np.delete(np.arange(3),np.argmax(abs(n)));p=[x.copy() for x in a[:,keep]];bb=b[:,keep];sg=np.sign(cross2(bb[1]-bb[0],bb[2]-bb[0]))
 for k in range(3):
  if not p:break
  c,e=bb[k],bb[(k+1)%3];out=[]
  for aa,zz in zip(p,p[1:]+p[:1]):
   da=sg*cross2(e-c,aa-c);dz=sg*cross2(e-c,zz-c);ina=da>=-1e-13;inz=dz>=-1e-13
   if ina:out.append(aa)
   if ina!=inz:out.append(aa+da/(da-dz)*(zz-aa))
  p=out
 if len(p)<3:return 0.
 return abs(sum(cross2(p[i],p[(i+1)%len(p)]) for i in range(len(p))))/2
for bucket in np.unique(bins):
 target=np.flatnonzero(bins==bucket);tree=cKDTree(ctr[target]);rmax=rad[target].max()
 for first in range(0,len(source),256):
  ids=source[first:first+256];hits=tree.query_ball_point(ctr[ids],rad[ids]+rmax+1e-8,workers=2);counts=np.array([len(h) for h in hits]);ii=np.repeat(ids,counts)
  if not len(ii):continue
  jj=target[np.concatenate(hits).astype(int)];mask=((jj>ii)|~root[parent[jj]])&(parent[ii]!=parent[jj]);ii=ii[mask];jj=jj[mask];mask=np.all(hi[jj]>=lo[ii]-1e-9,axis=1)&np.all(hi[ii]>=lo[jj]-1e-9,axis=1);ii=ii[mask];jj=jj[mask];candidates+=len(ii)
  for offset in range(0,len(ii),5000):
   i=ii[offset:offset+5000];j=jj[offset:offset+5000];bad=crossings(A[i],A[j])|crossings(A[j],A[i]);pairs.update(tuple(sorted(x)) for x in np.c_[parent[i[bad]],parent[j[bad]]]);parallel=np.linalg.norm(np.cross(normal[i],normal[j]),axis=1)<1e-10*norm[i]*norm[j];sep=abs(((A[i,0]-A[j,0])*normal[j]).sum(1))/np.maximum(norm[j],1e-30);cp=np.flatnonzero(parallel&(sep<1e-9))
   for k in cp:
    aa,bb=i[k],j[k]
    # Adjacent valid quads meet on their prescribed common edge or vertex.
    if np.intersect1d(Q[parent[aa]],Q[parent[bb]]).size:continue
    if coplanar_area(A[aa],A[bb],normal[aa])>1e-13:coplanar.add(tuple(sorted([int(parent[aa]),int(parent[bb])])) )
 print('surface intersection bin',int(bucket),'candidates',candidates,'crossings',len(pairs),'coplanar overlaps',len(coplanar),'seconds',round(time.time()-start),flush=True)
E=np.sort(np.stack([Q,np.roll(Q,-1,axis=1)],axis=2),axis=2).reshape(-1,2);E,counts=np.unique(E,axis=0,return_counts=True);sep=cKDTree(P).query(P,k=2)[0][:,1].min();assert np.all(counts==2) and sep>1e-9
report={'points_sha256':hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),'changed_or_adjacent_quads':int(root.sum()),'triangles':len(A),'candidate_pairs':candidates,'crossing_quad_pairs':[list(x) for x in sorted(pairs)],'coplanar_overlap_quad_pairs':[list(x) for x in sorted(coplanar)],'wall_topology':{'nodes':len(P),'quads':len(Q),'edges':len(E),'open_or_nonmanifold_edges':int(sum(counts!=2)),'Euler_characteristic':int(len(P)-len(E)+len(Q)),'minimum_distinct_node_separation_m':float(sep)},'scope':'All changed and adjacent bilinear quads sampled with a 3x3 subdivision, then triangle intersections checked against changed and adjacent samples and the remaining wall quad triangulation. Shared-edge/vertex coplanar neighbours excluded; local orientation checked separately. Finite surface diagnostic, not a proof of global injectivity.','seconds':time.time()-start};(O/'surface_intersections.json').write_text(json.dumps(report,indent=2,default=int));print(json.dumps(report,indent=2,default=int));assert not pairs and not coplanar
