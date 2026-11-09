"""Screen tessellated quadratic wall patches for intersections."""
from pathlib import Path
import numpy as np,json,time
from scipy.spatial import cKDTree
from quality import basis
O=Path('deliver/final/aircraft_quadratic')
import vtk
from vtk.util.numpy_support import vtk_to_numpy
r=vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(str(O/'aircraft_wall_quadratic_m.vtu'))
r.Update()
g=r.GetOutput()
P=vtk_to_numpy(g.GetPoints().GetData())
C=vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1,9)
Q=C[:,:4]
N=P[C[:,[0,7,3,4,8,6,1,5,2]]]
b,_=basis(np.linspace(-1,1,4))
W=np.einsum('ia,jb->ijab',b,b).reshape(16,9)
p=(W@N).reshape(-1,4,4,3)
stack=[]
for i in range(3):
    for j in range(3):
        x,y,z,w=p[:,i,j],p[:,i+1,j],p[:,i+1,j+1],p[:,i,j+1]
        stack.extend([np.stack([x,y,z],axis=1),np.stack([x,z,w],axis=1)])
A=np.concatenate(stack)
parent=np.tile(np.arange(len(Q)),18)
source=np.arange(len(A))
root=np.ones(len(Q),bool)
ctr=A.mean(1);rad=np.linalg.norm(A-ctr[:,None],axis=2).max(1);lo=A.min(1);hi=A.max(1);bins=np.floor(np.log2(np.maximum(rad,1e-12))).astype(int);normal=np.cross(A[:,1]-A[:,0],A[:,2]-A[:,0]);norm=np.linalg.norm(normal,axis=1);pairs=set();coplanar=set();candidates=0;start=time.time()
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
 for first in range(0,len(source),4096):
  ids=source[first:first+4096];hits=tree.query_ball_point(ctr[ids],rad[ids]+rmax+1e-8,workers=1);counts=np.array([len(h) for h in hits]);ii=np.repeat(ids,counts)
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
report={'triangles':len(A),'candidate_pairs':candidates,
        'crossing_quad_pairs':[list(map(int,x)) for x in sorted(pairs)],
        'coplanar_overlap_quad_pairs':[list(map(int,x)) for x in sorted(coplanar)],
        'scope':'3 by 3 tessellation per quadratic quad. Finite wall screen.'}
(O/'surface_intersections.json').write_text(json.dumps(report,indent=2))
assert not pairs and not coplanar,report
