from cad_surface import *
from scipy.spatial import cKDTree
s=np.load(O/'surface_tips_projected.npz');data={k:s[k] for k in s.files};P=data['points'];Q=data['quads'];d=np.load(O/'surface_input.npz');nom=d['nominal'];tt=d['circ'];fam=data['family'];records={int(r[0]):r.copy() for r in data['CAD_records']};modified=set(data['modified_nodes']);rootnodes=set(np.unique(Q[np.char.startswith(fam,'J_')]));ids=np.unique(Q[(fam=='W_H_TE')&np.all(nom[Q,1]>=0,axis=1)])
def ep(pair,span):
 for e,adj in edges.values():
  if not set(pair)<=adj:continue
  a,b=np.array(e.startPoint().toTuple())/1000,np.array(e.endPoint().toTuple())/1000
  if min(a[1],b[1])-1e-8<=span<=max(a[1],b[1])+1e-8:
   t=brentq(lambda t:e.positionAt(float(t)).toTuple()[1]/1000-span,0,1,xtol=1e-13);return np.array(e.positionAt(float(t)).toTuple())/1000
 raise ValueError((pair,span))
for i in ids:
 y=float(nom[i,1])
 if i in rootnodes or y>=20:continue
 if y<8:fi,up,low=25,12,29
 elif y<10.87:fi,up,low=23,12,27
 elif y<13.6:fi,up,low=15,4,21
 else:fi,up,low=0,4,1
 u,l=ep([fi,up],y),ep([fi,low],y);depth=(float(tt[i])-6)/2;p=l+depth*(u-l);uv=faces[fi].paramAt(tuple(p*1000));P[i]=np.array(faces[fi].positionAt(*uv).toTuple())/1000;records[int(i)]=np.array([i,fi,*uv]);modified.add(int(i))
ids=np.array(sorted(modified));dd,mate=cKDTree(nom).query(nom[ids]*[1,-1,1]);assert dd.max()<1e-8
for i,m in zip(ids,mate):
 if nom[i,1]<-1e-8:continue
 P[m]=P[i]*[1,-1,1]
 if int(i) in records:records[int(m)]=records[int(i)].copy();records[int(m)][0]=m
 modified.add(int(m))
data['modified_nodes']=np.array(sorted(modified));data['CAD_records']=np.array([records[i] for i in sorted(records)]);np.savez_compressed(O/'surface_tips_projected.npz',**data)
