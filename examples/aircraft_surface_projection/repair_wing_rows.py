from cad_surface import *
from scipy.spatial import cKDTree
s=np.load(O/'surface_tips_projected.npz');data={k:s[k] for k in s.files};b=np.load(O/'base_surface.npz');P=data['points'];Q=data['quads'];d=np.load(O/'surface_input.npz');nom=d['nominal'];tt=d['circ'];fam=data['family'];records={int(r[0]):r.copy() for r in data['CAD_records']};modified=set(data['modified_nodes']);root=set(np.unique(Q[np.char.startswith(fam,'J_')]));U=Chart(np.isin(FI,[4,12,13,22,33]),V[:,[0,1]],'xy');L=Chart(np.isin(FI,[1,7,19,20,21,27,28,29]),V[:,[0,1]],'xy');wn=np.unique(Q[np.char.startswith(fam,'W_')]);
for y in [8.75,10.375]:
 ids=wn[(abs(nom[wn,1]-y)<1e-8)&(tt[wn]<6)]
 for i in ids:
  t=float(tt[i])
  if i in root or t==0 or t==6:continue
  # Corresponding trailing-edge endpoint at the same span row and side.
  end=wn[(abs(nom[wn,1]-y)<1e-8)&(abs(tt[wn]-(8 if t<3 else 6))<1e-8)][0]
  shift=P[end,0]-b['points'][end,0];x=b['points'][i,0]+shift*chord_fraction('W',t);chart=U if t<3 else L;pos,fi,uv,dd=chart.evaluate([[x,y]]);P[i]=pos[0];records[int(i)]=np.array([i,int(fi[0]),*uv[0]]);modified.add(int(i))
ids=np.array(sorted(modified));dd,mate=cKDTree(nom).query(nom[ids]*[1,-1,1]);assert dd.max()<1e-8
for i,m in zip(ids,mate):
 if nom[i,1]<-1e-8:continue
 P[m]=P[i]*[1,-1,1]
 if int(i) in records:records[int(m)]=records[int(i)].copy();records[int(m)][0]=m
 modified.add(int(m))
data['modified_nodes']=np.array(sorted(modified));data['CAD_records']=np.array([records[i] for i in sorted(records)]);np.savez_compressed(O/'surface_tips_projected.npz',**data)
