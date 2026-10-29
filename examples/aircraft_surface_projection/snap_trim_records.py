from cad_surface import *
from scipy.optimize import minimize_scalar
from OCP.BRep import BRep_Tool
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN,TopAbs_ON
from OCP.gp import gp_Pnt2d
import json
s=np.load(O/'surface_tips_projected.npz');data={k:s[k] for k in s.files};P=data['points'];nom=np.load(O/'surface_input.npz')['nominal'];records=data['CAD_records'];changes=[]
for row in records:
 i,fi=map(int,row[:2]);sign=-1 if nom[i,1]<-1e-8 else 1;p=P[i]*[1,sign,1];f=faces[fi];uv=row[2:4];exact=np.array(f.positionAt(*uv).toTuple())/1000;state=BRepClass_FaceClassifier(f.wrapped,gp_Pnt2d(*uv),1e-4).State()
 if state in [TopAbs_IN,TopAbs_ON] and np.linalg.norm(exact-p)<1e-7:continue
 if fi not in [14,47,53]:continue
 best=None
 for e in f.Edges():
  ad=e._geomAdaptor();lo,hi=ad.FirstParameter(),ad.LastParameter();curve=BRep_Tool.CurveOnSurface_s(e.wrapped,f.wrapped,lo,hi)
  def ev(t):
   q=curve.Value(float(t));uv=np.array([q.X(),q.Y()]);return uv,np.array(f.positionAt(*uv).toTuple())/1000
  def fun(t):return np.sum((ev(t)[1]-p)**2)
  opt=minimize_scalar(fun,bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
  for t in [opt.x,lo,hi]:
   uv,xyz=ev(t);dist=np.linalg.norm(xyz-p)
   if best is None or dist<best[0]:best=(dist,uv,xyz)
 dist,uv,xyz=best;assert dist<2e-5,(i,fi,dist)
 row[2:4]=uv;P[i]=xyz*[1,sign,1];changes.append({'node':i,'face':fi,'shift_m':float(dist),'xyz_m':P[i].tolist()})
np.savez_compressed(O/'surface_tips_projected.npz',**data);(O/'CAD_boundary_tolerance_adjustments.json').write_text(json.dumps(changes,indent=2));print('cap trim adjustments',len(changes),'max m',max([c['shift_m'] for c in changes],default=0))
