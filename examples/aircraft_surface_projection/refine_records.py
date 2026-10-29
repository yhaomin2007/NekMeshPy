from cad_surface import *
from scipy.optimize import least_squares
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN,TopAbs_ON
from OCP.gp import gp_Pnt2d
s=np.load(O/'surface_tips_projected.npz');data={k:s[k] for k in s.files};P=data['points'];nom=np.load(O/'surface_input.npz')['nominal'];records=data['CAD_records'];outside=[]
for row in records:
 i,fi=map(int,row[:2]);p=P[i]*[1,-1 if nom[i,1]<-1e-8 else 1,1];f=faces[fi]
 def fun(uv):return np.array(f.positionAt(*uv).toTuple())/1000-p
 uv=least_squares(fun,row[2:4],xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=30).x;row[2:4]=uv
 if BRepClass_FaceClassifier(f.wrapped,gp_Pnt2d(*uv),1e-4).State() not in [TopAbs_IN,TopAbs_ON]:outside.append((i,fi,np.linalg.norm(fun(uv))))
print('outside after native refinement',outside,flush=True)
np.savez_compressed(O/'surface_tips_projected.npz',**data)
