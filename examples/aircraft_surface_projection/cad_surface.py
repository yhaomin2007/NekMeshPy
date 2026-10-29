"""Physical CAD charts with face identity and native UV interpolation."""
from pathlib import Path
import numpy as np,vtk,cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Pnt,gp_Vec
from scipy.optimize import brentq
from functools import lru_cache
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
R=Path(__file__).resolve().parent;O=R/'output';a=np.load(O/'cad_atlas.npz');V=a['points'];T=a['triangles'];FI=a['face_id'];UV=a['uv'];shape=cq.Shape.importBrep(str(R/'reference.brep'));faces=shape.Faces();body=set(range(35,47))|{26,54}
def polydata(points,triangles):
 p=vtk.vtkPoints();p.SetData(numpy_to_vtk(np.ascontiguousarray(points),deep=True));c=vtk.vtkCellArray();c.SetCells(len(triangles),numpy_to_vtkIdTypeArray(np.c_[np.full(len(triangles),3),triangles].ravel().astype(np.int64),deep=True));g=vtk.vtkPolyData();g.SetPoints(p);g.SetPolys(c);return g
def locator(g):
 loc=vtk.vtkStaticCellLocator();loc.SetDataSet(g);loc.BuildLocator();return loc
bodymask=np.isin(FI,list(body));bodypoly=polydata(V,T[bodymask]);bodyloc=locator(bodypoly)
centerfile=O/'body_centerline.npz'
if centerfile.exists():cc=np.load(centerfile);CX,CZ=cc['x'],cc['z']
else:
 CX=np.linspace(3,65.07,400);CZ=[];plane=vtk.vtkPlane();plane.SetNormal(1,0,0);cut=vtk.vtkCutter();cut.SetCutFunction(plane);cut.SetInputData(bodypoly)
 for x in CX:
  plane.SetOrigin(float(x),0,0);cut.Update();p=cut.GetOutput().GetPoints();z=np.array([p.GetPoint(i)[2] for i in range(p.GetNumberOfPoints())]);CZ.append((z.min()+z.max())/2)
 CZ=np.array(CZ);np.savez_compressed(centerfile,x=CX,z=CZ)
def center(x):return np.interp(x,CX,CZ)
def body_coordinates(p):return np.c_[p[:,0]/5,np.arctan2(p[:,2]-center(p[:,0]),np.maximum(p[:,1],0))]
adaptors=[BRepAdaptor_Surface(f.wrapped) for f in faces]
class Chart:
 def __init__(self,mask,coordinates,mode='body'):
  self.mode=mode;self.ids=np.flatnonzero(mask);self.tri=T[self.ids];self.coords=coordinates;tri=self.coords[self.tri];aa=tri[:,1]-tri[:,0];bb=tri[:,2]-tri[:,0];area=aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0];valid=abs(area)>1e-16;self.ids=self.ids[valid];self.tri=self.tri[valid];p=np.c_[coordinates,np.zeros(len(coordinates))];self.loc=locator(polydata(p,self.tri))
 def evaluate(self,points):
  result=[];fid=[];UVout=[];outside=[]
  for p in points:
   q=[0.,0.,0.];cid=vtk.reference(0);sid=vtk.reference(0);ds=vtk.reference(0.);self.loc.FindClosestPoint([float(p[0]),float(p[1]),0.],q,cid,sid,ds);ci=int(cid);tri=self.tri[ci];co=self.coords[tri];w=np.linalg.solve(np.c_[(co[1]-co[0]),(co[2]-co[0])],np.array(q[:2])-co[0]);weights=np.r_[1-w.sum(),w];uv=weights@UV[tri];fi=int(FI[self.ids[ci]])
   for it in range(8):
    pt=gp_Pnt();du=gp_Vec();dv=gp_Vec();adaptors[fi].D1(float(uv[0]),float(uv[1]),pt,du,dv);xyz=np.array(pt.Coord())/1000;der=np.array([du.Coord(),dv.Coord()]).T/1000
    if self.mode=='body':
     cz=center(xyz[0]);ii=np.clip(np.searchsorted(CX,xyz[0])-1,0,len(CX)-2);slope=(CZ[ii+1]-CZ[ii])/(CX[ii+1]-CX[ii]);dz=xyz[2]-cz;yy=xyz[1];coord=np.array([xyz[0]/5,np.arctan2(dz,yy)]);J=np.vstack([der[0]/5,(yy*(der[2]-slope*der[0])-dz*der[1])/(yy*yy+dz*dz)])
    else:
     axes=[0,1] if self.mode=='xy' else [0,2];coord=xyz[axes];J=der[axes]
    residual=coord-np.array(q[:2])
    if np.linalg.norm(residual)<1e-12:break
    try:uv-=np.linalg.solve(J,residual)
    except np.linalg.LinAlgError:break
   result.append(xyz);fid.append(fi);UVout.append(uv);outside.append(np.sqrt(float(ds)))
  return np.array(result),np.array(fid),np.array(UVout),np.array(outside)
bodychart=Chart(bodymask,body_coordinates(V))
def nearest_body(points):
 out=[]
 for p in points:
  q=[0.,0.,0.];cid=vtk.reference(0);sid=vtk.reference(0);ds=vtk.reference(0.);bodyloc.FindClosestPoint(p,q,cid,sid,ds);ids=T[np.flatnonzero(bodymask)[int(cid)]];tri=V[ids];mat=np.c_[tri[1]-tri[0],tri[2]-tri[0]];w=np.linalg.lstsq(mat,np.array(q)-tri[0],rcond=None)[0];uv=np.r_[1-w.sum(),w]@UV[ids];fi=int(FI[np.flatnonzero(bodymask)[int(cid)]]);out.append(np.array(faces[fi].positionAt(*uv).toTuple())/1000)
 return np.array(out)
edges={}
for fi,f in enumerate(faces):
 for e in f.Edges():edges.setdefault(e.hashCode(),[e,set()])[1].add(fi)
def chain(ids,start):
 es=[e for e,adj in edges.values() if adj&body and adj&set(ids)];ordered=[];current=np.array(start)
 while es:
  ends=[np.array([e.startPoint().toTuple(),e.endPoint().toTuple()])/1000 for e in es];k=np.argmin([np.linalg.norm(x-current,axis=1).min() for x in ends]);e=es.pop(k);rev=np.linalg.norm(ends[k][1]-current)<np.linalg.norm(ends[k][0]-current);ordered.append((e,rev));current=ends[k][0 if rev else 1]
 lengths=np.array([e.Length()/1000 for e,rev in ordered]);return ordered,np.r_[0,np.cumsum(lengths)]/lengths.sum()
chains={'WU':chain([12,22,33],[37.004,3.055,3.570]),'WL':chain([28,20,29],[24.701,3.058,4.275]),'WC':chain([25],[37.004,3.055,3.565]),'HU':chain([48],[62.199,.862,6.568]),'HL':chain([50],[56.891,1.805,6.493]),'HC':chain([49],[62.199,.862,6.563]),'VU':chain([51],[52.601,0,8.639]),'VC':chain([52],[61.3,.003,7.859])}
def chord_fraction(label,t):
 if label=='V':
  if t<=1:return .12*np.sin(np.pi*t/2)**2
  if t<=2:return .12+.6*(t-1)
  return .72+.28*(t-2)
 if t<=1:return 1-.28*t
 if t<=2:return .72-.6*(t-1)
 if t<=3:return .12*np.cos(np.pi*(t-2)/2)**2
 if t<=4:return .12*np.sin(np.pi*(t-3)/2)**2
 if t<=5:return .12+.6*(t-4)
 return .72+.28*(t-5)
def at_fraction(key,f):
 chain,arc=chains[key];i=min(np.searchsorted(arc,f,side='right')-1,len(chain)-1);i=max(i,0);u=np.clip((f-arc[i])/(arc[i+1]-arc[i]),0,1);e,rev=chain[i];return np.array(e.positionAt(float(1-u if rev else u)).toTuple())/1000
@lru_cache(None)
def seam(label,t):
 if label=='V':key='VU' if t<=3 else 'VC';fraction=t-3 if t>3 else None
 elif t<=3:key=label+'U';fraction=None
 elif t<=6:key=label+'L';fraction=None
 else:key=label+'C';fraction=(t-6)/2
 if fraction is not None:return at_fraction(key,fraction)
 a,b=at_fraction(key,0),at_fraction(key,1);q=chord_fraction(label,t);goal=min(a[0],b[0])+q*abs(b[0]-a[0]);f=brentq(lambda f:at_fraction(key,f)[0]-goal,0,1,xtol=1e-12);return at_fraction(key,f)
