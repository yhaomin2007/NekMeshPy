"""Project shared wall edge midpoints and face centers to the recovered CAD."""
import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
import cadquery as cq
from OCP.gp import gp_Pnt, gp_Vec, gp_Pnt2d
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN, TopAbs_ON

root=Path(sys.argv[1])
d=np.load(root/'aircraft_tail_refined_package/output/tail_refined.npz')
wall=d['points'][d['layer_nodes'][0]]
quads=d['wall_quads']
edges,idx=np.unique(np.sort(np.stack([quads,np.roll(quads,-1,axis=1)],axis=-1).reshape(-1,2),axis=1),axis=0,return_inverse=True)
query=np.vstack([wall[edges].mean(axis=1),wall[quads].mean(axis=1)])
a=np.load(root/'aircraft_surface_tips/output/cad_atlas.npz')
faces=cq.Shape.importBrep(str(root/'aircraft_surface_tips/reference.brep')).Faces()
p=vtk.vtkPoints();p.SetData(numpy_to_vtk(a['points'],deep=True))
tri=a['triangles']; ids=a['face_id']
cells=vtk.vtkCellArray(); cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,3*len(tri)+1,3,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(tri.astype(np.int64).ravel(),deep=True))
pd=vtk.vtkPolyData();pd.SetPoints(p);pd.SetPolys(cells)
loc=vtk.vtkCellLocator();loc.SetDataSet(pd);loc.CacheCellBoundsOn();loc.BuildLocator()
ad=[BRepAdaptor_Surface(f.wrapped) for f in faces]
verts=a['points']; uv_all=a['uv']
half=query.copy();half[:,1]=abs(half[:,1])
_,first,inv=np.unique(np.round(half,10),axis=0,return_index=True,return_inverse=True)

def project(pt):
    p=pt.copy();p[1]=abs(p[1])
    near=[0.,0.,0.];cell=vtk.reference(0);sub=vtk.reference(0);dist=vtk.reference(0.)
    loc.FindClosestPoint(p,near,cell,sub,dist)
    face=int(ids[int(cell)])
    t=tri[int(cell)]
    x=verts[t]
    uv=uv_all[t]
    ww=np.linalg.lstsq((x[1:]-x[0]).T,np.asarray(near)-x[0],rcond=None)[0]
    uv=uv[0]+ww@(uv[1:]-uv[0])
    initial=uv.copy()
    gp,du,dv=gp_Pnt(),gp_Vec(),gp_Vec()
    for _ in range(5):
        ad[face].D1(*map(float,uv),gp,du,dv)
        xyz=np.array(gp.Coord())/1000
        jac=np.array([du.Coord(),dv.Coord()]).T/1000
        step=np.linalg.lstsq(jac,p-xyz,rcond=None)[0]
        uv+=step
        if np.linalg.norm(jac@step)<1e-9:break
    best=np.inf
    xyz=p.copy()
    for candidate in (uv,initial,.99*initial+.01*uv_all[t].mean(0),uv_all[t].mean(0)):
        if not np.isfinite(candidate).all():continue
        state=BRepClass_FaceClassifier(faces[face].wrapped,gp_Pnt2d(*map(float,candidate)),1.e-6).State()
        if state not in (TopAbs_IN,TopAbs_ON):continue
        point=np.array(ad[face].Value(*map(float,candidate)).Coord())/1000
        dd=np.linalg.norm(point-p)
        if dd<best:xyz=point; best=dd
        if best<=np.sqrt(float(dist))+1.e-5:break
    if not np.isfinite(best):face=-1
    if pt[1]<0:xyz[1]*=-1
    if abs(pt[1])<1e-10:xyz[1]=0
    return xyz,face

if __name__ == "__main__":
    import multiprocessing as mp
    results=[]
    with mp.get_context("fork").Pool(6) as pool:
        for n, result in enumerate(pool.imap(project,half[first],chunksize=48),1):
            results.append(result)
            if n%2500==0: print("projected unique nodes",n,flush=True)
    target,face_ids=zip(*results)
    target=np.array(target)[inv]; target[query[:,1]<0,1]*=-1
    delta=target-query
    np.savez('mesh_upgrade/projection_bounded.npz',edges=edges,edge_index=idx.reshape(-1,4),delta=delta,face_id=np.array(face_ids)[inv])
    print('projection displacement [m]',np.quantile(np.linalg.norm(delta,axis=1),[0,.5,.95,.99,1]),flush=True)
