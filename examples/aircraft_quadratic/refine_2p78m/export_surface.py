from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
P=Path('/mnt/data/aircraft_refined_regenerated');N=43424
m=np.memmap(P/'aircraft.re2',dtype='<f8',mode='r',offset=84,shape=(N,25))
d=np.fromfile(P/'aircraft_q2.bin','<f8').reshape(N,5,3)
wall=np.stack([m[:,1:5],m[:,9:13],m[:,17:21]],axis=-1)
xyz=wall.reshape(-1,3)
_,unique_idx,conn=np.unique(np.round(xyz,9),axis=0,return_index=True,return_inverse=True)
cornerpoints=xyz[unique_idx]
conn=conn.reshape(N,4)
edge=np.stack((conn,np.roll(conn,-1,axis=1)),axis=-1).reshape(-1,2)
key=np.sort(edge,axis=1)
ukey,edgeconn=np.unique(key,axis=0,return_inverse=True)
edgecoord=(wall+np.roll(wall,-1,axis=1))*.5+d[:,:4,:]
edgepoints=np.zeros((len(ukey),3)); np.add.at(edgepoints,edgeconn,edgecoord.reshape(-1,3));edgepoints/=2
edgeconn=edgeconn.reshape(N,4)
facepoints=wall.mean(axis=1)+d[:,4,:]
points=np.vstack([cornerpoints,edgepoints,facepoints])
cell=np.column_stack([conn,edgeconn+len(cornerpoints),np.arange(N)+len(cornerpoints)+len(ukey)]).astype(np.int64)
p=vtk.vtkPoints();p.SetData(numpy_to_vtk(points,deep=True))
ids=vtk.vtkCellArray();ids.SetData(numpy_to_vtkIdTypeArray(np.arange(0,9*N+1,9,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(cell.ravel(),deep=True))
grid=vtk.vtkUnstructuredGrid();grid.SetPoints(p);grid.SetCells(vtk.VTK_BIQUADRATIC_QUAD,ids)
grid.GetCellData().AddArray(numpy_to_vtk(np.arange(N,dtype=np.int32),deep=True,array_type=vtk.VTK_INT))
arr=grid.GetCellData().GetArray(0);arr.SetName('wall_quad_id')
w=vtk.vtkXMLUnstructuredGridWriter();w.SetFileName(str(P/'aircraft_wall_quadratic.vtu'));w.SetCompressorTypeToZLib();w.SetInputData(grid);ret=w.Write()
assert ret==1
print('surface quadratic quad file',P/'aircraft_wall_quadratic.vtu','points',len(points),'cells',N)
