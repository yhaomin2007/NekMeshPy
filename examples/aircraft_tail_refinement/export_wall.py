"""Export the refined wall with exact volume-node correspondence."""
from pathlib import Path
import numpy as np,vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
O=Path(__file__).resolve().parent/'output';d=np.load(O/'tail_refined.npz');ids=d['layer_nodes'][0];P=d['points'][ids];Q=d['wall_quads'];g=vtk.vtkUnstructuredGrid();pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(P,deep=True));g.SetPoints(pts);cells=vtk.vtkCellArray();cells.SetCells(len(Q),numpy_to_vtkIdTypeArray(np.c_[np.full(len(Q),4),Q].ravel().astype(np.int64),deep=True));g.SetCells(vtk.VTK_QUAD,cells)
def field(name,data,cell=False):
 a=numpy_to_vtk(np.ascontiguousarray(data),deep=True);a.SetName(name);(g.GetCellData() if cell else g.GetPointData()).AddArray(a)
field('volume_node_id',ids);field('parent_surface_quad',d['parent_quad'],True);e=np.linalg.norm(P[Q]-np.roll(P[Q],-1,axis=1),axis=2);field('edge_aspect_ratio',e.max(1)/e.min(1),True);families=np.unique(d['family']);field('surface_family_id',np.searchsorted(families,d['family']),True);labels=vtk.vtkStringArray();labels.SetName('surface_family_names')
for f in families:labels.InsertNextValue(str(f))
g.GetFieldData().AddArray(labels);w=vtk.vtkXMLUnstructuredGridWriter();w.SetFileName(str(O/'aircraft_wall_tail_refined.vtu'));w.SetInputData(g);assert w.Write();reader=vtk.vtkXMLUnstructuredGridReader();reader.SetFileName(w.GetFileName());reader.Update();out=reader.GetOutput();assert out.GetNumberOfCells()==len(Q) and out.GetNumberOfPoints()==len(P);assert np.array_equal(vtk_to_numpy(out.GetPoints().GetData()),P);assert np.array_equal(vtk_to_numpy(out.GetCells().GetConnectivityArray()).reshape(-1,4),Q);print('Wall VTU round trip passed:',len(Q),'quads,',len(P),'nodes.')
