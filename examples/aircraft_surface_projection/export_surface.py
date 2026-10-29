from cad_surface import *
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
s=np.load(O/'surface_tips_projected.npz');b=np.load(O/'base_surface.npz');d=np.load(O/'surface_input.npz');P=s['points'];Q=s['quads'];f=np.load(O/'tip_validation_fields.npz');g=vtk.vtkPolyData();pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(P,deep=True));g.SetPoints(pts);cells=vtk.vtkCellArray();cells.SetCells(len(Q),numpy_to_vtkIdTypeArray(np.c_[np.full(len(Q),4),Q].ravel().astype(np.int64),deep=True));g.SetPolys(cells)
def field(name,a,cell=False):
 ar=numpy_to_vtk(a,deep=True);ar.SetName(name);(g.GetCellData() if cell else g.GetPointData()).AddArray(ar)
for name,key in [('CAD_distance_quad_centre_m','centre_error'),('CAD_normal_surface_scaled_Jacobian','surface_quality')]:
 a=np.full(len(Q),np.nan);a[f['checked_quad_ids']]=f[key];field(name,a,True)
a=np.full(len(P),np.nan);a[f['node_ids']]=f['node_error'];field('CAD_native_node_residual_m',a)
a=np.full(len(P),-1,dtype=np.int32);a[f['node_ids']]=f['node_face'];field('CAD_face_id',a)
field('displacement_from_approved_baseline_m',np.linalg.norm(P-b['points'],axis=1));field('source_volume_wall_node_id',d['source_volume_wall_node_ids'].astype(np.int64))
w=vtk.vtkXMLPolyDataWriter();w.SetFileName(str(O/'aircraft_surface_tips_corrected.vtp'));w.SetInputData(g);assert w.Write()
