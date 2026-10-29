from pathlib import Path
import numpy as np,vtk,sys
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'));P=np.load(O/(sys.argv[1] if len(sys.argv)>1 else 'gll_constrained.npz'))['points'];H=d['hexes'];points=vtk.vtkPoints();points.SetData(numpy_to_vtk(P,deep=True));cells=vtk.vtkCellArray();cells.SetCells(len(H),numpy_to_vtkIdTypeArray(np.c_[np.full(len(H),8),H].ravel().astype(np.int64),deep=True));g=vtk.vtkUnstructuredGrid();g.SetPoints(points);g.SetCells(vtk.VTK_HEXAHEDRON,cells)
for name,data in [('radial_layer',np.repeat(np.arange(60),len(d['wall_quads'])))]:
 ar=numpy_to_vtk(data,deep=True);ar.SetName(name);g.GetCellData().AddArray(ar)
fields=O/'volume_quality_fields.npz'
if fields.exists():
 f=np.load(fields)
 for name in f.files:
  ar=numpy_to_vtk(f[name],deep=True);ar.SetName(name);g.GetCellData().AddArray(ar)
w=vtk.vtkXMLUnstructuredGridWriter();w.SetFileName(str(O/'aircraft_volume_deformed.vtu'));w.SetInputData(g);assert w.Write();print('Exported',len(H),'hexes')
