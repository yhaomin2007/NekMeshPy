"""Export the current linear quad surface as VTU without changing geometry."""
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
O=Path(__file__).resolve().parent/'output'
r=vtk.vtkXMLPolyDataReader();r.SetFileName(str(O/'aircraft_surface_tips_corrected.vtp'));r.Update();s=r.GetOutput()
g=vtk.vtkUnstructuredGrid();g.SetPoints(s.GetPoints());g.SetCells(vtk.VTK_QUAD,s.GetPolys())
g.GetPointData().DeepCopy(s.GetPointData());g.GetCellData().DeepCopy(s.GetCellData());g.GetFieldData().DeepCopy(s.GetFieldData())
w=vtk.vtkXMLUnstructuredGridWriter();w.SetFileName(str(O/'aircraft_surface_tips_corrected.vtu'));w.SetInputData(g);assert w.Write()
r=vtk.vtkXMLUnstructuredGridReader();r.SetFileName(w.GetFileName());r.Update();result=r.GetOutput()
assert result.GetNumberOfPoints()==s.GetNumberOfPoints()==8834
assert result.GetNumberOfCells()==s.GetNumberOfCells()==8832
assert all(result.GetCellType(i)==vtk.VTK_QUAD for i in range(result.GetNumberOfCells()))
assert np.array_equal(vtk_to_numpy(result.GetPoints().GetData()),vtk_to_numpy(s.GetPoints().GetData()))
assert np.array_equal(vtk_to_numpy(result.GetCells().GetConnectivityArray()),vtk_to_numpy(s.GetPolys().GetConnectivityArray()))
for a,b in [(result.GetPointData(),s.GetPointData()),(result.GetCellData(),s.GetCellData())]:
 assert a.GetNumberOfArrays()==b.GetNumberOfArrays()
 for i in range(b.GetNumberOfArrays()):
  assert a.GetArrayName(i)==b.GetArrayName(i)
  assert np.array_equal(vtk_to_numpy(a.GetArray(i)),vtk_to_numpy(b.GetArray(i)),equal_nan=True)
print('VTU round-trip preserves all points, quads and validation arrays.')
