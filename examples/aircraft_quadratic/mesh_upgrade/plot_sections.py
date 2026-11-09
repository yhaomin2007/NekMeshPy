"""Plot a matched geometric section of both corner grids."""
from pathlib import Path
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
root=Path(sys.argv[1]); out=Path('deliver/final/aircraft_quadratic')
out.mkdir(parents=True,exist_ok=True)
d=np.load(root/'aircraft_tail_refined_package/output/tail_refined.npz')
h=d['hexes']; ids=d['layer_nodes']; p=d['points']
b=p.copy(); b[ids]=np.load('mesh_upgrade/candidate_checked.npz')['layers']
cells=vtk.vtkCellArray()
cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,8*(len(h)+1),8,dtype=np.int64),deep=True),
              numpy_to_vtkIdTypeArray(h.ravel(),deep=True))
def lines(x):
    g=vtk.vtkUnstructuredGrid(); pts=vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(x,deep=True));g.SetPoints(pts);g.SetCells(vtk.VTK_HEXAHEDRON,cells)
    plane=vtk.vtkPlane();plane.SetNormal(0,0,1);plane.SetOrigin(0,0,4.5)
    cut=vtk.vtkCutter();cut.SetInputData(g);cut.SetCutFunction(plane);cut.GenerateTrianglesOff();cut.Update()
    poly=cut.GetOutput()
    xyz=vtk_to_numpy(poly.GetPoints().GetData())[:,:2]
    e=poly.GetPolys()
    flat=vtk_to_numpy(e.GetConnectivityArray()); offsets=vtk_to_numpy(e.GetOffsetsArray())
    nxt=np.arange(len(flat))+1;nxt[offsets[1:]-1]=offsets[:-1]
    pairs=np.unique(np.sort(np.stack([flat,flat[nxt]],axis=1),axis=1),axis=0)
    return xyz[pairs]
fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained',sharex=True,sharey=True)
for ax,x,title in zip(axes,[p,b],['Accepted corner grid','Redistributed corner grid']):
    seg=lines(x); use=np.all((seg[:,:,0]>=-110)&(seg[:,:,0]<=180)&(abs(seg[:,:,1])<=145),axis=1)
    ax.add_collection(LineCollection(seg[use],colors='#213a4d',linewidths=.22))
    ax.set(xlim=(-110,180),ylim=(-145,145),aspect='equal',xlabel='x [m]',title=title)
axes[0].set_ylabel('y [m]')
fig.suptitle('Outer mesh redistribution | x-y section at z = 4.5 m')
fig.savefig(out/'aircraft_outer_sections.png',dpi=210)
