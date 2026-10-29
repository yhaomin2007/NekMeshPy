"""Sections of actual linear volume elements before/after multilayer deformation."""
from pathlib import Path
import numpy as np,vtk,sys
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'));old=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'))['points'];P=np.load(O/(sys.argv[1] if len(sys.argv)>1 else 'gll_constrained.npz'))['points'];H=d['hexes'];Q=d['wall_quads'];L=d['layer_nodes'];surf=np.load(R/'inputs/approved_surface.npz')['points'];cases=[('Wing/body',32.,(-7,7),(-1,9)),('Horizontal tail/body',60.5,(-6,6),(3,10)),('Vertical tail/body',62.14,(-2,2),(6.5,11))]
def grid(pp):
 pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(pp,deep=True));c=vtk.vtkCellArray();c.SetCells(len(H),numpy_to_vtkIdTypeArray(np.c_[np.full(len(H),8),H].ravel().astype(np.int64),deep=True));g=vtk.vtkUnstructuredGrid();g.SetPoints(pts);g.SetCells(vtk.VTK_HEXAHEDRON,c);return g
def cut(g,x):
 plane=vtk.vtkPlane();plane.SetNormal(1,0,0);plane.SetOrigin(x,0,0);c=vtk.vtkCutter();c.SetCutFunction(plane);c.SetInputData(g);c.GenerateTrianglesOff();c.Update();o=c.GetOutput();lines=[]
 for i in range(o.GetNumberOfCells()):
  ce=o.GetCell(i);ids=ce.GetPointIds();v=np.array([o.GetPoint(ids.GetId(j)) for j in range(ids.GetNumberOfIds())]);lines.append(np.r_[v[:,1:3],v[:1,1:3]])
 return lines
g0,g1=grid(old),grid(P);fig,axs=plt.subplots(2,3,figsize=(16,10))
for col,(name,x,ylim,zlim) in enumerate(cases):
 for row,g in enumerate([g0,g1]):
  ax=axs[row,col];ax.add_collection(LineCollection(cut(g,x),colors='#3e5c85',linewidths=.35));ax.set_xlim(*ylim);ax.set_ylim(*zlim);ax.set_aspect('equal');ax.set_title(f'{name}, x={x:g} m\n'+('Previous volume' if row==0 else 'Multilayer volume candidate'));ax.set_xlabel('y [m]');ax.set_ylabel('z [m]');ax.grid(alpha=.15)
fig.suptitle('Actual element sections — approved surface held fixed; 30 radial layers adjusted',fontsize=15);fig.tight_layout();fig.savefig(O/'volume_layer_sections.png',dpi=150);plt.close(fig)
# Close-up of all tips, actual first six layers in an x-normal slice.
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,(title,x,ylim,zlim) in zip(axs,[('Wing tip',46.4,(27.5,30.0),(6.0,7.6)),('Horizontal tail tip',65.5,(9,11.5),(7,8.6)),('Vertical tail tip',64.3,(-.7,.7),(17.2,19.1))]):
 ax.add_collection(LineCollection(cut(g1,x),colors='#3e5c85',linewidths=.45));ax.set_xlim(*ylim);ax.set_ylim(*zlim);ax.set_aspect('equal');ax.set_title(title+f', x={x:g} m');ax.set_xlabel('y [m]');ax.set_ylabel('z [m]');ax.grid(alpha=.15)
fig.suptitle('Volume layers at the corrected tips');fig.tight_layout();fig.savefig(O/'volume_tip_sections.png',dpi=160)
