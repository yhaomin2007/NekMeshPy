from pathlib import Path
import json,argparse,numpy as np,vtk
a=argparse.ArgumentParser();a.add_argument("--top",action="store_true");args=a.parse_args()
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
root=Path(__file__).resolve().parent;out=root/'output'
r=vtk.vtkXMLUnstructuredGridReader();r.SetFileName(str(out/'aircraft_farfield.vtu'));r.Update();g=r.GetOutput()
plane=vtk.vtkPlane();plane.SetOrigin(0,0,6);plane.SetNormal(*( (0,0,1) if args.top else (0,1,0)));c=vtk.vtkCutter();c.SetCutFunction(plane);c.SetInputData(g);c.GenerateTrianglesOff();c.Update();s=c.GetOutput();P=vtk_to_numpy(s.GetPoints().GetData());polys=[];ids=vtk.vtkIdList();ca=s.GetPolys();ca.InitTraversal()
while ca.GetNextCell(ids):polys.append(P[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]][:,[0,1] if args.top else [0,2]])
reg=vtk_to_numpy(s.GetCellData().GetArray('region'));colors=np.array(['#6b9fbe','#e1b56f','#d1e0e8'])[reg-1]
fig=plt.figure(figsize=(16,10),facecolor='#f7f9fb');gs=fig.add_gridspec(2,2,height_ratios=[1,1],left=.065,right=.97,bottom=.09,top=.85,hspace=.32,wspace=.23)
for ax,bounds,title in [(fig.add_subplot(gs[:,0]),[(-730,800),(-760,770)],'Full computational domain'),(fig.add_subplot(gs[0,1]),[(-30,100),(-60,70)],'Transition around aircraft'),(fig.add_subplot(gs[1,1]),[(0,70),(-5,25)],'Near-body / transition interface')]:
    ax.add_collection(PolyCollection(polys,facecolors=colors,edgecolors='#416171',linewidths=.20));ax.set_xlim(*bounds[0]);ax.set_ylim(*((-40,40) if args.top and title=='Near-body / transition interface' else bounds[1]));ax.set_aspect('equal');ax.set_title(title,loc='left',fontsize=12);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]' if args.top else 'z [m]')
info=json.loads((out/'checks.json').read_text());fig.suptitle('Full-aircraft all-hex mesh with far field',x=.065,ha='left',fontsize=23,weight='bold',color='#203d4a');fig.text(.065,.90,f"{info['hexes']:,} hexes | nominal far-field radius {info['radius_m']:g} m | centre-plane slice | geometry remains approximate",fontsize=12,color='#426373');fig.text(.065,.025,'Blue: unchanged near-body mesh. Orange: connecting transition. Light blue: stretched outer grid. No flow simulation performed.',fontsize=10.5,color='#79541d');fig.savefig(out/('farfield_top_slice.png' if args.top else 'farfield_mesh_preview.png'),dpi=170,facecolor=fig.get_facecolor())
print('Mesh preview saved',flush=True)
