"""Rebuild the featured network and render actual mesh faces and cross-sections."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nekmeshpy import hexmesh, quadmesh, writer
from build_network import build
import features

root=Path(__file__).resolve().parent
cfg=json.loads((root/'network.json').read_text())
mesh=build(cfg,root/'output')
ok=bool(hexmesh.is_overlap_free(mesh))
assert ok
checks=json.loads((root/'output/checks.json').read_text())
checks['overlap_check']={'corner_geometry_overlap_free':ok,'curved_global_intersection_certificate':False}
(root/'output/checks.json').write_text(json.dumps(checks,indent=2)+'\n')
writer.to_vtu(features.removed_baffle,str(root/'output/baffle_obstacle.vtu'))


def loops(blocks):
    return [np.concatenate([B[0],B[1:,-1],B[-1,-2::-1],B[-2:0:-1,0]]) for B in blocks]


def face_blocks(surface):
    return quadmesh.element_blocks(surface).reshape(-1,mesh.order+1,mesh.order+1,3)

fig=plt.figure(figsize=(15,8.8),facecolor='#f7f9fb')
gs=fig.add_gridspec(2,2,width_ratios=[1.45,1],left=.035,right=.97,bottom=.10,top=.84,wspace=.20,hspace=.45)
ax=fig.add_subplot(gs[:,0],projection='3d')
ax.set_facecolor('#f7f9fb')
for tag in mesh.face_group_tags:
    s=hexmesh.boundary_mesh(mesh,tag)
    color={'wall_bump':'#ef9b26','baffle':'#db694b','wall':'#75acbc',
           'inlet':'#d9ae51','main_outlet':'#6bad8d','branch_outlet':'#a288b6'}[tag]
    special=tag in ('wall_bump','baffle')
    ax.add_collection3d(Poly3DCollection(loops(face_blocks(s)),facecolor=color,
       edgecolor='#355660' if special else '#73949f',linewidths=.30 if special else .14,
       alpha=.95 if special else .12))
lo=mesh.points.min(axis=0);hi=mesh.points.max(axis=0)
ax.set_xlim(lo[0]-.02,hi[0]+.02);ax.set_ylim(lo[1]-.02,hi[1]+.02);ax.set_zlim(lo[2]-.02,hi[2]+.02)
ax.set_box_aspect(hi-lo,zoom=1.05);ax.view_init(elev=25,azim=-58)
ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_zlabel('z [m]');ax.grid(False)
ax.set_title('Connected network: walls shown translucent',loc='left',fontsize=12,pad=14)
ax.text(0,-.05,.27,'5 mm wall bump',color='#a86808',fontsize=10)
ax.text(.68,.30,.07,'Partial baffle',color='#aa432d',fontsize=10)
for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.fill=False

# Exact mesh-aligned cross-section through the branch centre plane x=0.
ax2=fig.add_subplot(gs[0,1]);ax2.set_facecolor('white')
q=mesh.quad_mesh;corners=mesh.points[q.corners]
mask=(np.max(np.abs(corners[:,:,0]),axis=1)<1e-8)&(corners[:,:,2].min(axis=1)>.159)&(corners[:,:,2].max(axis=1)<.321)
s=quadmesh.select(q,np.flatnonzero(mask))
polys=[P[:,[2,1]]*1000 for P in loops(face_blocks(s))]
ax2.add_collection(PolyCollection(polys,facecolor='#d5e8ee',edgecolor='#517c89',linewidths=.45))
z=np.linspace(.16,.32,400);t=(z-.15)/.18
rad=.05+(.03-.05)*t*t*(3-2*t)
u=(z-.24)/.06;shape=np.maximum(1-u*u,0)**3
ax2.plot(z*1000,-rad*1000,'--',color='#8a9299',lw=1.2,label='Original wall')
ax2.plot(z*1000,(-rad+.005*shape)*1000,color='#de8a13',lw=2.5,label='Bump design profile')
ax2.set_xlim(160,320);ax2.set_ylim(-49,0)
ax2.set_title('A. Branch centre-plane mesh (x = 0)',loc='left',fontsize=12)
ax2.set_xlabel('z [mm]');ax2.set_ylabel('y [mm]')
ax2.legend(loc='upper right',fontsize=8,frameon=False)
ax2.annotate('5 mm inward',xy=(240,-35),xytext=(262,-43),fontsize=9,
             arrowprops={'arrowstyle':'->','color':'#a86808'},color='#a86808')

# Exact cross-section halfway along the mesh-aligned baffle.
ax3=fig.add_subplot(gs[1,1]);ax3.set_facecolor('white')
ya=checks['features']['baffle']['y_start_m'];yb=checks['features']['baffle']['y_end_m']
cut=(ya+yb)/2
for obj,color,edge in [(mesh,'#d5e8ee','#517c89'),(features.removed_baffle,'#d9856e','#9c4c39')]:
    q=obj.quad_mesh; C=obj.points[q.corners]
    ids=np.flatnonzero(np.max(np.abs(C[:,:,1]-cut),axis=1)<1e-8)
    assert len(ids)>0
    blocks=quadmesh.element_blocks(quadmesh.select(q,ids)).reshape(-1,mesh.order+1,mesh.order+1,3)
    polys=[P[:,[0,2]]*1000-np.array([650,0]) for P in loops(blocks)]
    ax3.add_collection(PolyCollection(polys,facecolor=color,edgecolor=edge,linewidths=.45))
ax3.set_xlim(-65,65);ax3.set_ylim(-65,65);ax3.set_aspect('equal')
ax3.set_title('B. Baffle section: 25.2% blockage',loc='left',fontsize=12)
ax3.set_xlabel('x relative to pipe centre [mm]');ax3.set_ylabel('z [mm]')
ax3.text(-56,49,'Fluid',fontsize=10,color='#305f6f')
ax3.text(26,-3,'Solid\nrib',fontsize=9,color='#7d3021')
fig.suptitle('Small features in an all-hex pipe network',x=.035,ha='left',fontsize=22,weight='bold',color='#203c4a')
fig.text(.035,.89,'Smooth wall intrusion + a wall-attached internal obstruction; both are resolved by the fluid mesh.',fontsize=12,color='#46606d')
fig.text(.035,.035,'7,956 fluid hexes  |  Geometry order 3  |  Conformal, one connected volume  |  Min. sampled scaled Jacobian: 0.470',fontsize=11,color='#203c4a')
fig.savefig(root/'output/features_preview.png',dpi=170,facecolor=fig.get_facecolor())
print('Preview saved; corner geometry overlap-free:',ok)
