from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(R/'inputs/volume_parent.npz');old=np.load(R/'inputs/volume_parent.npz')['points'];s=np.load(O/'tail_refined.npz');target=np.array([61.,0,6.8]);back=np.array([.65,-.72,.25]);back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right);basis=np.stack([right,up,back],axis=1);fig,axs=plt.subplots(1,2,figsize=(16,7))
for ax,(P,Q,L,fam,title) in zip(axs,[(old,d['wall_quads'],d['layer_nodes'],d['family'],'Previous: 529,920 hexes'),(s['points'],s['wall_quads'],s['layer_nodes'],s['family'],'Directional refinement: 651,360 hexes')]):
 v=P[L[0,Q]];center=v.mean(1);mask=center[:,0]>51;normal=np.cross(v[:,1]-v[:,0],v[:,3]-v[:,0])+np.cross(v[:,2]-v[:,1],v[:,0]-v[:,1]);mask&=(normal@back)>0;vv=v[mask];proj=(vv-target)@basis;order=np.argsort(proj[:,:,2].mean(1));light=.75+.2*abs((normal[mask]/np.linalg.norm(normal[mask],axis=1)[:,None])@back);colors=np.c_[.83*light,.85*light,.9*light,np.ones(len(vv))];ax.add_collection(PolyCollection(proj[order,:,:2],facecolors=colors[order],edgecolors='#193b83',linewidths=.42));ax.set_xlim(-4.0,4.3);ax.set_ylim(-2.2,4.3);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=13)
fig.suptitle('Aft-fuselage stretching reduced; existing geometry preserved',fontsize=16);fig.tight_layout();fig.savefig(O/'tail_refinement_comparison.png',dpi=160);plt.close(fig)
