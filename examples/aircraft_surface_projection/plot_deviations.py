from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
R=Path(__file__).resolve().parent;O=R/'output';s=np.load(O/'surface_tips_projected.npz');d=np.load(O/'surface_input.npz');v=np.load(O/'tip_validation_fields.npz');P=s['points'];Q=s['quads'];ids=v['checked_quad_ids'];err=v['centre_error']*1000;fig,axs=plt.subplots(1,3,figsize=(16,6));norm=Normalize(0,max(err));cmap=plt.get_cmap('magma')
for ax,l,title,axis in zip(axs,'WHV',['Main wing','Horizontal tail','Vertical tail'],[1,1,2]):
 positive=np.all(d['nominal'][Q,1]>=-1e-8,axis=1);allmask=np.char.startswith(s['family'],l+'_')&positive;coords=P[Q][:,:,[axis,0]];ax.add_collection(PolyCollection(coords[allmask],facecolors='#e9edf0',edgecolors='#a8b0b7',linewidths=.2));m=np.char.startswith(s['family'][ids],l+'_')&positive[ids];sel=ids[m];ax.add_collection(PolyCollection(coords[sel],facecolors=cmap(norm(err[m])),edgecolors='#374151',linewidths=.3));ax.autoscale();ax.set_aspect('equal');ax.set_xlabel(('z' if axis==2 else 'y')+' [m]');ax.set_ylabel('x [m]');j=np.argmax(err[m]);p=P[Q[sel[j]]].mean(0);ax.plot(p[axis],p[0],'o',mfc='none',mec='#00bcd4',mew=1.7,ms=9);ax.set_title(f'{title}\nmax sampled centre deviation: {err[m][j]:.2f} mm',fontsize=11);ax.text(.02,.02,'Worst centre [m]:\n'+', '.join(f'{x:.4f}' for x in p),transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axs,label='Direct distance to physical trimmed CAD at quad centre [mm]',shrink=.8);fig.suptitle('Linear-surface deviation locations — positive half; mirrored side has the same geometry\nGrey: outside this validation scope. Centre samples are not a Hausdorff bound.',fontsize=12);fig.savefig(O/'CAD_deviation_locations.png',dpi=160,bbox_inches='tight')
