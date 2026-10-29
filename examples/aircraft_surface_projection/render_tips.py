"""Close-up comparisons of the actual quad mesh at all three CAD tip caps."""
from cad_surface import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
d=np.load(O/'surface_input.npz');s=np.load(O/'surface_tips_projected.npz');base=np.load(O/'base_surface.npz');P=s['points'];old=base['points'];Q=s['quads'];nom=d['nominal'];cases=[('W','Wing tip',14,27.5,[46.6,28.9,6.75],[50,34,9.3]),('H','Horizontal tail tip',47,8.5,[65.5,10.25,7.7],[69,15,10]),('V','Vertical tail tip',53,15.5,[64.2,0,17.9],[68,4,23])]
for label,title,cap,start,target,camera in cases:
 axis=2 if label=='V' else 1;roi=np.char.startswith(s['family'],label+'_')&np.all(abs(nom[Q,axis])>=start-1e-8,axis=1)&np.all(nom[Q,1]>=-1e-8,axis=1);q=Q[roi];target=np.array(target);back=np.array(camera)-target;back/=np.linalg.norm(back);up0=np.array([0,0,1]);right=np.cross(up0,back);right/=np.linalg.norm(right);up=np.cross(back,right);basis=np.stack([right,up,back],axis=1);both=np.concatenate([(pp[np.unique(q)]-target)@basis for pp in [old,P]]);lo=both[:,:2].min(0);hi=both[:,:2].max(0);margin=.04*(hi-lo);cadcurves=[]
 for e,adj in edges.values():
  if cap in adj:cadcurves.append((np.array([e.positionAt(float(t)).toTuple() for t in np.linspace(0,1,81)])/1000-target)@basis)
 fig,axes=plt.subplots(1,2,figsize=(15,6.3),facecolor='white')
 for ax,pp,heading in zip(axes,[old,P],['Previous projected surface','CAD tip-cap and outer-strip correction']):
  # Draw the full surface, then crop the view; no artificial open cut through the cap.
  v=pp[Q];proj=(v-target)@basis;order=np.argsort(proj[:,:,2].mean(1));normal=np.cross(v[:,1]-v[:,0],v[:,3]-v[:,0])+np.cross(v[:,2]-v[:,1],v[:,0]-v[:,1]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30);light=.55+.4*abs(normal@back);colors=np.c_[.81*light,.84*light,.88*light,np.ones(len(v))];ax.add_collection(PolyCollection(proj[order,:,:2],facecolors=colors[order],edgecolors='#17284d',linewidths=.4))
  for curve in cadcurves:ax.plot(curve[:,0],curve[:,1],color='#cc2020',lw=.9)
  ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1]);ax.set_aspect('equal');ax.set_axis_off();ax.set_title(heading,fontsize=12)
 fig.suptitle(title+' — red: original CAD cap boundary',fontsize=15);fig.subplots_adjust(left=.01,right=.99,top=.88,bottom=.02,wspace=.03);fig.savefig(O/(label+'_tip_comparison.png'),dpi=160);plt.close(fig)
def cut(C,x):
 e=np.stack([C,np.roll(C,-1,axis=1)],axis=2);dx=e[:,:,1,0]-e[:,:,0,0];t=np.divide(x-e[:,:,0,0],dx,out=np.full_like(dx,np.nan),where=abs(dx)>1e-12);mask=(t>=0)&(t<1);p=e[:,:,0]+t[:,:,None]*(e[:,:,1]-e[:,:,0]);return [a[b][:,[1,2]] for a,b in zip(p,mask) if sum(b)==2]
fig,axes=plt.subplots(1,3,figsize=(15,5));ccases=[('Wing tip',46.4,(27.5,29.7),(6.3,7.1)),('Horizontal tail tip',65.5,(8.5,10.95),(7.2,8.1)),('Vertical tail tip',64.3,(-.25,.25),(17.2,18.7))];Vfull=np.r_[V,V*[1,-1,1]];Tfull=np.r_[T,T[:,::-1]+len(V)]
for ax,(title,x,ylim,zlim) in zip(axes,ccases):
 for C,color,style,lw,label in [(old[Q],'#888888','--',1.,'Previous surface'),(Vfull[Tfull],'#cc2020','-',1.4,'Original CAD'),(P[Q],'#1269b0','-',1.0,'Corrected surface')]:
  mask=(C[:,:,0].min(1)<=x)&(C[:,:,0].max(1)>=x);ax.add_collection(LineCollection(cut(C[mask],x),colors=color,linestyles=style,linewidths=lw,label=label))
 ax.set_xlim(*ylim);ax.set_ylim(*zlim);ax.set_aspect('equal');ax.set_xlabel('y [m]');ax.set_ylabel('z [m]');ax.set_title(f'{title}\nx = {x:.3f} m');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Native CAD versus linear tip surface sections');fig.tight_layout();fig.savefig(O/'tip_CAD_sections.png',dpi=170);plt.close(fig)
