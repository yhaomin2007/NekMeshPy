"""Local shared-node trilinear-hex untangling, with the approved wall fixed."""
from pathlib import Path
import argparse,time,json
import numpy as np
from scipy.optimize import minimize
from quality import S
R=Path(__file__).resolve().parent;O=R/'output';a=argparse.ArgumentParser();a.add_argument('--input',default='harmonic_30_2_1.npz');a.add_argument('--rounds',type=int,default=6);a.add_argument('--iterations',type=int,default=350);a.add_argument('--floor',type=float,default=.06);a.add_argument('--name',default='optimized');a.add_argument('--gll',type=int,default=0);args=a.parse_args()
d=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'));H=d['hexes'];L=d['layer_nodes'];Q=d['wall_quads'];P=np.load(O/args.input)['points'].copy();target=np.load(R/'inputs/approved_surface.npz')['points'];assert np.array_equal(P[L[0]],target);depth=30;H=H[:depth*len(Q)];wallmask=np.zeros(len(P),bool);wallmask[L[0]]=True;wallmask[L[depth:]]=True
# Corners and centre; validation later uses GLL N=3,5,7.
T=np.r_[S,np.zeros((1,3))]
if args.gll:
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
 from nekmeshpy.core.interp import gll_nodes
 g=2*gll_nodes(args.gll)-1;T=np.array(np.meshgrid(g,g,g,indexing='ij')).reshape(3,-1).T
D=np.array([[S[:,j]*np.prod(1+S[:,[k for k in range(3) if k!=j]]*t[[k for k in range(3) if k!=j]],axis=1)/8 for j in range(3)] for t in T]).transpose(0,2,1)
def qualities(P,h):
 out=[]
 for k in range(0,len(h),2048):
  J=np.einsum('eni,tnj->etij',P[h[k:k+2048]],D);A,B,C=J[:,:,:,0],J[:,:,:,1],J[:,:,:,2];det=np.sum(np.cross(A,B)*C,axis=-1);length=np.linalg.norm(J,axis=2);out.append((det/np.maximum(np.prod(length,axis=-1),1e-30)).min(1))
 return np.concatenate(out)
old=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'))['points']
# Keep radial derivative lengths above 20% of the historical layer thickness.
radial_floor=[]
for k in range(0,len(H),2048):
 J=np.einsum('eni,tnj->etij',old[H[k:k+2048]],D);radial_floor.append(.2*np.linalg.norm(J[:,:,:,2],axis=2))
radial_floor=np.concatenate(radial_floor)
def thin_cells(P,h,rf):
 out=[]
 for k in range(0,len(h),2048):
  J=np.einsum('eni,tnj->etij',P[h[k:k+2048]],D);length=np.linalg.norm(J,axis=2);out.append(np.minimum((length[:,:,2]/rf[k:k+2048]).min(1),(length/5e-5).min((1,2))))
 return np.concatenate(out)
start=time.time();history=[]
for sweep in range(args.rounds):
 q=qualities(P,H);thin=thin_cells(P,H,radial_floor);bad=(q<args.floor*.8)|(thin<1.001);print('round',sweep,'minimum',q.min(),'inverted',sum(q<=0),'below floor',sum(bad),'seconds',round(time.time()-start),flush=True)
 if not np.any(bad):break
 active=np.unique(H[bad]);
 for _ in range(1):
  mask=np.zeros(len(P),bool);mask[active]=True;active=np.unique(H[np.any(mask[H],axis=1)])
 active=active[~wallmask[active]];mask=np.zeros(len(P),bool);mask[active]=True;hi=np.flatnonzero(np.any(mask[H],axis=1));h=H[hi];nodes=np.unique(h);lookup=np.full(len(P),-1);lookup[nodes]=np.arange(len(nodes));hc=lookup[h];free=lookup[active];ref=P[nodes].copy();points=ref.copy();node_scale=np.zeros(len(nodes));counts=np.zeros(len(nodes));span=np.linalg.norm(P[h[:,4:]]-P[h[:,:4]],axis=-1).mean(1)
 for col in hc.T:np.add.at(node_scale,col,span);np.add.at(counts,col,1)
 node_scale=np.maximum(.005,node_scale/np.maximum(counts,1));scale=node_scale[free];x0=np.zeros((len(free),3));idx=hc.ravel();calls=[0];best=[np.inf,None];last=[time.time()]
 def objective(x):
  x=x.reshape(-1,3);points[free]=ref[free]+x*scale[:,None];J=np.einsum('eni,tnj->etij',points[hc],D);A,B,C=J[:,:,:,0],J[:,:,:,1],J[:,:,:,2];len2=np.maximum(np.sum(J*J,axis=2),1e-24);den=np.sqrt(np.prod(len2,axis=-1));cof=np.stack([np.cross(B,C),np.cross(C,A),np.cross(A,B)],axis=-1);det=np.sum(A*cof[:,:,:,0],axis=-1);q=det/den;gap=np.minimum(q-args.floor,0);loss=np.sum(gap*gap);der=cof/den[:,:,None,None]-q[:,:,None,None]*J/len2[:,:,None,:];gradJ=2*gap[:,:,None,None]*der;length=np.sqrt(len2);floors=np.full_like(length,5e-5);floors[:,:,2]=radial_floor[hi];lgap=np.minimum(length/floors-1,0);loss+=np.sum(lgap*lgap);gradJ+=2*lgap[:,:,None,:]*J/(length*floors)[:,:,None,:];local=np.einsum('etij,tnj->eni',gradJ,D);grad=np.stack([np.bincount(idx,weights=local[:,:,j].ravel(),minlength=len(nodes)) for j in range(3)],axis=1)[free]*scale[:,None]
  tether=1e-7;loss+=tether*np.sum(x*x);grad+=2*tether*x;calls[0]+=1
  if loss<best[0]:best[:]=[loss,x.copy()]
  if time.time()-last[0]>25:print(' optimize',sweep,'evaluations',calls[0],'loss',float(loss),'min',q.min(),'bad',np.sum(q.min(1)<=0),flush=True);last[0]=time.time()
  return loss,grad.ravel()
 # Directional finite difference checks the analytic gradient on the first round.
 if sweep==0:
  f,g=objective(x0.ravel());rng=np.random.default_rng(4);direction=rng.normal(size=x0.size);direction/=np.linalg.norm(direction);eps=1e-8;fd=(objective(x0.ravel()+eps*direction)[0]-objective(x0.ravel()-eps*direction)[0])/(2*eps);ad=g@direction;print('gradient check',fd,ad,'relative',abs(fd-ad)/max(1,abs(fd)),flush=True);assert abs(fd-ad)<.01*max(1,abs(fd))
 print('subproblem',len(active),'free nodes',len(h),'hexes',flush=True);res=minimize(objective,x0.ravel(),jac=True,method='L-BFGS-B',options={'maxiter':args.iterations,'ftol':1e-12,'gtol':1e-8,'maxcor':12,'maxls':30});x=best[1];P[active]=ref[free]+x*scale[:,None];q=qualities(P,H);history.append({'round':sweep,'minimum':float(q.min()),'inverted':int(sum(q<=0)),'minimum_thickness_ratio':float(thin_cells(P,H,radial_floor).min()),'loss':float(best[0]),'iterations':res.nit,'message':str(res.message)});print(history[-1],flush=True);np.savez_compressed(O/(args.name+'.npz'),points=P);(O/(args.name+'.json')).write_text(json.dumps(history,indent=2));assert np.array_equal(P[L[0]],target)
