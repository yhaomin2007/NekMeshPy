"""Shared-node smoothing with sampled orientation and skew barriers."""
from pathlib import Path
import os,argparse,json,time
os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
import numpy as np,jax,jax.numpy as jnp
from scipy.optimize import minimize
p=argparse.ArgumentParser();p.add_argument('--init',default='baseline_transition.npz');p.add_argument('--sphere',default='sphere_balanced.npz');p.add_argument('--out',default='smoothed_transition.npz');p.add_argument('--grid',type=int,default=3);p.add_argument('--iterations',type=int,default=2500);p.add_argument('--smooth',type=float,default=0.);p.add_argument('--quality',type=float,default=.1);args=p.parse_args()
root=Path(__file__).resolve().parent;b=np.load(root/args.init);L=b['layers'];Q=b['quads'];C=b['center'];I=L[0];U=np.load(root/args.sphere)['unit_directions'];O=C+50*U;nr=len(L)-1;n=len(I);s=np.linspace(0,1,nr+1);blend=s**3*(10-15*s+6*s*s);X=L+blend[:,None,None]*(O-L[-1]);X[0]=I;X[-1]=O
H=np.concatenate([np.c_[Q+k*n,Q+(k+1)*n] for k in range(nr)])
S=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
TT=np.array(np.meshgrid(*([np.linspace(-1,1,args.grid)]*3),indexing='ij')).reshape(3,-1).T
DER=np.array([[(S[:,j]*np.prod(1+S[:,[k for k in range(3) if k!=j]]*t[[k for k in range(3) if k!=j]],axis=1)/8) for j in range(3)] for t in TT]).transpose(0,2,1)
def edge_scales(v):
 a=v[Q];return np.c_[.25*(np.linalg.norm(a[:,1]-a[:,0],axis=1)+np.linalg.norm(a[:,2]-a[:,3],axis=1)),.25*(np.linalg.norm(a[:,3]-a[:,0],axis=1)+np.linalg.norm(a[:,2]-a[:,1],axis=1))]
a=edge_scales(I);d=edge_scales(O);ds=np.maximum(np.linalg.norm(O[Q].mean(1)-I[Q].mean(1),axis=1)/nr/2,.02);scale=np.concatenate([np.c_[(1-t)*a+t*d,ds] for t in (s[:-1]+s[1:])/2]);norm=np.maximum(np.linalg.norm(O-I,axis=1)/nr,.1)
JJ=jnp.array(DER);HH=jnp.array(H);SC=jnp.array(scale);II=jnp.array(I);OO=jnp.array(O);NN=jnp.array(norm)
def calc(x):
 P=jnp.concatenate([II[None],x.reshape(nr-1,n,3),OO[None]]);J=jnp.einsum('bni,tnj->btij',P.reshape(-1,3)[HH],JJ)/SC[:,None,None,:];det=jnp.sum(jnp.cross(J[:,:,:,0],J[:,:,:,1])*J[:,:,:,2],axis=2);sj=det/jnp.maximum(jnp.prod(jnp.sqrt(jnp.sum(J*J,axis=2)+1e-20),axis=2),1e-25);return P,det,sj
def energy(x):
 P,det,sj=calc(x);curv=(P[2:]-2*P[1:-1]+P[:-2])/NN[None,:,None]
 return jnp.sum(jnp.minimum(sj-args.quality,0)**2+10000*jnp.minimum(det-.002,0)**2)/len(TT)+args.smooth*jnp.sum(curv*curv)
f=jax.jit(jax.value_and_grad(energy));qf=jax.jit(lambda x:calc(x)[2]);start=time.time();counter=[0]
def fg(x):v,g=f(x);return float(v),np.asarray(g)
def cb(x):
 counter[0]+=1
 if counter[0]%250==0:
  q=np.array(qf(x));print('it',counter[0],'energy',fg(x)[0],'SJ',q.min(),'bad',sum(np.any(q<=0,axis=1)),'seconds',round(time.time()-start),flush=True)
 if counter[0]%500==0:save(x)
def save(x):
 q=np.array(qf(x));np.savez_compressed(root/args.out,layers=np.concatenate([I[None],x.reshape(nr-1,n,3),O[None]]),quads=Q,source_ids=b['source_ids'],center=C,quality=q)
z=X[1:-1].ravel();q=np.array(qf(z));print('initial',q.min(),'bad',sum(np.any(q<=0,axis=1)),flush=True)
r=minimize(fg,z,jac=True,method='L-BFGS-B',callback=cb,options={'maxiter':args.iterations,'maxfun':args.iterations*3,'ftol':0.,'gtol':1e-10,'maxcor':12,'maxls':40});save(r.x);print('FINISHED',r.message,'energy',r.fun,'SJ',np.array(qf(r.x)).min(),'seconds',time.time()-start,flush=True)
