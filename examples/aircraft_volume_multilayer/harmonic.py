"""Weighted graph deformation across many layers, with exact approved wall coordinates."""
from pathlib import Path
import sys,argparse,time,json
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg,LinearOperator
R=Path(__file__).resolve().parent;O=R/'output';from quality import quality
p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=24);p.add_argument('--power',type=float,default=2);p.add_argument('--tangent',type=float,default=1);a=p.parse_args();depth=a.depth
d=np.load(Path(str(Path(__file__).resolve().parent/'inputs/volume_input.npz')));old=np.load(Path(str(Path(__file__).resolve().parent/'inputs/volume_input.npz')))['points'];target=np.load(R/'inputs/approved_surface.npz')['points'];L=d['layer_nodes'];Q=d['wall_quads'];H=d['hexes'];n=len(target);nq=len(Q);X=old[L[:depth+1]].reshape(-1,3);edge=np.sort(np.stack([Q,np.roll(Q,-1,axis=1)],axis=2),axis=2).reshape(-1,2);edge=np.unique(edge,axis=0);E=np.concatenate([np.concatenate([edge+k*n for k in range(depth+1)]),np.c_[np.arange(depth*n),np.arange(depth*n)+n]]);length=np.linalg.norm(X[E[:,1]]-X[E[:,0]],axis=1);weights=1/np.maximum(length,1e-4)**a.power;weights[:(depth+1)*len(edge)]*=a.tangent;row=np.r_[E[:,0],E[:,1]];col=np.r_[E[:,1],E[:,0]];val=np.r_[weights,weights];A=coo_matrix((-val,(row,col)),shape=(len(X),len(X))).tocsr();A.setdiag(np.bincount(row,weights=val,minlength=len(X)));unknown=np.arange(n,depth*n);fixed=np.r_[np.arange(n),np.arange(depth*n,(depth+1)*n)];K=A[unknown][:,unknown];bc=np.r_[target-old[L[0]],np.zeros_like(target)];rhs=-A[unknown][:,fixed]@bc;diag=K.diagonal();M=LinearOperator(K.shape,matvec=lambda x:x/diag);D=np.zeros_like(X);D[fixed]=bc;start=time.time()
for axis in range(3):
 steps=[0]
 def progress(x):
  steps[0]+=1
  if steps[0]%250==0:print('harmonic axis',axis,'iteration',steps[0],'seconds',round(time.time()-start),flush=True)
 sol,info=cg(K,rhs[:,axis],M=M,rtol=1e-8,atol=1e-10,maxiter=4000,callback=progress);print('CG',axis,info,steps[0],flush=True);D[unknown,axis]=sol
P=old.copy();P[L[:depth+1]]=X.reshape(depth+1,n,3)+D.reshape(depth+1,n,3);P[L[0]]=target;P[L[depth:]]=old[L[depth:]];q,_=quality(P,H[:depth*nq]);report={'depth':depth,'power':a.power,'tangential_weight':a.tangent,'minimum_SJ':float(q.min()),'nonpositive':int(sum(q<=0)),'bad_per_layer':[int(sum(q[k*nq:(k+1)*nq]<=0)) for k in range(depth)],'seconds':time.time()-start};print(report,flush=True);name=f'harmonic_{depth}_{a.power:g}_{a.tangent:g}';np.savez_compressed(O/(name+'.npz'),points=P);(O/(name+'.json')).write_text(json.dumps(report,indent=2))
