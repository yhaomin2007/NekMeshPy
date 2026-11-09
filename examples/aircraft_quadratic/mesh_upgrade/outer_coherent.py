"""Spread the fine tangential bands using a common shell displacement field."""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from quality import L,rules,evaluate,bernstein_setup,bernstein_coeff,certify
root=Path(sys.argv[1])
d=np.load(root/'aircraft_tail_refined_package/output/tail_refined.npz')
Q=d['wall_quads']; A=d['points'][d['layer_nodes']]
E=np.unique(np.sort(np.stack([Q,np.roll(Q,-1,axis=1)],axis=-1).reshape(-1,2),axis=1),axis=0)
i=np.r_[E[:,0],E[:,1]]; j=np.r_[E[:,1],E[:,0]]
degree=np.bincount(i,minlength=A.shape[1])
G=csr_matrix((1/degree[i],(i,j)),shape=(A.shape[1],)*2)
ref=45
x=A[ref].copy()
normal=np.zeros_like(x)
corners=x[Q]
n=np.cross(corners[:,2]-corners[:,0],corners[:,3]-corners[:,1])
for j in range(4):np.add.at(normal,Q[:,j],n)
normal/=np.linalg.norm(normal,axis=1)[:,None]
for j in range(90):
    v=G@x-x
    x+=.5*(v-(v*normal).sum(axis=1)[:,None]*normal)
displacement=x-A[ref]
radius_scale=np.median(np.linalg.norm(A[:,E[:,0]]-A[:,E[:,1]],axis=-1),axis=1)
radius_scale/=radius_scale[ref]
def ramp(x):
    s=np.clip(x,0,1);return s*s*(3-2*s)
k=np.arange(61)
weights=ramp((k-22)/16)*ramp((60-k)/10)
delta=radius_scale[:,None,None]*weights[:,None,None]*displacement
g,W,D=rules(3)
setup=bernstein_setup((2,2,2))
base_min=np.full(60,np.nan)
def check(B,baseline):
    worst=np.full(60,np.nan)
    for k in range(22,60):
        minimum=np.inf
        for i in range(0,len(Q),1024):
            q=Q[i:i+1024]
            nodes=L@np.concatenate([B[k,q],B[k+1,q]],axis=1)
            dj,sj,_,_=evaluate(nodes,W,D,g)
            minimum=min(minimum,sj.min())
            if not baseline:
                if (dj<=0).any():return False,k,minimum
                co=bernstein_coeff(nodes,setup)
                tol=np.max(abs(co),axis=(1,2,3))*1.e-10
                if any(not certify(co[c],tol[c]) for c in np.flatnonzero(co.min(axis=(1,2,3))<=tol)):
                    return False,k,minimum
        worst[k]=minimum
        if not baseline and minimum<min(.045,base_min[k]*.90):return False,k,minimum
    return True,None,worst
_,_,base_min=check(A,True)
for factor in (1.,.8,.6,.45,.3,.2,.1):
    B=A+factor*delta
    ok,k,score=check(B,False)
    print('coherent outer',factor,ok,k,float(np.nanmin(score)),flush=True)
    if ok:
        np.savez('mesh_upgrade/outer_coherent.npz',shell=B,scale=factor*weights)
        break
else: raise RuntimeError('No outer trial passed')
