"""Compare candidate and baseline at increasingly fine tessellation near flagged pairs."""
import json,sys
from pathlib import Path
import numpy as np
from quality import basis
root=Path(sys.argv[1]); out=Path('deliver/final/aircraft_quadratic')
d=np.load(root/'aircraft_tail_refined_package/output/tail_refined.npz'); q=d['wall_quads']; x=d['points'][d['layer_nodes'][0]]
proj=np.load('mesh_upgrade/projection_checked.npz'); can=np.load('mesh_upgrade/candidate_guarded.npz')
ei=proj['edge_index']; ne=len(proj['edges'])
v=np.c_[q,ei+len(x),np.arange(len(q))+len(x)+ne]
p0=np.vstack([x,x[proj['edges']].mean(1),x[q].mean(1)])
scale=np.r_[can['edscale'],can['fcscale']]
p1=p0.copy();p1[len(x):]+=proj['delta']*scale[:,None]
a=Path('mesh_upgrade/audit_wall.py').read_text()
exec(a[a.index('def crossings'):a.index('def cross2')])
pairs=np.loadtxt('mesh_upgrade/wall_crossing_quads.csv',delimiter=',',dtype=int).reshape(-1,2)
print('pairs',len(pairs),'involving',len(np.unique(pairs)),'quads',flush=True)
def find(p,pairs,n):
    z,_=basis(np.linspace(-1,1,n+1))
    W=np.einsum('ia,jb->ijab',z,z).reshape((n+1)**2,9)
    ids,idx=np.unique(pairs,return_inverse=True)
    N=p[v[ids][:,[0,7,3,4,8,6,1,5,2]]]
    g=(W@N).reshape(-1,n+1,n+1,3)
    a,b,c,d=g[:,:-1,:-1],g[:,1:,:-1],g[:,1:,1:],g[:,:-1,1:]
    t=np.stack([np.stack([a,b,c],axis=-2),np.stack([a,c,d],axis=-2)],axis=1).reshape(len(ids),-1,3,3)
    lo=t.min(2);hi=t.max(2); bad=[]
    for i,j in idx.reshape(-1,2):
        hit=np.all(hi[i,:,None]>=lo[j,None,:]-1e-9,axis=-1)&np.all(hi[j,None,:]>=lo[i,:,None]-1e-9,axis=-1)
        aa,bb=np.nonzero(hit); a=t[i,aa];b=t[j,bb]
        bad.append(np.any(crossings(a,b)|crossings(b,a)))
    return np.asarray(bad)
for n in (3,6,12):
    old=find(p0,pairs,n);new=find(p1,pairs,n)
    print('resolution',n,'old',int(old.sum()),'new',int(new.sum()),'new only',int((new&~old).sum()),flush=True)
    assert not new.any()
