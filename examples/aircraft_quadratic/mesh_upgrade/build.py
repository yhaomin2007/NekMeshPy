"""Redistribute the outer shell and limit quadratic wall curvature against Jacobians."""
from pathlib import Path
import sys,json
import numpy as np
from quality import L,rules,evaluate,bernstein_setup,bernstein_coeff,certify

R=Path(sys.argv[1]); O=Path('mesh_upgrade')
d=np.load(R/'aircraft_tail_refined_package/output/tail_refined.npz')
Q=d['wall_quads']; nq=len(Q)
def ramp(x):
    s=np.clip(x,0,1);return s*s*(3-2*s)

def coords(B,k,index):
    q=Q[index]
    return L@np.concatenate([B[k,q],B[k+1,q]],axis=1)

outer=np.load(O/'outer_coherent.npz')
shell=outer['shell']
f=outer['scale']
proj=np.load(O/'projection_checked.npz'); E=proj['edges']; ei=proj['edge_index']
delta=proj['delta']; nc=len(E)
# Relax higher-order projection at shared nodes only where full curving would invert cells.
# Keep the accepted corners and taper wall displacements smoothly through eight layers.
profile=1-ramp(np.arange(61)/8.)
edscale=np.ones(nc);fcscale=np.ones(nq)
pending=np.arange(nq)
if '--intersection-guard' in sys.argv:
    original=np.load(O/'candidate_checked.npz')
    edscale=original['edscale'].copy()
    fcscale=original['fcscale'].copy()
    marked=np.unique(np.loadtxt(O/'wall_crossing_quads.csv',delimiter=',',dtype=int))
    # Include two edge-neighbour rings so the linear patches meet the
    # curved wall away from the thin opposing surfaces.
    for _ in range(2):
        marked=np.flatnonzero(np.any(np.isin(ei,np.unique(ei[marked])),axis=1))
    touched=np.unique(ei[marked])
    edscale[touched]=0
    fcscale[marked]=0
    pending=np.flatnonzero(np.any(np.isin(ei,touched),axis=1))
    print('wall intersections guard',len(marked),'linear wall quads',flush=True)
# tensor layout uses t as the fastest coordinate
edge_slots=[9,21,15,3]
def curvature(k,index):
    dq=np.zeros((len(index),27,3))
    for j,slot in enumerate(edge_slots):
        e=ei[index,j]
        dq[:,slot:slot+3]=delta[e,None,:]*edscale[e,None,None]
    dq[:,12:15]=delta[nc+index,None,:]*fcscale[index,None,None]
    weight=np.tile([profile[k],(profile[k]+profile[k+1])/2,profile[k+1]],9)
    return dq*weight[None,:,None]
baseline=np.full((8,4,nq),np.nan)
quadratures=[rules(5),rules(7)]
for k in range(8):
    g,W,D=quadratures[k==0]
    for i in range(0,len(pending),384):
        index=pending[i:i+384]
        baseline[k,:,index]=np.asarray(evaluate(coords(shell,k,index),W,D,g)).T
setup=bernstein_setup((5,5,2))
if "--intersection-guard" in sys.argv:
    pending=np.arange(nq)
    for k in range(8):
        g,W,D=quadratures[k==0]
        for i in range(0,nq,384):
            index=pending[i:i+384]
            if np.isnan(baseline[k,:,index]).any():
                baseline[k,:,index]=np.asarray(evaluate(coords(shell,k,index),W,D,g)).T
    pending=np.flatnonzero(np.any(np.isin(ei,touched),axis=1))
for it in range(36):
    bad=np.zeros(nq,dtype=bool)
    for k in range(8):
        g,W,D=quadratures[k==0]
        for i in range(0,len(pending),384):
            index=pending[i:i+384]
            linear_nodes=coords(shell,k,index)
            nodes=linear_nodes+curvature(k,index)
            det,sj,spacing,rate=evaluate(nodes,W,D,g)
            _,base_sj,base_spacing,base_rate=baseline[k][:,index]
            fail=(sj<np.minimum(.025,.80*base_sj))|(det<=0)
            fail|=spacing<np.maximum(.85*base_spacing,.98*baseline[k,2].min())
            fail|=rate>np.maximum(1.04*base_rate,1.02*baseline[k,3].max())
            if k==0:
                # Compare the wall normals at matched parametric points.
                dr,ds=D[0][::len(g)],D[1][::len(g)]
                n0=np.cross(dr@linear_nodes,ds@linear_nodes)
                n1=np.cross(dr@nodes,ds@nodes)
                dot=np.sum(n0*n1,axis=-1)
                mag=np.linalg.norm(n0,axis=-1)*np.linalg.norm(n1,axis=-1)
                fail|=np.any(dot<.2*mag,axis=1)
            co=bernstein_coeff(nodes,setup)
            tol=np.max(abs(co),axis=(1,2,3))*1.e-10
            for c in np.flatnonzero((co.min(axis=(1,2,3))<=tol)&~fail):
                fail[c]=not certify(co[c],tol[c])
            bad[index]|=fail
    print('curving attempt',it,'tested',len(pending),'limited quads',bad.sum(),flush=True)
    if not bad.any():break
    changed=np.unique(ei[bad])
    edscale[changed]*=.65
    fcscale[bad]*=.65
    pending=np.flatnonzero(bad|np.any(np.isin(ei,changed),axis=1))
else: raise RuntimeError('quadratic curving did not pass')
name='candidate_guarded.npz' if '--intersection-guard' in sys.argv else 'candidate_checked.npz'
np.savez(O/name,layers=shell,profile=profile,edscale=edscale,fcscale=fcscale,outer_scale=f)
