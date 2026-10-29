from pathlib import Path
import sys
import numpy as np
R=Path(__file__).resolve().parent;O=R/'output'
sys.path.insert(0,str(R.parents[1]))
S=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
g=np.r_[-1.,-1/np.sqrt(5),1/np.sqrt(5),1.];T=np.array(np.meshgrid(g,g,g,indexing='ij')).reshape(3,-1).T
D=np.array([[S[:,j]*np.prod(1+S[:,[k for k in range(3) if k!=j]]*t[[k for k in range(3) if k!=j]],axis=1)/8 for j in range(3)] for t in T]).transpose(0,2,1)
def quality(P,H):
 sj=[];spacing=[]
 for first in range(0,len(H),512):
  J=np.einsum('eni,tnj->etij',P[H[first:first+512]],D,optimize=True)
  lengths=np.linalg.norm(J,axis=2);det=np.sum(np.cross(J[:,:,:,0],J[:,:,:,1])*J[:,:,:,2],axis=2)
  sj.append((det/np.maximum(np.prod(lengths,axis=2),1e-30)).min(1));spacing.append(lengths.min((1,2)))
 return np.concatenate(sj),np.concatenate(spacing)
def field(P0,L,target,alpha):
 wall=L[0];dist=np.linalg.norm(P0[L[:13]]-P0[wall][None,:,:],axis=2);length=dist[12];s=np.minimum(1,dist/np.maximum(length,1e-30));weight=1-3*s*s+2*s*s*s
 P=P0.copy();P[L[:13]]=P0[L[:13]]+weight[:,:,None]*(alpha[:,None]*(target-P0[wall]))[None,:,:]
 return P
def linear_scan(P,H,order):
 """Analytic trilinear Jacobian at tensor GLL nodes, with shared derivative factors."""
 from nekmeshpy.core.interp import gll_nodes
 g=2*gll_nodes(order)-1;x=g[None,:,None,None,None];y=g[None,None,:,None,None];z=g[None,None,None,:,None]
 M=np.c_[S,S[:,0]*S[:,1],S[:,0]*S[:,2],S[:,1]*S[:,2],S.prod(1)]/8;out=[]
 for first in range(0,len(H),512):
  c=np.einsum('enk,nm->emk',P[H[first:first+512]],M)
  a=[c[:,j,None,None,None,:] for j in range(7)]
  A=a[0]+a[3]*y+a[4]*z+a[6]*y*z;B=a[1]+a[3]*x+a[5]*z+a[6]*x*z;C=a[2]+a[4]*x+a[5]*y+a[6]*x*y
  det=np.sum(np.cross(A,B)*C,axis=-1);q=det/(np.linalg.norm(A,axis=-1)*np.linalg.norm(B,axis=-1)*np.linalg.norm(C,axis=-1));out.append(q.min((1,2,3)))
 return np.concatenate(out)
def linear_spacing(P,H,order=7):
 """Exact nearest-neighbour tensor GLL spacing for a trilinear hex map."""
 from nekmeshpy.core.interp import gll_nodes
 g=2*gll_nodes(order)-1;x=g[None,:,None,None,None];y=g[None,None,:,None,None];z=g[None,None,None,:,None];delta=np.diff(g).min()
 M=np.c_[S,S[:,0]*S[:,1],S[:,0]*S[:,2],S[:,1]*S[:,2],S.prod(1)]/8;out=[]
 for first in range(0,len(H),1024):
  c=np.einsum('enk,nm->emk',P[H[first:first+1024]],M);a=[c[:,j,None,None,None,:] for j in range(7)]
  A=a[0]+a[3]*y+a[4]*z+a[6]*y*z;B=a[1]+a[3]*x+a[5]*z+a[6]*x*z;C=a[2]+a[4]*x+a[5]*y+a[6]*x*y
  out.append(delta*np.minimum.reduce([np.linalg.norm(t,axis=-1).min((1,2,3)) for t in [A,B,C]]))
 return np.concatenate(out)
