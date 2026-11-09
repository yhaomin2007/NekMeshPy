"""Tensor-product geometry evaluation and consistent baseline/candidate metrics."""
import numpy as np
from scipy.special import roots_jacobi
from itertools import product

def basis(x):
    return np.array([x*(x-1)/2,1-x*x,x*(x+1)/2]).T, np.array([x-.5,-2*x,x+.5]).T

def rules(n):
    g=np.r_[-1.,roots_jacobi(n-1,1,1)[0],1.]
    b,d=basis(g)
    mat=lambda a,b,c: np.einsum('ia,jb,kc->ijkabc',a,b,c).reshape(-1,27)
    return g,mat(b,b,b),[mat(d,b,b),mat(b,d,b),mat(b,b,d)]

def eval_q2(x,n=7):
    g,W,D=rules(n)
    return evaluate(x,W,D,g)

def evaluate(x,W,D,g):
    x=x-x[:,:1]
    j=np.stack([M@x for M in D],axis=2)
    a,b,c=j[:,:,0],j[:,:,1],j[:,:,2]
    bc=np.cross(b,c); det=np.einsum('egc,egc->eg',a,bc)
    norms=np.linalg.norm(j,axis=-1).prod(axis=-1)
    sj=det/norms
    # NekRS CFL stencil for a uniform unit-x velocity.
    invx=np.stack([bc[:,:,0],np.cross(c,a)[:,:,0],np.cross(a,b)[:,:,0]],axis=-1)/det[:,:,None]
    h=np.r_[g[1]-g[0],.5*(g[2:]-g[:-2]),g[-1]-g[-2]]
    h=np.array(np.meshgrid(h,h,h,indexing='ij')).reshape(3,-1).T
    cfl=np.sum(abs(invx)/h,axis=-1).max(axis=1)
    shape=(len(x),len(g),len(g),len(g),3)
    xyz=(W@x).reshape(shape)
    spacing=np.minimum.reduce([np.linalg.norm(np.diff(xyz,axis=i),axis=-1).reshape(len(x),-1).min(axis=1) for i in (1,2,3)])
    return det.min(axis=1),sj.min(axis=1),spacing,cfl

UVW=np.array(list(product((0.,.5,1.),repeat=3)))
u,v,w=UVW.T
L=np.array([(1-u)*(1-v)*(1-w),u*(1-v)*(1-w),u*v*(1-w),(1-u)*v*(1-w),(1-u)*(1-v)*w,u*(1-v)*w,u*v*w,(1-u)*v*w]).T
# VTK's full triquadratic ordering, in tensor indexing.
VTK=np.array([0,18,24,6,2,20,26,8,9,21,15,3,11,23,17,5,1,19,25,7,4,22,10,16,12,14,13])
WALL=np.array([0,6,8,2,3,7,5,1,4])

def determinant_matrices(grids):
    (a,da),(b,db),(c,dc)=map(basis,grids)
    mat=lambda a,b,c: np.einsum('ia,jb,kc->ijkabc',a,b,c).reshape(-1,27)
    return [mat(da,b,c),mat(a,db,c),mat(a,b,dc)]

def det_at(x,D):
    x=x-x[:,:1]
    a,b,c=[m@x for m in D]
    return np.einsum('egc,egc->eg',a,np.cross(b,c))

def bernstein_setup(degrees):
    from math import comb
    grids=[np.linspace(0,1,d+1) for d in degrees]
    transforms=[np.linalg.inv(np.array([comb(d,i)*u**i*(1-u)**(d-i) for i in range(d+1)]).T)
                for d,u in zip(degrees,grids)]
    return determinant_matrices([2*u-1 for u in grids]),transforms

def bernstein_coeff(x,setup):
    D,(A,B,C)=setup
    y=det_at(x,D).reshape(-1,len(A),len(B),len(C))
    return np.einsum('ia,jb,kc,eabc->eijk',A,B,C,y,optimize=True)

def bisect(coef,axis):
    z=np.moveaxis(coef,axis,0)
    left=[z[0]]; right=[z[-1]]
    while len(z)>1:
        z=.5*(z[:-1]+z[1:])
        left.append(z[0]);right.append(z[-1])
    return [np.moveaxis(np.stack(u),0,axis) for u in (left,right[::-1])]

def certify(c,margin,depth=0):
    if c.min()>margin:return True
    if depth==18 or c.max()<=margin:return False
    return all(certify(child,margin,depth+1) for child in bisect(c,depth%3))
