import numpy as np,json
from pathlib import Path
P=Path('/mnt/data/aircraft_refined_regenerated');rng=np.random.default_rng(42)
NQ=10856;newn=43424
old=np.memmap('/mnt/data/aircraft.re2',dtype='<f8',mode='r',offset=84,shape=(60,NQ,25))
new=np.memmap(P/'aircraft.re2',dtype='<f8',mode='r',offset=84,shape=(64,newn,25))
od=np.fromfile('/mnt/data/aircraft_q2.bin','<f8').reshape(NQ,5,3)
nd=np.fromfile(P/'aircraft_q2.bin','<f8').reshape(newn,5,3)
cuts=np.r_[0,.018,.065,.19,.46,np.arange(1,61)]
a0=np.zeros(61)
for k in range(9):
 t=k/8;a0[k]=1-t*t*(3-2*t)
def aa(t):
 k=min(int(t),59);return a0[k]*(1-(t-k))+a0[k+1]*(t-k)
def bw(u,v):return np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v])
def qw(u,v):
 p=np.array([(2*u-1)*(u-1),4*u*(1-u),u*(2*u-1)]);q=np.array([(2*v-1)*(v-1),4*v*(1-v),v*(2*v-1)])
 return np.array([p[1]*q[0],p[2]*q[1],p[1]*q[2],p[0]*q[1],p[1]*q[1]])
def geom(m,dd,u,v,w,a,b):
 vertices=np.stack([m[1:9],m[9:17],m[17:25]],axis=-1)
 return (1-w)*(bw(u,v)@vertices[:4])+w*(bw(u,v)@vertices[4:]) + ((1-w)*a+w*b)*(qw(u,v)@dd)
maxdist=0
for i in range(4000):
 k=int(rng.integers(0,64));q=int(rng.integers(0,10856));c=int(rng.integers(0,4));u,v,w=rng.random(3)
 t=cuts[k]+w*(cuts[k+1]-cuts[k]);ok=min(int(t),59);ow=t-ok
 ou=.5*((c%2)+u);ov=.5*((c//2)+v)
 pold=geom(old[ok,q],od[q],ou,ov,ow,a0[ok],a0[ok+1])
 pnew=geom(new[k,4*q+c],nd[4*q+c],u,v,w,aa(cuts[k]),aa(cuts[k+1]))
 maxdist=max(maxdist,float(np.linalg.norm(pold-pnew)))
res={'random_sample_count':4000,'max_parent_geometry_reconstruction_error_nondimensional':maxdist,'maximum_error_m':maxdist*7.00532}
(P/'reconstruction.json').write_text(json.dumps(res,indent=2));print(res)
