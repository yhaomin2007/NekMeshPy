#!/usr/bin/env python3
"""Conformal 2x2 tangential refinement + 4 additional first-parent-layer divisions.

No CAD is used. Preserves the geometry defined by supplied RE2+USR+Q2 binary.
Builds a new standard Nek5000 RE2 with linear corners sampled from the original
curved mapping and a new Q2 correction file and USR for smooth curved mapping.
Original 651360 hexes = 10856 wall quads * 60 wall-normal layers.
Output 2779136 hexes = 43424 wall quads * 64 layers.
"""
import os, sys, json, struct, time
from pathlib import Path
import numpy as np

SRC=Path('/mnt/data'); OUT=Path('/mnt/data/aircraft_refined_regenerated')
OUT.mkdir(parents=True,exist_ok=True)
NQ=10856; NS=4; NQN=NQ*NS; KOLD=60; KNEW=64; NH=NQN*KNEW
S=np.r_[0.,.018,.065,.19,.46,np.arange(1,61,dtype=float)]
assert len(S)==KNEW+1 and S[-1]==60
OLD=np.memmap(SRC/'aircraft.re2',dtype='<f8',mode='r',offset=84,shape=(60,NQ,25))
DELTA=np.fromfile(SRC/'aircraft_q2.bin','<f8').reshape(NQ,5,3)
# Mesh is in nondimensional length (reference = 7.00532 m).
PARENTS=np.stack([OLD[:,:,1:9],OLD[:,:,9:17],OLD[:,:,17:25]],axis=-1)
UV=np.array([(u,v) for v in [0.,.5,1.] for u in [0.,.5,1.]],float)
CHILD=np.array([[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]],int)
SLOTS=np.array([[.5,0],[1,.5],[.5,1],[0,.5],[.5,.5]])

def quad_weights(uv):
 u,v=uv[:,0],uv[:,1]
 return np.stack(((1-u)*(1-v),u*(1-v),u*v,(1-u)*v),axis=-1)

def q2_weights(uv):
 u,v=uv[:,0],uv[:,1]
 pu=np.stack(((2*u-1)*(u-1),4*u*(1-u),u*(2*u-1)),axis=-1)
 pv=np.stack(((2*v-1)*(v-1),4*v*(1-v),v*(2*v-1)),axis=-1)
 return np.stack((pu[:,1]*pv[:,0],pu[:,2]*pv[:,1],pu[:,1]*pv[:,2],pu[:,0]*pv[:,1],pu[:,1]*pv[:,1]),axis=-1)

def sub_coords(child,uv):
 u0,v0=(child%2)*.5,(child//2)*.5
 return np.c_[u0+uv[:,0]*.5,v0+uv[:,1]*.5]

subdelta=np.empty((NQN,5,3),dtype='<f8')
for c in range(4):
 uvc=sub_coords(c, np.array([[0,0],[1,0],[1,1],[0,1]],float))
 wco=q2_weights(uvc); wcand=q2_weights(sub_coords(c,SLOTS))
 wres=wcand-quad_weights(SLOTS)@wco
 subdelta[c::4]=np.einsum('ij,qjk->qik',wres,DELTA,optimize=True)
subdelta.tofile(OUT/'aircraft_q2.bin')

alpha_levels=np.zeros(61)
for k in range(9):
 t=k/8.;alpha_levels[k]=1-t*t*(3-2*t)

def af(t):
 k=min(int(t),59); f=t-k
 return alpha_levels[k]*(1-f)+alpha_levels[k+1]*f

D9=np.einsum('ij,qjk->qik',q2_weights(UV),DELTA,optimize=True)
W9=quad_weights(UV)
with open(SRC/'aircraft.re2','rb') as f: header=f.read(84)
newhead=(f'#v004 {NH:15d}  3 {NH:15d}   1 hdr').ljust(80).encode('ascii')+header[80:84]
assert len(newhead)==84
geo=OUT/'aircraft.re2'
with geo.open('wb') as f:
 f.write(newhead)
 f.truncate(84+NH*25*8)
DEST=np.memmap(geo,dtype='<f8',mode='r+',offset=84,shape=(NH,25))
DEST[:,0]=0.
start=time.time()
for layer in range(KNEW):
 t0,t1=S[layer:layer+2]
 def xy_at(t):
  k=min(int(t),59);f=t-k
  old0=PARENTS[k,:,:4,:];old1=PARENTS[k,:,4:,:]
  corners=(1-f)*old0+f*old1
  base=np.einsum('up,qpc->quc',W9,corners,optimize=True)
  return base+af(t)*D9
 bot=xy_at(t0);top=xy_at(t1)
 children=np.concatenate((bot[:,CHILD,:],top[:,CHILD,:]),axis=2).reshape(NQN,8,3)
 a=DEST[layer*NQN:(layer+1)*NQN]
 a[:,1:9]=children[:,:,0];a[:,9:17]=children[:,:,1];a[:,17:25]=children[:,:,2]
 if layer%8==0:print('written layer',layer,'elapsed',round(time.time()-start,1),flush=True)
DEST.flush();del DEST

with open(SRC/'aircraft.re2','rb') as f:
 f.seek(84+651360*25*8);original_tail=f.read()
assert struct.unpack('<d',original_tail[:8])[0]==0
assert int(struct.unpack('<d',original_tail[8:16])[0])==2*NQ
records=np.frombuffer(original_tail[16:],dtype='V64')
assert len(records)==2*NQ
with geo.open('ab') as f:
 f.write(struct.pack('<dd',0.,2*NQN))
 for side in range(2):
  rec=np.repeat(records[side*NQ:(side+1)*NQ],4)
  rr=np.frombuffer(rec.tobytes(),dtype='u1').reshape(-1,64).copy()
  elems=np.arange(1,NQN+1,dtype='<f8')+(NH-NQN if side else 0)
  rr[:,:8]=elems.view('u1').reshape(-1,8)
  f.write(rr.tobytes())
assert geo.stat().st_size==84+NH*25*8+16+2*NQN*64

user=(SRC/'aircraft.usr').read_text()
user=user.replace('parameter (nq=10856,nlayers=8,nrecords=5*nq)',f'parameter (nq={NQN},nlayers=12,nrecords=5*nq)')
user=user.replace('60*nq',f'{KNEW}*nq')
user=user.replace('         alpha0=real(layer)/real(nlayers)\n         alpha1=real(layer+1)/real(nlayers)\n         alpha0=1.0-alpha0**2*(3.0-2.0*alpha0)\n         alpha1=1.0-alpha1**2*(3.0-2.0*alpha1)',
'''         call aircraft_refined_alpha(layer,alpha0)
         call aircraft_refined_alpha(layer+1,alpha1)''')
user+='''\n      subroutine aircraft_refined_alpha(k,a)
      integer k,oldk
      real a,t,part,a0,a1,ff
      real cuts(6)
      data cuts /0.0,0.018,0.065,0.19,0.46,1.0/
      if (k.le.5) then
         t=cuts(k+1)
      else
         t=real(k-4)
      endif
      if (t.ge.8.0) then
         a=0.0
         return
      endif
      oldk=int(t)
      ff=t-real(oldk)
      part=real(oldk)/8.0
      a0=1.0-part*part*(3.0-2.0*part)
      part=real(oldk+1)/8.0
      a1=1.0-part*part*(3.0-2.0*part)
      a=(1.0-ff)*a0+ff*a1
      return
      end
'''
(OUT/'aircraft.usr').write_text(user)
print('COMPLETE',NH,geo.stat().st_size,'bytes',round(time.time()-start,1),'seconds',flush=True)
print('Q2',(OUT/'aircraft_q2.bin').stat().st_size)
