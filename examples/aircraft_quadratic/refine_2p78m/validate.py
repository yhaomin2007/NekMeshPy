import json, time
from pathlib import Path
import numpy as np
P=Path('/mnt/data/aircraft_refined_regenerated')
NQ=43424; K=64;NH=NQ*K
geom=np.memmap(P/'aircraft.re2',dtype='<f8',mode='r',offset=84,shape=(NH,25))
delta=np.memmap(P/'aircraft_q2.bin',dtype='<f8',mode='r',shape=(NQ,5,3))
profile=np.r_[0.,.018,.065,.19,.46,np.arange(1,61,dtype=float)]
oldprof=np.zeros(61)
for k in range(9):
 t=k/8;oldprof[k]=1-t*t*(3-2*t)
def alpha(t):
 k=min(int(t),59);f=t-k
 return (1-f)*oldprof[k]+f*oldprof[k+1]

def q2_uv(u,v):
 pu=np.array(((2*u-1)*(u-1),4*u*(1-u),u*(2*u-1)))
 du=np.array((4*u-3,4-8*u,4*u-1))
 pv=np.array(((2*v-1)*(v-1),4*v*(1-v),v*(2*v-1)))
 dv=np.array((4*v-3,4-8*v,4*v-1))
 def w(a,b):return np.array([a[1]*b[0],a[2]*b[1],a[1]*b[2],a[0]*b[1],a[1]*b[1]])
 return w(pu,pv),w(du,pv),w(pu,dv)

def coef(u,v,w):
 a=np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v]);
 au=np.array([-(1-v),1-v,v,-v]);av=np.array([-(1-u),-u,u,1-u]);
 b=np.r_[(1-w)*a,w*a];bu=np.r_[(1-w)*au,w*au];bv=np.r_[(1-w)*av,w*av];bw=np.r_[-a,a]
 return bu,bv,bw
samples=[(u,v,w) for u in (.1127016654,.5,.8872983346) for v in (.1127016654,.5,.8872983346) for w in (.1127016654,.5,.8872983346)]
report={'element_count':NH,'samples_per_curved_cell':27,'samples_per_outer_cell':8,'layer_metrics':[]}
start=time.time();badids=[];global_min=1e9;global_sj=1e9;cntbad=0;cntweak=0;total_samples=0
for k in range(K):
 x=geom[k*NQ:(k+1)*NQ]
 xyz=np.stack((x[:,1:9],x[:,9:17],x[:,17:25]),axis=-1)
 chunk=2048
 amin=1e99;asj=1e99;bn=0;weak=0
 samp=samples if k<12 else [(u,v,w) for u in (.2113248654,.7886751346) for v in (.2113248654,.7886751346) for w in (.2113248654,.7886751346)]
 a0=alpha(profile[k]);a1=alpha(profile[k+1])
 for q0 in range(0,NQ,chunk):
  q1=min(q0+chunk,NQ)
  xx=xyz[q0:q1];dd=delta[q0:q1]
  cmin=np.full(len(xx),1e99);sjmin=np.full(len(xx),1e99)
  for u,v,w in samp:
   bu,bv,bw=coef(u,v,w)
   xu=np.einsum('i,eic->ec',bu,xx);xv=np.einsum('i,eic->ec',bv,xx);xw=np.einsum('i,eic->ec',bw,xx)
   if k<12:
    _,qu,qv=q2_uv(u,v)
    aa=a0+(a1-a0)*w
    xu=xu+aa*np.einsum('i,eic->ec',qu,dd)
    xv=xv+aa*np.einsum('i,eic->ec',qv,dd)
    qw=q2_uv(u,v)[0]
    xw=xw+(a1-a0)*np.einsum('i,eic->ec',qw,dd)
   cr=np.cross(xu,xv); det=np.sum(cr*xw,axis=1)
   sj=det/np.maximum(np.linalg.norm(xu,axis=1)*np.linalg.norm(xv,axis=1)*np.linalg.norm(xw,axis=1),1e-250)
   cmin=np.minimum(cmin,det);sjmin=np.minimum(sjmin,sj)
  amin=min(amin,cmin.min());asj=min(asj,sjmin.min())
  bn+=np.count_nonzero(cmin<=0);weak+=np.count_nonzero(sjmin<0.01)
  if np.any(cmin<=0) and len(badids)<100:
   badids.extend((k*NQ+q0+np.flatnonzero(cmin<=0)[:30]).tolist())
 global_min=min(global_min,amin);global_sj=min(global_sj,asj);cntbad+=bn;cntweak+=weak;total_samples+=NQ*len(samp)
 report['layer_metrics'].append({'layer':k,'min_jacobian':float(amin),'min_scaled_jacobian':float(asj),'inverted_sampled_cells':int(bn),'scaled_j_less_0p01':int(weak)})
 if k%8==0:print('quality layer',k,'bad',cntbad,'min_sj',round(global_sj,5),'elapsed',round(time.time()-start,1),flush=True)
report.update({'min_sampled_det_jacobian':float(global_min),'min_sampled_scaled_jacobian':float(global_sj),'cells_with_nonpositive_sampled_det':int(cntbad),'cells_with_scaled_j_below_0p01':int(cntweak),'sampled_evaluation_count':total_samples,'first_bad_cell_ids_zero_based':badids})
(P/'validation.json').write_text(json.dumps(report,indent=2))
print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='layer_metrics'},indent=2))
