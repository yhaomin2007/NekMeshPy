"""Project wing/tail caps and their adjoining outer strips to the correct CAD sheets."""
from cad_surface import *
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from untangle import smooth
import json,hashlib
d=np.load(O/'surface_input.npz');base=np.load(O/'base_surface.npz');P=base['points'].copy();nom=d['nominal'];Q=base['quads'];fam=base['family'];tt=d['circ'];labelnode=d['label'];spec={'W':{'cap':14,'upper':[4,13],'lower':[7,19],'te':11,'start':27.5,'end':29.2,'axis':1},'H':{'cap':47,'upper':[48],'lower':[50],'te':49,'start':8.5,'end':10.2,'axis':1},'V':{'cap':53,'upper':[51],'lower':[],'te':52,'start':15.5,'end':18.,'axis':2}}
CADrecords={};affected=set();cap_node_sets={};cap_quads={};band_quads={};charts={};cap_reports={};band_reports={}
def ordered(ids,start):
 es=list(ids);out=[];current=np.array(start)
 while es:
  ends=[np.array([e.startPoint().toTuple(),e.endPoint().toTuple()])/1000 for e in es];k=np.argmin([np.linalg.norm(x-current,axis=1).min() for x in ends]);e=es.pop(k);rev=np.linalg.norm(ends[k][1]-current)<np.linalg.norm(ends[k][0]-current);out.append((e,rev));current=ends[k][0 if rev else 1]
 lengths=np.array([e.Length()/1000 for e,rev in out]);return out,np.r_[0,np.cumsum(lengths)]/sum(lengths)
def evalchain(chain,f):
 es,arc=chain;k=np.clip(np.searchsorted(arc,f,side='right')-1,0,len(es)-1);u=np.clip((f-arc[k])/(arc[k+1]-arc[k]),0,1);e,rev=es[k];return np.array(e.positionAt(float(1-u if rev else u)).toTuple())/1000
capchains={};sideedges={}
for label,c in spec.items():
 cap=c['cap'];upper=[e for e,adj in edges.values() if cap in adj and adj&set(c['upper'])];lo=[e for e,adj in edges.values() if cap in adj and adj&set(c['lower'])];te=[e for e,adj in edges.values() if {cap,c['te']}<=adj];verts=np.array([x.toTuple() for x in faces[cap].Vertices()])/1000;le=verts[np.argmin(verts[:,0])];tes=np.array([e.startPoint().toTuple() for e in te]+[e.endPoint().toTuple() for e in te])/1000
 if label=='V':up=tes[np.argmax(tes[:,1])];low=tes[np.argmin(tes[:,1])];capchains[label+'U']=ordered(upper,le);sym=[e for e,adj in edges.values() if {cap,55}<=adj];capchains[label+'S']=ordered(sym,le)
 else:up=tes[np.argmax(tes[:,2])];low=tes[np.argmin(tes[:,2])];capchains[label+'U']=ordered(upper,up);capchains[label+'L']=ordered(lo,le)
 capchains[label+'T']=ordered(te,low if label!='V' else up);c['le']=le;c['teu']=up;c['tel']=low
 pairs={'LE':[(13,7)] if label=='W' else ([(48,50)] if label=='H' else [(51,55)]),'TEU':[(4,11)] if label=='W' else ([(48,49)] if label=='H' else [(51,52)]),'TEL':[(19,11)] if label=='W' else ([(50,49)] if label=='H' else [(52,55)])}
 for key,ps in pairs.items():sideedges[label+key]=[e for e,adj in edges.values() if any(set(pair)<=adj for pair in ps)]
 capaxes=[0,1] if label=='V' else [0,2];charts[label+'C']=Chart(FI==cap,V[:,capaxes],'xy' if label=='V' else 'xz');charts[label+'U']=Chart(np.isin(FI,c['upper']),V[:,[0,2] if label=='V' else [0,1]],'xz' if label=='V' else 'xy')
 if label!='V':charts[label+'L']=Chart(np.isin(FI,c['lower']),V[:,[0,1]],'xy')
@lru_cache(None)
def capboundary(label,t):
 c=spec[label]
 if label=='V' and t>=3:return evalchain(capchains[label+'T'],t-3)
 if label!='V' and t>=6:return evalchain(capchains[label+'T'],(t-6)/2)
 key=label+('U' if label=='V' or t<=3 else 'L');q=chord_fraction(label,t);a=evalchain(capchains[key],0);b=evalchain(capchains[key],1);x=min(a[0],b[0])+q*abs(a[0]-b[0]);f=brentq(lambda u:evalchain(capchains[key],u)[0]-x,0,1,xtol=1e-12);return evalchain(capchains[key],f)
@lru_cache(None)
def edgepoint(key,value):
 axis=spec[key[0]]['axis']
 for e in sideedges[key]:
  a=np.array(e.startPoint().toTuple())/1000;b=np.array(e.endPoint().toTuple())/1000;lo,hi=sorted([a[axis],b[axis]])
  if lo-1e-7<=value<=hi+1e-7:
   if abs(value-a[axis])<1e-7:return a
   if abs(value-b[axis])<1e-7:return b
   u=brentq(lambda u:e.positionAt(float(u)).toTuple()[axis]/1000-value,0,1,xtol=1e-12);return np.array(e.positionAt(float(u)).toTuple())/1000
 raise ValueError((key,value))
@lru_cache(None)
def inboard(label,t):
 c=spec[label];v=c['start']
 if label=='V':
  if t==0:return edgepoint(label+'LE',v)
  if t>=3:return (4-t)*edgepoint(label+'TEU',v)+(t-3)*edgepoint(label+'TEL',v)
  key=label+'U'
 else:
  if t==3:return edgepoint(label+'LE',v)
  if t==0:return edgepoint(label+'TEU',v)
  if t>=6:return (1-(t-6)/2)*edgepoint(label+'TEL',v)+((t-6)/2)*edgepoint(label+'TEU',v)
  key=label+('U' if t<3 else 'L')
 le=edgepoint(label+'LE',v);te=edgepoint(label+('TEU' if key.endswith('U') else 'TEL'),v);x=le[0]+chord_fraction(label,t)*(te[0]-le[0]);return charts[key].evaluate([[x,v]])[0][0]
for label,c in spec.items():
 cap=c['cap'];mask=(fam==label+'_tip')&np.all(nom[Q,1]>=-1e-8,axis=1);qi=np.flatnonzero(mask);ids=np.unique(Q[qi]);lookup=np.full(len(P),-1);lookup[ids]=np.arange(len(ids));qq=lookup[Q[qi]];E=np.sort(np.stack([qq,np.roll(qq,-1,axis=1)],axis=2),axis=2).reshape(-1,2);E,count=np.unique(E,axis=0,return_counts=True);boundary=np.unique(E[count==1]);axes=[0,1] if label=='V' else [0,2];coords=nom[ids][:,axes];lo=coords.min(0);scale=np.ptp(coords,axis=0);ref=(coords-lo)/scale;goal=ref.copy();fixed=np.zeros(len(ids),bool);fixed[boundary]=True;boundaryP={}
 for k in boundary:
  i=int(ids[k])
  if label=='V' and abs(nom[i,1])<1e-8 and abs(nom[i,2]-c['end'])>1e-7:
   fraction=(nom[i,0]-nom[ids,0].min())/np.ptp(nom[ids,0]);chain=capchains[label+'S'];point=evalchain(chain,float(fraction))
  else:point=capboundary(label,float(tt[i]))
  boundaryP[i]=point;goal[k]=(point[axes]-lo)/scale
 weights=1/np.maximum(np.linalg.norm(ref[E[:,1]]-ref[E[:,0]],axis=1),1e-5);r=np.r_[E[:,0],E[:,1]];col=np.r_[E[:,1],E[:,0]];v=np.r_[weights,weights];A=coo_matrix((-v,(r,col)),shape=(len(ids),len(ids))).tocsr();A.setdiag(np.bincount(r,weights=v,minlength=len(ids)));free=~fixed;uv=goal.copy();uv[free]=spsolve(A[free][:,free],-A[free][:,fixed]@goal[fixed]);cc=ref[qq];a=cc[:,1]-cc[:,0];b=cc[:,3]-cc[:,0];sign=np.sign(np.median(a[:,0]*b[:,1]-a[:,1]*b[:,0]));uv=smooth(uv,qq,fixed,sign,maxiter=500);pos,ff,native,out=charts[label+'C'].evaluate(uv*scale+lo)
 for k,i in enumerate(ids):
  if int(i) in boundaryP:pos[k]=boundaryP[int(i)];native[k]=faces[cap].paramAt(tuple(pos[k]*1000))
  P[i]=pos[k];CADrecords[int(i)]=(cap,*native[k]);affected.add(int(i))
 cap_node_sets[label]=ids;cap_quads[label]=qi;cap_reports[label]={'positive_cap_quads':len(qi),'positive_cap_nodes':len(ids),'interior_nodes_outside_atlas':int(sum(out[~fixed]>1e-5)),'max_interior_chart_distance':float(out[~fixed].max(initial=0))};print('cap',label,cap_reports[label],flush=True)
 # Reparameterize the adjoining last span interval using actual cap/LE/TE boundaries.
 band=(np.char.startswith(fam,label+'_')&(fam!=label+'_tip')&np.all(nom[Q,1]>=-1e-8,axis=1)&np.all(nom[Q,c['axis']]>=c['start']-1e-8,axis=1));bqi=np.flatnonzero(band);bnodes=np.unique(Q[bqi]);outside=[]
 for i in bnodes:
  t=float(tt[i]);p=(nom[i,c['axis']]-c['start'])/(c['end']-c['start']);p=float(np.clip(p,0,1));aa=[0,2] if label=='V' else [0,1]
  if p>=1-1e-10:
   point=capboundary(label,t);fi=cap;native=faces[cap].paramAt(tuple(point*1000));dist=0.
  elif label=='V' and t>=3 or label!='V' and t>=6:
   span=(1-p)*c['start']+p*capboundary(label,t)[c['axis']];u=edgepoint(label+'TEU',float(span));low=edgepoint(label+'TEL',float(span));depth=4-t if label=='V' else (t-6)/2;point=low+depth*(u-low);fi=c['te'];native=faces[fi].paramAt(tuple(point*1000));point=np.array(faces[fi].positionAt(*native).toTuple())/1000;dist=0.
  else:
   if label=='V':key=label+'U';ta,tb=0,3;k0,k1='LE','TEU';frac=t/3
   elif t<=3:key=label+'U';ta,tb=0,3;k0,k1='TEU','LE';frac=t/3
   else:key=label+'L';ta,tb=3,6;k0,k1='LE','TEL';frac=(t-3)/3
   def side(k,tcorner):return edgepoint(label+k,float((1-p)*c['start']+p*capboundary(label,tcorner)[c['axis']]))
   c00=inboard(label,ta);c01=inboard(label,tb);c10=capboundary(label,ta);c11=capboundary(label,tb);bil=(1-p)*((1-frac)*c00+frac*c01)+p*((1-frac)*c10+frac*c11);xyz=(1-p)*inboard(label,t)+p*capboundary(label,t)+(1-frac)*side(k0,ta)+frac*side(k1,tb)-bil;pos,ff,uvv,dd=charts[key].evaluate([xyz[aa]]);point=pos[0];fi=int(ff[0]);native=uvv[0];dist=float(dd[0])
   if t==ta or t==tb:point=side(k0 if t==ta else k1,ta if t==ta else tb);native=faces[fi].paramAt(tuple(point*1000))
  P[i]=point;CADrecords[int(i)]=(fi,*native);affected.add(int(i));outside.append(dist)
 band_quads[label]=bqi;band_reports[label]={'positive_band_quads':len(bqi),'positive_band_nodes':len(bnodes),'nodes_outside_atlas':int(sum(np.array(outside)>1e-5)),'max_chart_distance':float(max(outside))};print('band',label,band_reports[label],flush=True)
# Carry the thin trailing-edge correspondence inboard to the approved collar.
rootnodes=np.unique(Q[np.char.startswith(fam,'J_')])
for label,c in spec.items():
 ids_te=np.unique(Q[(fam==label+'_H_TE') & np.all(nom[Q,1]>=-1e-8,axis=1)])
 for i in ids_te:
  if i in rootnodes or nom[i,c['axis']]>=c['start']-1e-8 or (label=='W' and nom[i,1]<20):continue
  span=float(nom[i,c['axis']]);t=float(tt[i]);u=edgepoint(label+'TEU',span);low=edgepoint(label+'TEL',span)
  depth=4-t if label=='V' else (t-6)/2
  point=low+depth*(u-low);fi=c['te'];native=faces[fi].paramAt(tuple(point*1000));point=np.array(faces[fi].positionAt(*native).toTuple())/1000
  P[i]=point;CADrecords[int(i)]=(fi,*native);affected.add(int(i))
# Mirror the same CAD correspondence to the negative half without changing connectivity.
ids=np.array(sorted(affected));tree=cKDTree(nom);dd,mate=tree.query(nom[ids]*[1,-1,1]);assert dd.max()<1e-8
for i,m in zip(ids,mate):P[m]=P[i]*[1,-1,1];CADrecords[int(m)]=CADrecords[int(i)]
modified=np.unique(np.r_[ids,mate]);rootnodes=np.unique(Q[np.char.startswith(fam,'J_')]);assert np.array_equal(P[rootnodes],base['points'][rootnodes]);unchanged=np.setdiff1d(np.arange(len(P)),modified);assert np.array_equal(P[unchanged],base['points'][unchanged]);np.savez_compressed(O/'surface_tips_projected.npz',points=P,quads=Q,family=fam,modified_nodes=modified,CAD_records=np.array([[i,*CADrecords[i]] for i in sorted(CADrecords)]),positive_cap_quads_W=cap_quads['W'],positive_cap_quads_H=cap_quads['H'],positive_cap_quads_V=cap_quads['V'],positive_band_quads_W=band_quads['W'],positive_band_quads_H=band_quads['H'],positive_band_quads_V=band_quads['V']);(O/'tip_projection_summary.json').write_text(json.dumps({'cap':cap_reports,'band':band_reports,'modified_nodes':len(modified),'approved_root_nodes_unchanged':True,'all_other_nodes_exactly_unchanged':True,'connectivity_retained':True,'points_sha256':hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest()},indent=2))
