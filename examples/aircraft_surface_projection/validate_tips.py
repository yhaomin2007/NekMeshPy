"""Validate changed tip nodes on native CAD, and sample the adjoining linear quads."""
from cad_surface import *
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN,TopAbs_ON,TopAbs_REVERSED
from OCP.gp import gp_Pnt2d
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
import multiprocessing as mp,json,csv,hashlib,time
d=np.load(O/'surface_input.npz');base=np.load(O/'base_surface.npz');s=np.load(O/'surface_tips_projected.npz');P=s['points'];Q=s['quads'];fam=s['family'];nom=d['nominal'];records=s['CAD_records'];modified=s['modified_nodes'];nodeerr=[];outside=[];nodeids=[];nodeface=[]
for row in records:
 i,fi=map(int,row[:2]);uv=row[2:4];sign=-1 if nom[i,1]<-1e-8 else 1;point=P[i]*[1,sign,1];exact=np.array(faces[fi].positionAt(*uv).toTuple())/1000;nodeerr.append(np.linalg.norm(exact-point));state=BRepClass_FaceClassifier(faces[fi].wrapped,gp_Pnt2d(*uv),1e-4).State();nodeids.append(i);nodeface.append(fi)
 if state not in [TopAbs_IN,TopAbs_ON]:outside.append(i)
print('tip CAD node reconstruction max m',max(nodeerr),'outside trim',len(outside),flush=True)
mask=np.zeros(len(P),bool);mask[modified]=True;affected=np.any(mask[Q],axis=1);ids=np.flatnonzero(affected);groups={'WC':[14],'WU':[4,12,13],'WL':[1,7,19,21,27,29],'WT':[11,25,23,15,0,3,10],'HC':[47],'HU':[48],'HL':[50],'HT':[49],'VC':[53],'VU':[51],'VT':[52]};tables={k:np.flatnonzero(np.isin(FI,fs)) for k,fs in groups.items()};locs={k:locator(polydata(V,T[table])) for k,table in tables.items()};owners=[];quality=[];tridist=[];labels=[]
for i in ids:
 label=str(fam[i])[0];sector=d['quad_sector'][i];key=label+('C' if fam[i]==label+'_tip' else ('T' if sector>=(3 if label=='V' else 6) else ('U' if label=='V' or sector<3 else 'L')));sign=-1 if nom[Q[i],1].mean()<-1e-8 else 1;center=P[Q[i]].mean(0)*[1,sign,1];q=[0.,0.,0.];ci=vtk.reference(0);si=vtk.reference(0);ds=vtk.reference(0.);locs[key].FindClosestPoint(center,q,ci,si,ds);idx=tables[key][int(ci)];tri=T[idx];fi=int(FI[idx]);v=V[tri];a=np.linalg.lstsq(np.c_[v[1]-v[0],v[2]-v[0]],np.array(q)-v[0],rcond=None)[0];uv=np.r_[1-a.sum(),a]@UV[tri];pt=gp_Pnt();aa=gp_Vec();bb=gp_Vec();adaptors[fi].D1(float(uv[0]),float(uv[1]),pt,aa,bb);normal=np.cross(aa.Coord(),bb.Coord());normal*=(-1 if faces[fi].wrapped.Orientation()==TopAbs_REVERSED else 1);normal/=np.linalg.norm(normal);normal*=[1,sign,1];c=P[Q[i]];values=[]
 for u in np.linspace(0,1,5):
  for vv in np.linspace(0,1,5):
   du=-(1-vv)*c[0]+(1-vv)*c[1]+vv*c[2]-vv*c[3];dv=-(1-u)*c[0]-u*c[1]+u*c[2]+(1-u)*c[3];values.append(np.dot(np.cross(du,dv),normal)/(np.linalg.norm(du)*np.linalg.norm(dv)))
 quality.append(min(values));owners.append(fi);tridist.append(np.sqrt(float(ds)));labels.append(label)
quality=np.array(quality);owners=np.array(owners);labels=np.array(labels);bad=ids[quality<=0];print('affected quads',len(ids),'minimum signed surface Jacobian',quality.min(),'nonpositive',len(bad),flush=True)
bounds=[]
for face in faces[:-1]:
 b=face.BoundingBox();bounds.append([[b.xmin,b.ymin,b.zmin],[b.xmax,b.ymax,b.zmax]])
bounds=np.array(bounds)/1000;centres=P[Q[ids]].mean(1);cqpoints=centres.copy();cqpoints[:,1]=abs(cqpoints[:,1]);unique,inv=np.unique(np.round(cqpoints,10),axis=0,return_inverse=True);first=np.unique(inv,return_index=True)[1];seed=owners[first]
def query(job):
 p,fi=job;vertex=cq.Vertex.makeVertex(*(p*1000));lower=np.linalg.norm(np.maximum(np.maximum(bounds[:,0]-p,p-bounds[:,1]),0),axis=1);best=np.inf
 for k in np.r_[fi,np.argsort(lower)]:
  if lower[k]>best+1e-8:continue
  dd=BRepExtrema_DistShapeShape(vertex.wrapped,faces[int(k)].wrapped);dd.Perform()
  if dd.IsDone() and dd.NbSolution():best=min(best,dd.Value()/1000)
 return best
sha=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest();cadsha=hashlib.sha256((R/'reference.brep').read_bytes()).hexdigest();cache=O/'exact_tip_centres.npz'
if cache.exists() and str(np.load(cache)['points_sha256'])==sha:values=np.load(cache)['values']
else:
 values=[];start=time.time()
 with mp.get_context('fork').Pool(4) as pool:
  for k,v in enumerate(pool.imap(query,zip(unique,seed),chunksize=8)):
   values.append(v)
   if k%200==0:print('direct CAD centres',k,len(unique),'seconds',round(time.time()-start),flush=True)
 values=np.array(values);np.savez_compressed(cache,values=values,query_points=unique,points_sha256=sha,CAD_sha256=cadsha)
error=values[inv];assert np.isfinite(error).all();C=P[Q[ids]];areas=np.zeros(len(ids));gx,gw=np.polynomial.legendre.leggauss(3);gx=(gx+1)/2;gw=gw/2
for a,u in enumerate(gx):
 for b,v in enumerate(gx):
  du=-(1-v)*C[:,0]+(1-v)*C[:,1]+v*C[:,2]-v*C[:,3];dv=-(1-u)*C[:,0]-u*C[:,1]+u*C[:,2]+(1-u)*C[:,3];areas+=np.linalg.norm(np.cross(du,dv),axis=1)*gw[a]*gw[b]
regions={}
for label in ['W','H','V']:
 for part in ['cap','band_and_neighbours','all']:
  m=(labels==label)
  if part=='cap':m&=fam[ids]==label+'_tip'
  elif part=='band_and_neighbours':m&=fam[ids]!=label+'_tip'
  e=error[m];ix=np.argsort(e);w=areas[m];regions[label+'_'+part]={'quad_count':int(sum(m)),'mean_m':float(np.average(e,weights=w)),'p95_m':float(np.interp(.95,np.cumsum(w[ix])/sum(w),e[ix])),'max_m':float(e.max()),'worst_xyz_m':centres[m][np.argmax(e)].tolist()}
E=np.sort(np.stack([Q,np.roll(Q,-1,axis=1)],axis=2),axis=2).reshape(-1,2);E,count=np.unique(E,axis=0,return_counts=True);rootids=np.unique(Q[np.char.startswith(fam,'J_')]);assert np.array_equal(P[rootids],base['points'][rootids]);report={'points_sha256':sha,'CAD_sha256':cadsha,'wall_nodes':len(P),'wall_quads':len(Q),'checked_nodes':len(nodeids),'max_native_CAD_node_residual_m':float(max(nodeerr)),'nodes_outside_trimmed_CAD_faces':outside,'checked_quads':len(ids),'minimum_CAD_normal_signed_surface_scaled_Jacobian':float(quality.min()),'nonpositive_surface_quad_ids':bad.tolist(),'approved_junction_nodes_exactly_unchanged':True,'connectivity_exactly_unchanged':bool(np.array_equal(Q,base['quads'])),'open_or_nonmanifold_edges':int(sum(count!=2)),'regions':regions,'scope':'Surface-only correction of W/H/V tip caps, last span interval and shared inboard interface nodes. Native UV/trim ownership checked. Surface orientation sampled on 5x5 bilinear grids. Distances to all 55 physical original CAD faces at quad centres; 3x3 area weights; finite samples, not a Hausdorff bound. No volume mesh updated.'};(O/'tip_CAD_validation.json').write_text(json.dumps(report,indent=2));np.savez_compressed(O/'tip_validation_fields.npz',checked_quad_ids=ids,surface_quality=quality,centre_error=error,node_ids=np.array(nodeids),node_error=np.array(nodeerr),node_face=np.array(nodeface),areas=areas)
with (O/'tip_deviation_regions.csv').open('w') as f:
 w=csv.writer(f,lineterminator="\n");w.writerow(['region','quad_count','mean_m','p95_m','max_m','worst_x_m','worst_y_m','worst_z_m'])
 for key,v in regions.items():w.writerow([key,v['quad_count'],v['mean_m'],v['p95_m'],v['max_m'],*v['worst_xyz_m']])
print(json.dumps(report,indent=2),flush=True)
