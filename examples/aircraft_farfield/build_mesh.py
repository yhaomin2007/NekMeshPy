"""Assemble a conformal near-body/transition/farfield all-hex mesh."""
from pathlib import Path
import json,argparse,time
import numpy as np,vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
from nekmeshpy import hexmesh,writer
p=argparse.ArgumentParser();p.add_argument('--radius',type=float,default=750.);p.add_argument('--bridge',default='transition_geometry.npz');args=p.parse_args()
root=Path(__file__).resolve().parent;out=root/'output';out.mkdir(exist_ok=True)
near=np.load(root/'nearbody_mesh.npz');b=np.load(root/args.bridge);L=b['layers'];Q=b['quads'];C=b['center'];sub=int(b['subdivision']) if 'subdivision' in b else 1;n=4//sub
old_patch=b['old_patch'] if 'old_patch' in b else np.arange(len(Q));parent_ij=b['local_ij'] if 'local_ij' in b else np.zeros((len(Q),2),int)
P0=near['points'];H0=near['hexes'];bid0=near['block_id'];ijk0=near['ijk'];lookup={tuple(np.round(p,10)):k for k,p in enumerate(P0)}
def bilinear(v,u,w):return (1-u)*(1-w)*v[0]+u*(1-w)*v[1]+u*w*v[2]+(1-u)*w*v[3]
surface_ids=[];unique_surface={};tables=[];qfine=[];patch=[];ij=[]
for k,q in enumerate(Q):
    tab=np.empty((n+1,n+1),int)
    for j,v in enumerate(np.linspace(0,1,n+1)):
        for i,u in enumerate(np.linspace(0,1,n+1)):
            point=bilinear(L[0,q],u,v);key=tuple(np.round(point,10));nid=lookup[key]
            if nid not in unique_surface:unique_surface[nid]=len(surface_ids);surface_ids.append(nid)
            tab[j,i]=unique_surface[nid]
    tables.append(tab)
    for j in range(n):
        for i in range(n):qfine.append([tab[j,i],tab[j,i+1],tab[j+1,i+1],tab[j+1,i]]);patch.append(old_patch[k]);ij.append((parent_ij[k]*n+[i,j]).tolist())
surface_ids=np.array(surface_ids);qfine=np.array(qfine);patch=np.array(patch);ij=np.array(ij);nf=len(surface_ids)
def interpolate(layer):
    x=np.zeros((nf,3))
    for k,q in enumerate(Q):
        for j,v in enumerate(np.linspace(0,1,n+1)):
            for i,u in enumerate(np.linspace(0,1,n+1)):x[tables[k][j,i]]=bilinear(layer[q],u,v)
    return x
fine=[P0[surface_ids]]+[interpolate(l) for l in L[1:]]
base=fine[-1];rr=np.linalg.norm(base-C,axis=1);U=(base-C)/rr[:,None]
ratio=1.18;delta=ratio**np.arange(30);fractions=np.r_[0,np.cumsum(delta)/sum(delta)]
for f in fractions[1:]:fine.append(C+U*(rr+(args.radius-rr)*f)[:,None])
points=[P0];hexes=[H0];blocks=[bid0];ijk=[ijk0];region=[np.ones(len(H0),int)];levels=[np.full(len(H0),-1,int)];previous=surface_ids
for k,x in enumerate(fine[1:]):
    ids=np.arange(nf)+len(P0)+k*nf;points.append(x);hexes.append(np.c_[previous[qfine],ids[qfine]]);blocks.append(680+(k//3)*340+patch+1);ijk.append(np.c_[ij,np.full(len(ij),k%3)]);region.append(np.full(len(qfine),2 if k<len(L)-1 else 3));levels.append(np.full(len(qfine),k));previous=ids
P=np.concatenate(points);H=np.concatenate(hexes);blocks=np.concatenate(blocks);ijk=np.concatenate(ijk);region=np.concatenate(region);levels=np.concatenate(levels)
mesh=hexmesh.HexMesh.from_corners(P,H);sj=hexmesh.scaled_jacobian(mesh,order=3);print('cells',len(H),'min SJ',sj.min(),'bad',sum(sj<=0),flush=True)
if not np.isfinite(sj).all() or np.any(sj<=0):
    np.savez_compressed(root/'failed_assembly.npz',points=P,hexes=H,quality=sj,block_id=blocks,region=region)
    raise RuntimeError('Assembled mesh has invalid cells; no solver mesh exported')
# Batch sampling to bound memory while checking every element.
from nekmeshpy.core.interp import sampled_scaled_jacobian,corner_indices
from nekmeshpy.core.quality import OrderScan
mins=[];invs=[]
for order in [5,7]:
    vals=[]
    for start in range(0,len(H),2048):
        hh=H[start:start+2048];block=np.empty((len(hh),8,3));block[:,corner_indices(1,3),:]=P[hh]
        vals.append(sampled_scaled_jacobian(block,1,order,3))
    quality=np.concatenate(vals);assert np.isfinite(quality).all()
    mins.append(float(quality.min()));invs.append(int(sum(quality<=0)))
    print('order',order,'min SJ',mins[-1],'bad',invs[-1],flush=True)
    if invs[-1]:
        np.savez_compressed(root/'failed_assembly.npz',points=P,hexes=H,quality=quality,block_id=blocks,region=region)
        raise RuntimeError('Invalid higher-order samples; no solver mesh exported')
scan=OrderScan((5,7),tuple(mins),tuple(invs),())
top=hexmesh.topology_report(mesh);assert top.n_components==1 and top.conformal and top.watertight,top
fc=mesh.quad_mesh.corners;bd=hexmesh.boundary_face_ids(mesh);wallnodes=set(np.flatnonzero(near['radial_coordinate']==0));far_nodes=set(previous.tolist())
wall=bd&np.array([all(i in wallnodes for i in f) for f in fc]);far=bd&np.array([all(i in far_nodes for i in f) for f in fc]);assert np.all((wall|far)==bd)
for label,mask in [('aircraft_wall',wall),('farfield',far)]:mesh=hexmesh.tag_faces(mesh,np.flatnonzero(mask),label)
tags=hexmesh.tag_report(mesh);assert tags.n_untagged_boundary==tags.n_tagged_interior==0
writer.to_re2(mesh,str(out/'aircraft_farfield.re2'),bc={'aircraft_wall':('W  ',1),'farfield':('O  ',2)})
pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(P),deep=True));cells=vtk.vtkCellArray();pack=np.c_[np.full(len(H),8),H].ravel().astype(np.int64);cells.SetCells(len(H),numpy_to_vtkIdTypeArray(pack,deep=True));g=vtk.vtkUnstructuredGrid();g.SetPoints(pts);g.SetCells(vtk.VTK_HEXAHEDRON,cells)
for name,data in [('block_id',blocks),('region',region),('ijk',ijk),('scaled_jacobian_order3',sj),('extension_layer',levels)]:a=numpy_to_vtk(np.ascontiguousarray(data),deep=True);a.SetName(name);g.GetCellData().AddArray(a)
w=vtk.vtkXMLUnstructuredGridWriter();w.SetFileName(str(out/'aircraft_farfield.vtu'));w.SetInputData(g);w.Write()
np.savez_compressed(out/'aircraft_farfield.npz',points=P,hexes=H,block_id=blocks,ijk=ijk,region=region,quality=sj,interface_nodes=surface_ids,interface_quads=qfine,farfield_nodes=previous,farfield_quads=qfine)
stats={'hexes':len(H),'points':len(P),'nearbody_hexes':len(H0),'transition_hexes':int(sum(region==2)),'outer_hexes':int(sum(region==3)),'blocks':int(blocks.max()),'radius_m':args.radius,'center_m':C.tolist(),'nearbody_unchanged':bool(np.array_equal(P[:len(P0)],P0) and np.array_equal(H[:len(H0)],H0)),'interface_faces':len(qfine),'scaled_jacobian_order3_min':float(sj.min()),'quality_scan':scan._asdict(),'topology':top._asdict(),'tags':tags._asdict(),'outer_radial_growth_ratio':ratio,'outer_radial_layers':30,'transition_layers':len(L)-1,'boundary_ids':{'aircraft_wall':1,'farfield':2},'bc_note':'O is an export placeholder, not a complete external-flow boundary condition setup','cad_fitted':False,'solver_run':False}
(out/'checks.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2),flush=True)
