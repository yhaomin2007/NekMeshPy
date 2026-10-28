from pathlib import Path
import json,time,numpy as np,vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
from nekmeshpy import hexmesh,reader
from nekmeshpy.core import topology
root=Path(__file__).resolve().parent;out=root/'output'
m=np.load(out/'aircraft_farfield.npz');near=np.load(root/'nearbody_mesh.npz');b=np.load(root/'transition_geometry.npz');info=json.loads((out/'checks.json').read_text());P=m['points'];H=m['hexes'];C=np.array(info['center_m'])
# The former outer boundary must be represented by exactly one face from each
# mesh portion, using identical shared node IDs.
old=hexmesh.HexMesh.from_corners(near['points'],near['hexes']);fc=old.quad_mesh.corners;mask=hexmesh.boundary_face_ids(old)&np.all(near['radial_coordinate'][fc]==1,axis=1)
keys=lambda a:{tuple(sorted(row)) for row in a}
expected=keys(fc[mask]);actual=keys(H[len(near['hexes']):len(near['hexes'])+len(m['interface_quads']),:4]);assert expected==actual and len(actual)==5440
info['interface_audit']={'matched_faces':len(actual),'missing_or_extra_faces':len(expected^actual),'nearbody_coordinates_and_connectivity_unchanged':bool(np.array_equal(P[:len(near['points'])],near['points']) and np.array_equal(H[:len(near['hexes'])],near['hexes']))}
assert info['interface_audit']['nearbody_coordinates_and_connectivity_unchanged']
# Check clearance conservatively: minimum centre-to-faceted-farfield distance
# minus the maximum radius of every vertex in the original CAD reference.
fq=m['farfield_nodes'][m['farfield_quads']];poly=vtk.vtkPolyData();pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(P),deep=True));poly.SetPoints(pts);ca=vtk.vtkCellArray();tri=np.r_[fq[:,[0,1,2]],fq[:,[0,2,3]]];pack=np.c_[np.full(len(tri),3),tri].ravel().astype(np.int64);ca.SetCells(len(tri),numpy_to_vtkIdTypeArray(pack,deep=True));poly.SetPolys(ca);dist=vtk.vtkImplicitPolyDataDistance();dist.SetInput(poly);rmin=abs(dist.EvaluateFunction(C))
print('Interface checked',flush=True)
cad=np.load(root/'cad_surface.npz')['points']/1000;rgeom=np.linalg.norm(cad-C,axis=1).max();clearance=rmin-rgeom
info['domain_clearance']={'min_faceted_boundary_radius_m':float(rmin),'CAD_max_radius_m':float(rgeom),'conservative_body_to_boundary_distance_lower_bound_m':float(clearance),'lower_bound_in_reference_chords':float(clearance/7.00532),'reference_chord_m':7.00532,'meets_100_chord_clearance_bound':bool(clearance>=700.532)}
assert clearance>=700.532
# Read exported corners directly: avoid a size-based re-weld tolerance for
# this domain's very wide range of element sizes.
from nekmeshpy.io.reader import _read_re2_raw
corners,nelv,bcs=_read_re2_raw(str(out/'aircraft_farfield.re2'),3,'validation')
assert nelv==len(H) and np.array_equal(corners,P[H])
from collections import Counter
codes=Counter(code for blk in bcs for _,_,code in blk)
assert codes=={'W':5440,'O':5440},codes
info['re2_roundtrip']={'all_element_corners_exact':True,'elements':nelv,'boundary_codes':dict(codes),'scope':'raw reader comparison without coordinate welding; topology audited on the identical source connectivity'}
del corners
print('Interface, clearance and RE2 checks passed',flush=True)
# Macro-level overlap sampling: the near/transition cells are trilinear macro
# maps; the outer macro hull approximates the spherical logical blocks.
MP=near['macro_points'];MH=near['macro_hexes'];raw=[MP[MH]];L=b['layers'];Q=b['quads']
for a,z in zip(L[:-1],L[1:]):raw.append(np.concatenate([a[Q],z[Q]],1))
end=L[-1];U=(end-C)/np.linalg.norm(end-C,axis=1)[:,None];g=1.18**np.arange(30);f=np.r_[0,np.cumsum(g)/g.sum()];R=info['radius_m'];r0=np.linalg.norm(end-C,axis=1)
for a,z in zip(f[:-1],f[1:]):raw.append(np.concatenate([(C+U*(r0+a*(R-r0))[:,None])[Q],(C+U*(r0+z*(R-r0))[:,None])[Q]],1))
v=np.concatenate(raw);pts=v.reshape(-1,3);_,ix,inv=np.unique(np.round(pts,10),axis=0,return_index=True,return_inverse=True);start=time.time();count=0;bad_pairs=[]
# Bounded-memory AABB broad phase, then the repository's original sample/plane test.
X=pts[ix];HC=inv.reshape(-1,8);centroid,radius,vertex,normal=topology._hex_geometry(X,HC);samples=topology._hex_sample_points(X,HC)
lo=X[HC].min(1);hi=X[HC].max(1);tol=1e-6*float(np.linalg.norm(np.ptp(X,axis=0)));candidates=0
for first in range(0,len(HC),32):
    ids=np.arange(first,min(first+32,len(HC)))
    mask=np.all(hi[ids,None,:]>=lo[None,:,:],axis=2)&np.all(hi[None,:,:]>=lo[ids,None,:],axis=2)
    mask&=np.arange(len(HC))[None,:]>ids[:,None]
    aa,bb=np.nonzero(mask);aa=ids[aa];candidates+=len(aa)
    for k in range(0,len(aa),1000):
        i=aa[k:k+1000];j=bb[k:k+1000]
        a=topology._points_inside(samples[i],vertex[j],normal[j],tol);z=topology._points_inside(samples[j],vertex[i],normal[i],tol)
        bad=a|z|(np.linalg.norm(centroid[i]-centroid[j],axis=1)<tol);count+=int(sum(bad));bad_pairs.extend(np.c_[i[bad],j[bad]].tolist())
info['macro_overlap_sampling']={'macro_hexes':len(v),'overlapping_pairs':int(count),'candidate_pairs':candidates,'tolerance_m':tol,'method':'batched AABB broad phase and repository sample/plane narrow phase','seconds':time.time()-start,'scope':'near-body and transition macro maps plus linear corner approximations of the spherical outer blocks; not an exhaustive fine-cell intersection proof'}
np.save(root/'macro_flagged_pairs.npy',np.array(bad_pairs,dtype=int))
# Planar macro approximations can flag warped, valid trilinear faces.
# Retain the flags and audit the actual exported children, then invert their
# trilinear maps for all remaining polyhedral flags.
import subprocess,sys
subprocess.run([sys.executable,str(root/'check_flagged_pairs.py')],check=True)
subprocess.run([sys.executable,str(root/'check_trilinear_pairs.py')],check=True)
info['fine_pair_audit']=json.loads((out/'flagged_pair_audit.json').read_text())
info['trilinear_pair_audit']=json.loads((out/'trilinear_pair_audit.json').read_text())
info['intersection_audit_note']='Polyhedral flags are retained: planar approximations differ from warped trilinear faces. Inverse-map sampling found no strictly interior crossings among the remaining flagged fine pairs. This is a finite diagnostic, not an exhaustive geometric proof.'
(out/'checks.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
assert info['trilinear_pair_audit']['strictly_inside_count']==0
