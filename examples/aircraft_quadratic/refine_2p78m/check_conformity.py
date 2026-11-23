import numpy as np,json
from pathlib import Path
P=Path('/mnt/data/aircraft_refined_regenerated');Q=43424;K=64
m=np.memmap(P/'aircraft.re2',dtype='<f8',mode='r',offset=84,shape=(K,Q,25));d=np.fromfile(P/'aircraft_q2.bin','<f8').reshape(Q,5,3)
wall=np.stack((m[0,:,1:5],m[0,:,9:13],m[0,:,17:21]),axis=-1)
xyz=wall.reshape(-1,3)
_,ids=np.unique(np.round(xyz,9),axis=0,return_inverse=True)
conn=ids.reshape(Q,4)
edges=np.stack([conn,np.roll(conn,-1,axis=1)],axis=-1).reshape(-1,2)
edgekeys=np.sort(edges,axis=-1)
unique_edge,inv,counts=np.unique(edgekeys,axis=0,return_inverse=True,return_counts=True)
mid=.5*(wall+np.roll(wall,-1,axis=1))+d[:,:4,:]
mx=mid.reshape(-1,3)
acc=np.zeros((len(unique_edge),3));np.add.at(acc,inv,mx)
avg=acc[inv]/counts[inv,None]
dist=np.linalg.norm(mx-avg,axis=-1)
kerr=0
for k in range(K-1):
 up=np.stack((m[k,:,5:9],m[k,:,13:17],m[k,:,21:25]),axis=-1)
 down=np.stack((m[k+1,:,1:5],m[k+1,:,9:13],m[k+1,:,17:21]),axis=-1)
 kerr=max(kerr,float(np.abs(up-down).max()))
h=np.stack((m[0,:,5:9]-m[0,:,1:5],m[0,:,13:17]-m[0,:,9:13],m[0,:,21:25]-m[0,:,17:21]),axis=-1)
sp=np.linalg.norm(h,axis=-1)*7.00532
re=1.5e6;cf=0.0592*(re**(-.2)); ut=np.sqrt(cf/2.)
yplus=sp*.06412992575*re/7.00532*ut
res={'wall_quads':Q,'unique_wall_corner_vertices':int(len(np.unique(ids))), 'wall_edges':int(len(unique_edge)),
'wall_edge_multiplicity':{str(x):int(np.sum(counts==x)) for x in np.unique(counts)},
'max_shared_quadratic_edge_midpoint_disagreement_mesh_units':float(dist.max()*2),
'max_radial_layer_interface_corner_difference_mesh_units':kerr,
'first_hex_radial_edge_m_percentiles':np.percentile(sp,[0,5,50,95,99,100]).tolist(),
'estimated_first_gll_yplus_percentiles':np.percentile(yplus,[0,5,50,95,99,100]).tolist(),
'reynolds':re,'reference_length_m':7.00532,'friction_velocity_over_freestream_estimate':float(ut)}
(P/'conformity.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
