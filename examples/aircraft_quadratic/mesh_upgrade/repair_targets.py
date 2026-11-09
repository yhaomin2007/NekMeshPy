"""Resolve large seeded displacements using native trimmed-face distances."""
import multiprocessing as mp
import sys
from pathlib import Path
import cadquery as cq
import numpy as np
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.gp import gp_Pnt
root=Path(sys.argv[1])
p=Path('mesh_upgrade/projection_bounded.npz')
d=np.load(root/'aircraft_tail_refined_package/output/tail_refined.npz')
pr=dict(np.load(p))
x=d['points'][d['layer_nodes'][0]]
q=d['wall_quads']
query=np.vstack([x[pr['edges']].mean(axis=1),x[q].mean(axis=1)])
before=np.linalg.norm(pr['delta'],axis=1)
idx=np.flatnonzero(before>.08)
ref=query[idx].copy()
ref[:,1]=abs(ref[:,1])
uniq,first,inv=np.unique(np.round(ref,10),axis=0,return_index=True,return_inverse=True)
faces=cq.Shape.importBrep(str(root/'aircraft_surface_tips/reference.brep')).Faces()
def exact(job):
    point,face=job
    v=BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float,point*1000))).Vertex()
    result=BRepExtrema_DistShapeShape(v,faces[face].wrapped)
    if not result.IsDone():return np.full(3,np.nan)
    return np.asarray(result.PointOnShape2(1).Coord())/1000
with mp.get_context('fork').Pool(5) as pool:
    result=np.asarray(list(pool.imap(exact,zip(ref[first],pr['face_id'][idx[first]]),chunksize=2)))
target=result[inv]
target[query[idx,1]<0,1]*=-1
target[abs(query[idx,1])<1e-10,1]=0
proposal=target-query[idx]
change=np.linalg.norm(proposal,axis=1)<before[idx]
pr['delta'][idx[change]]=proposal[change]
np.savez('mesh_upgrade/projection_checked.npz',**pr)
print('large seeded distances corrected',int(change.sum()),'of',len(idx),'max before and after',before.max(),np.linalg.norm(pr['delta'],axis=1).max(),flush=True)
