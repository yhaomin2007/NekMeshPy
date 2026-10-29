"""Refine stretched aft-body directions; propagate splits across opposite edges."""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
R=Path(__file__).resolve().parent;O=R/'output';src=R;d=np.load(R/'inputs/volume_parent.npz');P=np.load(R/'inputs/volume_parent.npz')['points'];Q=d['wall_quads'];wall=P[d['layer_nodes'][0]];E=np.sort(np.stack([Q,np.roll(Q,-1,axis=1)],axis=2),axis=2);edges,inv=np.unique(E.reshape(-1,2),axis=0,return_inverse=True);ids=inv.reshape(-1,4);a=np.r_[ids[:,0],ids[:,1]];b=np.r_[ids[:,2],ids[:,3]];graph=coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(edges),len(edges)));n,labels=connected_components(graph,directed=False);length=np.linalg.norm(wall[Q]-np.roll(wall[Q],-1,axis=1),axis=2);C=wall[Q].mean(1);ar=length.max(1)/length.min(1);m=(d['family']=='F_body_O')&(C[:,0]>60)&(ar>8);counts=np.ones(n,int)
for i in np.flatnonzero(m):
 axis=np.argmax(length[i]);edgeclass=labels[ids[i,axis]];counts[edgeclass]=max(counts[edgeclass],min(4,int(2**np.ceil(np.log2(ar[i]/6)))))
shape=counts[labels[ids[:,:2]]];assert np.all(counts[labels[ids[:,0]]]==counts[labels[ids[:,2]]]);assert np.all(counts[labels[ids[:,1]]]==counts[labels[ids[:,3]]]);np.savez_compressed(O/'selected_plan.npz',subdivisions=shape,target_parent_quads=np.flatnonzero(m),aft_parent_quads=np.flatnonzero((d['family']=='F_body_O')&(C[:,0]>58)));print('planned hexes',60*np.prod(shape,axis=1).sum())
