"""Add a smooth inward bump and a mesh-aligned wall-attached partial baffle."""
import numpy as np
from nekmeshpy import hexmesh

removed_baffle = None


def add_features(mesh, cfg, out):
    global removed_baffle
    f=cfg['features']; bump=f['wall_bump']; baffle=f['baffle']
    H=cfg['junction_branch_height']; ell=cfg['branch_transition_length']
    R0=cfg['junction_diameter']/2; R1=cfg['branch_outlet_diameter']/2
    z0=bump['z_center']; length=bump['axial_length']; height=bump['height']
    assert H < z0-length/2 < z0+length/2 < H+ell
    assert 0 < height < 0.25*R1
    def warp(P):
        Q=P.copy()
        u=(P[:,2]-z0)/(length/2)
        # Compact cap with displacement and two derivatives zero at its ends.
        shape=np.maximum(1-u*u,0)**3
        t=np.clip((P[:,2]-H)/ell,0,1)
        radius=R0+(R1-R0)*t*t*(3-2*t)
        side=(np.maximum(-P[:,1],0)/radius)**3
        active=(np.abs(P[:,0]) < R0*1.05) & (np.abs(u)<1)
        Q[:,1]+=height*shape*side*active
        return Q
    mesh=hexmesh.transform_fn(mesh,warp)
    fc=mesh.points[mesh.quad_mesh.corners].mean(axis=1)
    patch=(hexmesh.boundary_face_ids(mesh) & (np.abs(fc[:,2]-z0)<length/2)
           & (fc[:,1]<-0.005) & (np.abs(fc[:,0])<R0))
    mesh=hexmesh.tag_faces(mesh,np.flatnonzero(patch),'wall_bump')
    # Main tail is parallel to +y, centred at x=L+transition+Rb.
    rb=cfg['main_bend_radius']; Rt=cfg['main_outlet_diameter']/2
    X=cfg['junction_main_half_length']+cfg['main_transition_length']+rb
    tail=cfg['main_tail_length']
    nl=max(2,int(np.ceil(tail/cfg['mesh']['target_axial_length'])))
    i0=int(round(nl*baffle['tail_fraction_start']))
    i1=int(round(nl*baffle['tail_fraction_end']))
    assert 0 < i0 < i1 < nl
    ya=rb+tail*i0/nl; yb=rb+tail*i1/nl
    cut=X+Rt*baffle['cut_fraction_of_radius']
    centers=mesh.points[mesh.corners].mean(axis=1)
    remove=((centers[:,1]>ya+1e-9)&(centers[:,1]<yb-1e-9)&
            (centers[:,0]>cut)&(np.abs(centers[:,2])<Rt+1e-9))
    ids=np.flatnonzero(remove)
    assert ids.size>0
    removed_baffle=hexmesh.select(mesh,ids)
    # Excise whole cells: exposed faces already match remaining high-order nodes.
    mesh=hexmesh.remove(mesh,ids)
    newly_exposed=hexmesh.boundary_face_ids(mesh)&~mesh.face_tags.is_tagged()
    mesh=hexmesh.tag_faces(mesh,np.flatnonzero(newly_exposed),'baffle')
    assert np.count_nonzero(newly_exposed)>0
    removed_volume=hexmesh.volume(removed_baffle)
    area=removed_volume/(yb-ya)
    info={'wall_bump':{'location':'negative-y wall of vertical branch transition',
                       'z_center_m':z0,'axial_support_m':length,
                       'design_peak_height_m':height,
                       'construction':'smooth shared-node warp, inward into fluid'},
          'baffle':{'location':'positive-x side of main outlet straight tail',
                    'y_start_m':ya,'y_end_m':yb,'thickness_m':yb-ya,
                    'main_tail_center_x_m':X,'selection_cut_x_m':cut,
                    'removed_hexes':int(ids.size),
                    'exposed_wall_faces':int(np.count_nonzero(newly_exposed)),
                    'approx_blocked_area_m2':float(area),
                    'approx_blocked_area_fraction':float(area/(np.pi*Rt*Rt)),
                    'construction':'mesh-aligned stepped rib, not an exact planar chord plate'}}
    return mesh,info
