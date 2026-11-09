"""Audit the candidate and write the wall VTU and NekRS geometry pair."""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from quality import L, rules, evaluate, eval_q2

root=Path(sys.argv[1]); work=Path('mesh_upgrade'); out=Path('deliver/final/aircraft_quadratic')
out.mkdir(parents=True,exist_ok=True)
d=np.load(root/'aircraft_tail_refined_package/output/tail_refined.npz')
p=d['points']; layer_ids=d['layer_nodes']; A=p[layer_ids]; H=d['hexes']; Q=d['wall_quads']; nq=len(Q)
guarded='--guarded' in sys.argv
reuse='--reuse-outer' in sys.argv
if reuse:
    old_report=json.loads((out/'validation.json').read_text())
    assert old_report['whole_cell_Bernstein_unresolved']==0
    old_metrics=np.loadtxt(out/'layer_metrics.csv',delimiter=',',skiprows=1)
    assert np.array_equal(np.load(work/'candidate_checked.npz')['layers'],np.load(work/'candidate_guarded.npz')['layers'])
candidate=np.load(work/('candidate_guarded.npz' if guarded else 'candidate_checked.npz')); B=candidate['layers']; profile=candidate['profile']
pr=np.load(work/'projection_checked.npz'); E=pr['edges']; ei=pr['edge_index']; delta=pr['delta']; ne=len(E)
scale=np.r_[candidate['edscale'],candidate['fcscale']]
matched=pr['face_id']>=0
moves=delta*scale[:,None]
wall_nodes=np.vstack([A[0],A[0,E].mean(axis=1)+moves[:ne],A[0,Q].mean(axis=1)+moves[ne:]])
wall_cells=np.c_[Q,ei+len(A[0]),np.arange(nq)+len(A[0])+ne]
wall_fields=np.r_[np.zeros(len(A[0])),np.linalg.norm(delta-moves,axis=1)]
vpoints=vtk.vtkPoints(); vpoints.SetData(numpy_to_vtk(wall_nodes,deep=True))
cells=vtk.vtkCellArray();cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,9*nq+1,9,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(wall_cells.ravel(),deep=True))
grid=vtk.vtkUnstructuredGrid();grid.SetPoints(vpoints);grid.SetCells(vtk.VTK_BIQUADRATIC_QUAD,cells)
for name,values in [('projection_fraction',np.r_[np.zeros(len(A[0])),scale*matched]),('distance_to_assigned_CAD_target_m',wall_fields),('cad_target_available',np.r_[np.zeros(len(A[0]),dtype=np.uint8),matched.astype(np.uint8)])]:
    array=numpy_to_vtk(values,deep=True);array.SetName(name);grid.GetPointData().AddArray(array)
writer=vtk.vtkXMLUnstructuredGridWriter();writer.SetFileName(str(out/'aircraft_wall_quadratic_m.vtu'));writer.SetInputData(grid)
if writer.Write()!=1:raise RuntimeError('VTU write failed')
# A shared wall edge has one displacement, and its radial face trace is the
# same on both neighbouring hexes. The radial profile is identical at joins.
correction=np.concatenate([moves[ei],moves[ne:,None,:]],axis=1)
(correction/7.00532).astype('<f8').tofile(out/'aircraft_q2.bin')
template=root/'aircraft_nekrs'
old=(template/'aircraft.re2').read_bytes()
raw=np.frombuffer(old,dtype='<f8',offset=84,count=len(H)*25).reshape(-1,25)
assert np.array_equal(raw[:,1:].reshape(-1,3,8).transpose(0,2,1),p[H]/7.00532)
new=raw.copy()
xyz=np.empty_like(p);xyz[layer_ids]=B
new[:,1:]=(xyz[H]/7.00532).transpose(0,2,1).reshape(-1,24)
with (out/'aircraft.re2').open('wb') as f:
    f.write(old[:84]);new.tofile(f);f.write(old[84+raw.nbytes:])
with (out/'aircraft.re2').open('rb') as f:
    assert f.read(84)==old[:84]
    assert np.array_equal(np.fromfile(f,'<f8',len(H)*25).reshape(-1,25),new)
    assert f.read()==old[84+raw.nbytes:]
for name in ('aircraft.par','aircraft_smoke.par'):
    s=(template/name).read_text()
    (out/name).write_text(s.replace('[GENERAL]','[GENERAL]\nusr = aircraft.usr',1))
shutil.copyfile(work/'aircraft.usr',out/'aircraft.usr')
udf=(template/'aircraft.udf').read_text()
# Retain the established freestream initialization and exact-length diagnostic.
a=udf.index('    // Smooth');b=udf.index('    bc->uyFluid',a)
udf=udf[:a]+'    bc->uxFluid = 1.0;\n'+udf[b:]
s='    platform->linAlg->fill(nrs->fluid->o_P.size(), 0.0, nrs->fluid->o_P);'
assert s in udf
udf=udf.replace(s,'    auto o_ux = nrs->fluid->o_solution().slice(0, nrs->fluid->fieldOffset);\n    platform->linAlg->fill(nrs->meshV->Nlocal, 1.0, o_ux);\n'+s)
udf=udf.replace('nrs->fluid->fieldOffset, o_id, nrs->fluid->o_solution());','nrs->fluid->fieldOffset, o_id, nrs->fluid->o_solution().slice(\n              0, mesh->dim * nrs->fluid->fieldOffset));')
(out/'aircraft.udf').write_text(udf)
if guarded: (work/'guarded_export_ready').touch()
from quality import bernstein_setup,bernstein_coeff,certify
bset=[bernstein_setup((2,2,2)),bernstein_setup((5,5,2))]
g,W,D=rules(7)
linear_W=W@L
linear_D=[m@L for m in D]
cfl0=np.empty((60,nq)); cfl1=np.empty_like(cfl0)
stats=[];uncertified=[];bad_count=0; split_count=0
def audit_layer(k):
    unresolved=[]
    count=0
    score=np.empty((2,4,nq))
    for i in range(0,nq,384):
        j=min(i+384,nq); q=Q[i:j]
        ac=np.concatenate([A[k,q],A[k+1,q]],axis=1)
        bc=np.concatenate([B[k,q],B[k+1,q]],axis=1)
        b=L@bc
        if k<8:
            w=np.array([profile[k],(profile[k]+profile[k+1])/2,profile[k+1]])
            for m,slot in enumerate((9,21,15,3,12)):
                b[:,slot:slot+3]+=correction[i:j,m,None,:]*w[None,:,None]
        score[0,:,i:j]=evaluate(ac,linear_W,linear_D,g)
        if k<8:
            score[1,:,i:j]=evaluate(b,W,D,g)
        elif np.array_equal(ac,bc):
            score[1,:,i:j]=score[0,:,i:j]
        else:
            score[1,:,i:j]=evaluate(bc,linear_W,linear_D,g)
        co=bernstein_coeff(b,bset[k<8])
        tol=np.max(abs(co),axis=(1,2,3))*1.e-10
        for c in np.flatnonzero(co.min(axis=(1,2,3))<=tol):
            count+=1
            if not certify(co[c],tol[c]):unresolved.append(k*nq+i+int(c))
    def edges(x):
        xx=x[k,Q]; return np.linalg.norm(xx-np.roll(xx,1,axis=1),axis=2)
    e0=edges(A); e1=edges(B)
    stats=[k,score[0,1].min(),score[1,1].min(),score[0,2].min(),score[1,2].min(),
                  7.00532*score[0,3].max(),7.00532*score[1,3].max(),np.quantile(e0,.05),np.quantile(e1,.05),
                  e0.min(),e1.min()]
    return k,score[:,3],stats,unresolved,count,int(np.sum(score[1,0]<=0))

import multiprocessing as mp
with mp.get_context('fork').Pool(5) as pool:
    for k,rates,metrics,fail,split,bad in pool.imap(audit_layer,range(8 if reuse else 60)):
        cfl0[k],cfl1[k]=rates
        stats.append(metrics)
        uncertified.extend(fail)
        split_count+=split
        bad_count+=bad
        if k%10==0: print('audited layer',k,metrics,flush=True)
assert bad_count==0 and not uncertified, (bad_count,uncertified[:20])
if reuse: stats.extend(old_metrics[8:].tolist())
with (out/'layer_metrics.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['layer_zero_based','baseline_min_scaled_J','candidate_min_scaled_J',
                'baseline_min_axis_GLL_spacing_m','candidate_min_axis_GLL_spacing_m',
                'baseline_max_uniform_x_CFL_proxy_per_tstar','candidate_max_uniform_x_CFL_proxy_per_tstar',
                'baseline_p05_tangential_edge_m','candidate_p05_tangential_edge_m',
                'baseline_min_tangential_edge_m','candidate_min_tangential_edge_m'])
    w.writerows(stats)
s=np.asarray(stats)
relative=np.ones(1) if reuse else cfl1[22:60]/cfl0[22:60]
report={'hexes':len(H),'wall_quads':nq,'curved_hexes':8*nq,'gll_degree':7,
        'sampled_nonpositive_jacobian_cells':bad_count,'whole_cell_Bernstein_unresolved':len(uncertified),
        'Bernstein_cells_requiring_subdivision_in_audited_layers':split_count,
        'reused_identical_outer_certified_layers':52 if reuse else 0,
        'min_scaled_jacobian_baseline_candidate':s[:,1:3].min(axis=0).tolist(),
        'min_axis_adjacent_spacing_m_baseline_candidate':s[:,3:5].min(axis=0).tolist(),
        'global_uniform_x_CFL_proxy_baseline_candidate':(s[:,5:7].max(axis=0)).tolist(),
        'layer_22_to_59_uniform_x_CFL_proxy_baseline_candidate':s[22:60,5:7].max(axis=0).tolist(),
        'layer_35_to_55_uniform_x_CFL_proxy_baseline_candidate':s[35:56,5:7].max(axis=0).tolist(),
        'outer_cell_fraction_CFL_proxy_improved':float(np.mean(relative<.999)),
        'outer_cell_fraction_CFL_proxy_worsened':float(np.mean(relative>1.001)),
        'outer_cell_median_CFL_proxy_ratio':float(np.median(relative)),
        'higher_order_nodes':len(scale),'max_applied_wall_correction_mm':1000*float(np.linalg.norm(moves,axis=1).max()),'unresolved_CAD_targets':int(sum(~matched)),
        'relaxed_higher_order_nodes':int(sum((scale<1)&matched)),
        'CAD_target_distance_mm_baseline_candidate':(1000*np.quantile(np.c_[np.linalg.norm(delta[matched],axis=1),np.linalg.norm((delta-moves)[matched],axis=1)],[.5,.95,1],axis=0)).tolist(),
        'wall_corners_exact':bool(np.array_equal(A[0],B[0])),
        'farfield_corners_exact':bool(np.array_equal(A[-1],B[-1])),
        'max_corner_movement_m':float(np.linalg.norm(B-A,axis=-1).max())}
if reuse:
    for key in ('outer_cell_fraction_CFL_proxy_improved','outer_cell_fraction_CFL_proxy_worsened','outer_cell_median_CFL_proxy_ratio'):
        report[key]=old_report[key]
if guarded:
    report['intersection_guard_linear_wall_quads']=int(np.sum((scale[ne:]==0)&np.all(scale[ei]==0,axis=1)))
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
