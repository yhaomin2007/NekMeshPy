#!/usr/bin/env python3
"""Static v26 source/schema and binary boundary checks; does not run NekRS."""
import configparser
import json
from pathlib import Path
import re
import sys
import numpy as np

root = Path(__file__).resolve().parent
src = Path(sys.argv[1])
par_source = (src/'src/platform/par/par.cpp').read_text()

def keys(name):
    section = re.search(r'\b'+name+r'\s*=\s*\{(.*?)\n\};',par_source,re.S)
    assert section, name
    return set(re.findall(r'"([^"]+)"',section[1].lower()))

allowed = {
    'GENERAL': keys('generalKeys') | keys('commonKeys'),
    'MESH': keys('meshKeys'),
    'FLUID PRESSURE': keys('ellipticKeys') | keys('commonKeys'),
    'FLUID VELOCITY': keys('velocityKeys') | keys('commonKeys'),
}
for name in ('aircraft.par','aircraft_smoke.par'):
    c = configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',))
    c.read(root/name)
    assert set(c.sections()) == set(allowed)
    for section in c:
        if section != 'DEFAULT':
            assert not set(c[section])-allowed[section], (name,section,set(c[section])-allowed[section])
    g,v = c['GENERAL'],c['FLUID VELOCITY']
    assert g['checkpointcontrol'].lower() in ('steps','simulationtime')
    assert g['timestepper'].lower() == 'tombo2'
    assert v['boundarytypemap'].replace(' ','') == 'zeroDirichlet,udfDirichlet,zeroNeumann'
    assert v['rho']=='1.0' and v['viscosity']=='1/5900000'
    udf = root/g.get('udf',Path(name).stem+'.udf')
    assert udf.exists()

path=root/'aircraft.re2'
with path.open('rb') as f:
    header=f.read(80).decode()
    assert header.startswith('#v004') and int(header[21:24])==3
    ne=int(header[5:21])
    f.seek(84)
    xyz=np.fromfile(f,'<f8',ne*25).reshape(ne,25)[:,1:].reshape(ne,3,8).transpose(0,2,1)
    assert np.fromfile(f,'<f8',1)[0]==0
    nbc=int(np.fromfile(f,'<f8',1)[0])
    b=np.fromfile(f,'<f8',nbc*8).reshape(nbc,8)
    assert f.read()==b''
assert ne==651360 and nbc==21712
ids,counts=np.unique(b[:,6].astype(int),return_counts=True)
assert dict(zip(ids,counts))=={1:10856,2:4434,3:6422}
assert len(set(map(tuple,b[:,:2])))==nbc
codes={1:b'W  ',2:b'v  ',3:b'O  '}
for row in b:
    bid=int(row[6])
    assert row[7:8].tobytes()[:3]==codes[bid]
    assert row[1]==(5 if bid==1 else 6)
# Independent normal check from exported coordinates, not original checkpoint.
outer=b[b[:,6]>1]
q=xyz[outer[:,0].astype(int)-1,4:8]
norms=[]
for u,v in ((-1,-1),(1,-1),(1,1),(-1,1)):
    du=((1-v)*(q[:,1]-q[:,0])+(1+v)*(q[:,2]-q[:,3]))/4
    dv=((1-u)*(q[:,3]-q[:,0])+(1+u)*(q[:,2]-q[:,1]))/4
    norms.append(np.cross(du,dv))
norms=np.stack(norms,axis=1)
is_out=outer[:,6]==3
assert np.all(norms[is_out,:,0]>0)
# Integral area vector equals four times centre cross product for bilinear face.
area_x=4*norms[:,:,0].mean(axis=1)
flux_sum=float(area_x.sum())
relative=abs(flux_sum)/float(abs(area_x).sum())
assert relative<1e-12
report={'static_v26_input_check':True,'binary_boundary_check':True,
        'outlet_normals_positive_everywhere':True,
        'unit_freestream_closed_sphere_relative_flux_error':relative,
        'hexes':ne,'boundary_faces':dict(zip(map(str,ids),map(int,counts))),
        'compiled_udf':False,'solver_run':False,
        'api_tag':'v26.0','api_commit':'96b3cf9e5bacede16568826c04a21bc0fe50dc7d'}
(root/'input_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
