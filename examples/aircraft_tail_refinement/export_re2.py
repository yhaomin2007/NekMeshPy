from pathlib import Path
import sys,gzip,shutil,numpy as np,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from nekmeshpy import hexmesh,writer
from nekmeshpy.io.reader import _read_re2_raw
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(O/'tail_refined.npz');P=d['points'];H=d['hexes'];L=d['layer_nodes'];mesh=hexmesh.HexMesh.from_corners(P,H);faces=mesh.quad_mesh.corners;boundary=hexmesh.boundary_face_ids(mesh)
for name,ids in [('aircraft_wall',L[0]),('farfield',L[-1])]:mesh=hexmesh.tag_faces(mesh,np.flatnonzero(boundary&np.all(np.isin(faces,ids),axis=1)),name)
writer.to_re2(mesh,str(O/'aircraft_tail_refined.re2'),bc={'aircraft_wall':('W  ',1),'farfield':('O  ',2)});raw,count,_=_read_re2_raw(str(O/'aircraft_tail_refined.re2'),3,'deformed mesh validation');assert count==len(H) and np.array_equal(raw,P[H]);del raw
with (O/'aircraft_tail_refined.re2').open('rb') as src,gzip.open(O/'aircraft_tail_refined.re2.gz','wb',compresslevel=6) as dst:shutil.copyfileobj(src,dst)
(O/'RE2_export_validation.json').write_text(json.dumps({'points_sha256':hashlib.sha256(P.tobytes()).hexdigest(),'hexes':len(H),'geometry_order':1,'exact_corner_roundtrip':True,'boundary_ids':{'wall':1,'farfield':2},'boundary_codes':'W/O export placeholders; configure BCs for the intended flow.','solver_run':False},indent=2));print('RE2 exact corner roundtrip passed',flush=True)
