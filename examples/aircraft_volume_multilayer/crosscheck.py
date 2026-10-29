from pathlib import Path
import sys,json,hashlib,numpy as np
from quality import linear_scan
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from nekmeshpy.core.interp import sampled_scaled_jacobian,corner_indices
R=Path(__file__).resolve().parent;O=R/'output';d=np.load(str(Path(__file__).resolve().parent/'inputs/volume_input.npz'));P=np.load(O/'gll_constrained.npz')['points'];H=d['hexes'];ids=np.r_[np.random.default_rng(42).choice(len(H),192,replace=False),5740,54586];corners=np.empty((len(ids),8,3));corners[:,corner_indices(1,3)]=P[H[ids]];r={'points_sha256':hashlib.sha256(P.tobytes()).hexdigest(),'deterministic_elements':len(ids),'crosscheck':{}}
for n in [3,5,7]:
 a=sampled_scaled_jacobian(corners,1,n,3);b=linear_scan(P,H[ids],n);err=float(abs(a-b).max());assert np.allclose(a,b,atol=1e-9,rtol=1e-8);r['crosscheck'][str(n)]=err
(O/'independent_jacobian_crosscheck.json').write_text(json.dumps(r,indent=2));print(r)
