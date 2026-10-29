"""Recompute the 30-layer volume deformation and all checks from supplied inputs."""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parent
(R/'output').mkdir(exist_ok=True)
commands=[['harmonic.py','--depth','30'],['optimize.py','--input','harmonic_30_2_1.npz','--name','constrained','--floor','.02','--iterations','500','--rounds','6'],['optimize.py','--input','constrained.npz','--name','gll_constrained','--gll','3','--floor','.012','--iterations','500','--rounds','4'],['scan.py','gll_constrained.npz'],['gll_spacing.py'],['crosscheck.py'],['verify_topology.py','gll_constrained.npz'],['audit_volume.py'],['document_changes.py'],['export_preview.py'],['export_re2.py'],['render_layers.py']]
for command in commands:
 print('Running',*command,flush=True);subprocess.run([sys.executable,*command],cwd=R,check=True)
