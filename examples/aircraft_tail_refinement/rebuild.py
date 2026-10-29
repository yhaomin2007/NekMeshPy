from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parent;(R/'output').mkdir(exist_ok=True)
for name in ['build_plan.py','refine_tail.py','scan.py','measure_refinement.py','verify_topology.py','gll_spacing.py','export_vtu.py','export_wall.py','export_re2.py','render_comparison.py','verify_checkpoint.py']:
 print('Running',name,flush=True);subprocess.run([sys.executable,name],cwd=R,check=True)
