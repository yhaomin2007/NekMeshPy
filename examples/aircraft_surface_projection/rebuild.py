"""Reproduce the final surface from the approved junction baseline."""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parent
for name in ['project_tips.py','refine_records.py','snap_trim_records.py','repair_wing_te.py','repair_wing_rows.py','snap_other_records.py','validate_tips.py','topology_check.py','audit_tips.py','render_tips.py','export_surface.py','export_vtu.py','plot_deviations.py','finalize_status.py']:
 print('Running',name,flush=True);subprocess.run([sys.executable,str(R/name)],cwd=R,check=True)
