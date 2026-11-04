from pathlib import Path
import subprocess,time,json
q=Path(__file__).resolve().parent
while True:
    subprocess.run(['/home/xyw/miniforge3/envs/malr/bin/python','/home/xyw/paper/E23/scripts/summarize_exchange_csdi_budget_comparison.py'],check=True)
    status=json.loads((q/'status.json').read_text())['status']
    if status!='running':break
    time.sleep(30)
