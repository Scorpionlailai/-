from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'position_interventions.csv').open()))
variables=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
time_labels=['1-16','17-32','33-48','49-64','65-80','81-96']
fig,axes=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
fields=['repair_mse_global','mean_forecast_mse_gain']
for i,source in enumerate(('v29_r1_stage1','v30_l4_g8_stage1')):
 selected=[r for r in rows if r['source']==source and r['predictor']=='v29_r1']
 for j,field in enumerate(fields):
  values=np.zeros((7,6))
  for r in selected:values[variables.index(r['variable']),int(r['history_start'])//16]=float(r[field])*(1000 if j==1 else 1)
  ax=axes[i,j]
  im=ax.imshow(values,aspect='auto',cmap='RdBu_r' if j==1 else 'YlOrRd',norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=20) if j==1 else None,vmin=0 if j==0 else None,vmax=.7 if j==0 else None)
  ax.set_xticks(range(6),time_labels);ax.set_yticks(range(7),variables)
  for y in range(7):
   for x in range(6):ax.text(x,y,f'{values[y,x]:.1f}' if j else f'{values[y,x]:.3f}',ha='center',va='center',fontsize=8,color='white' if (values[y,x]>.5 if j==0 else values[y,x]>14) else 'black')
  ax.set_xlabel('History positions: older -> more recent')
  ax.set_title(('v29-r1 Stage 1' if i==0 else 'v30 (4 layers x 8 groups) Stage 1')+'\n'+('Regional hidden-value reconstruction MSE' if j==0 else 'Oracle-restoration forecast MSE gain x 1000'),fontsize=11)
  fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Reconstruction quality and forecasting value vary by position\nOriginal validation; the same frozen v29-r1 predictor for all panels',fontsize=13)
fig.savefig(root/'reconstruction_vs_forecast_value.png',dpi=180)
fig.savefig(root/'reconstruction_vs_forecast_value.pdf')
