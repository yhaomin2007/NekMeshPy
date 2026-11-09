"""Compare the short tangential edges and directional CFL proxy by layer."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
O=Path('deliver/final/aircraft_quadratic')
a=np.loadtxt(O/'layer_metrics.csv',delimiter=',',skiprows=1)
fig,axes=plt.subplots(1,2,figsize=(10.8,3.8),layout='constrained')
for ax,cols,yl in [
        (axes[0],(7,8),'5th-percentile tangential edge [m]'),
        (axes[1],(5,6),'Maximum uniform-flow CFL proxy per unit $t^*$')]:
    ax.semilogy(a[:,0],a[:,cols[0]],color='#52687e',ls='--',label='Accepted linear mesh')
    ax.semilogy(a[:,0],a[:,cols[1]],color='#007c83',label='Redistributed Q2 candidate')
    ax.set(xlabel='Radial layer (0 = aircraft wall)',ylabel=yl)
    ax.axvspan(0,7,facecolor='#d6e9ef',alpha=.5)
    ax.grid(alpha=.2)
axes[0].legend(loc='lower right',frameon=False)
fig.savefig(O/'aircraft_layer_comparison.png',dpi=210)
