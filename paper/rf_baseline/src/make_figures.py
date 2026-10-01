"""Data-derived, deterministic v16 figures (no fitted-window selection)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from common import HERE

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
OUT=HERE/'figures';A=HERE/'results/analysis_run03'
COLORS=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#333333']
FLAGS=['coordinate','dct','dft','polynomial','random_orthogonal','permuted_polynomial']
NAMES=['Coordinate','DCT','DFT','Polynomial','Random orthogonal','Permuted polynomial']


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def applicability():
    data=pd.read_csv(A/'E1_SOURCE_FLAG.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.7),sharey=True)
    for ax,source in zip(axes,['S1','S4']):
        for flag,label,color in zip(FLAGS,NAMES,COLORS):
            f=data[(data.source==source)&(data.flag==flag)].sort_values('delta')
            ax.plot(f.delta,100*f.fraction,'o-',ms=2.7,lw=1.1,color=color,label=label)
        ax.set_xscale('log');ax.set_xlabel(r'Observation tolerance $\delta$');ax.set_title(source,loc='left',fontsize=9)
        ax.grid(True,alpha=.18);ax.set_ylim(-2,103)
    axes[0].set_ylabel('Conservatively applicable (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.04),frameon=False)
    fig.tight_layout(rect=(0,.13,1,1));save(fig,'v16_applicability')


def perturbation():
    data=pd.read_csv(HERE/'results/e2_run02/CANONICAL_INTERVALS.csv')
    families=[('mixed',k) for k in ['preserving','cross','tangent','residual']]+[('quadratic',k) for k in ['preserving','cross','tangent','residual','mixed']]
    labels=[('3/2' if o=='mixed' else '2')+' / '+k for o,k in families]
    values=np.array([[data[(data.order==o)&(data.perturbation==k)&(data.epsilon_index==e)].window_25.sum() for e in range(9)] for o,k in families])
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.35),gridspec_kw={'width_ratios':[1.15,1]})
    im=axes[0].imshow(values,vmin=0,vmax=12,cmap='Blues',aspect='auto')
    axes[0].set_yticks(range(9),labels);axes[0].set_xticks(range(9),['0']+[f'$10^{{{e}}}$' for e in range(-8,0)],rotation=45)
    for r in range(9):
        for c in range(9):axes[0].text(c,r,str(values[r,c]),ha='center',va='center',fontsize=6.5,color='white' if values[r,c]>7 else 'black')
    axes[0].set_xlabel(r'Perturbation amplitude $\varepsilon$');axes[0].set_title('(a) Eligible radii / 12',loc='left',fontsize=9)
    ax=axes[1];cmap=plt.get_cmap('viridis')
    for e in range(9):
        g=data[(data.order=='quadratic')&(data.perturbation=='cross')&(data.epsilon_index==e)].sort_values('delta')
        color='black' if e==0 else cmap((e-1)/7)
        ax.plot(g.delta,g.loss_upper/g.unperturbed_loss,color=color,lw=1.1,label='0' if e==0 else f'$10^{{{e-9}}}$')
        sel=g[g.window_25];ax.plot(sel.delta,sel.loss_upper/sel.unperturbed_loss,'o',ms=3,color=color)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'Observation tolerance $\delta$')
    ax.set_ylabel(r'Downward loss / $(\delta^2/2)$');ax.set_title('(b) Quadratic task + cross leakage',loc='left',fontsize=9)
    ax.grid(True,alpha=.18);ax.legend(title=r'$\varepsilon$',ncol=3,fontsize=6,title_fontsize=7,loc='upper right',frameon=False)
    fig.tight_layout(w_pad=1.5);save(fig,'v16_perturbation')


def design():
    data=pd.read_csv(A/'E3_PAIRED_ROWS.csv')
    data['alpha_ratio']=data.pair_alpha_upper/data.pair_alpha_upper_baseline
    random=data[data.method.str.startswith('random')].groupby(['source','anchor','k'],as_index=False)[['alpha_ratio','multiclass_radius_lower_ratio']].mean()
    random['method']='Random (3-seed mean)'
    main=data[data.method.isin(['difference','raw_scores'])][['source','anchor','k','alpha_ratio','multiclass_radius_lower_ratio','method']].copy()
    main.method=main.method.map({'difference':'Difference head','raw_scores':'Raw scores'})
    merged=pd.concat([main,random],ignore_index=True)
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.1))
    for i,source in enumerate(['S1','S4']):
        for j,metric in enumerate(['alpha_ratio','multiclass_radius_lower_ratio']):
            ax=axes[i,j]
            for name,color in zip(['Difference head','Raw scores','Random (3-seed mean)'],[COLORS[0],COLORS[1],COLORS[5]]):
                g=merged[(merged.source==source)&(merged.method==name)]
                med=[1.]+[g[g.k==k][metric].median() for k in [1,2]]
                lo=[1.]+[g[g.k==k][metric].quantile(.25) for k in [1,2]]
                hi=[1.]+[g[g.k==k][metric].quantile(.75) for k in [1,2]]
                ax.plot([0,1,2],med,'o-',lw=1.2,ms=3,color=color,label=name)
                ax.fill_between([0,1,2],lo,hi,color=color,alpha=.12)
            ax.set_xticks([0,1,2]);ax.grid(True,alpha=.18);ax.set_title(source,loc='left',fontsize=9)
            ax.set_ylabel(r'Target $\alpha$ / baseline' if j==0 else r'Multiclass $\delta_L$ / baseline')
            if i==1:ax.set_xlabel('Added temporal directions k')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.015),frameon=False)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'v16_design')


if __name__=='__main__':applicability();perturbation();design()
