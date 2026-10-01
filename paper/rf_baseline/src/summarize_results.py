"""Preserve all source/fold/device strata; descriptive paired comparisons."""
import json
import numpy as np
import pandas as pd
from common import HERE,write_csv,write_json,now


def run():
    out=HERE/'results/analysis_run03';out.mkdir(exist_ok=False)
    e1=pd.read_csv(HERE/'results/e1_run01/STRATIFIED_APPLICABILITY.csv')
    grouped=e1.groupby(['source','flag','delta'],as_index=False)[['denominator','applicable','outer_bound_available']].sum()
    grouped['fraction']=grouped.applicable/grouped.denominator
    write_csv(out/'E1_SOURCE_FLAG.csv',grouped.to_dict('records'))
    natural=pd.read_csv(HERE/'results/e1_run01/NATURAL_ROWS.csv')
    mass=[]
    for (source,flag),g in natural.groupby(['source','flag']):
        mass.append(dict(source=source,flag=flag,rows=len(g),s_min=g.residual_mass.min(),s_q25=g.residual_mass.quantile(.25),
            s_median=g.residual_mass.median(),s_q75=g.residual_mass.quantile(.75),s_max=g.residual_mass.max(),
            tau_over_delta_01_median=g.tau.median()/.1))
    write_csv(out/'E1_RESIDUAL_DISTRIBUTIONS.csv',mass)
    # Deterministic outer/witness gaps; no claim of global optimization.
    gaps=[]
    for p in (HERE/'results/e1_run01').glob('*/*/*.npz'):
        with np.load(p) as packed:f={key:packed[key] for key in packed.files}
        r=np.arange(len(f['row_ids']));target=f['target_pair_column']
        for j,d in enumerate(f['delta']):
            mask=f['applicable'][:,j]
            ow=f['outer_upper'][r,j,target]-f['outer_lower'][r,j,target]
            ww=f['witness_upper'][r,j,target]-f['witness_lower'][r,j,target]
            for i in np.flatnonzero(mask):
                gaps.append(dict(source=p.parent.parent.name,fold=p.parent.name,flag=p.stem,row_id=int(f['row_ids'][i]),delta=float(d),
                    outer_width=float(ow[i]),witness_span=float(ww[i]),width_ratio=float(ow[i]/ww[i]) if ww[i]>0 else None,
                    lower_endpoint_bracket=float(f['witness_lower'][i,j,target[i]]-f['outer_lower'][i,j,target[i]]),
                    upper_endpoint_bracket=float(f['outer_upper'][i,j,target[i]]-f['witness_upper'][i,j,target[i]])))
    write_csv(out/'E1_FINITE_BRACKETS.csv',gaps)
    e2=pd.read_csv(HERE/'results/e2_run02/CANONICAL_INTERVALS.csv')
    win=e2.groupby(['order','perturbation','epsilon'],as_index=False).agg(window_cells=('window_25','sum'),grid_cells=('window_25','size'))
    write_csv(out/'E2_WINDOWS.csv',win.to_dict('records'))
    e3=pd.read_csv(HERE/'results/e3_run01/DESIGN_RESULTS.csv')
    base=e3[e3.k==0].set_index('anchor')
    compare=e3[e3.k>0].copy()
    for metric in ['pair_alpha_upper','pair_radius_lower','multiclass_radius_lower']:
        compare[metric+'_baseline']=compare.anchor.map(base[metric])
        compare[metric+'_ratio']=compare[metric]/compare[metric+'_baseline']
    raw=compare[compare.method=='raw_scores'].set_index(['anchor','k'])
    for metric in ['pair_radius_lower','multiclass_radius_lower']:
        compare[metric+'_vs_raw']=[
            r[metric]/raw.loc[(r['anchor'],r['k']),metric] for _,r in compare.iterrows()]
    write_csv(out/'E3_PAIRED_ROWS.csv',compare.where(pd.notna(compare),None).to_dict('records'))
    metrics=['pair_alpha_upper','pair_radius_lower','multiclass_radius_lower','pair_radius_lower_ratio','multiclass_radius_lower_ratio',
             'pair_radius_lower_vs_raw','multiclass_radius_lower_vs_raw']
    for strata,name in [(['source','fold','method','k'],'FOLDS'),(['source','device','method','k'],'DEVICES'),(['source','method','k'],'SOURCE')]:
        agg=compare.groupby(strata)[metrics].agg(['mean','std','median','min','max']).reset_index()
        agg.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in agg.columns]
        write_csv(out/f'E3_{name}.csv',agg.where(pd.notna(agg),None).to_dict('records'))
    findings={}
    for source in ['S1','S4']:
        findings[source]={}
        for k in [1,2]:
            g=compare[(compare.source==source)&(compare.k==k)&(compare.method=='difference')]
            findings[source][f'k{k}']=dict(anchors=len(g),pair_vs_raw_median=float(g.pair_radius_lower_vs_raw.median()),
                multi_vs_raw_median=float(g.multiclass_radius_lower_vs_raw.median()),
                pair_gain_over_raw=int((g.pair_radius_lower_vs_raw>1+1e-8).sum()),
                multi_gain_over_raw=int((g.multiclass_radius_lower_vs_raw>1+1e-8).sum()),
                pair_vs_base_median=float(g.pair_radius_lower_ratio.median()),multi_vs_base_median=float(g.multiclass_radius_lower_ratio.median()))
    gapsdf=pd.DataFrame(gaps)
    write_json(out/'SUMMARY.json',dict(created_at=now(),E3=findings,
        E1_applicable_row_radius_cells=int(e1.applicable.sum()),E1_total_row_radius_cells=int(e1.denominator.sum()),
        E1_target_outer_to_witness_median=float(gapsdf.width_ratio.median()),
        E1_negative_endpoint_brackets=int(((gapsdf.lower_endpoint_bracket < -1e-8)|(gapsdf.upper_endpoint_bracket < -1e-8)).sum()),
        E2_all_cells=len(e2),E2_windows=int(e2.window_25.sum()),
        interpretation='descriptive paired source/fold/device results; no independent flag-row inference'))


if __name__=='__main__':run()
