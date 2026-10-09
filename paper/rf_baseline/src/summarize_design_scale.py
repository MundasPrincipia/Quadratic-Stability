"""Absolute scales of the already declared lower/upper radius endpoints."""
import pandas as pd
from common import HERE,sha,now,write_json,write_csv

if __name__=='__main__':
    p=HERE/'results/e3_run01/DESIGN_RESULTS.csv';data=pd.read_csv(p)
    out=HERE/'results/design_scale_run01';out.mkdir(exist_ok=False);rows=[]
    for (source,method,k),g in data.groupby(['source','method','k']):
        for task in ['pair','multiclass']:
            low=g[task+'_radius_lower'];upper=g[task+'_radius_upper'];censor=g[task+'_censored']
            ratios=upper[~censor]/low[~censor]
            row=dict(source=source,method=method,k=int(k),task=task,anchors=len(g),censored=int(censor.sum()))
            for label,arr in [('delta_lower',low),('delta_upper',upper[~censor]),('upper_lower_ratio',ratios)]:
                for suffix,q in [('q25',.25),('median',.5),('q75',.75)]:row[label+'_'+suffix]=float(arr.quantile(q))
            rows.append(row)
    write_csv(out/'ABSOLUTE_RADII_AND_BRACKETS.csv',rows)
    summary=[r for r in rows if r['method'] in ['baseline','difference'] and r['task']=='multiclass']
    lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{Absolute multiclass lower radii and witness brackets. Entries are median [interquartile range]; $k=0$ is the shared baseline and $k=1,2$ use difference design.}',r'\label{tab:v16-absolute-radii}',r'\begin{tabular}{llrr}',r'\toprule',r'Source & $k$ & $10^4\delta_L$ & $\delta_U/\delta_L$\\',r'\midrule']
    for r in summary:
        low='%.2f [%.2f, %.2f]'%tuple(1e4*r['delta_lower_'+s] for s in ['median','q25','q75'])
        ratio='%.2f [%.2f, %.2f]'%tuple(r['upper_lower_ratio_'+s] for s in ['median','q25','q75'])
        lines.append(f"{r['source']} & {r['k']} & {low} & {ratio}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\parbox{\columnwidth}{\footnotesize No target-pair or multiclass row is censored. One of the 15,840 individual competitor comparisons is censored; another witnessed competitor still supplies its multiclass upper bound. All methods and both pair/multiclass quantities are in the accompanying CSV.}',r'\end{table}']
    (HERE/'tables/v16_absolute_radii.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'MANIFEST.json',dict(created_at=now(),issuance='post-outcome descriptive presentation of predeclared endpoints',
        input_sha256=sha(p),runner_sha256=sha(HERE/'src/summarize_design_scale.py'),
        output_sha256=sha(out/'ABSOLUTE_RADII_AND_BRACKETS.csv'),table_sha256=sha(HERE/'tables/v16_absolute_radii.tex')))
    for r in summary:print(r['source'],r['k'],r['delta_lower_median'],r['upper_lower_ratio_median'])
