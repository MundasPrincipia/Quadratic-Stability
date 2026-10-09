"""Presentation-only revision; original experimental sources remain unchanged."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from make_figures import A, COLORS, FLAGS, NAMES, save, perturbation
from common import HERE, sha, now, write_json


def applicability():
    data = pd.read_csv(A / 'E1_SOURCE_FLAG.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharey=True)
    markers = ['o', 's', '^', 'D', 'v', 'X']
    styles = ['-', '--', '-.', ':', '--', '-.']
    for ax, source in zip(axes, ['S1', 'S4']):
        ax.axvspan(.19, .31, color='.93', zorder=0)
        ax.axvline(.25, color='.6', lw=.65, ls=':')
        for flag, label, color, marker, style in zip(FLAGS, NAMES, COLORS, markers, styles):
            f = data[(data.source == source) & (data.flag == flag)].sort_values('delta')
            interior = f[f.delta < .25]
            cap = f[f.delta == .25]
            assert len(interior) == 11 and len(cap) == 1
            ax.plot(interior.delta, 100 * interior.fraction, marker=marker,
                    ls=style, ms=3.2, lw=1.05, color=color, label=label,
                    markerfacecolor='none')
            ax.plot(cap.delta, 100 * cap.fraction, marker=marker, ls='none',
                    ms=3.2, color='.45', markerfacecolor='none')
        ax.text(.25, 99, 'Cap', ha='center', va='top', color='.35', fontsize=7)
        ax.set_xscale('log')
        ax.set_xlabel(r'Observation tolerance $\delta$')
        ax.set_title(source, loc='left', fontsize=9)
        ax.grid(True, alpha=.18)
        ax.set_ylim(-2, 103)
        ax.set_xlim(7e-7, .31)
    axes[0].set_ylabel('Conservatively applicable (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
               ncol=3, bbox_to_anchor=(.5, -.04), frameon=False)
    fig.tight_layout(rect=(0, .13, 1, 1))
    save(fig, 'v16_applicability')


def design():
    data = pd.read_csv(A / 'E3_PAIRED_ROWS.csv')
    data['alpha_ratio'] = data.pair_alpha_upper / data.pair_alpha_upper_baseline
    metrics = ['alpha_ratio', 'multiclass_radius_lower_ratio']
    random = data[data.method.str.startswith('random')].groupby(
        ['source', 'anchor', 'k'], as_index=False)[metrics].mean()
    random['method'] = 'Random (3-seed mean)'
    main = data[data.method.isin(['difference', 'raw_scores'])][
        ['source', 'anchor', 'k', 'method'] + metrics].copy()
    main.method = main.method.map({'difference': 'Difference head', 'raw_scores': 'Raw scores'})
    merged = pd.concat([main, random], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.1))
    for i, source in enumerate(['S1', 'S4']):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for name, color, marker, style in zip(
                    ['Difference head', 'Raw scores', 'Random (3-seed mean)'],
                    [COLORS[0], COLORS[1], COLORS[5]], ['o', 's', '^'], ['-', '--', '-.']):
                g = merged[(merged.source == source) & (merged.method == name)]
                med = [1.] + [g[g.k == k][metric].median() for k in [1, 2]]
                lo = [1.] + [g[g.k == k][metric].quantile(.25) for k in [1, 2]]
                hi = [1.] + [g[g.k == k][metric].quantile(.75) for k in [1, 2]]
                ax.plot([0, 1, 2], med, marker=marker, ls=style, lw=1.2,
                        ms=3.5, color=color, label=name)
                ax.fill_between([0, 1, 2], lo, hi, color=color, alpha=.12)
            ax.set_xticks([0, 1, 2])
            ax.grid(True, alpha=.18)
            ax.set_title(source, loc='left', fontsize=9)
            ax.set_ylabel(r'Target $\alpha$ / baseline' if j == 0 else
                          r'Multiclass $\delta_L$ / baseline')
            if i == 1:
                ax.set_xlabel('Added temporal directions k')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
               ncol=3, bbox_to_anchor=(.5, -.015), frameon=False)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, 'v16_design')


if __name__ == '__main__':
    applicability()
    design()
    # Unchanged canonical data and transformation for the perturbation figure.
    perturbation()
    files = [HERE / 'figures' / (stem + '.' + ext)
             for stem in ['v16_applicability', 'v16_perturbation', 'v16_design']
             for ext in ['pdf', 'svg', 'png']]
    inputs = [A / 'E1_SOURCE_FLAG.csv', A / 'E3_PAIRED_ROWS.csv',
              HERE / 'results/e2_run02/CANONICAL_INTERVALS.csv']
    write_json(HERE / 'provenance/FINAL_FIGURE_RECEIPT.json', {
        'created_at': now(), 'scope': 'presentation-only, complete grids unchanged',
        'source_sha256': sha(__file__),
        'inputs': {p.relative_to(HERE).as_posix(): sha(p) for p in inputs},
        'outputs': {p.relative_to(HERE).as_posix(): sha(p) for p in files}})
