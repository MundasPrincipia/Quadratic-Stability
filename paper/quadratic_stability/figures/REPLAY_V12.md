# Figure replay in the v12 package

Figures 1–4 are inherited images, not new experiments. Their unchanged
`v11_*_provenance.json` files record paths relative to the **v11 root**, which
is packaged here as `inherited_v11/`. Their script fingerprint identifies
`inherited_v11/src/plot_v11.py`, not a nonexistent current `src/plot_v11.py`.
The historical provenance is preserved rather than rewritten.

| Figure | Result directory relative to the v12 package |
|---|---|
| 1 | `inherited_v11/results/budget_rates/run_20260905_045339_758316/` |
| 2 | `inherited_v11/results/positive_rank/run_20260905_124913_710642/` |
| 3 | `inherited_v11/results/weak_margin/run_20260905_130201/` |
| 4 | `inherited_v11/results/design/run_20260905_044629_927159/` |

From the extracted v12 package root, run:

```text
python src/replay_figures.py --output-dir tmp/my_new_figure_replay
```

The directory must be new. This reads the existing rows and the original
plotting functions, redirects only their image output, and does not call
the historical main program that also rewrites its summary table. It does
not train heads, solve design problems or regenerate experiment results.
Requirements are Python, NumPy, pandas, Matplotlib and Pillow. The original
font preference is Times New Roman, with DejaVu Serif fallback; font and
library changes may change rendering. PDF metadata may differ across runs.

Individual entries work with an explicit fresh output directory, for example:

```text
python figures/gen_fig1.py --output-dir tmp/my_new_figure1_replay
```

The analogous entries for Figures 2, 3 and 4 have the same interface. Do not
write into `figures/`, `inherited_v11/` or an earlier release. The main TeX
roots continue to use the preserved images; no replay is needed to compile.
The v12 caption's solver-status clarification does not change plotted data.
