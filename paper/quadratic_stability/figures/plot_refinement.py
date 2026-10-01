"""Render the four-panel figure exclusively from the saved example records."""
import argparse
import json
from pathlib import Path
from matplotlib.lines import Line2D
from paper_plot_style import plt,COLORS,MARKERS

HERE=Path(__file__).resolve().parents[1]

def run(results,output):
    endpoints=json.loads((results/"ENDPOINTS.json").read_text(encoding="utf-8"))
    margins=json.loads((results/"MARGINS.json").read_text(encoding="utf-8"))
    summary=json.loads((results/"SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["status"]=="PASS" and len(endpoints)==39 and len(margins)==27
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.2))
    labels={"A":"A: base, D = 2","B":"B: add coupled direction, D = 5",
            "C":"C: add control direction, D = 5"}
    for design in ("A","B","C"):
        e=[r for r in endpoints if r["design"]==design and r["delta"]["plot_midpoint"]>0]
        m=[r for r in margins if r["design"]==design]
        color,marker=COLORS[design],MARKERS[design]
        axes[0,0].loglog([r["delta"]["plot_midpoint"] for r in e],
                        [r["loss"]["plot_midpoint"] for r in e],
                        color=color,marker=marker,ms=3,lw=1.2)
        axes[0,1].loglog([r["margin"]["plot_midpoint"] for r in m],
                        [r["critical_radius"]["plot_midpoint"] for r in m],
                        color=color,marker=marker,ms=3,lw=1.2)
        for ax,lo,hi in ((axes[1,0],"N_lower","N_upper"),
                         (axes[1,1],"scalar_lower","scalar_upper")):
            x=[r["margin"]["plot_midpoint"] for r in m]
            lower=[r[lo] for r in m];upper=[r[hi] for r in m]
            ax.fill_between(x,lower,upper,color=color,alpha=.09,lw=0)
            ax.loglog(x,lower,color=color,ls="--",lw=1.2)
            ax.loglog(x,upper,color=color,marker=marker,ms=3,lw=1.2)
    for ax,title in zip(axes.flat,("(a) Exact downward loss","(b) First-failure radius",
                                   "(c) Repetition bounds","(d) Scalar-cost bounds")):
        ax.set_title(title,loc="left")
        ax.grid(which="major",color=".9",lw=.5)
        ax.tick_params(which="both",direction="out")
    axes[0,0].set_xlabel(r"Observation tolerance $\delta$")
    axes[0,0].set_ylabel(r"$\ell_i(\delta)$")
    axes[0,1].set_ylabel(r"$r_i(m)$")
    axes[1,0].set_ylabel("Repetitions")
    axes[1,1].set_ylabel("Scalar observations")
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_xlabel(r"Reference margin $m$")
    handles=[Line2D([0],[0],color=COLORS[d],marker=MARKERS[d],lw=1.2,ms=4,label=labels[d])
             for d in ("A","B","C")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,1.015),
               ncol=1,frameon=False,labelspacing=.25)
    axes[1,0].legend(handles=[
        Line2D([0],[0],color=".25",ls="--",label="Necessary lower"),
        Line2D([0],[0],color=".25",ls="-",label="Sufficient upper")],
        loc="upper right",frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.88),h_pad=1.15,w_pad=1.8)
    fig.savefig(output/"wang2.pdf")
    fig.savefig(output/"wang2.png",dpi=300)
    plt.close(fig)
    print("Rendered wang2.pdf and .png from all saved grid records")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=HERE/"results/boundary_refinement_run01")
    p.add_argument("--output",type=Path,default=HERE/"figures")
    a=p.parse_args();run(a.results.resolve(),a.output.resolve())
