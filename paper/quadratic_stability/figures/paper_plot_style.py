"""Print style for the v19 exact refinement figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "font.size":10,"axes.labelsize":10,"axes.titlesize":10,
    "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,
    "mathtext.fontset":"stix","pdf.fonttype":42,"ps.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False,
    "savefig.bbox":"tight","savefig.pad_inches":0.05,
})
COLORS={"A":"#0072B2","B":"#D55E00","C":"#009E73"}
MARKERS={"A":"o","B":"s","C":"^"}
