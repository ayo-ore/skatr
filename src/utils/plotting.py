import matplotlib.pyplot as plt

# pyplot config
pyplot_cfg = {
    # "axes.formatter.limits": [-2, 2.99],
    "axes.titlesize": 14,
    "font.size": 14,
    "font.family": "serif",
    "legend.fontsize": 12,
    "figure.subplot.top": 0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.96,
    "savefig.pad_inches": 0.1,
    "savefig.dpi": 300,
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath}"
        r"\usepackage[bitstream-charter]{mathdesign}"
        r"\DeclareSymbolFont{usualmathcal}{OMS}{cmsy}{m}{n}"
        r"\DeclareSymbolFontAlphabet{\mathcal}{usualmathcal}"
    ),
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(pyplot_cfg)
