"""
Evaluation helpers for the PID model: the MC-vs-data comparison plot
(Figure 32 in Niels den Besten's thesis, built like his fiducial_analysis notebook).

No torch needed: everything works on numpy arrays / pandas columns.

Workflow:
    logit = neutrino_logit(p_neutrino)                      # per event, MC and data
    mc_class = mc_class_from_pid(truth_pid)                 # noise / muon / nue / numu / nutau
    result = compare_mc_data(mc_logit, mc_class, mc_w, data_logit, data_w)
    plot_mc_data_comparison(result, title="...")
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

EPS = 1e-7
MC_CLASSES = ("noise", "muon", "nue", "numu", "nutau")
COLORS = {"noise": "tab:blue", "muon": "tab:orange",
          "nue": "#1b5e20", "numu": "#43a047", "nutau": "#a5d6a7"}
LABELS = {"noise": "noise", "muon": "muon", "nue": "nue", "numu": "numu", "nutau": "nutau"}


def neutrino_logit(p, eps: float = EPS) -> np.ndarray:
    """log(p / (1 - p)), clamped with eps so that p = 0 or 1 gives about -/+16.1."""
    p = np.asarray(p, dtype=float)
    return np.log((p * (1 - 2 * eps) + eps) / ((1 - p) * (1 - 2 * eps) + eps))


def mc_class_from_pid(pid) -> np.ndarray:
    """Map the true `pid` to the five MC classes of the plot.

    -1 = noise, +-13 = muon, +-12 = nue, +-14 = numu, +-16 = nutau (anything else: 'other').
    """
    pid = np.asarray(pid, dtype=float)
    out = np.full(pid.shape, "other", dtype=object)
    out[pid == -1] = "noise"
    out[np.abs(pid) == 13] = "muon"
    out[np.abs(pid) == 12] = "nue"
    out[np.abs(pid) == 14] = "numu"
    out[np.abs(pid) == 16] = "nutau"
    return out

def mc_class_from_label(label) -> np.ndarray:
    """Map the 5-class `label` of the prepared meta files (0 noise, 1 muon, 2 nue, 3 numu, 4 nutau) to names."""
    return np.asarray(MC_CLASSES, dtype=object)[np.asarray(label).astype(int)]

def load_prepared_summary(path):
    """Read summary.csv of the preparation script; the index is the dataset NUMBER (121122, ...)."""
    import pandas as pd
    df = pd.read_csv(path)
    df["dataset_no"] = df["dataset"].str.split("/").str[-1].astype(int)
    return df.set_index("dataset_no")

def rescale_weights_by_dataset(class_weight, dataset, summary):
    """Rescale the weights of a random subset, per dataset folder.
    new weight = class_weight * (n_events_total - n_dropped_no_pulses) / (events of this dataset in the input).
    Call it BEFORE any physics cut."""
    import pandas as pd
    class_weight = np.asarray(class_weight, dtype=float)
    dataset = np.asarray(dataset).astype(int)
    new_w = class_weight.copy()
    rows = []
    for d in np.unique(dataset):
        sel = dataset == d
        n_kept = int(sel.sum())
        r = summary.loc[d]
        n_full = float(r["n_events_total"] - r["n_dropped_no_pulses"])
        new_w[sel] = class_weight[sel] * n_full / n_kept
        rows.append(dict(dataset=r["dataset"], n_kept=n_kept, n_full_with_pulses=int(n_full),
                         factor=n_full / n_kept, sum_new_w=new_w[sel].sum(),
                         sum_class_weight_all=r["sum_class_weight_all"]))
    table = pd.DataFrame(rows).set_index("dataset")
    table["new_w / full"] = table["sum_new_w"] / table["sum_class_weight_all"]
    return new_w, table


def weighted_hist(values, weights, bins):
    """Weighted histogram and its statistical error sqrt(sum of w^2) per bin."""
    weights = np.asarray(weights, dtype=float)
    h, _ = np.histogram(values, bins=bins, weights=weights)
    var, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return h, np.sqrt(var)


def fit_scale_factors(components, h_data) -> np.ndarray:
    """Poisson-likelihood fit of one scale factor per MC component to the data histogram.

    Same fit as in the thesis notebook: minimise sum(pred - data * log(pred)) over
    log10(scale factors).
    """
    def nll(log10_s):
        pred = sum(10.0 ** l * h for l, h in zip(log10_s, components))
        return np.sum(pred - h_data * np.log(np.maximum(pred, 1e-10)))

    res = minimize(nll, np.zeros(len(components)), method="L-BFGS-B")
    return 10.0 ** res.x


def compare_mc_data(mc_logit, mc_class, mc_w, data_logit, data_w,
                    n_bins: int = 62, bins=None, fit: bool = True) -> dict:
    """Histogram MC (per class) and data on the same bins, and fit the MC scale factors.

    With fit=True one scale factor each is fitted for noise, muon and neutrinos
    (nue, numu and nutau share one factor, as in Figure 32). fit=False uses 1.0.
    """
    mc_logit, mc_class, mc_w = np.asarray(mc_logit), np.asarray(mc_class), np.asarray(mc_w, dtype=float)
    data_logit, data_w = np.asarray(data_logit), np.asarray(data_w, dtype=float)

    if bins is None:
        all_l = np.concatenate([mc_logit, data_logit])
        bins = np.linspace(all_l.min(), all_l.max(), n_bins + 1)

    hs = {k: weighted_hist(mc_logit[mc_class == k], mc_w[mc_class == k], bins) for k in MC_CLASSES}
    h_nu = hs["nue"][0] + hs["numu"][0] + hs["nutau"][0]
    h_data, h_data_err = weighted_hist(data_logit, data_w, bins)

    if fit:
        s = fit_scale_factors([hs["noise"][0], hs["muon"][0], h_nu], h_data)
        sf = {"noise": s[0], "muon": s[1], "nue": s[2], "numu": s[2], "nutau": s[2]}
    else:
        sf = {k: 1.0 for k in MC_CLASSES}

    mc = {k: hs[k][0] * sf[k] for k in MC_CLASSES}
    return dict(
        bins=bins, centers=0.5 * (bins[1:] + bins[:-1]),
        mc=mc, mc_total=sum(mc.values()),
        mc_err=np.sqrt(sum((hs[k][1] * sf[k]) ** 2 for k in MC_CLASSES)),
        data=h_data, data_err=h_data_err,
        scale_factors=sf, n_data=len(data_logit), n_mc=len(mc_logit),
    )


def plot_mc_data_comparison(result: dict, title: str = None, save_path: str = None,
                            ylim=(1e-5, 1e2), ratio_ylim=(0, 2)):
    """Figure 32 style plot: stacked MC classes, data points, and a Data/MC ratio panel."""
    bc = result["centers"]
    bw = result["bins"][1] - result["bins"][0]
    tot, err = result["mc_total"], result["mc_err"]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})

    bottom = np.zeros_like(bc)
    for k in MC_CLASSES:
        h = result["mc"][k]
        ax.bar(bc, h, width=bw, bottom=bottom, color=COLORS[k], alpha=0.75,
               label=f"{LABELS[k]} (s={result['scale_factors'][k]:.2f})")
        bottom += h
    ax.fill_between(bc, np.maximum(tot - err, 1e-10), tot + err,
                    color="gray", alpha=0.35, label="MC stat. unc.")
    ax.errorbar(bc, result["data"], yerr=result["data_err"], fmt="o", ms=3.5, color="black",
                elinewidth=1, capsize=2, zorder=5, label=f"Data (n={result['n_data']:,})")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_ylabel("Rate [Hz]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tot > 0, result["data"] / tot, np.nan)
        ratio_err = np.where(tot > 0, result["data_err"] / tot, np.nan)
        mc_band = np.where(tot > 0, err / tot, np.nan)
    axr.fill_between(bc, 1 - mc_band, 1 + mc_band, color="gray", alpha=0.35)
    axr.errorbar(bc, ratio, yerr=ratio_err, fmt="o", ms=3, color="red", elinewidth=1, capsize=2)
    axr.axhline(1, color="black", ls="--", lw=1)
    axr.set_ylim(*ratio_ylim)
    axr.set_ylabel("Data/MC")
    axr.set_xlabel("neutrino logit")
    axr.grid(alpha=0.25)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, (ax, axr)
