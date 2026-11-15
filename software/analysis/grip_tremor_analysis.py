"""
Grip Strength & Tremor — Final Publication Analysis
=====================================================
Literature:
  Celis-Morales et al. (2018)     — weakness thresholds <26/16 kg; HR per 5 kg
  Rantanen et al. (1999)          — grip predicts disability 25 yr later
  Vaillancourt & Newell (2003)    — aging shifts force power to 0–4 Hz band
  Elble (2003)                    — central-neurogenic tremor 8–12 Hz young,
                                    5–7 Hz elderly
  McAuley & Marsden (2000)        — pathological tremor 3–6 Hz (Parkinson)
  Silva-Santos et al. (2024)      — force-time indicators beyond fmax
  Newell & Carlton (1988)         — force variability vs force level

Four figures, white theme:
  Fig 1  — Strength overview  (scatter Fmax×age + boxplot Fmax×group + boxplot RFD×group)
  Fig 2  — Force variability  (scatter CV×Fmax + scatter CV×age + boxplot PSD_force×group)
  Fig 3  — Tremor fingerprint (scatter freq×age with bands + bubble freq×FWHM×TFR + boxplot TFR×group)
  Fig 4  — Correlation matrix (Spearman, sig mask)
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from scipy import stats

# ─────────────────────────────────────────────
# STYLE  (white theme, matches analisev2)
# ─────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222",
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CCCCCC",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────

AGE_BINS = [0, 17, 39, 64, 120]
AGE_LABELS = [
    "Youth\n(<18)",
    "Adults\n(18–39)",
    "Middle-age\n(40–64)",
    "Elderly\n(≥65)",
]
AGE_LABELS_LEGEND = [
    "Youth (<18)",
    "Adults (18–39)",
    "Middle-age (40–64)",
    "Elderly (≥65)",
]

PALETTE = {
    "Youth\n(<18)": "#4FC3F7",
    "Adults\n(18–39)": "#66BB6A",
    "Middle-age\n(40–64)": "#FFA726",
    "Elderly\n(≥65)": "#EF5350",
}
COLORS = list(PALETTE.values())

WEAK_MEN = 26.0
WEAK_WOMEN = 16.0

# ─────────────────────────────────────────────
# LOAD & PREPARE
# ─────────────────────────────────────────────

df = pd.read_csv("results.csv")
df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
df["RFD"] = df["fmax"] / df["ttp"]
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def spearman_annot(ax, x, y, xpos=0.05, ypos=0.93, fontsize=9):
    mask = (~np.isnan(np.array(x, dtype=float))) & (~np.isnan(np.array(y, dtype=float)))
    if mask.sum() < 5:
        return
    r, p = stats.spearmanr(np.array(x)[mask], np.array(y)[mask])
    sig = "*" if p < 0.05 else "n.s."
    ax.text(
        xpos,
        ypos,
        f"r = {r:.2f}, p = {p:.3f} ({sig})",
        transform=ax.transAxes,
        fontsize=fontsize,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.85),
    )


def styled_boxplot(ax, data_list, labels, colors):
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "#AAAAAA",
            "alpha": 0.7,
        },
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y")
    return bp


def kruskal_annot(ax, groups_dict, ypos=0.97):
    clean = {
        k: v.dropna().values for k, v in groups_dict.items() if len(v.dropna()) >= 3
    }
    if len(clean) < 2:
        return
    stat, p = stats.kruskal(*clean.values())
    sig = "*" if p < 0.05 else "n.s."
    ax.text(
        0.97,
        ypos,
        f"Kruskal–Wallis p = {p:.3f} ({sig})",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.85),
    )


# ═══════════════════════════════════════════════════════
# FIG 1  —  STRENGTH OVERVIEW
#   Panel A: scatter fmax × age, by sex, weakness thresholds
#   Panel B: boxplot fmax × age group
#   Panel C: boxplot RFD × age group
# ═══════════════════════════════════════════════════════


def fig1_strength():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Maximum Grip Strength and Rate of Force Development",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # ── A: scatter fmax vs age ──
    ax = axes[0]
    markers = {"M": ("o", "#1565C0", "Male"), "F": ("^", "#C62828", "Female")}
    for sex, (mk, col, lbl) in markers.items():
        sub = df[df["sex"] == sex]
        ax.scatter(
            sub["age"],
            sub["fmax"],
            marker=mk,
            color=col,
            s=60,
            alpha=0.80,
            edgecolors="white",
            lw=0.5,
            label=lbl,
            zorder=4,
        )
    ax.axhline(
        WEAK_MEN,
        color="#1565C0",
        ls="--",
        lw=1.3,
        alpha=0.6,
        label=f"Weakness ♂ ({WEAK_MEN} kg)",
    )
    ax.axhline(
        WEAK_WOMEN,
        color="#C62828",
        ls="--",
        lw=1.3,
        alpha=0.6,
        label=f"Weakness ♀ ({WEAK_WOMEN} kg)",
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Fmax (kg)")
    ax.set_title("A  —  Fmax vs Age by Sex")
    ax.legend(fontsize=8)
    ax.grid(True)
    spearman_annot(ax, df["age"], df["fmax"])

    # ── B: boxplot fmax × age group ──
    ax = axes[1]
    data_fmax = [df[df["age_group"] == g]["fmax"].dropna() for g in AGE_LABELS]
    styled_boxplot(ax, data_fmax, AGE_LABELS, COLORS)
    ax.set_ylabel("Fmax (kg)")
    ax.set_title("B  —  Fmax by Age Group")
    kruskal_annot(ax, {g: df[df["age_group"] == g]["fmax"] for g in AGE_LABELS})

    # ── C: boxplot RFD × age group ──
    ax = axes[2]
    data_rfd = [df[df["age_group"] == g]["RFD"].dropna() for g in AGE_LABELS]
    styled_boxplot(ax, data_rfd, AGE_LABELS, COLORS)
    ax.set_ylabel("RFD (kg/s)")
    ax.set_title("C  —  Rate of Force Development by Age Group")
    kruskal_annot(ax, {g: df[df["age_group"] == g]["RFD"] for g in AGE_LABELS})

    plt.tight_layout()
    plt.savefig("fig_forca.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig1_forca.png ✓")


# ═══════════════════════════════════════════════════════
# FIG 2  —  FORCE VARIABILITY
#   Panel A: scatter CV × fmax  (colour = age group)
#   Panel B: scatter CV × age   (colour = age group)
#   Panel C: scatter PSD_force × age  (colour = age group)
#            — shows that 0–4 Hz power increases with age
#              (Vaillancourt & Newell 2003)
# ═══════════════════════════════════════════════════════


def fig2_variability():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Force Variability and Neuromuscular Control",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    handles = [
        mpatches.Patch(color=PALETTE[g], label=l)
        for g, l in zip(AGE_LABELS, AGE_LABELS_LEGEND)
    ]

    # Panel A & B: standard scatter plots
    for ax, xcol, ycol, xlabel, ylabel, title in [
        (
            axes[0],
            "fmax",
            "cv",
            "Fmax (kg)",
            "CV (%)",
            "A  —  CV vs Fmax",
        ),
        (axes[1], "age", "cv", "Age (years)", "CV (%)", "B  —  CV vs Age"),
    ]:
        ax.grid(True)
        for grp, color in PALETTE.items():
            sub = df[df["age_group"] == grp]
            ax.scatter(
                sub[xcol],
                sub[ycol],
                color=color,
                s=60,
                alpha=0.85,
                edgecolors="white",
                lw=0.5,
                zorder=4,
            )
        spearman_annot(ax, df[xcol], df[ycol])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Panel C: PSD Force by age group (boxplot)
    ax = axes[2]
    data_psd_f = [df[df["age_group"] == g]["psd_force"].dropna() for g in AGE_LABELS]
    styled_boxplot(ax, data_psd_f, AGE_LABELS, COLORS)
    ax.set_ylabel("PSD Force 0–4 Hz (kg²/Hz)")
    ax.set_title("C  —  Force spectral power by Age Group")
    ax.set_yscale("log")
    kruskal_annot(ax, {g: df[df["age_group"] == g]["psd_force"] for g in AGE_LABELS})

    # Shared legend below
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig("fig2_variabilidade.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig_variabilidade.png ✓")


# ═══════════════════════════════════════════════════════
# FIG 2B  —  PSD FORCE CHARACTERISATION
#   Panel A: PSD Force (0–4 Hz) distribution by age group
#   Panel B: PSD Force vs Peak Freq Force correlation
# ═══════════════════════════════════════════════════════


def fig2b_psd_force():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Force Spectral Power",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    handles = [
        mpatches.Patch(color=PALETTE[g], label=l)
        for g, l in zip(AGE_LABELS, AGE_LABELS_LEGEND)
    ]

    # Panel A: Distribution by age group
    ax = axes[0]
    data_psd_f = [df[df["age_group"] == g]["psd_force"].dropna() for g in AGE_LABELS]
    styled_boxplot(ax, data_psd_f, AGE_LABELS, COLORS)
    ax.set_ylabel("PSD Force 0–4 Hz (kg²/Hz)")
    ax.set_title("A  —  Force spectral power by Age Group")
    ax.set_yscale("log")
    kruskal_annot(ax, {g: df[df["age_group"] == g]["psd_force"] for g in AGE_LABELS})

    # Panel B: Correlation with peak frequency
    ax = axes[1]
    ax.grid(True)
    for grp, color in PALETTE.items():
        sub = df[df["age_group"] == grp]
        ax.scatter(
            sub["Peak_Freq_Force_Hz"],
            sub["psd_force"],
            color=color,
            s=60,
            alpha=0.85,
            edgecolors="white",
            lw=0.5,
            zorder=4,
        )
    spearman_annot(ax, df["Peak_Freq_Force_Hz"], df["psd_force"])
    ax.set_yscale("log")
    ax.set_xlabel("Peak Frequency — Force (Hz)")
    ax.set_ylabel("PSD Force 0–4 Hz (kg²/Hz)")
    ax.set_title("B  —  Force power vs Peak Force Frequency")

    # Shared legend below
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig("fig2b_psd_force.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig2b_psd_force.png ✓")


# ═══════════════════════════════════════════════════════
# FIG 2C  —  PSD ACCELEROMETER CHARACTERISATION
#   Panel A: PSD Acc (8–12 Hz) distribution by age group
#   Panel B: PSD Acc vs Peak Freq Acc correlation
# ═══════════════════════════════════════════════════════


def fig2c_psd_acc():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Hand Tremor Spectral Power and Force–Accelerometer Agreement",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    handles = [
        mpatches.Patch(color=PALETTE[g], label=l)
        for g, l in zip(AGE_LABELS, AGE_LABELS_LEGEND)
    ]

    # Panel A: Distribution by age group
    ax = axes[0]
    data_psd_a = [
        df[df["age_group"] == g]["PSD_Acc_8_12Hz"].dropna() for g in AGE_LABELS
    ]
    styled_boxplot(ax, data_psd_a, AGE_LABELS, COLORS)
    ax.set_ylabel("PSD Acc (g²/Hz)")
    ax.set_title("A  —  Tremor spectral power by Age Group")
    ax.set_yscale("log")
    kruskal_annot(
        ax, {g: df[df["age_group"] == g]["PSD_Acc_8_12Hz"] for g in AGE_LABELS}
    )

    # Panel B: Correlation with peak frequency
    ax = axes[1]
    ax.grid(True)
    for grp, color in PALETTE.items():
        sub = df[df["age_group"] == grp]
        ax.scatter(
            sub["Peak_Freq_Acc_Hz"],
            sub["PSD_Acc_8_12Hz"],
            color=color,
            s=60,
            alpha=0.85,
            edgecolors="white",
            lw=0.5,
            zorder=4,
        )
    spearman_annot(ax, df["Peak_Freq_Acc_Hz"], df["PSD_Acc_8_12Hz"])
    ax.set_yscale("log")
    ax.set_xlabel("Peak Frequency — Tremor (Hz)")
    ax.set_ylabel("PSD Acc (g²/Hz)")
    ax.set_title("B  —  Tremor power vs Peak Tremor Frequency")

    # Shared legend below
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig("fig_psd_acc.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig2c_psd_acc.png ✓")


# ═══════════════════════════════════════════════════════
# FIG 3  —  TREMOR FINGERPRINT
#   Panel A: scatter Peak_Freq_Acc × age
#            — reference bands from Elble (2003) and McAuley & Marsden (2000)
#   Panel B: bubble plot Peak_Freq_Acc (x) × FWHM (y) × TFR (bubble)
#            — the "tremor type space" figure
#   Panel C: boxplot TFR × age group
#            — tremor burden across the lifespan
# ═══════════════════════════════════════════════════════


def fig3_tremor():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Hand Tremor Characterisation",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    handles = [
        mpatches.Patch(color=PALETTE[g], label=l)
        for g, l in zip(AGE_LABELS, AGE_LABELS_LEGEND)
    ]

    # ── A: frequency vs age ──
    ax = axes[0]
    ax.grid(True)
    for grp, color in PALETTE.items():
        sub = df[df["age_group"] == grp]
        ax.scatter(
            sub["age"],
            sub["Peak_Freq_Acc_Hz"],
            color=color,
            s=60,
            alpha=0.85,
            edgecolors="white",
            lw=0.5,
            zorder=4,
        )
    spearman_annot(ax, df["age"], df["Peak_Freq_Acc_Hz"])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Dominant tremor frequency — Acc (Hz)")
    ax.set_title("A  —  Tremor frequency vs Age")
    ax.set_ylim(3, 13)

    # ── B: tremor fingerprint bubble ──
    ax = axes[1]
    ax.grid(True)

    for grp, color in PALETTE.items():
        sub = df[df["age_group"] == grp]
        ax.scatter(
            sub["Peak_Freq_Acc_Hz"],
            sub["FWHM"],
            s=60,
            color=color,
            alpha=0.80,
            edgecolors="white",
            lw=0.6,
            zorder=4,
        )

    ax.set_xlabel("Dominant tremor frequency — Acc (Hz)")
    ax.set_ylabel("FWHM — Spectral bandwidth (Hz)")
    ax.set_title("B  —  Tremor type space\nFrequency × Regularity × Age group")
    ax.set_xlim(3.5, 13)

    # ── C: TFR boxplot × age group ──
    ax = axes[2]
    data_tfr = [df[df["age_group"] == g]["TFR"].dropna() for g in AGE_LABELS]
    styled_boxplot(ax, data_tfr, AGE_LABELS, COLORS)
    ax.set_ylabel("TFR")
    ax.set_title("C  —  Tremor-Force Ratio by Age Group")
    kruskal_annot(ax, {g: df[df["age_group"] == g]["TFR"] for g in AGE_LABELS})

    # Shared legend
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig("fig_tremor.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig3_tremor.png ✓")


# ═══════════════════════════════════════════════════════
# FIG 4  —  SPEARMAN CORRELATION MATRIX
#   Sig mask: non-significant cells (p ≥ 0.05) are shown
#   in light grey text so the reader's eye goes immediately
#   to the meaningful associations.
# ═══════════════════════════════════════════════════════


def fig4_correlations():
    cols = [
        "age",
        "fmax",
        "z_Score",
        "ttp",
        "RFD",
        "cv",
        "TFR",
        "psd_force",
        "Peak_Freq_Force_Hz",
        "PSD_Acc_8_12Hz",
        "Peak_Freq_Acc_Hz",
        "FWHM",
    ]
    labels = [
        "Age",
        "Fmax",
        "z-Score",
        "TTP",
        "RFD",
        "CV",
        "TFR",
        "PSD Force",
        "Peak Freq Force",
        "PSD Acc",
        "Peak Freq Acc",
        "FWHM",
    ]

    sub = df[cols].copy()
    n = len(cols)

    r_mat = np.zeros((n, n))
    p_mat = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            mask = sub.iloc[:, i].notna() & sub.iloc[:, j].notna()
            if mask.sum() >= 5:
                r, p = stats.spearmanr(sub.iloc[mask, i], sub.iloc[mask, j])
                r_mat[i, j] = r
                p_mat[i, j] = p

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(
        "Spearman Correlation Matrix\n" "(grey text = p ≥ 0.05, not significant)",
        fontsize=13,
        fontweight="bold",
    )

    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            r = r_mat[i, j]
            sig = p_mat[i, j] < 0.05
            txt_color = ("#111111" if abs(r) < 0.5 else "white") if sig else "#CCCCCC"
            fw = "bold" if sig else "normal"
            ax.text(
                j,
                i,
                f"{r:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
                fontweight=fw,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, rotation=90)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Spearman r", fontsize=9)

    plt.tight_layout()
    plt.savefig("fig_correlacoes.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig_correlacoes.png ✓")


# ─────────────────────────────────────────────
# STATISTICS PRINT — used to write the results text
# ─────────────────────────────────────────────


def print_stats():
    print("\n" + "=" * 65)
    print("  KEY STATISTICS")
    print("=" * 65)

    # fmax by sex
    adults = df[df["age"] >= 18]
    m = adults[adults["sex"] == "M"]["fmax"].dropna()
    f = adults[adults["sex"] == "F"]["fmax"].dropna()
    nm_a = (stats.shapiro(m).pvalue > 0.05) if len(m) <= 50 else True
    nm_b = (stats.shapiro(f).pvalue > 0.05) if len(f) <= 50 else True
    if nm_a and nm_b:
        stat, p = stats.ttest_ind(m, f, equal_var=False)
        test = "Welch t"
    else:
        stat, p = stats.mannwhitneyu(m, f, alternative="two-sided")
        test = "Mann–Whitney U"
    print(f"\nfmax M vs F ({test}): stat={stat:.3f}, p={p:.4f}")
    print(f"  Male median: {m.median():.2f} kg  |  Female median: {f.median():.2f} kg")

    # Weakness
    def wflag(r):
        if r["age"] < 18:
            return np.nan
        return int(r["fmax"] < (WEAK_MEN if r["sex"] == "M" else WEAK_WOMEN))

    df["wk"] = df.apply(wflag, axis=1)
    a18 = df[df["age"] >= 18]
    n_wk = int(a18["wk"].sum())
    n_tot = int(a18["wk"].notna().sum())
    print(f"\nWeakness screening (adults ≥18): {n_wk}/{n_tot} ({100*n_wk/n_tot:.0f}%)")
    eld = a18[a18["age"] >= 65]
    if len(eld) > 0:
        n_wk_e = int(eld["wk"].sum())
        print(f"  Elderly subset: {n_wk_e}/{len(eld)} ({100*n_wk_e/len(eld):.0f}%)")

    # Kruskal-Wallis for main variables
    print("\nKruskal–Wallis across age groups:")
    for col, lbl in [
        ("fmax", "fmax"),
        ("cv", "CV"),
        ("TFR", "TFR"),
        ("Peak_Freq_Acc_Hz", "Freq. tremor"),
        ("FWHM", "FWHM"),
        ("psd_force", "PSD force"),
    ]:
        groups = [
            df[df["age_group"] == g][col].dropna()
            for g in AGE_LABELS
            if len(df[df["age_group"] == g][col].dropna()) >= 3
        ]
        if len(groups) < 2:
            continue
        h, p = stats.kruskal(*groups)
        print(f"  {lbl:<18} H={h:.2f}, p={p:.4f} {'*' if p<0.05 else ''}")

    # Spearman correlations
    print("\nSpearman correlations:")
    pairs = [
        ("age", "fmax", "age vs Fmax"),
        ("age", "Peak_Freq_Acc_Hz", "age vs tremor freq"),
        ("age", "FWHM", "age vs FWHM"),
        ("fmax", "cv", "Fmax vs CV"),
        ("fmax", "TFR", "Fmax vs TFR"),
        ("cv", "TFR", "CV vs TFR"),
        ("psd_force", "PSD_Acc_8_12Hz", "PSD force vs PSD acc"),
    ]
    for x, y, lbl in pairs:
        mask = df[x].notna() & df[y].notna()
        r, p = stats.spearmanr(df.loc[mask, x], df.loc[mask, y])
        print(f"  {lbl:<28} r={r:+.3f}, p={p:.4f} {'*' if p<0.05 else ''}")

    # Tremor outliers
    q75, q25 = df["TFR"].quantile(0.75), df["TFR"].quantile(0.25)
    iqr = q75 - q25
    thresh = q75 + 1.5 * iqr
    outliers = df[df["TFR"] > thresh]
    print(f"\nTFR outliers (> {thresh:.3f}):")
    if len(outliers):
        print(
            outliers[
                ["age", "sex", "fmax", "TFR", "Peak_Freq_Acc_Hz", "FWHM", "cv"]
            ].to_string(index=False)
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  GRIP STRENGTH & TREMOR — FINAL ANALYSIS")
    print("=" * 65)

    fig1_strength()
    fig2_variability()
    fig2b_psd_force()
    fig2c_psd_acc()
    fig3_tremor()
    fig4_correlations()
    print_stats()

    print("\n✓  Done. 6 figures saved.")
