from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("docs/figures")
PNG_OUT = OUT_DIR / "rna_stability_elements_workflow.png"
SVG_OUT = OUT_DIR / "rna_stability_elements_workflow.svg"

COLORS = {
    "ink": "#0B1220",
    "muted": "#667085",
    "line": "#1F2937",
    "blue": "#3F73C5",
    "blue_light": "#9FB9E8",
    "orange": "#D85B0D",
    "orange_light": "#F6C7A5",
    "green": "#A7BF76",
    "green_dark": "#4C7A33",
    "gray": "#BFBFBF",
    "gray_light": "#EFEFEF",
    "white": "#FFFFFF",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 12.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_frame(ax)
    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)

    fig.savefig(PNG_OUT, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_OUT)
    print(SVG_OUT)


def draw_frame(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((0.01, 0.015), 0.98, 0.97, fill=False, ec="black", lw=1.25))
    ax.plot([0.01, 0.99], [0.535, 0.535], color="black", lw=1.25)
    ax.plot([0.01, 0.99], [0.315, 0.315], color="black", lw=1.25)
    label(ax, 0.018, 0.955, "A", size=22, weight="bold")
    label(ax, 0.018, 0.515, "B", size=22, weight="bold")
    label(ax, 0.018, 0.292, "C", size=22, weight="bold")


def draw_panel_a(ax: plt.Axes) -> None:
    label(ax, 0.06, 0.845, "Target\nconstruction", size=20, color=COLORS["blue"], weight="bold", ha="left")

    rounded_box(
        ax,
        0.055,
        0.925,
        0.49,
        0.045,
        "ENCODE BrU-seq / BruChase-seq   |   16 cell lines   |   0h, 2h, 6h",
        fc=COLORS["blue_light"],
        ec=COLORS["blue_light"],
        size=13,
        weight="bold",
    )

    rounded_box(ax, 0.19, 0.845, 0.21, 0.055, "Metadata discovery\n48 experiments", fc="white")
    arrow(ax, (0.295, 0.925), (0.295, 0.902), color=COLORS["blue"])
    arrow(ax, (0.295, 0.845), (0.295, 0.805), color=COLORS["blue"])

    rounded_box(ax, 0.15, 0.745, 0.29, 0.065, "Download quantification TSV\n96 gene quant + 96 genic feature files", fc=COLORS["gray_light"])
    arrow(ax, (0.295, 0.745), (0.295, 0.707), color=COLORS["blue"])

    draw_timecourse(ax, 0.08, 0.635, 0.42, 0.075)
    label(ax, 0.505, 0.672, "log2 stability scores", size=13, weight="bold")
    rounded_box(
        ax,
        0.48,
        0.59,
        0.28,
        0.075,
        "2h/0h, 6h/2h, 6h/0h\npseudo = 0.1; min 0h signal = 0.5",
        fc="white",
        size=10.5,
    )
    arrow(ax, (0.435, 0.67), (0.48, 0.63), color=COLORS["blue"])

    rounded_box(
        ax,
        0.60,
        0.84,
        0.28,
        0.12,
        "Quality gates\nreplicate Pearson median = 0.972\nexon vs gene consensus Pearson = 0.868",
        fc="#F7F9FC",
        ec=COLORS["blue"],
        size=11,
        weight="bold",
    )

    rounded_box(
        ax,
        0.79,
        0.61,
        0.16,
        0.105,
        "gene_sense\ntarget rows\n150,233",
        fc=COLORS["gray"],
        size=12,
        weight="bold",
    )
    arrow(ax, (0.76, 0.628), (0.79, 0.655), color=COLORS["blue"])
    arrow(ax, (0.87, 0.61), (0.87, 0.565), color=COLORS["blue"])

    rounded_box(
        ax,
        0.725,
        0.545,
        0.25,
        0.055,
        "Consensus target: median across cell lines\n10,907 genes",
        fc=COLORS["orange_light"],
        ec=COLORS["orange"],
        size=11,
        weight="bold",
    )

    draw_transcript(ax, 0.19, 0.555, 0.36, 0.035)
    label(ax, 0.205, 0.548, "GENCODE v29 transcript", size=10.5, weight="bold")
    arrow(ax, (0.55, 0.575), (0.725, 0.575), color=COLORS["blue"])


def draw_panel_b(ax: plt.Axes) -> None:
    label(
        ax,
        0.055,
        0.505,
        "Sequence feature and baseline prediction",
        size=14.5,
        color=COLORS["orange"],
        weight="bold",
    )

    draw_transcript(ax, 0.095, 0.412, 0.25, 0.03)
    label(ax, 0.105, 0.475, "full transcript", size=11, weight="bold", color=COLORS["blue"])
    label(ax, 0.12, 0.37, "5'UTR", size=10, color=COLORS["muted"], weight="bold")
    label(ax, 0.205, 0.37, "CDS", size=10, color=COLORS["blue"], weight="bold")
    label(ax, 0.305, 0.37, "3'UTR", size=10, color=COLORS["orange"], weight="bold")

    arrow(ax, (0.365, 0.428), (0.43, 0.428))
    rounded_box(
        ax,
        0.43,
        0.39,
        0.16,
        0.08,
        "Feature builder\nlength / GC / AU\n3-mer + 4-mer\nconfigured motifs",
        fc=COLORS["gray_light"],
        size=9.5,
        weight="bold",
    )
    arrow(ax, (0.59, 0.428), (0.65, 0.428))
    rounded_box(
        ax,
        0.65,
        0.395,
        0.14,
        0.065,
        "Feature matrix\n10,907 x 1,346",
        fc=COLORS["gray"],
        size=10,
        weight="bold",
    )

    arrow(ax, (0.79, 0.428), (0.825, 0.428))
    rounded_box(
        ax,
        0.825,
        0.392,
        0.145,
        0.075,
        "Baseline models\nRidge | ElasticNet\nRandomForest",
        fc=COLORS["orange"],
        ec=COLORS["line"],
        size=9.5,
        weight="bold",
    )
    arrow(ax, (0.895, 0.392), (0.895, 0.375), color=COLORS["blue"])

    rounded_box(
        ax,
        0.80,
        0.326,
        0.19,
        0.052,
        "Stability prediction\nRF Pearson = 0.480; R2 = 0.220",
        fc=COLORS["blue_light"],
        ec=COLORS["blue"],
        size=10,
        weight="bold",
    )


def draw_panel_c(ax: plt.Axes) -> None:
    label(ax, 0.055, 0.29, "Interpretation and downstream tasks", size=16, color=COLORS["green"], weight="bold")

    draw_rna_scene(ax, 0.12, 0.205, 0.36, 0.055)
    rounded_box(
        ax,
        0.08,
        0.085,
        0.21,
        0.095,
        "Strict validation\nrepeated random split\nchromosome holdout\nfeature ablation",
        fc="white",
        ec=COLORS["green"],
        size=10.5,
        weight="bold",
    )
    rounded_box(
        ax,
        0.335,
        0.085,
        0.22,
        0.095,
        "Interpretability\nmotif enrichment\nregion importance\nin silico mutagenesis",
        fc="white",
        ec=COLORS["green"],
        size=10.5,
        weight="bold",
    )
    arrow(ax, (0.25, 0.2), (0.19, 0.18), color=COLORS["green_dark"])
    arrow(ax, (0.37, 0.2), (0.445, 0.18), color=COLORS["green_dark"])

    rounded_box(
        ax,
        0.62,
        0.19,
        0.16,
        0.065,
        "Sequence\nencoder",
        fc=COLORS["blue_light"],
        ec=COLORS["blue"],
        size=12,
        weight="bold",
    )
    rounded_box(
        ax,
        0.62,
        0.095,
        0.16,
        0.065,
        "Context\nencoder",
        fc=COLORS["green"],
        ec=COLORS["green_dark"],
        size=12,
        weight="bold",
    )
    label(ax, 0.81, 0.22, "RBP / miRNA\nexpression", size=10.5, color=COLORS["green_dark"], weight="bold")
    label(ax, 0.81, 0.125, "eCLIP / binding\ncell-line context", size=10.5, color=COLORS["green_dark"], weight="bold")
    arrow(ax, (0.78, 0.222), (0.86, 0.18), color=COLORS["green_dark"])
    arrow(ax, (0.78, 0.128), (0.86, 0.18), color=COLORS["green_dark"])
    rounded_box(
        ax,
        0.84,
        0.145,
        0.13,
        0.075,
        "sequence x context\ninteraction head",
        fc=COLORS["orange_light"],
        ec=COLORS["orange"],
        size=10.5,
        weight="bold",
    )
    arrow(ax, (0.56, 0.13), (0.62, 0.13), color=COLORS["green_dark"])
    arrow(ax, (0.55, 0.132), (0.62, 0.222), color=COLORS["green_dark"])

    label(
        ax,
        0.31,
        0.045,
        "candidate stability motifs    |    cell-line-specific RNA stability elements    |    RNA vaccine UTR/module design",
        size=13,
        color="#B00000",
        weight="bold",
        style="italic",
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str | None = None,
    size: float = 10,
    color: str = COLORS["ink"],
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.006",
        linewidth=1.1,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    label(ax, x + w / 2, y + h / 2, text, size=size, color=color, weight=weight, ha="center", va="center")


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 10,
    color: str = COLORS["ink"],
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    style: str = "normal",
) -> None:
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "black",
    lw: float = 1.7,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_timecourse(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    xs = [x + 0.08 * w, x + 0.50 * w, x + 0.92 * w]
    labels = ["0h\nBrU-seq", "2h\nBruChase", "6h\nBruChase"]
    values = ["baseline", "chase", "chase"]
    ax.plot([xs[0], xs[-1]], [y + 0.5 * h, y + 0.5 * h], color=COLORS["line"], lw=1.5)
    for xpos, lab, val in zip(xs, labels, values):
        ax.add_patch(Circle((xpos, y + 0.5 * h), 0.014, fc=COLORS["blue"], ec="white", lw=1))
        label(ax, xpos, y + h * 1.02, lab, size=10, weight="bold", ha="center")
        label(ax, xpos, y + h * 0.02, val, size=9, color=COLORS["muted"], ha="center")
    arrow(ax, (xs[0] + 0.025, y + 0.5 * h), (xs[1] - 0.025, y + 0.5 * h), color=COLORS["blue"])
    arrow(ax, (xs[1] + 0.025, y + 0.5 * h), (xs[2] - 0.025, y + 0.5 * h), color=COLORS["blue"])


def draw_transcript(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.plot([x, x + w], [y + h / 2, y + h / 2], color="#6B7280", lw=3)
    parts = [
        (0.09, 0.23, COLORS["gray"]),
        (0.23, 0.70, COLORS["blue"]),
        (0.70, 0.90, COLORS["orange"]),
    ]
    for start, end, color in parts:
        ax.add_patch(Rectangle((x + w * start, y), w * (end - start), h, fc=color, ec=color))
    ax.add_patch(Circle((x + 0.04 * w, y + h / 2), h * 0.37, fc=COLORS["gray"], ec=COLORS["gray"]))
    label(ax, x + w * 0.92, y + h / 2, "AAAA", size=10, weight="bold")


def draw_rna_scene(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    points = []
    for i in range(120):
        t = i / 119
        points.append((x + w * t, y + h * (0.5 + 0.34 * math.sin(t * 6.5 * math.pi))))
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color="#A8B0BA",
        lw=5,
        solid_capstyle="round",
    )
    for t in [0.27, 0.45, 0.68]:
        xpos = x + w * t
        ypos = y + h * (0.5 + 0.34 * math.sin(t * 6.5 * math.pi))
        ax.add_patch(Circle((xpos, ypos), 0.018, fc="#0086C9", ec="#2F80ED", lw=1.2))
        ax.add_patch(Circle((xpos + 0.02, ypos + 0.012), 0.006, fc=COLORS["orange"], ec=COLORS["orange"]))
    for t in [0.31, 0.72, 0.75, 0.78]:
        xpos = x + w * t
        ypos = y + h * (0.5 + 0.34 * math.sin(t * 6.5 * math.pi))
        ax.add_patch(Circle((xpos, ypos), 0.006, fc=COLORS["orange"], ec=COLORS["orange"]))
    label(ax, x - 0.035, y + h * 0.45, "m7G", size=10, weight="bold")
    label(ax, x + w + 0.008, y + h * 0.45, "(A)n", size=10, weight="bold")
    label(ax, x + 0.18 * w, y + h * 1.14, "ribosome", size=9, weight="bold")


if __name__ == "__main__":
    main()
