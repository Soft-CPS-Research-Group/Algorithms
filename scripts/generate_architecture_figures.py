"""Generate architecture diagrams for the thesis.

Produces four matplotlib-based figures:

  fig_arch_entity_interface.png   – entity payload → adapter → per-agent vectors
  fig_arch_pipeline.png           – offline RL pipeline (collect → train → eval)
  fig_arch_iql_network.png        – IQL actor-critic-value network structure
  fig_arch_cql_network.png        – CQL actor-critic-value + conservative penalty

Outputs go to --output-dir (default: thesis/ch4/assets).

Usage:
    python scripts/generate_architecture_figures.py \
        --output-dir /path/to/thesis/ch4/assets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C = {
    "blue":    "#1565C0",
    "lblue":   "#90CAF9",
    "green":   "#2E7D32",
    "lgreen":  "#A5D6A7",
    "orange":  "#E65100",
    "lorange": "#FFCC80",
    "purple":  "#6A1B9A",
    "lpurple": "#CE93D8",
    "red":     "#C62828",
    "lred":    "#EF9A9A",
    "grey":    "#546E7A",
    "lgrey":   "#CFD8DC",
    "bg":      "#FAFAFA",
}


def _add_box(ax, x, y, w, h, text, fc, ec, fontsize=9, bold=False):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.5, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, zorder=4, wrap=True,
            multialignment="center")


def _arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.5, arrowstyle="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))


# ---------------------------------------------------------------------------
# Figure 1 — Entity Interface
# ---------------------------------------------------------------------------

def fig_entity_interface(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("Entity Interface Architecture", fontsize=13, fontweight="bold", pad=10)

    # Simulator box
    _add_box(ax, 1.2, 2.5, 2.0, 3.5,
             "Simulator\n(CityLearn)\n\nentity payload:\n• tables\n• edges\n• meta",
             C["lgrey"], C["grey"], fontsize=8)

    _arrow(ax, 2.2, 2.5, 3.2, 2.5, C["grey"])

    # Entity Adapter
    _add_box(ax, 4.0, 2.5, 1.4, 1.6,
             "Entity\nAdapter",
             C["lblue"], C["blue"], bold=True, fontsize=9)
    ax.text(3.65, 3.55, "topology\n→ layout\nper agent group", fontsize=7,
            ha="center", va="center", color=C["blue"], style="italic")

    _arrow(ax, 4.7, 2.5, 5.5, 2.5, C["grey"])

    # Per-agent vectors
    _add_box(ax, 6.5, 2.5, 1.7, 3.0,
             "Per-agent\nobservation\nvectors\n\nGroup A: 627-d\nGroup B: 706-d\nGroup C: 749-d\nGroup D: 785-d",
             C["lgreen"], C["green"], fontsize=8)

    _arrow(ax, 7.35, 2.5, 8.2, 2.5, C["grey"])

    # Agent policies
    _add_box(ax, 9.2, 3.8, 1.8, 1.0, "Policy A\n(obs=627)", C["lorange"], C["orange"], fontsize=8)
    _add_box(ax, 9.2, 2.9, 1.8, 0.8, "Policy B\n(obs=706)", C["lorange"], C["orange"], fontsize=8)
    _add_box(ax, 9.2, 2.1, 1.8, 0.8, "Policy C\n(obs=749)", C["lorange"], C["orange"], fontsize=8)
    _add_box(ax, 9.2, 1.3, 1.8, 0.8, "Policy D\n(obs=785)", C["lorange"], C["orange"], fontsize=8)

    for y in [3.8, 2.9, 2.1, 1.3]:
        _arrow(ax, 8.2, 2.5, 8.7, y, C["orange"])
        _arrow(ax, 10.1, y, 10.8, y, C["grey"])

    # Action collection
    _add_box(ax, 11.5, 2.5, 1.4, 1.6,
             "Action\nCollection",
             C["lpurple"], C["purple"], bold=True, fontsize=9)

    _arrow(ax, 12.2, 2.5, 13.0, 2.5, C["grey"])

    # Back to simulator
    _add_box(ax, 13.5, 2.5, 0.8, 1.2,
             "env\nstep()",
             C["lgrey"], C["grey"], fontsize=8)

    # Labels
    ax.text(4.0, 0.45, "Converts entity payload\nto per-agent obs vectors", fontsize=7,
            ha="center", va="bottom", color=C["blue"])
    ax.text(11.5, 0.85, "Assembles actions\nback to entity tables", fontsize=7,
            ha="center", va="bottom", color=C["purple"])

    fig.tight_layout()
    out_path = output_dir / "fig_arch_entity_interface.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — Offline RL Pipeline
# ---------------------------------------------------------------------------

def fig_pipeline(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_title("Offline Reinforcement Learning Pipeline", fontsize=13, fontweight="bold", pad=10)

    stages = [
        (1.4, "Data\nCollection\n\nRBCSmart policy\nseeds 22–31\n(9 training seeds)",
         C["lgreen"], C["green"]),
        (4.0, "Offline\nDataset\n\nParquet files\nper agent group\n~87k steps/seed",
         C["lblue"], C["blue"]),
        (7.0, "Training\n\nIQL or CQL\n50k gradient steps\nper group × seed",
         C["lorange"], C["orange"]),
        (10.0, "Policy\nAssembly\n\nLoad best checkpoint\nper group\nmap to buildings",
         C["lpurple"], C["purple"]),
        (12.8, "Evaluation\n\n5 eval seeds\n(200–204)\n3-way benchmark",
         C["lred"], C["red"]),
    ]

    for x, text, fc, ec in stages:
        _add_box(ax, x, 2.0, 2.2, 3.2, text, fc, ec, fontsize=8, bold=False)

    xs = [s[0] for s in stages]
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + 1.1, 2.0, xs[i + 1] - 1.1, 2.0, C["grey"], lw=2)

    # Validation feedback arrow
    ax.annotate("", xy=(7.0, 0.15), xytext=(10.0, 0.15),
                arrowprops=dict(arrowstyle="<-", color=C["orange"],
                                lw=1.2, connectionstyle="arc3,rad=0"))
    ax.text(8.5, 0.3, "validation MSE during training (seed 31)", fontsize=7,
            ha="center", va="bottom", color=C["orange"], style="italic")

    fig.tight_layout()
    out_path = output_dir / "fig_arch_pipeline.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — IQL Network Architecture
# ---------------------------------------------------------------------------

def _network_diagram(ax, title, nodes_per_layer, layer_labels, color, extra_text=None):
    """Generic layered network diagram."""
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, len(nodes_per_layer) - 0.5)
    ax.set_ylim(-0.5, max(nodes_per_layer) + 0.5)
    ax.axis("off")

    positions = {}  # layer -> list of (x, y)
    max_n = max(nodes_per_layer)
    for l_idx, n in enumerate(nodes_per_layer):
        positions[l_idx] = []
        for n_idx in range(n):
            y = (max_n - 1) / 2 - (n - 1) / 2 + n_idx
            positions[l_idx].append((l_idx, y))

    # Draw edges
    for l_idx in range(len(nodes_per_layer) - 1):
        for x0, y0 in positions[l_idx]:
            for x1, y1 in positions[l_idx + 1]:
                ax.plot([x0, x1], [y0, y1], "-", color="#BDBDBD", lw=0.5, zorder=1)

    # Draw nodes
    for l_idx, pts in positions.items():
        for x, y in pts:
            circle = plt.Circle((x, y), 0.22, fc=color, ec="white", lw=1.5, zorder=2)
            ax.add_patch(circle)

    # Layer labels
    for l_idx, label in enumerate(layer_labels):
        ax.text(l_idx, -0.3, label, ha="center", va="top", fontsize=7, color="#333")

    if extra_text:
        ax.text(len(nodes_per_layer) / 2 - 0.5, max_n + 0.2,
                extra_text, ha="center", va="bottom", fontsize=7,
                color=C["grey"], style="italic")


def fig_iql_network(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("IQL Network Architecture", fontsize=13, fontweight="bold")

    # Three sub-networks: Actor, Q-function (×2), Value
    specs = [
        ("Actor  π(a|s)\n[Gaussian policy]",
         [4, 5, 5, 3], ["obs", "256", "256", "action\n(tanh)"], C["lorange"]),
        ("Q-function  Q(s,a)\n[×2, double critics]",
         [4, 5, 5, 2], ["obs+act", "256", "256", "Q-value"], C["lblue"]),
        ("Value  V(s)\n[expectile regression]",
         [4, 5, 5, 2], ["obs", "256", "256", "V-value"], C["lgreen"]),
    ]
    for i, (title, nodes, labels, color) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_facecolor(C["bg"])
        _network_diagram(ax, title, nodes, labels, color)

    # Note about node count being schematic
    fig.text(0.5, 0.01,
             "Node counts are schematic. Actual hidden layers: 256 units each.",
             ha="center", fontsize=8, color=C["grey"], style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = output_dir / "fig_arch_iql_network.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4 — CQL Network Architecture
# ---------------------------------------------------------------------------

def fig_cql_network(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(14, 5.5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("CQL Network Architecture", fontsize=13, fontweight="bold")

    specs = [
        ("Actor  π(a|s)\n[Gaussian policy]",
         [4, 5, 5, 3], ["obs", "256", "256", "action\n(tanh)"], C["lorange"]),
        ("Q-function  Q(s,a)\n[×2 + conservative penalty]",
         [4, 5, 5, 2], ["obs+act", "256", "256", "Q-value"], C["lred"]),
        ("Value  V(s)\n[expectile regression]",
         [4, 5, 5, 2], ["obs", "256", "256", "V-value"], C["lgreen"]),
    ]
    for i, (title, nodes, labels, color) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_facecolor(C["bg"])
        _network_diagram(ax, title, nodes, labels, color,
                         extra_text="+ CQL penalty: α·E[Q(s,ã)] − E[Q(s,a)]"
                         if i == 1 else None)

    # CQL penalty annotation
    fig.text(0.5, 0.01,
             "Conservative penalty pushes Q-values down for out-of-distribution actions ã.",
             ha="center", fontsize=8, color=C["red"], style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = output_dir / "fig_arch_cql_network.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("thesis/ch4/assets"),
                   help="Destination for generated PNG files.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[arch-figures] output = {args.output_dir}\n")

    fig_entity_interface(args.output_dir)
    fig_pipeline(args.output_dir)
    fig_iql_network(args.output_dir)
    fig_cql_network(args.output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
