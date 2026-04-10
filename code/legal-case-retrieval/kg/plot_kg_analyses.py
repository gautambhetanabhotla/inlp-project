#!/usr/bin/env python3
"""
Generate visualisations for kg_analyses_results.json.
Saves 8 PNG files to Eval/plots/.

Usage:
  python Eval/plot_kg_analyses.py
"""
from __future__ import annotations
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = Path(__file__).parent / "kg_analyses_results.json"
OUT  = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

with open(DATA) as f:
    d = json.load(f)

def short(label: str) -> str:
    """Strip 'in The Indian Penal Code' suffix for axis labels."""
    return label.replace(" in The Indian Penal Code", "")

# ── helpers ──────────────────────────────────────────────────────────────────
def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. Graph Summary — annotated bar of node/edge counts (log scale)
# ═════════════════════════════════════════════════════════════════════════════
g = d["1_graph_summary"][0]
fig, ax = plt.subplots(figsize=(7, 4))
labels  = ["Case nodes", "Label nodes", "HAS_LABEL\nedges", "SIMILAR_LABELS\nedges"]
values  = [g["cases"], g["labels"], g["has_label_edges"], g["similar_labels_edges"]]
colors  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
bars = ax.bar(labels, values, color=colors, width=0.5)
ax.set_yscale("log")
ax.set_ylabel("Count (log scale)")
ax.set_title("1. KG Graph Summary")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.15,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(top=max(values)*20)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "01_graph_summary.png")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Top-20 Labels by Case Frequency — horizontal bar
# ═════════════════════════════════════════════════════════════════════════════
rows  = d["2_top20_labels_by_freq"]
names = [short(r["label"]) for r in rows][::-1]
counts = [r["case_count"] for r in rows][::-1]
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(names, counts, color="#4C72B0")
ax.set_xlabel("Number of cases")
ax.set_title("2. Top-20 IPC Labels by Case Frequency")
for bar, v in zip(bars, counts):
    ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontsize=8)
ax.set_xlim(right=max(counts)*1.12)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "02_top20_labels_freq.png")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Label Count Distribution per Case — bar
# ═════════════════════════════════════════════════════════════════════════════
rows = d["3_label_count_distribution"]
xs   = [r["num_labels"] for r in rows]
ys   = [r["num_cases"]  for r in rows]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([str(x) for x in xs], ys, color="#55A868")
ax.set_xlabel("Number of IPC labels per case")
ax.set_ylabel("Number of cases")
ax.set_title("3. Label Count Distribution")
for bar, v in zip(bars, ys):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{v:,}", ha="center", fontsize=9)
ax.set_ylim(top=max(ys)*1.12)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "03_label_count_distribution.png")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Top-20 Hub Cases by SIMILAR_LABELS degree — horizontal bar
# ═════════════════════════════════════════════════════════════════════════════
rows    = d["4_top20_hub_cases"]
case_ids = [r["case_id"].replace(".txt","") for r in rows][::-1]
degrees  = [r["degree"] for r in rows][::-1]
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(case_ids, degrees, color="#C44E52")
ax.set_xlabel("SIMILAR_LABELS degree")
ax.set_title("4. Top-20 Hub Cases by Degree")
for bar, v in zip(bars, degrees):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontsize=8)
ax.set_xlim(right=max(degrees)*1.08)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "04_top20_hub_cases.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. Top-20 Label Co-occurrence Pairs — horizontal bar
# ═════════════════════════════════════════════════════════════════════════════
rows  = d["5_top20_label_cooccurrence"]
pair_labels = [f"{short(r['label_a'])} +\n{short(r['label_b'])}" for r in rows][::-1]
co_counts   = [r["co_cases"] for r in rows][::-1]
fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(pair_labels, co_counts, color="#DD8452")
ax.set_xlabel("Cases sharing both labels")
ax.set_title("5. Top-20 Label Co-occurrence Pairs")
for bar, v in zip(bars, co_counts):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontsize=7.5)
ax.set_xlim(right=max(co_counts)*1.12)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "05_top20_label_cooccurrence.png")

# ═════════════════════════════════════════════════════════════════════════════
# 6. Shared-count Histogram — bar
# ═════════════════════════════════════════════════════════════════════════════
rows   = d["6_shared_count_histogram"]
bkts   = [r["bucket"] for r in rows]
ecnts  = [r["edge_count"] for r in rows]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(bkts, ecnts, color="#8172B2")
ax.set_xlabel("Shared label count (edge weight)")
ax.set_ylabel("Number of SIMILAR_LABELS edges")
ax.set_title("6. SIMILAR_LABELS Edge Weight Distribution")
for bar, v in zip(bars, ecnts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f"{v:,}", ha="center", fontsize=9)
ax.set_ylim(top=max(ecnts)*1.12)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "06_edge_weight_histogram.png")

# ═════════════════════════════════════════════════════════════════════════════
# 7+8. Edge weight stats + zero-label cases — summary table as figure
# ═════════════════════════════════════════════════════════════════════════════
stats = d["8_edge_weight_stats"][0]
zero  = d["7_cases_with_zero_labels"][0]["zero_label_cases"]
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis("off")
table_data = [
    ["Metric", "Value"],
    ["Total Case nodes",           f"{g['cases']:,}"],
    ["Total Label nodes",          f"{g['labels']:,}"],
    ["HAS_LABEL edges",            f"{g['has_label_edges']:,}"],
    ["SIMILAR_LABELS edges",       f"{g['similar_labels_edges']:,}"],
    ["Cases with zero labels",     str(zero)],
    ["Avg shared_count (edge wt)", str(stats["avg_shared"])],
    ["Max shared_count",           str(stats["max_shared"])],
    ["Min shared_count",           str(stats["min_shared"])],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center",
               colWidths=[0.6, 0.35])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f0f4fa")
ax.set_title("7+8. KG Statistics Summary", pad=12, fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "07_08_stats_summary.png")

# ═════════════════════════════════════════════════════════════════════════════
# 9. Top strongest case pairs — bubble chart (shared count as size)
# ═════════════════════════════════════════════════════════════════════════════
rows   = d["9_top20_strongest_pairs"]
shared = [r["shared"] for r in rows]
pair_n = [f"{r['case_a'].replace('.txt','')} –\n{r['case_b'].replace('.txt','')}" for r in rows]
fig, ax = plt.subplots(figsize=(5, 9))
y_pos = list(range(len(rows)))
scatter = ax.scatter([0]*len(rows), y_pos,
                     s=[v*300 for v in shared],
                     c=shared, cmap="Reds", alpha=0.85,
                     edgecolors="grey", linewidths=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(pair_n, fontsize=7)
ax.set_xticks([])
ax.set_title("9. Top-20 Strongest Case Pairs\n(bubble size = shared label count)")
cb = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02)
cb.set_label("Shared labels")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
save(fig, "09_top20_strongest_pairs.png")

# ═════════════════════════════════════════════════════════════════════════════
# 10. Strong-neighbour degree CDF — how many cases have >= N strong neighbours
# ═════════════════════════════════════════════════════════════════════════════
rows = d["10_dense_cluster_sizes"]
if rows:
    # rows: each unique degree (110–129), num_cases=1 each.
    # Build cumulative: for threshold t, how many cases have degree >= t
    all_degs = sorted([r["min_3_shared_degree"] for r in rows], reverse=True)
    thresholds = list(range(50, max(all_degs)+2, 10))
    cum_counts = [sum(1 for deg in all_degs if deg >= t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(t) for t in thresholds], cum_counts, color="#64B5CD", width=0.6)
    ax.set_xlabel("Min strong-neighbour degree threshold (shared_count ≥ 3)")
    ax.set_ylabel("Number of cases above threshold")
    ax.set_title("10. Cases with ≥N Strong Neighbours (shared_count ≥ 3)")
    for bar, v in zip(ax.patches, cum_counts):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    str(v), ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "10_dense_clusters.png")
else:
    print("  (no data for plot 10)")

print(f"\nAll plots saved to {OUT}/")
