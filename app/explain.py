"""SHAP explanation for a single input row."""
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
from matplotlib.lines import Line2D

# Diverging pair: red = pushes risk up, blue = pulls risk down. Values per
# mode from the project's palette (light chart chrome / dark chart chrome).
_COLORS = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "up": "#e34948",
        "down": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "up": "#e66767",
        "down": "#3987e5",
    },
}


def _theme():
    return "dark" if st.get_option("theme.base") == "dark" else "light"


@st.cache_resource
def get_explainer(_model):
    """Cached TreeExplainer. Prefixing the arg with `_` tells st.cache_resource
    not to hash the (unhashable) model object itself."""
    return shap.TreeExplainer(_model)


def _positive_class_row(model, input_row):
    """SHAP explanation for the single input row, positive (diabetes) class."""
    explainer = get_explainer(model)
    explanation = explainer(input_row)
    if len(explanation.shape) == 3:
        explanation = explanation[:, :, 1]
    return explanation[0]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def waterfall_figure(model, input_row, max_display=8):
    """Returns a matplotlib Figure: a horizontal diverging bar chart of SHAP
    contributions for one row (log-odds units — same scale XGBoost's
    TreeExplainer returns by default), each bar measured from zero rather
    than stacked, so bar length always reads as that feature's own effect.
    """
    row = _positive_class_row(model, input_row)
    values = row.values
    base_value = float(row.base_values)
    feature_values = row.data
    feature_names = list(input_row.columns)
    final_value = base_value + values.sum()

    order = np.argsort(-np.abs(values))
    top, rest = order[:max_display], order[max_display:]

    rows = [(feature_names[i], feature_values[i], values[i]) for i in top]
    if len(rest):
        rows.append((f"{len(rest)} other features", None, values[rest].sum()))

    # Largest impact at the top.
    rows = list(reversed(rows))

    c = _COLORS[_theme()]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.5))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    y_pos = np.arange(len(rows))
    max_abs = max(abs(sval) for _, _, sval in rows) or 1.0
    for y, (_name, _fval, sval) in zip(y_pos, rows):
        color = c["up"] if sval >= 0 else c["down"]
        ax.plot([0, sval], [y, y], color=color, linewidth=12,
                 solid_capstyle="round", zorder=3)
        offset = 0.03 * max_abs
        ax.text(sval + (offset if sval >= 0 else -offset), y, f"{sval:+.2f}",
                 va="center", ha="left" if sval >= 0 else "right",
                 fontsize=9, color=c["text"])

    ytick_labels = [
        name if fval is None else f"{name} = {fval:g}"
        for name, fval, _ in rows
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ytick_labels, color=c["text"], fontsize=9)
    ax.tick_params(axis="y", length=0)

    ax.axvline(0, color=c["baseline"], linewidth=1, zorder=1)

    ax.set_title(
        f"Baseline risk {_sigmoid(base_value):.0%}  →  This patient {_sigmoid(final_value):.0%}",
        loc="left", fontsize=10.5, color=c["text"], fontweight="bold", pad=12,
    )
    ax.set_xlabel("Contribution to risk score (log-odds)", color=c["muted"], fontsize=9)
    ax.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(axis="x", colors=c["muted"], labelsize=8.5)
    ax.margins(x=0.2, y=0.1)

    legend_handles = [
        Line2D([0], [0], color=c["up"], linewidth=6, solid_capstyle="round"),
        Line2D([0], [0], color=c["down"], linewidth=6, solid_capstyle="round"),
    ]
    ax.legend(legend_handles, ["Increases risk", "Decreases risk"],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=c["text"])

    fig.tight_layout()
    return fig


def summarize_top_drivers(model, input_row, top_n=3):
    """Plain-English sentence naming the top features pushing risk up/down."""
    row = _positive_class_row(model, input_row)
    pairs = sorted(zip(input_row.columns, row.values), key=lambda p: abs(p[1]), reverse=True)[:top_n]

    increasing = [n for n, v in pairs if v > 0]
    decreasing = [n for n, v in pairs if v < 0]

    parts = []
    if increasing:
        parts.append(f"raised primarily by {', '.join(increasing)}")
    if decreasing:
        parts.append(f"lowered by {', '.join(decreasing)}")
    return " and ".join(parts) if parts else "close to the model's baseline risk"
