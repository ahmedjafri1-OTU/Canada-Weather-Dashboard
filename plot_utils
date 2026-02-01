# plot_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _prep_df(df: pd.DataFrame, y_cols):
    if df is None or df.empty or "time" not in df.columns:
        return None
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"])
    for c in y_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _format_time_axis(fig, ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=20, ha="right")


def row_line_violin(df: pd.DataFrame, y: str, title: str, ylabel: str):
    d = _prep_df(df, [y])
    if d is None or y not in d.columns:
        return None
    d = d[["time", y]].dropna()
    if d.empty:
        return None

    fig = plt.figure(figsize=(12, 3.0), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.2], wspace=0.25)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])

    # left: line
    ax_ts.plot(d["time"], d[y], linewidth=2)
    ax_ts.set_title(title, fontsize=12)
    ax_ts.set_ylabel(ylabel)
    ax_ts.grid(True, alpha=0.25)
    _format_time_axis(fig, ax_ts)

    # right: violin
    ax_v.violinplot(d[y].values, showmeans=True, showmedians=True, vert=True)
    ax_v.set_title("Distribution", fontsize=11)
    ax_v.set_ylabel(ylabel)
    ax_v.set_xticks([1])
    ax_v.set_xticklabels([""])
    ax_v.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(pad=0.6)
    return fig


def row_bar_violin(df: pd.DataFrame, y: str, title: str, ylabel: str):
    d = _prep_df(df, [y])
    if d is None or y not in d.columns:
        return None
    d = d[["time", y]].dropna()
    if d.empty:
        return None

    fig = plt.figure(figsize=(12, 3.0), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.2], wspace=0.25)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])

    # left: bar
    ax_ts.bar(d["time"], d[y])
    ax_ts.set_title(title, fontsize=12)
    ax_ts.set_ylabel(ylabel)
    ax_ts.grid(True, axis="y", alpha=0.25)
    _format_time_axis(fig, ax_ts)

    # right: violin
    ax_v.violinplot(d[y].values, showmeans=True, showmedians=True, vert=True)
    ax_v.set_title("Distribution", fontsize=11)
    ax_v.set_ylabel(ylabel)
    ax_v.set_xticks([1])
    ax_v.set_xticklabels([""])
    ax_v.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(pad=0.6)
    return fig


def row_windrose_violin(
    df: pd.DataFrame,
    speed_col: str = "wind_speed_10m",
    dir_col: str = "wind_direction_10m",
    n_sectors: int = 16,
    violin_col: str = "wind_speed_10m",
    violin_label: str = "m/s",
):
    d = _prep_df(df, [speed_col, dir_col, violin_col])
    if d is None or speed_col not in d.columns or dir_col not in d.columns:
        return None

    d2 = d[[speed_col, dir_col]].dropna()
    if d2.empty:
        return None

    dirs_rad = np.deg2rad((d2[dir_col].values % 360))
    sector_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)
    counts, _ = np.histogram(dirs_rad, bins=sector_edges)

    width = (2 * np.pi) / n_sectors
    centers = sector_edges[:-1] + width / 2

    fig = plt.figure(figsize=(12, 3.2), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.2], wspace=0.25)

    ax_wr = fig.add_subplot(gs[0, 0], polar=True)
    ax_v = fig.add_subplot(gs[0, 1])

    # left: wind rose (frequency by direction)
    ax_wr.bar(centers, counts, width=width, align="center")
    ax_wr.set_theta_zero_location("N")
    ax_wr.set_theta_direction(-1)
    ax_wr.set_title("Wind Rose (direction frequency)", fontsize=12, pad=12)

    # right: violin of wind speed (or any chosen column)
    if violin_col in d.columns:
        vv = d[violin_col].dropna()
        if not vv.empty:
            ax_v.violinplot(vv.values, showmeans=True, showmedians=True, vert=True)
            ax_v.set_title("Distribution", fontsize=11)
            ax_v.set_ylabel(violin_label)
            ax_v.set_xticks([1])
            ax_v.set_xticklabels([""])
            ax_v.grid(True, axis="y", alpha=0.25)
        else:
            ax_v.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_v.set_axis_off()
    else:
        ax_v.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_v.set_axis_off()

    fig.tight_layout(pad=0.6)
    return fig
