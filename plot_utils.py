# plot_utils.py
# Python 3.8+
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Set, Dict, Tuple


# ============================================================
# CORE HELPERS
# ============================================================

def _prep_df(df: pd.DataFrame, y_cols):
    if df is None or df.empty or "time" not in df.columns:
        return None
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    for c in y_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _format_time_axis(fig, ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=20, ha="right")


def _clean_nonneg(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    return x


# ============================================================
# DASHBOARD ROWS (single location)
# ============================================================

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

    ax_ts.plot(d["time"], d[y], linewidth=2)
    ax_ts.set_title(title, fontsize=12)
    ax_ts.set_ylabel(ylabel)
    ax_ts.grid(True, alpha=0.25)
    _format_time_axis(fig, ax_ts)

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

    ax_ts.bar(d["time"], d[y])
    ax_ts.set_title(title, fontsize=12)
    ax_ts.set_ylabel(ylabel)
    ax_ts.grid(True, axis="y", alpha=0.25)
    _format_time_axis(fig, ax_ts)

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

    ax_wr.bar(centers, counts, width=width, align="center")
    ax_wr.set_theta_zero_location("N")
    ax_wr.set_theta_direction(-1)
    ax_wr.set_title("Wind Rose (direction frequency)", fontsize=12, pad=12)

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


# ============================================================
# MULTI-VARIABLE (one dataset) with multiple y-axes
# ============================================================

def multi_timeseries(
    df: pd.DataFrame,
    y_cols: List[str],
    title: str = "Multi-variable time series",
    labels: Optional[Dict[str, str]] = None,
):
    if not y_cols:
        return None

    d = _prep_df(df, y_cols)
    if d is None:
        return None

    keep = ["time"] + [c for c in y_cols if c in d.columns]
    if len(keep) < 2:
        return None

    d = d[keep].dropna(subset=["time"])
    if d.empty:
        return None

    fig, ax0 = plt.subplots(figsize=(12, 4.2), dpi=140)
    ax0.set_title(title, fontsize=12)
    ax0.grid(True, alpha=0.25)

    axes = [ax0]

    y0 = keep[1]
    ax0.plot(d["time"], d[y0], linewidth=2, label=(labels.get(y0, y0) if labels else y0))
    ax0.set_ylabel(labels.get(y0, y0) if labels else y0)
    _format_time_axis(fig, ax0)

    for i, y in enumerate(keep[2:], start=1):
        axn = ax0.twinx()
        axn.spines["right"].set_position(("axes", 1.0 + 0.10 * (i - 1)))
        axn.plot(d["time"], d[y], linewidth=2, label=(labels.get(y, y) if labels else y))
        axn.set_ylabel(labels.get(y, y) if labels else y)
        axes.append(axn)

    handles, legend_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        legend_labels.extend(l)
    ax0.legend(handles, legend_labels, fontsize=9, loc="upper left", frameon=True)

    fig.tight_layout(pad=0.8)
    return fig


# ============================================================
# A vs B OVERLAY LINE PLOT (same variable, same axes)
# ============================================================

def compare_timeseries_ab(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    y: str,
    label_a: str = "Location A",
    label_b: str = "Location B",
    title: str = "",
    ylabel: str = "",
    agg: str = "none",  # "none" or "daily"
):
    """
    Plots A and B on the same chart (two lines) for a single variable y.
    agg="daily" will resample to daily mean (or daily sum for rain/snow/precip if you choose).
    """
    da = _prep_df(df_a, [y])
    db = _prep_df(df_b, [y])
    if da is None or db is None:
        return None
    if y not in da.columns or y not in db.columns:
        return None

    da = da[["time", y]].dropna()
    db = db[["time", y]].dropna()
    if da.empty or db.empty:
        return None

    if agg == "daily":
        da = da.set_index("time").resample("D")[y].mean().dropna().reset_index()
        db = db.set_index("time").resample("D")[y].mean().dropna().reset_index()

    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=140)
    ax.plot(da["time"], da[y], linewidth=2, label=label_a)
    ax.plot(db["time"], db[y], linewidth=2, label=label_b)

    ax.set_title(title if title else f"{y}: {label_a} vs {label_b}", fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    _format_time_axis(fig, ax)

    fig.tight_layout(pad=0.7)
    return fig


# ============================================================
# A vs B WINDROSE (side-by-side)
# ============================================================

def compare_windrose_ab(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Location A",
    label_b: str = "Location B",
    speed_col: str = "wind_speed_10m",
    dir_col: str = "wind_direction_10m",
    n_sectors: int = 16,
):
    da = _prep_df(df_a, [speed_col, dir_col])
    db = _prep_df(df_b, [speed_col, dir_col])
    if da is None or db is None:
        return None
    if speed_col not in da.columns or dir_col not in da.columns:
        return None
    if speed_col not in db.columns or dir_col not in db.columns:
        return None

    da = da[[speed_col, dir_col]].dropna()
    db = db[[speed_col, dir_col]].dropna()
    if da.empty or db.empty:
        return None

    def counts(d):
        dirs_rad = np.deg2rad((d[dir_col].values % 360))
        edges = np.linspace(0, 2 * np.pi, n_sectors + 1)
        c, _ = np.histogram(dirs_rad, bins=edges)
        width = (2 * np.pi) / n_sectors
        centers = edges[:-1] + width / 2
        return centers, c, width

    centers_a, counts_a, width = counts(da)
    centers_b, counts_b, _ = counts(db)

    fig = plt.figure(figsize=(12, 4.2), dpi=140)
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    ax2 = fig.add_subplot(gs[0, 1], polar=True)

    ax1.bar(centers_a, counts_a, width=width, align="center")
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_title(f"{label_a} Wind Rose", fontsize=12, pad=12)

    ax2.bar(centers_b, counts_b, width=width, align="center")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_title(f"{label_b} Wind Rose", fontsize=12, pad=12)

    fig.tight_layout(pad=0.8)
    return fig


# ============================================================
# WEIBULL PARAMS (for display)
# ============================================================

def weibull_fit_params(series: pd.Series, force_loc0: bool = True):
    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required. Install with: pip install scipy") from e

    x = _clean_nonneg(series)
    if x.empty:
        return None

    if force_loc0:
        c, loc, scale = stats.weibull_min.fit(x.values, floc=0)
    else:
        c, loc, scale = stats.weibull_min.fit(x.values)

    return {"shape_k": float(c), "loc": float(loc), "scale_lambda": float(scale), "n": int(x.size)}


# ============================================================
# 3-PANEL WIND FIT REPORT (PDF -> CDF -> Tail)
# ============================================================

def wind_fit_report_3panel(
    df: pd.DataFrame,
    speed_col: str = "wind_speed_10m",
    months_set: Optional[Set[int]] = None,
    season_label: str = "Season",
    bin_width: float = 0.5,
    x_min: float = 0.0,
    x_max: Optional[float] = None,
    n_grid: int = 600,
):
    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required. Install with: pip install scipy") from e

    d = _prep_df(df, [speed_col])
    if d is None or speed_col not in d.columns:
        return None

    d = d[["time", speed_col]].dropna()
    d = d[d["time"].notna()]
    if months_set is not None:
        d = d[d["time"].dt.month.isin(months_set)]
    if d.empty:
        return None

    xdata = _clean_nonneg(d[speed_col])
    xdata = xdata[xdata >= x_min]
    if xdata.empty:
        return None

    xmax = float(xdata.max()) if x_max is None else float(x_max)
    xmax = max(x_min + bin_width, xmax)
    xmax = np.ceil(xmax / bin_width) * bin_width

    bins = np.arange(x_min, xmax + bin_width, bin_width)
    xgrid = np.linspace(x_min, xmax, int(n_grid))

    fits = []

    for name, dist, ls, fit_kwargs in [
        ("Weibull 2", stats.weibull_min, "-", {"floc": 0}),
        ("Champernowne", getattr(stats, "champernowne", None), "-", {"floc": 0}),
        ("Rayleigh", stats.rayleigh, "-", {"floc": 0}),
        ("Rice", stats.rice, "-", {"floc": 0}),
        ("Weibull 3", stats.weibull_min, "--", {}),
    ]:
        if dist is None:
            continue
        try:
            params = dist.fit(xdata.values, **fit_kwargs)
            fits.append((name, ls, dist, params))
        except Exception:
            pass

    xs = np.sort(xdata.values)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), dpi=140, sharex=True)
    ax_pdf, ax_cdf, ax_tail = axes

    ax_pdf.hist(xdata.values, bins=bins, density=True, alpha=0.7, edgecolor="black", label="Histogram")
    for name, ls, dist, p in fits:
        pdf = np.asarray(dist.pdf(xgrid, *p), dtype=float)
        pdf[~np.isfinite(pdf)] = 0.0
        ax_pdf.plot(xgrid, pdf, linewidth=2, linestyle=ls, label=name)
    ax_pdf.set_ylabel("Probability Density")
    ax_pdf.set_title(f"{season_label}: PDF fits", fontsize=12)
    ax_pdf.grid(True, alpha=0.25)
    ax_pdf.legend(loc="upper right", fontsize=9, frameon=True)

    ax_cdf.step(xs, ys, where="post", linewidth=2, label="Empirical CDF")
    for name, ls, dist, p in fits:
        cdf = np.asarray(dist.cdf(xgrid, *p), dtype=float)
        cdf[~np.isfinite(cdf)] = 0.0
        cdf = np.clip(cdf, 0.0, 1.0)
        ax_cdf.plot(xgrid, cdf, linewidth=2, linestyle=ls, label=name)
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title(f"{season_label}: CDF fits", fontsize=12)
    ax_cdf.set_ylim(0, 1.0)
    ax_cdf.grid(True, alpha=0.25)
    ax_cdf.legend(loc="lower right", fontsize=9, frameon=True)

    ax_tail.hist(xdata.values, bins=bins, density=True, alpha=0.35, edgecolor="black", label="Histogram")
    for name, ls, dist, p in fits:
        pdf = np.asarray(dist.pdf(xgrid, *p), dtype=float)
        pdf[~np.isfinite(pdf)] = 0.0
        ax_tail.plot(xgrid, pdf, linewidth=2, linestyle=ls, label=name)
    ax_tail.set_yscale("log")
    ax_tail.set_ylabel("PDF (log scale)")
    ax_tail.set_xlabel("Wind Speed (m/s)")
    ax_tail.set_title(f"{season_label}: Tail comparison (log-PDF)", fontsize=12)
    ax_tail.grid(True, alpha=0.25)
    ax_tail.legend(loc="upper right", fontsize=9, frameon=True)

    ax_tail.set_xlim(x_min, xmax)
    fig.tight_layout(pad=0.8)
    return fig


# ============================================================
# WINTER vs SUMMER FITS: PDF + CDF (2x2)
# ============================================================

def seasonal_pdf_cdf_comparison(
    df: pd.DataFrame,
    speed_col: str = "wind_speed_10m",
    winter_months: Optional[Set[int]] = None,
    summer_months: Optional[Set[int]] = None,
    dist_names: Optional[List[str]] = None,
    x_min: float = 0.0,
    x_max: Optional[float] = None,
    n_grid: int = 500,
    min_n: int = 30,
):
    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required. Install with: pip install scipy") from e

    if winter_months is None:
        winter_months = {11, 12, 1, 2, 3}
    if summer_months is None:
        summer_months = {4, 5, 6, 7, 8, 9, 10}
    if dist_names is None:
        dist_names = ["weibull2", "weibull3", "champernowne", "rayleigh", "rice", "gamma", "lognorm"]

    d = _prep_df(df, [speed_col])
    if d is None or speed_col not in d.columns:
        return None

    d = d[["time", speed_col]].dropna()
    d = d[d["time"].notna()]
    if d.empty:
        return None

    winter = _clean_nonneg(d[d["time"].dt.month.isin(winter_months)][speed_col])
    summer = _clean_nonneg(d[d["time"].dt.month.isin(summer_months)][speed_col])

    winter = winter[winter >= x_min]
    summer = summer[summer >= x_min]

    if winter.size < min_n and summer.size < min_n:
        return None

    overall_max = 0.0
    if winter.size:
        overall_max = max(overall_max, float(winter.max()))
    if summer.size:
        overall_max = max(overall_max, float(summer.max()))

    x_max_plot = max(x_min + 1.0, overall_max * 1.05) if x_max is None else float(x_max)
    x = np.linspace(x_min, x_max_plot, int(n_grid))

    def fit_one(sample: pd.Series, name: str):
        if sample is None or sample.size < min_n:
            return None
        data = sample.values

        if name == "weibull2":
            try:
                params = stats.weibull_min.fit(data, floc=0)
                pdf = stats.weibull_min.pdf(x, *params)
                cdf = stats.weibull_min.cdf(x, *params)
                peak = float(np.nanmax(pdf))
                return {"label": "weibull2", "pdf": pdf, "cdf": cdf, "peak": peak}
            except Exception:
                return None

        if name == "weibull3":
            try:
                params = stats.weibull_min.fit(data)
                pdf = stats.weibull_min.pdf(x, *params)
                cdf = stats.weibull_min.cdf(x, *params)
                peak = float(np.nanmax(pdf))
                return {"label": "weibull3", "pdf": pdf, "cdf": cdf, "peak": peak}
            except Exception:
                return None

        if not hasattr(stats, name):
            return None
        dist = getattr(stats, name)

        try:
            params = dist.fit(data, floc=0)
        except Exception:
            try:
                params = dist.fit(data)
            except Exception:
                return None

        try:
            pdf = dist.pdf(x, *params)
            cdf = dist.cdf(x, *params)
            peak = float(np.nanmax(pdf))
        except Exception:
            return None

        return {"label": name, "pdf": pdf, "cdf": cdf, "peak": peak}

    def make_curves(sample: pd.Series):
        curves = []
        for nm in dist_names:
            out = fit_one(sample, nm)
            if out is None:
                continue
            if not np.isfinite(out["peak"]) or out["peak"] <= 0:
                continue
            pdf = np.asarray(out["pdf"], dtype=float)
            cdf = np.asarray(out["cdf"], dtype=float)
            pdf[~np.isfinite(pdf)] = 0.0
            cdf[~np.isfinite(cdf)] = 0.0
            cdf = np.clip(cdf, 0.0, 1.0)
            curves.append({"label": out["label"], "pdf": pdf, "cdf": cdf, "peak": out["peak"]})
        curves.sort(key=lambda z: z["peak"], reverse=True)
        return curves

    winter_curves = make_curves(winter)
    summer_curves = make_curves(summer)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=140, sharex=True)
    ax_w_pdf, ax_s_pdf = axes[0, 0], axes[0, 1]
    ax_w_cdf, ax_s_cdf = axes[1, 0], axes[1, 1]

    def plot_pdf(ax, curves, title, n_points):
        ax.set_title(f"{title} PDF (n={n_points:,})", fontsize=12)
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("PDF")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_min, x_max_plot)
        if not curves:
            ax.text(0.5, 0.5, "Not enough data / fit failed", ha="center", va="center", transform=ax.transAxes)
            return
        for c in curves:
            ax.plot(x, c["pdf"], linewidth=2, label=c["label"])
        ax.legend(fontsize=8, loc="upper right", frameon=True)

    def plot_cdf(ax, curves, title, n_points):
        ax.set_title(f"{title} CDF (n={n_points:,})", fontsize=12)
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_min, x_max_plot)
        ax.set_ylim(0.0, 1.0)
        if not curves:
            ax.text(0.5, 0.5, "Not enough data / fit failed", ha="center", va="center", transform=ax.transAxes)
            return
        for c in curves:
            ax.plot(x, c["cdf"], linewidth=2, label=c["label"])
        ax.legend(fontsize=8, loc="lower right", frameon=True)

    plot_pdf(ax_w_pdf, winter_curves, "Winter", int(winter.size))
    plot_pdf(ax_s_pdf, summer_curves, "Summer", int(summer.size))
    plot_cdf(ax_w_cdf, winter_curves, "Winter", int(winter.size))
    plot_cdf(ax_s_cdf, summer_curves, "Summer", int(summer.size))

    fig.tight_layout(pad=0.8)
    return fig


# ============================================================
# COMPARE TWO LOCATIONS: PDF + CDF overlay
# ============================================================

def compare_two_locations_pdf_cdf(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Location A",
    label_b: str = "Location B",
    speed_col: str = "wind_speed_10m",
    months_set: Optional[Set[int]] = None,
    bin_width: float = 0.5,
    x_min: float = 0.0,
    x_max: Optional[float] = None,
):
    da = _prep_df(df_a, [speed_col])
    db = _prep_df(df_b, [speed_col])
    if da is None or db is None:
        return None
    if speed_col not in da.columns or speed_col not in db.columns:
        return None

    da = da[["time", speed_col]].dropna()
    db = db[["time", speed_col]].dropna()
    da = da[da["time"].notna()]
    db = db[db["time"].notna()]

    if months_set is not None:
        da = da[da["time"].dt.month.isin(months_set)]
        db = db[db["time"].dt.month.isin(months_set)]

    xa = _clean_nonneg(da[speed_col])
    xb = _clean_nonneg(db[speed_col])
    xa = xa[xa >= x_min]
    xb = xb[xb >= x_min]
    if xa.empty or xb.empty:
        return None

    if x_max is None:
        x_max = max(float(xa.max()), float(xb.max())) * 1.05
    x_max = max(x_min + bin_width, float(x_max))
    x_max = np.ceil(x_max / bin_width) * bin_width

    bins = np.arange(x_min, x_max + bin_width, bin_width)

    xsa = np.sort(xa.values)
    ysa = np.arange(1, len(xsa) + 1) / len(xsa)
    xsb = np.sort(xb.values)
    ysb = np.arange(1, len(xsb) + 1) / len(xsb)

    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=140)

    ax_pdf.hist(xa.values, bins=bins, density=True, alpha=0.45, edgecolor="black",
                label=f"{label_a} (n={len(xa):,})")
    ax_pdf.hist(xb.values, bins=bins, density=True, alpha=0.45, edgecolor="black",
                label=f"{label_b} (n={len(xb):,})")
    ax_pdf.set_title("PDF (Histogram density)", fontsize=12)
    ax_pdf.set_xlabel("Wind speed (m/s)")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.grid(True, alpha=0.25)
    ax_pdf.legend(fontsize=9, frameon=True)

    ax_cdf.step(xsa, ysa, where="post", linewidth=2, label=label_a)
    ax_cdf.step(xsb, ysb, where="post", linewidth=2, label=label_b)
    ax_cdf.set_title("CDF (Empirical)", fontsize=12)
    ax_cdf.set_xlabel("Wind speed (m/s)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(True, alpha=0.25)
    ax_cdf.legend(fontsize=9, frameon=True)

    fig.tight_layout(pad=0.8)
    return fig
