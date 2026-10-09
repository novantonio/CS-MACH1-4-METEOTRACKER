"""
MT4CSMACH1 — MeteoTracker → CS-MACH1 Streamlit pipeline
------------------------------------------------------
Mobile-first layout: logo/title → instructions → upload → controls →
combined summary plots → per-file metrics + expandable charts.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore")

# Optional cartopy (maps). If missing, maps are skipped gracefully.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

apply_cs_mach1_theme(
    page_title="MT4CSMACH1 — MeteoTracker pipeline",
    main_title="🌤 MT4CSMACH1: MeteoTracker → CS-MACH1",
    subtitle="Multi-parameter atmospheric sessions (AirLogger / MeteoTracker)",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────

PARAMS = [
    ("Temp[°C]",    "Temperature",     "°C",   "steelblue"),
    ("Hum[%]",      "Humidity",        "%",    "mediumseagreen"),
    ("Alt[m]",      "Altitude",        "m",    "slategray"),
    ("Press[mbar]", "Pressure",        "mbar", "mediumpurple"),
    ("DP[°C]",      "Dew Point",       "°C",   "cadetblue"),
    ("θ[K]",        "Pot. Temp θ",     "K",    "sandybrown"),
    ("HDX[°C]",     "Heat Discomfort", "°C",   "indianred"),
    ("Speed[km/h]", "Wind Speed",      "km/h", "royalblue"),
    ("Radiation[]", "Radiation",       "a.u.", "goldenrod"),
]

PARAM_COLS = [p[0] for p in PARAMS]


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_airlog_csv(file) -> pd.DataFrame:
    """
    Parse a MeteoTracker / AirLogger CSV.
    Expected columns (subset ok): Time, Temp[°C], Hum[%], Alt[m], Press[mbar],
    DP[°C], θ[K], HDX[°C], Speed[km/h], Radiation[], Lat, Lon.
    """
    df = pd.read_csv(file)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=False)
    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)

    for col in PARAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for lat_c, lon_c in (("Lat", "Lon"), ("lat", "lon"), ("Latitude", "Longitude")):
        if lat_c in df.columns and lon_c in df.columns:
            df["Lat"] = pd.to_numeric(df[lat_c], errors="coerce")
            df["Lon"] = pd.to_numeric(df[lon_c], errors="coerce")
            break

    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def add_rolling(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df[col].rolling(window=window, min_periods=1).mean()


# ── Figures ───────────────────────────────────────────────────────────────────

def make_combined_parameter_figures(
    logger_dfs: dict[str, pd.DataFrame],
    params_list: list[tuple],
) -> list[tuple[str, plt.Figure]]:
    """
    One figure per parameter: raw series of all sessions overlaid.
    Returns list of (parameter_name, figure).
    """
    figures: list[tuple[str, plt.Figure]] = []
    if not logger_dfs:
        return figures

    file_names = list(logger_dfs.keys())
    n = max(len(file_names), 1)
    # Matplotlib ≥ 3.7 prefers plt.colormaps; keep fallback for older builds
    try:
        cmap = plt.colormaps["tab10"].resampled(n)
    except Exception:
        cmap = plt.cm.get_cmap("tab10", n)
    file_color_map = {name: cmap(i) for i, name in enumerate(file_names)}

    for col, name, unit, _ in params_list:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(
            f"📈 Combined {name} ({unit}) — all sessions",
            fontsize=13, fontweight="bold", y=1.02,
        )

        has_data = False
        for fname in file_names:
            df = logger_dfs[fname]
            if col not in df.columns or df[col].isna().all():
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            times = df.loc[series.index, "Time"]
            ax.plot(
                times, series,
                alpha=0.65, linewidth=1.1,
                color=file_color_map[fname],
                label=Path(fname).stem,
            )
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel(f"{name} ({unit})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        if len(file_names) <= 12:
            ax.legend(fontsize=7, loc="best", title="Sessions")
        fig.tight_layout()
        figures.append((name, fig))

    return figures


def make_3x3_figure(df: pd.DataFrame, label: str, window: int) -> plt.Figure:
    """3×3 grid — raw + rolling mean + mean/median reference lines per parameter."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(f"📋 {label}", fontsize=13, fontweight="bold", y=1.01)

    for ax, (col, name, unit, color) in zip(axes.flat, PARAMS):
        if col not in df.columns or df[col].isna().all():
            ax.set_visible(False)
            continue

        series = df[col].dropna()
        times = df.loc[series.index, "Time"]
        rolling = add_rolling(df.loc[series.index], col, window)

        p_mean = series.mean()
        p_med = series.median()

        ax.plot(times, series, alpha=0.30, linewidth=0.7, color=color, label="Raw")
        ax.plot(
            times, rolling,
            linewidth=1.8, color="tomato",
            label=f"Rolling (w={window})",
        )
        ax.axhline(p_mean, color="crimson", linewidth=1.2, linestyle="--",
                   label=f"Mean {p_mean:.2f}")
        ax.axhline(p_med, color="darkorange", linewidth=1.2, linestyle="--",
                   label=f"Med {p_med:.2f}")

        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("Time", fontsize=7)
        ax.set_ylabel(f"{name} ({unit})", fontsize=7)
        ax.set_title(f"{name}", fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    return fig


def make_trajectory_map(df: pd.DataFrame, label: str) -> plt.Figure | None:
    """Cartopy trajectory map (start ▲ / end ▼). Returns None if cartopy or coords missing."""
    if not HAS_CARTOPY:
        return None
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None
    if df["Lat"].isna().all() or df["Lon"].isna().all():
        return None

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.plot(
        df["Lon"], df["Lat"],
        marker="o", markersize=2.5, linestyle="-", alpha=0.7, color="steelblue",
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        df["Lon"].iloc[0], df["Lat"].iloc[0],
        color="green", marker="^", s=100, label="Start", zorder=5,
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        df["Lon"].iloc[-1], df["Lat"].iloc[-1],
        color="crimson", marker="v", s=100, label="End", zorder=5,
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    lon_min, lon_max = df["Lon"].min(), df["Lon"].max()
    lat_min, lat_max = df["Lat"].min(), df["Lat"].max()
    pad_lon = max((lon_max - lon_min) * 0.15, 0.02)
    pad_lat = max((lat_max - lat_min) * 0.15, 0.02)
    ax.set_extent(
        [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat],
        crs=ccrs.PlateCarree(),
    )

    ax.set_title(f"Trajectory — {label}")
    ax.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False,
        alpha=0.6, linestyle="--",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ── Instructions & upload ─────────────────────────────────────────────────────

st.markdown(
    """
    ### How to use
    1. **Upload** one or more MeteoTracker / AirLogger CSV files below.
    2. Optionally adjust the **rolling window** (used for smoothed traces).
    3. Tap **Start Processing**.
    4. Scroll: first the **combined overview** (all sessions overlaid),  
       then each session with mean values and expandable parameter charts.

    **Expected columns** (any subset is fine):  
    `Time`, `Temp[°C]`, `Hum[%]`, `Alt[m]`, `Press[mbar]`, `DP[°C]`,  
    `θ[K]`, `HDX[°C]`, `Speed[km/h]`, `Radiation[]`, and optionally `Lat` / `Lon`.
    """
)

uploaded_files = st.file_uploader(
    "Upload MeteoTracker / AirLogger CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files
    st.caption(f"{len(uploaded_files)} file(s) selected")

# Controls under the upload area (mobile-friendly, no sidebar)
window_size = st.number_input(
    "Rolling window (samples)",
    min_value=1,
    max_value=50,
    value=st.session_state.get("window_size", 5),
    step=1,
)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)
with col_btn2:
    reset_button = st.button("🧹 Reset", use_container_width=True)

if reset_button:
    st.session_state.clear()
    st.rerun()

if not HAS_CARTOPY:
    st.caption("Note: cartopy not installed — trajectory maps will be skipped.")


# ── Processing ────────────────────────────────────────────────────────────────

if start_button:
    if "uploaded_files" not in st.session_state or not st.session_state["uploaded_files"]:
        st.warning("Please upload at least one CSV file first.")
    else:
        raw_files = st.session_state["uploaded_files"]
        total = len(raw_files)
        logger_dfs: dict[str, pd.DataFrame] = {}

        progress = st.progress(0, text="Starting…")
        status = st.empty()

        for i, file in enumerate(raw_files):
            pct = int(((i + 1) / total) * 100)
            text = f"Parsing {i + 1}/{total}: {file.name}"
            progress.progress(pct, text=text)
            status.caption(text)
            try:
                df = parse_airlog_csv(file)
                if df.empty:
                    st.warning(f"Empty after parse: **{file.name}**")
                    continue
                logger_dfs[file.name] = df
            except Exception as exc:
                st.warning(f"Failed parsing **{file.name}**: {exc}")

        progress.progress(100, text="✅ Done!")
        status.caption(f"Parsed {len(logger_dfs)}/{total} file(s) OK.")

        if not logger_dfs:
            st.error("No valid sessions found.")
        else:
            st.session_state["logger_dfs"] = logger_dfs
            st.session_state["window_size"] = int(window_size)


# Keep window_size updated when the number input changes after processing
if "logger_dfs" in st.session_state:
    st.session_state["window_size"] = int(window_size)


# ── Results ───────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state:
    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    win = int(st.session_state.get("window_size", 5))

    # ── Combined overview (all sessions overlaid) ─────────────────────────────
    st.header("📊 Combined overview — all sessions")
    st.caption("One plot per parameter; each colour is a different session.")

    combined_figs = make_combined_parameter_figures(logger_dfs, PARAMS)
    if not combined_figs:
        st.info("No parameter data available for combined plots.")
    else:
        for pname, fig in combined_figs:
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # ── Per-session: metrics + expandable charts ──────────────────────────────
    st.header("📄 Sessions")

    for fname, df in logger_dfs.items():
        label = Path(fname).stem
        st.subheader(f"📄 {fname}")

        # Mean metrics
        present = [
            (col, name, unit)
            for col, name, unit, _ in PARAMS
            if col in df.columns and not df[col].isna().all()
        ]
        if present:
            n_cols = min(3, len(present))
            cols = st.columns(n_cols)
            for i, (col, name, unit) in enumerate(present):
                cols[i % n_cols].metric(name, f"{df[col].mean():.2f} {unit}")

        # Meta line
        t0 = str(df["Time"].iloc[0])[:19]
        t1 = str(df["Time"].iloc[-1])[:19]
        duration_min = (df["Time"].iloc[-1] - df["Time"].iloc[0]).total_seconds() / 60
        st.caption(
            f"{t0} → {t1}  ·  {len(df)} samples  ·  {duration_min:.1f} min"
        )

        # Expandable detail charts
        with st.expander("Show parameter charts & map", expanded=False):
            fig_map = make_trajectory_map(df, label)
            if fig_map is not None:
                st.pyplot(fig_map)
                plt.close(fig_map)
            elif HAS_CARTOPY:
                st.caption("No valid Lat/Lon — map skipped.")

            fig = make_3x3_figure(df, label=label, window=win)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

    st.info(
        "🔴 crimson dashed = mean  ·  🟠 darkorange dashed = median  ·  "
        "tomato solid = rolling mean  ·  pale = raw"
    )

    cs_mach1_footer(text="MT4CSMACH1 · CS-MACH1 Project · MeteoTracker Pipeline")
else:
    cs_mach1_footer(text="MT4CSMACH1 · CS-MACH1 Project · MeteoTracker Pipeline")
