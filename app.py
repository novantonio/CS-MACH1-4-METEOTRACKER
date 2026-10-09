"""
MT4CSMACH1 — MeteoTracker → CS-MACH1 Streamlit pipeline
------------------------------------------------------
Convert MeteoTracker / AirLogger multi-parameter sessions into
interactive diagnostics (time series, trajectory maps, session stats)
with the CS-MACH1 look & feel.
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
    subtitle="Multi-parameter atmospheric sessions (AirLogger / MeteoTracker) — diagnostics & maps",
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

    # Lat / Lon (common variants)
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

def make_3x3_figure(df: pd.DataFrame, label: str, window: int) -> plt.Figure:
    """3×3 grid — raw + rolling mean + mean/median reference lines per parameter."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(f"📋 {label}", fontsize=14, fontweight="bold", y=1.01)

    for ax, (col, name, unit, color) in zip(axes.flat, PARAMS):
        if col not in df.columns or df[col].isna().all():
            ax.set_visible(False)
            continue

        series = df[col].dropna()
        times = df.loc[series.index, "Time"]
        rolling = add_rolling(df.loc[series.index], col, window)

        p_mean = series.mean()
        p_med = series.median()

        ax.plot(times, series, alpha=0.30, linewidth=0.7, color=color, label="Raw data")
        ax.plot(
            times, rolling,
            linewidth=2, color="tomato",
            label=f"Rolling mean (w={window})",
        )
        ax.axhline(p_mean, color="crimson", linewidth=1.4, linestyle="--",
                   label=f"Mean   {p_mean:.2f} {unit}")
        ax.axhline(p_med, color="darkorange", linewidth=1.4, linestyle="--",
                   label=f"Median {p_med:.2f} {unit}")

        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel(f"{name} ({unit})", fontsize=8)
        ax.set_title(f"{name}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
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

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.plot(
        df["Lon"], df["Lat"],
        marker="o", markersize=3, linestyle="-", alpha=0.7, color="steelblue",
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        df["Lon"].iloc[0], df["Lat"].iloc[0],
        color="green", marker="^", s=120, label="Start", zorder=5,
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        df["Lon"].iloc[-1], df["Lat"].iloc[-1],
        color="crimson", marker="v", s=120, label="End", zorder=5,
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Auto-zoom with a small margin
    lon_min, lon_max = df["Lon"].min(), df["Lon"].max()
    lat_min, lat_max = df["Lat"].min(), df["Lat"].max()
    pad_lon = max((lon_max - lon_min) * 0.15, 0.02)
    pad_lat = max((lat_max - lat_min) * 0.15, 0.02)
    ax.set_extent(
        [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat],
        crs=ccrs.PlateCarree(),
    )

    ax.set_title(f"Trajectory — {label}")
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 alpha=0.6, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    window_size = st.slider("Rolling window", min_value=1, max_value=50, value=5)

    st.divider()
    n_files = len(st.session_state.get("logger_dfs", {}))
    if n_files > 0:
        st.success(f"📂 {n_files} session{'s' if n_files != 1 else ''} loaded")
    else:
        st.info("📂 No files loaded yet")

    st.divider()
    sidebar_progress = st.empty()
    sidebar_status = st.empty()

    st.divider()
    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)

    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if not HAS_CARTOPY:
        st.warning("cartopy not installed — trajectory maps disabled.")


# ── File uploader ─────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or more MeteoTracker / AirLogger CSV files, then press **Start Processing**",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files


# ── Processing ────────────────────────────────────────────────────────────────

if start_button and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    total = len(raw_files)
    logger_dfs: dict[str, pd.DataFrame] = {}

    pbar = sidebar_progress.progress(0, text="Starting…")

    for i, file in enumerate(raw_files):
        pct = int((i / total) * 100)
        text = f"Parsing {i + 1}/{total}: {file.name}"
        pbar.progress(pct, text=text)
        sidebar_status.caption(text)
        try:
            df = parse_airlog_csv(file)
            if df.empty:
                st.warning(f"Empty after parse: **{file.name}**")
                continue
            logger_dfs[file.name] = df
        except Exception as exc:
            st.warning(f"Failed parsing **{file.name}**: {exc}")

    pbar.progress(100, text="✅ Done!")
    sidebar_status.caption(
        f"Parsed {len(logger_dfs)}/{total} file{'s' if total != 1 else ''} OK."
    )

    if not logger_dfs:
        st.error("No valid sessions found.")
        st.stop()

    st.session_state["logger_dfs"] = logger_dfs
    st.session_state["window_size"] = window_size


# Keep window_size in sync when slider moves after processing
if "logger_dfs" in st.session_state:
    st.session_state["window_size"] = window_size


# ── Display ───────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state:
    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    win = st.session_state.get("window_size", 5)

    # ── Per-session ───────────────────────────────────────────────────────────
    for fname, df in logger_dfs.items():
        label = Path(fname).stem
        st.subheader(f"📄 {fname}")

        # Metrics row (up to 3 visible at a time, then more)
        present = [(col, name, unit) for col, name, unit, _ in PARAMS
                   if col in df.columns and not df[col].isna().all()]
        if present:
            cols = st.columns(min(3, len(present)))
            for i, (col, name, unit) in enumerate(present[:3]):
                cols[i].metric(f"{name}", f"{df[col].mean():.2f} {unit}")
            if len(present) > 3:
                with st.expander("More metrics"):
                    extra = st.columns(min(3, len(present) - 3))
                    for i, (col, name, unit) in enumerate(present[3:]):
                        extra[i % 3].metric(f"{name}", f"{df[col].mean():.2f} {unit}")

        # Trajectory map
        fig_map = make_trajectory_map(df, label)
        if fig_map is not None:
            st.pyplot(fig_map)
            plt.close(fig_map)
        elif HAS_CARTOPY:
            st.caption("No valid Lat/Lon columns — map skipped.")

        # 3×3 parameter panel
        fig = make_3x3_figure(df, label=label, window=win)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.header("📊 Summary — All Sessions")

    rows = []
    for fname, df in logger_dfs.items():
        row = {
            "File": fname,
            "Start": str(df["Time"].iloc[0])[:19],
            "End": str(df["Time"].iloc[-1])[:19],
            "N samples": len(df),
            "Duration (min)": round(
                (df["Time"].iloc[-1] - df["Time"].iloc[0]).total_seconds() / 60, 1
            ),
        }
        for col, name, unit, _ in PARAMS:
            if col in df.columns and not df[col].isna().all():
                row[f"{name} mean ({unit})"] = round(df[col].mean(), 2)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.info(
        "🔴 crimson dashed = mean  ·  🟠 darkorange dashed = median  ·  "
        "tomato solid = rolling mean  ·  pale = raw"
    )

    st.divider()
    cs_mach1_footer(text="MT4CSMACH1 · CS-MACH1 Project · MeteoTracker Pipeline")
else:
    st.markdown(
        """
        ### How to use
        1. Upload one or more **MeteoTracker / AirLogger** CSV files (multi-parameter).
        2. Adjust the **rolling window** in the sidebar.
        3. Press **Start Processing**.

        Expected columns (subset is fine):  
        `Time`, `Temp[°C]`, `Hum[%]`, `Alt[m]`, `Press[mbar]`, `DP[°C]`,  
        `θ[K]`, `HDX[°C]`, `Speed[km/h]`, `Radiation[]`, and optionally `Lat` / `Lon`.
        """
    )
    cs_mach1_footer(text="MT4CSMACH1 · CS-MACH1 Project · MeteoTracker Pipeline")
