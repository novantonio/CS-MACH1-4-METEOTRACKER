# MT4CSMACH1

**MeteoTracker → CS-MACH1** — Streamlit pipeline for multi-parameter atmospheric sessions (AirLogger / MeteoTracker).

Part of the [CS-MACH1](https://github.com/) project family.  
Style & branding shared with the EnvLogger temperature pipeline (`cs_mach1_theme`).

## Features

- Upload one or more MeteoTracker / AirLogger CSV files
- Rolling-mean window (reactive slider)
- Per-session metrics (mean of each available parameter)
- Trajectory map (Cartopy) when `Lat` / `Lon` are present
- 3×3 diagnostic panel: raw + rolling mean + mean/median reference lines for  
  Temperature, Humidity, Altitude, Pressure, Dew Point, Potential Temperature,  
  Heat Discomfort, Wind Speed, Radiation
- Summary table of all sessions (start/end, duration, means)

## Quick start (local)

```bash
git clone <this-repo-url> MT4CSMACH1
cd MT4CSMACH1
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> **Note on Cartopy**  
> Cartopy needs system libraries (GEOS, PROJ). On Debian/Ubuntu:
> ```bash
> sudo apt-get install -y libgeos-dev libproj-dev proj-data proj-bin
> ```
> If Cartopy is missing, the app still runs; trajectory maps are simply skipped.

## Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. In [share.streamlit.io](https://share.streamlit.io) → **New app** → select the repo, branch, and `app.py`.
3. Cartopy may require a `packages.txt` (system deps) on Streamlit Cloud:

```text
# packages.txt (optional, place next to app.py)
libgeos-dev
libproj-dev
proj-data
proj-bin
```

## Expected CSV columns

| Column        | Unit   | Required |
|---------------|--------|----------|
| `Time`        | ISO    | yes      |
| `Temp[°C]`    | °C     | optional |
| `Hum[%]`      | %      | optional |
| `Alt[m]`      | m      | optional |
| `Press[mbar]` | mbar   | optional |
| `DP[°C]`      | °C     | optional |
| `θ[K]`        | K      | optional |
| `HDX[°C]`     | °C     | optional |
| `Speed[km/h]` | km/h   | optional |
| `Radiation[]` | a.u.   | optional |
| `Lat` / `Lon` | deg    | optional (for map) |

Any subset of the parameter columns is accepted.

## Project layout

```
MT4CSMACH1/
├── app.py              # Streamlit entry point
├── cs_mach1_theme.py   # Shared CS-MACH1 branding
├── logo.png
├── requirements.txt
├── LICENSE             # CC0 1.0
└── README.md
```

## Licence

[CC0 1.0 Universal](LICENSE) — public domain dedication.

---

CS-MACH1 Project · Ocean & atmosphere temperature monitoring platform
