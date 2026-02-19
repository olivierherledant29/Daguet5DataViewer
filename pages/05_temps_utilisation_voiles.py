import streamlit as st
import pandas as pd
import numpy as np

from lib.db_client import fetch_aggregated

st.set_page_config(page_title="Temps d'utilisation voiles", layout="wide")
st.title("Temps d'utilisation des voiles")

# --------------------
# Paramètres fixes
# --------------------
START_CUTOFF_UTC = pd.Timestamp("2026-02-14T00:00:00Z")
BSP_THRESH = 4.0          # ✅ seuil BSP > 4 nds
BUCKET = "1m"             # lecture 1 point par minute
DT_MINUTES = 1.0          # 1 minute par ligne retenue

FIELDS_MEAN = [
    "SilverData.BSP_BoatSpeed",
    "SilverData.WIND_TWS",
    "SilverData.WIND_TWA",
]
FIELDS_LAST = [
    "SilverData.PERF_MainSail",
    "SilverData.PERF_StaySail",
    "SilverData.PERF_Jib",
    "SilverData.PERF_HeadSail",
]

POS_COLS = {
    "Main": "SilverData.PERF_MainSail",
    "Stay": "SilverData.PERF_StaySail",
    "Jib":  "SilverData.PERF_Jib",
    "Head": "SilverData.PERF_HeadSail",
}

MOIS_FR = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

def format_date_fr(ts_utc: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts_utc)
    return f"{ts.day} {MOIS_FR[ts.month]} {ts.year}"

def day_to_range_utc(day_utc: pd.Timestamp) -> tuple[str, str]:
    d = pd.Timestamp(day_utc)
    start = d.floor("D")
    end = start + pd.Timedelta(days=1)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

def ensure_nav_days() -> list[pd.Timestamp]:
    nav_days = st.session_state.get("nav_days", [])
    return sorted(nav_days, reverse=True)

def clean_sail(x) -> str | None:
    """
    Retourne un sailcode propre ou None si pas de voile.
    - None/NaN -> None
    - "" -> None
    - "none" (toutes casses) -> None
    - chaînes qui commencent par "nan" ou "none" -> None (ex: 'NaN2025_b')
    """
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    s_low = s.lower()
    if s_low == "none":
        return None
    if s_low.startswith("nan") or s_low.startswith("none"):
        return None

    return s

def minutes_to_hm(minutes: float) -> str:
    minutes = int(round(minutes))
    h = minutes // 60
    m = minutes % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m} min"

# --------------------
# UI
# --------------------
nav_days = ensure_nav_days()
if not nav_days:
    st.warning("Librairie des jours navigués indisponible. Ouvre d’abord la page d’accueil.")
    st.stop()

# filtre cutoff : garder uniquement >= 14/02/2026
nav_days = [pd.Timestamp(d) for d in nav_days]
nav_days = [d for d in nav_days if d >= START_CUTOFF_UTC]

if not nav_days:
    st.warning("Aucun jour navigué >= 14 février 2026 dans la librairie.")
    st.stop()

nav_labels = [format_date_fr(d) for d in nav_days]
label_to_day = {format_date_fr(d): d for d in nav_days}

with st.sidebar:
    st.header("Période analysée")
    st.caption("Lecture 1 minute. Une minute compte si BSP > 4 nds.")
    days_sel = st.multiselect(
        "Jours navigués (UTC)",
        options=nav_labels,
        default=nav_labels[:10],
        help="Sélectionne les jours que tu veux analyser.",
    )
    top_n = st.slider("Top N voiles", 10, 200, 50, 5)

if not days_sel:
    st.info("Sélectionne au moins un jour dans la barre latérale.")
    st.stop()

# --------------------
# Chargement
# --------------------
@st.cache_data(show_spinner=False)
def load_days_1m(days_labels: tuple[str, ...]) -> pd.DataFrame:
    dfs = []
    for lab in days_labels:
        day = label_to_day[lab]
        s_iso, e_iso = day_to_range_utc(day)
        df = fetch_aggregated(
            fields_mean=FIELDS_MEAN,
            fields_last=FIELDS_LAST,
            start_utc_iso=s_iso,
            end_utc_iso=e_iso,
            bucket=BUCKET,
        )
        if not df.empty:
            df["day_utc"] = pd.Timestamp(day).floor("D")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

with st.spinner("Chargement des données (1 minute)…"):
    df = load_days_1m(tuple(days_sel))

if df.empty:
    st.warning("Aucune donnée retournée.")
    st.stop()

# --------------------
# Filtre BSP>4 + nettoyage
# --------------------
needed = ["time_utc", "SilverData.BSP_BoatSpeed", "SilverData.WIND_TWS", "SilverData.WIND_TWA"] + list(POS_COLS.values())
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes dans les données: {missing}")
    st.stop()

df = df.dropna(subset=["time_utc", "SilverData.BSP_BoatSpeed"]).copy()
df = df[df["SilverData.BSP_BoatSpeed"] > BSP_THRESH].copy()

if df.empty:
    st.warning("Aucune minute BSP>4 sur la période sélectionnée.")
    st.stop()

# --------------------
# Accumulation : 1 minute par voile en position
# --------------------
rows = []
for _, r in df.iterrows():
    tws = r["SilverData.WIND_TWS"]
    twa = r["SilverData.WIND_TWA"]
    abs_twa = abs(float(twa)) if pd.notna(twa) else np.nan

    for pos, col in POS_COLS.items():
        sail = clean_sail(r[col])
        if sail is None:
            continue
        rows.append({
            "sail": sail,
            "pos": pos,
            "tws": tws,
            "abs_twa": abs_twa,
            "minutes": DT_MINUTES,
        })

long = pd.DataFrame(rows)
if long.empty:
    st.warning("Aucune voile détectée pendant les minutes BSP>4.")
    st.stop()

# --------------------
# Stats par voile
# --------------------
agg = (long.groupby("sail", as_index=False)
       .agg(
           minutes_total=("minutes", "sum"),
           tws_mean=("tws", "mean"),
           tws_min=("tws", "min"),
           tws_max=("tws", "max"),
           abs_twa_mean=("abs_twa", "mean"),
           samples=("minutes", "count"),
       )
       .sort_values("minutes_total", ascending=False)
       .reset_index(drop=True))

out = agg.copy()
out["temps"] = out["minutes_total"].apply(minutes_to_hm)
out["tws_mean"] = out["tws_mean"].round(2)
out["tws_min"] = out["tws_min"].round(2)
out["tws_max"] = out["tws_max"].round(2)
out["abs_twa_mean"] = out["abs_twa_mean"].round(1)

out = out.rename(columns={"abs_twa_mean": "abs(TWA)_mean"})
out = out[["sail", "temps", "minutes_total", "samples", "tws_mean", "tws_min", "tws_max", "abs(TWA)_mean"]]
out = out.head(top_n)

# --------------------
# Affichage
# --------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jours analysés", str(len(days_sel)))
c2.metric("Seuil BSP", f">{BSP_THRESH:.1f} nds")
c3.metric("Minutes retenues", f"{len(df):,}")
c4.metric("Voiles distinctes", f"{out['sail'].nunique():,}")

st.subheader("Temps d'utilisation par voile (minutes BSP>4)")
st.caption("Chaque minute où BSP>4 compte +1 minute pour chaque voile présente (Main/Stay/Jib/Head).")

st.dataframe(out, width="stretch", hide_index=True)

with st.expander("Détail long (debug)", expanded=False):
    st.dataframe(long.head(500), width="stretch", hide_index=True)
