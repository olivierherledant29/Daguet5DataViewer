import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from lib.db_client import fetch_aggregated

st.set_page_config(page_title="Custom", layout="wide")
st.title("Custom")

# --------------------
# Helpers
# --------------------
MOIS_FR = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

def format_date_fr(ts_utc: pd.Timestamp) -> str:
    return f"{ts_utc.day} {MOIS_FR[ts_utc.month]} {ts_utc.year}"

def day_to_range_utc(day_utc: pd.Timestamp) -> tuple[str, str]:
    start = day_utc.floor("D")
    end = start + pd.Timedelta(days=1)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

def ensure_nav_days() -> list[pd.Timestamp]:
    nav_days = st.session_state.get("nav_days", [])
    return sorted(nav_days, reverse=True)

def build_bins_2units_with_optional_first_1(lo_int: int, hi_int_exclusive: int) -> list[tuple[float, float]]:
    """
    Bins sur [lo_int, hi_int_exclusive) en pas 2.
    Si la largeur totale n'est pas multiple de 2, première bin de 1.
    """
    width = hi_int_exclusive - lo_int
    if width <= 0:
        return []

    bins = []
    cur = float(lo_int)

    if width % 2 == 1:
        bins.append((cur, cur + 1.0))
        cur += 1.0

    while cur < hi_int_exclusive:
        nxt = min(cur + 2.0, float(hi_int_exclusive))
        bins.append((cur, nxt))
        cur = nxt

    return bins

def build_abs_twa_bins_10deg() -> list[tuple[float, float]]:
    """
    Demande: commencer à 40° (35-45), puis 10° en 10° jusqu'à 160°.
    Donc: [35,45), [45,55), ..., [155,165)
    """
    edges = list(range(35, 166, 10))  # 35,45,...,165
    return [(float(edges[i]), float(edges[i+1])) for i in range(len(edges) - 1)]

def fmt_range(lo: float, hi: float, unit: str) -> str:
    if unit in ("nds", "deg"):
        return f"{int(lo)} à {int(hi)} {unit}"
    return f"{lo} à {hi}"

# --------------------
# Plot settings
# --------------------
SCATTER_S = 10
SCATTER_ALPHA = 0.4

# --------------------
# Channels à charger (10s)
# --------------------
FIELDS_MEAN = [
    "SilverData.WIND_TWA",
    "SilverData.BSP_BoatSpeed",
    "SilverData.WIND_TWS",
    "SilverData.PERF_BSP_PolarRatio",
    "SilverData.AHRS_Heel",
    "SilverData.AHRS_Trim",
]
FIELDS_LAST = []

# --------------------
# Sélection des journées
# --------------------
nav_days = ensure_nav_days()
if not nav_days:
    st.warning("Librairie des jours navigués indisponible. Ouvre d’abord la page d’accueil.")
    st.stop()

nav_labels = [format_date_fr(d) for d in nav_days]
label_to_day = {format_date_fr(d): d for d in nav_days}

st.markdown("**1: sélectionner journées, 2: appliquer filtres**")
st.markdown("**data lissées sur 10 secondes**")

default_days = [nav_labels[0]] if nav_labels else []

days_labels = st.multiselect(
    "Journées — coche au moins 1",
    options=nav_labels,
    default=default_days,
    key="custom_days_v3",
)

load_clicked = st.button("Sélectionner journées", type="primary")

def load_days_10s(selected_labels: list[str]) -> pd.DataFrame:
    dfs = []
    for lab in selected_labels:
        day = label_to_day[lab]
        s_iso, e_iso = day_to_range_utc(day)
        df = fetch_aggregated(
            fields_mean=FIELDS_MEAN,
            fields_last=FIELDS_LAST,
            start_utc_iso=s_iso,
            end_utc_iso=e_iso,
            bucket="10s",
        )
        if not df.empty:
            df["day_utc"] = day.floor("D")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if load_clicked:
    if not days_labels:
        st.error("Sélectionne au moins une journée.")
    else:
        with st.spinner("Chargement des données (10 s)…"):
            st.session_state["custom_df_raw_v3"] = load_days_10s(days_labels)
        st.session_state.pop("custom_df_f_v3", None)
        st.success("Données chargées.")

df_raw = st.session_state.get("custom_df_raw_v3", pd.DataFrame())

# --------------------
# Filtres + binning
# --------------------
st.subheader("Filtres")

colF1, colF2, colF3, colF4, colF5 = st.columns([1, 1, 1, 1, 1.2])
with colF1:
    twa_min, twa_max = st.slider("abs(TWA) — degrés", 0, 180, (0, 180), step=1, key="custom_twa_v3")
with colF2:
    bsp_min, bsp_max = st.slider("BSP — nds", 0, 30, (0, 30), step=1, key="custom_bsp_v3")
with colF3:
    tws_min, tws_max = st.slider("TWS — nds", 0, 40, (0, 40), step=1, key="custom_tws_v3")
with colF4:
    pr_min, pr_max = st.slider("BSP_polarRatio", 0, 160, (70, 130), step=1, key="custom_pr_v3")
with colF5:
    bin_mode = st.radio("Bin", options=["TWS", "BSP", "abs(TWA)"], horizontal=True, index=0, key="custom_bin_mode_v3")

apply_filters = st.button("Appliquer filtres", key="custom_apply_v3")

def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.dropna().copy()
    out["abs_twa"] = out["SilverData.WIND_TWA"].abs()
    out["abs_heel"] = out["SilverData.AHRS_Heel"].abs()

    return out[
        (out["abs_twa"] >= twa_min) & (out["abs_twa"] <= twa_max) &
        (out["SilverData.BSP_BoatSpeed"] >= bsp_min) & (out["SilverData.BSP_BoatSpeed"] <= bsp_max) &
        (out["SilverData.WIND_TWS"] >= tws_min) & (out["SilverData.WIND_TWS"] <= tws_max) &
        (out["SilverData.PERF_BSP_PolarRatio"] >= pr_min) & (out["SilverData.PERF_BSP_PolarRatio"] <= pr_max)
    ]

if apply_filters:
    st.session_state["custom_df_f_v3"] = apply_common_filters(df_raw)

df_f = st.session_state.get("custom_df_f_v3", pd.DataFrame())

# --------------------
# Checks
# --------------------
if df_raw.empty:
    st.info("Sélectionne des journées puis clique sur **Sélectionner journées**.")
    st.stop()

if df_f.empty:
    st.warning("Aucun point après filtrage.")
    st.stop()

# --------------------
# Choix X / Y / Color (après bin)
# --------------------
# Note: Heel remplacé par abs(Heel)
CHANNELS = {
    "Trim": "SilverData.AHRS_Trim",
    "abs(Heel)": "abs_heel",
    "BSP": "SilverData.BSP_BoatSpeed",
    "TWS": "SilverData.WIND_TWS",
    "TWA": "SilverData.WIND_TWA",
    "abs(TWA)": "abs_twa",
    "BSP_polarRatio": "SilverData.PERF_BSP_PolarRatio",
}

if bin_mode == "TWS":
    bin_key = "TWS"
    bin_col = CHANNELS["TWS"]
    unit = "nds"
    bins = build_bins_2units_with_optional_first_1(
        int(float(df_f[bin_col].min())),
        int(float(df_f[bin_col].max())) + 1
    )
elif bin_mode == "BSP":
    bin_key = "BSP"
    bin_col = CHANNELS["BSP"]
    unit = "nds"
    bins = build_bins_2units_with_optional_first_1(
        int(float(df_f[bin_col].min())),
        int(float(df_f[bin_col].max())) + 1
    )
else:
    bin_key = "abs(TWA)"
    bin_col = CHANNELS["abs(TWA)"]
    unit = "deg"
    bins = build_abs_twa_bins_10deg()

bins_non_empty = []
for lo, hi in bins:
    if not df_f[(df_f[bin_col] >= lo) & (df_f[bin_col] < hi)].empty:
        bins_non_empty.append((lo, hi))

if not bins_non_empty:
    st.warning("Aucun bin non-vide avec les filtres actuels.")
    st.stop()

# Options X/Y/Color: enlever le channel utilisé pour bin
options = [k for k in CHANNELS.keys() if k != bin_key]

# Defaults : X=abs(TWA), Y=abs(Heel), Color=BSP_polarRatio (si possible)
default_x = "abs(TWA)" if "abs(TWA)" in options else options[0]
default_y = "abs(Heel)" if "abs(Heel)" in options else options[0]
default_c = "BSP_polarRatio" if "BSP_polarRatio" in options else options[0]

st.subheader("Scatter plot")

colXYC1, colXYC2, colXYC3 = st.columns(3)
with colXYC1:
    x_key = st.selectbox("X", options=options, index=options.index(default_x), key="custom_x_v3")
with colXYC2:
    y_key = st.selectbox("Y", options=options, index=options.index(default_y), key="custom_y_v3")
with colXYC3:
    c_key = st.selectbox("Color", options=options, index=options.index(default_c), key="custom_c_v3")

x_col = CHANNELS[x_key]
y_col = CHANNELS[y_key]
c_col = CHANNELS[c_key]

# Échelle couleur globale (pour comparer entre bins)
c_min = float(df_f[c_col].min())
c_max = float(df_f[c_col].max())
norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
cmap = plt.get_cmap()

st.caption(f"Binning: **{bin_mode}** — X: **{x_key}** — Y: **{y_key}** — Color: **{c_key}**")

# --------------------
# Plots (2 par ligne) + 1 colorbar par ligne
# --------------------
for i in range(0, len(bins_non_empty), 2):
    row_bins = bins_non_empty[i:i+2]
    c1, c2, ccb = st.columns([1, 1, 0.18])
    cols = [c1, c2]
    have_any = False

    for j, (lo, hi) in enumerate(row_bins):
        sub = df_f[(df_f[bin_col] >= lo) & (df_f[bin_col] < hi)]
        if sub.empty:
            continue

        fig = plt.figure()
        plt.scatter(
            sub[x_col],
            sub[y_col],
            s=SCATTER_S,
            alpha=SCATTER_ALPHA,
            c=sub[c_col],
            cmap=cmap,
            norm=norm,
        )
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.title(fmt_range(lo, hi, unit))
        plt.tight_layout()

        cols[j].pyplot(fig)
        plt.close(fig)
        have_any = True

    # Colorbar unique par ligne (si au moins un plot)
    if have_any:
        fig_cb, ax_cb = plt.subplots(figsize=(1.0, 2.6))
        fig_cb.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb,
            label=c_key,
        )
        fig_cb.tight_layout()
        ccb.pyplot(fig_cb)
        plt.close(fig_cb)

with st.expander("Aperçu données filtrées", expanded=False):
    preview_cols = ["time_utc", "day_utc"] + [v for v in CHANNELS.values() if v in df_f.columns]
    st.dataframe(df_f[preview_cols].head(500), width="stretch")
