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

def build_bins_2knots_with_optional_first_1(lo_int: int, hi_int_exclusive: int):
    """
    Construit des bins sur [lo_int, hi_int_exclusive) en pas 2.
    Si la largeur totale n'est pas multiple de 2, la première bin fait 1.
    Exemple: lo=5, hi=12 => largeur=7 (impair) => [5,6) puis [6,8) [8,10) [10,12)
    """
    width = hi_int_exclusive - lo_int
    bins = []
    cur = float(lo_int)

    if width <= 0:
        return bins

    if width % 2 == 1:
        bins.append((cur, cur + 1.0))
        cur += 1.0

    while cur < hi_int_exclusive:
        nxt = min(cur + 2.0, float(hi_int_exclusive))
        bins.append((cur, nxt))
        cur = nxt

    return bins

# --------------------
# Constantes plot
# --------------------
SCATTER_S = 6
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
    st.warning("Librairie des jours navigués indisponible.")
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
            st.session_state["custom_df_raw"] = load_days_10s(days_labels)
        st.session_state.pop("custom_df_f", None)
        st.success("Données chargées.")

df_raw = st.session_state.get("custom_df_raw", pd.DataFrame())

# --------------------
# Filtres + choix binning
# --------------------
st.subheader("Filtres")

colF1, colF2, colF3, colF4, colF5 = st.columns([1, 1, 1, 1, 1.2])
with colF1:
    twa_min, twa_max = st.slider("abs(TWA) — degrés", 0, 180, (0, 180))
with colF2:
    bsp_min, bsp_max = st.slider("BSP — nds", 0, 30, (0, 30))
with colF3:
    tws_min, tws_max = st.slider("TWS — nds", 0, 40, (0, 40))
with colF4:
    pr_min, pr_max = st.slider("BSP_polarRatio", 0, 160, (70, 130))
with colF5:
    bin_mode = st.radio(
        "Bins",
        options=["BSP", "TWS"],
        horizontal=True,
        index=0,
    )

apply_filters = st.button("Appliquer filtres")

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
    st.session_state["custom_df_f"] = apply_common_filters(df_raw)

df_f = st.session_state.get("custom_df_f", pd.DataFrame())

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
# Bins (BSP ou TWS) sur data filtrées
# --------------------
if bin_mode == "BSP":
    bin_col = "SilverData.BSP_BoatSpeed"
    unit = "nds"
else:
    bin_col = "SilverData.WIND_TWS"
    unit = "nds"

lo_int = int(float(df_f[bin_col].min()))
hi_int_excl = int(float(df_f[bin_col].max())) + 1  # demandé : int(max)+1

bins = build_bins_2knots_with_optional_first_1(lo_int, hi_int_excl)

# Conserve seulement les bins non vides
bins_non_empty = []
for lo, hi in bins:
    sub = df_f[(df_f[bin_col] >= lo) & (df_f[bin_col] < hi)]
    if not sub.empty:
        bins_non_empty.append((lo, hi))

# Échelle couleur globale : BSP_polarRatio
pr_global_min = float(df_f["SilverData.PERF_BSP_PolarRatio"].min())
pr_global_max = float(df_f["SilverData.PERF_BSP_PolarRatio"].max())

def plot_grid_one_cbar_per_row(title: str, y_col: str, y_label: str):
    st.subheader(title)

    cmap = plt.get_cmap()
    norm = mpl.colors.Normalize(vmin=pr_global_min, vmax=pr_global_max)

    # 2 plots par ligne
    for i in range(0, len(bins_non_empty), 2):
        row_bins = bins_non_empty[i:i+2]
        col1, col2, colcb = st.columns([1, 1, 0.18])
        plot_cols = [col1, col2]

        first = None
        for j, (lo, hi) in enumerate(row_bins):
            sub = df_f[(df_f[bin_col] >= lo) & (df_f[bin_col] < hi)]
            if sub.empty:
                continue

            fig = plt.figure()
            sc = plt.scatter(
                sub["abs_twa"],
                sub[y_col],
                s=SCATTER_S,
                alpha=SCATTER_ALPHA,
                c=sub["SilverData.PERF_BSP_PolarRatio"],
                cmap=cmap,
                norm=norm,
            )

            plt.xlabel("abs(TWA) [deg]")
            plt.ylabel(y_label)
            # ex: "6 à 7 nds" ou "7 à 9 nds"
            lo_i = int(lo) if float(lo).is_integer() else lo
            hi_i = int(hi) if float(hi).is_integer() else hi
            plt.title(f"{lo_i} à {hi_i} {unit}")
            plt.tight_layout()

            plot_cols[j].pyplot(fig)
            plt.close(fig)

            if first is None:
                first = sc

        # Une seule colorbar pour la ligne
        if first is not None:
            fig_cb, ax_cb = plt.subplots(figsize=(1.0, 2.6))
            fig_cb.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax_cb,
                label="BSP_polarRatio",
            )
            fig_cb.tight_layout()
            colcb.pyplot(fig_cb)
            plt.close(fig_cb)

# --------------------
# Plots
# --------------------
plot_grid_one_cbar_per_row(
    title=f"Trim vs abs(TWA) — couleur = BSP_polarRatio (bins {bin_mode})",
    y_col="SilverData.AHRS_Trim",
    y_label="Trim",
)

plot_grid_one_cbar_per_row(
    title=f"abs(Heel) vs abs(TWA) — couleur = BSP_polarRatio (bins {bin_mode})",
    y_col="abs_heel",
    y_label="abs(Heel)",
)
