import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydeck as pdk

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
    edges = list(range(35, 166, 10))  # 35,45,...,165
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

def fmt_range(lo: float, hi: float, unit: str) -> str:
    if unit in ("nds", "deg"):
        return f"{int(lo)} à {int(hi)} {unit}"
    return f"{lo} à {hi}"

# --------------------
# Plot settings
# --------------------
SCATTER_S = 10
SCATTER_ALPHA = 0.4

# Map point sizes (divisé par 2 vs précédent)
MAP_RADIUS_BG_M = 4   # non retenus = noir
MAP_RADIUS_FG_M = 5   # retenus = colorés

# Map color scale (fixe)
MAP_PR_MIN = 70.0
MAP_PR_MAX = 130.0

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
    "SilverData.GPS_Latitude",
    "SilverData.GPS_Longitude",
    "SilverData.WIND_AWA",
    "SilverData.WIND_AWS",
    "SilverData.PERF_VMG",
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
    key="custom_days_v5",
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
            st.session_state["custom_df_raw_v5"] = load_days_10s(days_labels)
        st.session_state.pop("custom_df_f_v5", None)
        st.success("Données chargées.")

df_raw = st.session_state.get("custom_df_raw_v5", pd.DataFrame())
if df_raw.empty:
    st.info("Sélectionne des journées puis clique sur **Sélectionner journées**.")
    st.stop()

# --------------------
# Filtres + carte à droite
# --------------------
st.subheader("Filtres")

left, right = st.columns([1.0, 1.35], vertical_alignment="top")

with left:
    twa_min, twa_max = st.slider("abs(TWA) — degrés", 0, 180, (0, 180), step=1, key="custom_twa_v5")
    bsp_min, bsp_max = st.slider("BSP — nds", 0, 30, (0, 30), step=1, key="custom_bsp_v5")
    tws_min, tws_max = st.slider("TWS — nds", 0, 40, (0, 40), step=1, key="custom_tws_v5")
    pr_min, pr_max = st.slider("BSP_polarRatio", 0, 160, (70, 130), step=1, key="custom_pr_v5")

    bin_mode = st.radio("Bin", options=["TWS", "BSP", "abs(TWA)"], horizontal=True, index=0, key="custom_bin_mode_v5")

    apply_filters = st.button("Appliquer filtres", key="custom_apply_v5")

def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna().copy()
    out["abs_twa"] = out["SilverData.WIND_TWA"].abs()
    out["abs_heel"] = out["SilverData.AHRS_Heel"].abs()

    return out[
        (out["abs_twa"] >= twa_min) & (out["abs_twa"] <= twa_max) &
        (out["SilverData.BSP_BoatSpeed"] >= bsp_min) & (out["SilverData.BSP_BoatSpeed"] <= bsp_max) &
        (out["SilverData.WIND_TWS"] >= tws_min) & (out["SilverData.WIND_TWS"] <= tws_max) &
        (out["SilverData.PERF_BSP_PolarRatio"] >= pr_min) & (out["SilverData.PERF_BSP_PolarRatio"] <= pr_max)
    ]

if apply_filters or ("custom_df_f_v5" not in st.session_state):
    st.session_state["custom_df_f_v5"] = apply_common_filters(df_raw)

df_f = st.session_state.get("custom_df_f_v5", pd.DataFrame())
if df_f.empty:
    st.warning("Aucun point après filtrage.")
    st.stop()

# --------------------
# Carte GPS + colorbar à droite
# --------------------
def pr_to_rgba_clamped(pr: float, alpha: int, cmap) -> list[int]:
    """
    Map pr sur [70..130] avec clamp.
    """
    if pr is None or (isinstance(pr, float) and np.isnan(pr)):
        pr = MAP_PR_MIN

    pr_c = float(np.clip(pr, MAP_PR_MIN, MAP_PR_MAX))
    t = (pr_c - MAP_PR_MIN) / (MAP_PR_MAX - MAP_PR_MIN)  # 0..1
    r, g, b, _ = cmap(t)
    return [int(255 * r), int(255 * g), int(255 * b), alpha]

def build_map_frames(df_all: pd.DataFrame, df_filtered: pd.DataFrame):
    lat_col = "SilverData.GPS_Latitude"
    lon_col = "SilverData.GPS_Longitude"
    pr_col = "SilverData.PERF_BSP_PolarRatio"

    all_map = df_all.dropna(subset=["time_utc", lat_col, lon_col, pr_col]).copy()
    fil_map = df_filtered.dropna(subset=["time_utc", lat_col, lon_col, pr_col]).copy()

    # non retenues par filtres = all - filtered (par time_utc en ns)
    fil_times = set(fil_map["time_utc"].astype("int64"))
    all_map["_t"] = all_map["time_utc"].astype("int64")
    bg_map = all_map[~all_map["_t"].isin(fil_times)].copy()

    # standardize
    fil_map = fil_map.rename(columns={lat_col: "lat", lon_col: "lon", pr_col: "pr"})
    bg_map = bg_map.rename(columns={lat_col: "lat", lon_col: "lon", pr_col: "pr"})

    return bg_map, fil_map

bg_map, fil_map = build_map_frames(df_raw, df_f)

with right:
    st.markdown("**Cartographie GPS**")

    map_col, cbar_col = st.columns([1.0, 0.12], vertical_alignment="top")

    with map_col:
        if fil_map.empty and bg_map.empty:
            st.info("Pas de points GPS disponibles sur la sélection.")
        else:
            ref = fil_map if not fil_map.empty else bg_map
            center_lat = float(ref["lat"].mean())
            center_lon = float(ref["lon"].mean())
            view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0)

            cmap = plt.get_cmap()  # colormap matplotlib par défaut

            # BG: non retenus = noir
            if not bg_map.empty:
                bg_map = bg_map.copy()
                bg_map["color"] = bg_map["pr"].apply(lambda _: [0, 0, 0, 60])
            else:
                bg_map = pd.DataFrame(columns=["lat", "lon", "pr", "color"])

            # FG: retenus = colorés (pr clamped à [70..130])
            if not fil_map.empty:
                fil_map = fil_map.copy()
                fil_map["color"] = fil_map["pr"].apply(lambda v: pr_to_rgba_clamped(float(v), alpha=200, cmap=cmap))
            else:
                fil_map = pd.DataFrame(columns=["lat", "lon", "pr", "color"])

            layer_bg = pdk.Layer(
                "ScatterplotLayer",
                data=bg_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=MAP_RADIUS_BG_M,
                radius_units="meters",
                pickable=False,
            )
            layer_fg = pdk.Layer(
                "ScatterplotLayer",
                data=fil_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=MAP_RADIUS_FG_M,
                radius_units="meters",
                pickable=True,
            )

            deck = pdk.Deck(
                layers=[layer_bg, layer_fg],
                initial_view_state=view,
                tooltip={"text": "BSP_polarRatio: {pr}\nlat: {lat}\nlon: {lon}"},
            )
            st.pydeck_chart(deck, width="stretch")

    with cbar_col:
        # Colorbar de l'échelle 70..130
        fig_cb, ax_cb = plt.subplots(figsize=(1.0, 3.2))
        cmap = plt.get_cmap()
        norm = mpl.colors.Normalize(vmin=MAP_PR_MIN, vmax=MAP_PR_MAX)
        fig_cb.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb,
            label="BSP_polarRatio",
        )
        fig_cb.tight_layout()
        st.pyplot(fig_cb)
        plt.close(fig_cb)

# --------------------
# Choix X / Y / Color (après bin)
# --------------------
CHANNELS = {
    "Trim": "SilverData.AHRS_Trim",
    "abs(Heel)": "abs_heel",
    "BSP": "SilverData.BSP_BoatSpeed",
    "TWS": "SilverData.WIND_TWS",
    "TWA": "SilverData.WIND_TWA",
    "abs(TWA)": "abs_twa",
    "BSP_polarRatio": "SilverData.PERF_BSP_PolarRatio",
    "AWA": "SilverData.WIND_AWA",
    "AWS": "SilverData.WIND_AWS",
    "VMG": "SilverData.PERF_VMG",
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

options = [k for k in CHANNELS.keys() if k != bin_key]

default_x = "abs(TWA)" if "abs(TWA)" in options else options[0]
default_y = "abs(Heel)" if "abs(Heel)" in options else options[0]
default_c = "BSP_polarRatio" if "BSP_polarRatio" in options else options[0]

st.subheader("Scatter plot")

colXYC1, colXYC2, colXYC3 = st.columns(3)
with colXYC1:
    x_key = st.selectbox("X", options=options, index=options.index(default_x), key="custom_x_v5")
with colXYC2:
    y_key = st.selectbox("Y", options=options, index=options.index(default_y), key="custom_y_v5")
with colXYC3:
    c_key = st.selectbox("Color", options=options, index=options.index(default_c), key="custom_c_v5")

x_col = CHANNELS[x_key]
y_col = CHANNELS[y_key]
c_col = CHANNELS[c_key]

# couleur globale (pour les scatters)
c_min = float(df_f[c_col].min())
c_max = float(df_f[c_col].max())
norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
cmap = plt.get_cmap()

st.caption(f"Binning: **{bin_mode}** — X: **{x_key}** — Y: **{y_key}** — Color: **{c_key}**")

for i in range(0, len(bins_non_empty), 2):
    row_bins = bins_non_empty[i:i + 2]
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
