import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydeck as pdk

from lib.db_client import fetch_aggregated, run_influxql, parse_series_to_df

st.set_page_config(page_title="Custom", layout="wide")
st.title("Custom")

# --------------------
# Constantes / helpers
# --------------------
MEASUREMENT = "C54"
SCATTER_S = 14
SCATTER_ALPHA = 0.45

MAP_RADIUS_BG_M = 2
MAP_RADIUS_FG_M = 3
MAP_PR_MIN = 70.0
MAP_PR_MAX = 130.0

MOIS_FR = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

BASE_FIELDS_MEAN = [
    "SilverData.WIND_TWA",
    "SilverData.BSP_BoatSpeed",
    "SilverData.WIND_TWS",
    "SilverData.PERF_BSP_PolarRatio",
    "SilverData.AHRS_Heel",
    "SilverData.AHRS_Trim",
    "SilverData.GPS_Latitude",
    "SilverData.GPS_Longitude",
]

BASE_FIELDS_LAST = [
    "SilverData.PERF_MainSail",
    "SilverData.PERF_StaySail",
    "SilverData.PERF_Jib",
    "SilverData.PERF_HeadSail",
]

COMBO_SOURCE_COLS = [
    "SilverData.PERF_MainSail",
    "SilverData.PERF_StaySail",
    "SilverData.PERF_Jib",
    "SilverData.PERF_HeadSail",
]

# Alias utiles en tête de liste
FRIENDLY_CHANNELS = {
    "BSP": "SilverData.BSP_BoatSpeed",
    "TWS": "SilverData.WIND_TWS",
    "TWA": "SilverData.WIND_TWA",
    "abs(TWA)": "__abs_twa__",
    "BSP_polarRatio": "SilverData.PERF_BSP_PolarRatio",
    "Trim": "SilverData.AHRS_Trim",
    "abs(Heel)": "__abs_heel__",
    "Combo": "__combo_id__",
}

def format_date_fr(ts_utc: pd.Timestamp) -> str:
    return f"{ts_utc.day} {MOIS_FR[ts_utc.month]} {ts_utc.year}"

def day_to_range_utc(day_utc: pd.Timestamp) -> tuple[str, str]:
    start = pd.Timestamp(day_utc).floor("D")
    end = start + pd.Timedelta(days=1)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

def ensure_nav_days() -> list[pd.Timestamp]:
    nav_days = st.session_state.get("nav_days", [])
    return sorted(nav_days, reverse=True)

def clean_sail(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, float) and np.isnan(x):
        return "none"
    s = str(x).strip()
    if not s:
        return "none"
    s_low = s.lower()
    if s_low == "none" or s_low.startswith("nan") or s_low.startswith("none"):
        return "none"
    return s

def combo_tuple_from_row(row: pd.Series) -> tuple[str, str, str, str]:
    return tuple(clean_sail(row.get(c, "none")) for c in COMBO_SOURCE_COLS)

def build_abs_twa_bins_10deg() -> list[tuple[float, float]]:
    edges = list(range(35, 166, 10))
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

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

def fmt_range(lo: float, hi: float, unit: str) -> str:
    if unit in ("nds", "deg"):
        return f"{int(lo)} à {int(hi)} {unit}"
    return f"{lo} à {hi}"

def rgba_to_css(rgba, alpha=0.95):
    r, g, b = rgba[:3]
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"

def build_combo_palette(n: int):
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for cm in cmaps:
        colors.extend([cm(i) for i in range(cm.N)])
    if n <= len(colors):
        return colors[:n]
    rep = int(np.ceil(n / len(colors)))
    return (colors * rep)[:n]

# --------------------
# Détection des fields disponibles
# --------------------
@st.cache_data(show_spinner=False)
def show_all_field_keys() -> pd.DataFrame:
    q = f'SHOW FIELD KEYS FROM "{MEASUREMENT}"'
    data = run_influxql(q)
    df = parse_series_to_df(data)
   
    if df.empty:
        return pd.DataFrame(columns=["fieldKey", "fieldType"])
    return df[["fieldKey", "fieldType"]].dropna().drop_duplicates().reset_index(drop=True)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=False)
def detect_present_fields_for_window(start_iso: str, end_iso: str) -> list[str]:
    meta = show_all_field_keys()
    all_fields = meta["fieldKey"].astype(str).tolist()
    present = []

    for chunk in chunked(all_fields, 25):
        select = ", ".join([f'COUNT("{f}") AS "{f}"' for f in chunk])
        q = (
            f'SELECT {select} FROM "{MEASUREMENT}" '
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}'"
        )

        data = run_influxql(q)
        df = parse_series_to_df(data)

        if df.empty:
            continue

        # 🔥 FIX PRINCIPAL ICI
        for f in chunk:
            if f not in df.columns:
                continue

            col = df[f]

            # cas colonnes dupliquées -> DataFrame
            if isinstance(col, pd.DataFrame):
                values = col.iloc[0].values
            else:
                values = [col.iloc[0]]

            for v in values:
                try:
                    if pd.notna(v) and float(v) > 0:
                        present.append(f)
                        break
                except Exception:
                    continue

    return sorted(list(dict.fromkeys(present)))

@st.cache_data(show_spinner=False)
def detect_present_fields_union(windows: tuple[tuple[str, str], ...]) -> list[str]:
    union = set()

    for start_iso, end_iso in windows:
        fields = detect_present_fields_for_window(start_iso, end_iso)
        if isinstance(fields, list):
            union.update(fields)

    return sorted(list(union))

# --------------------
# Chargement data
# --------------------
def field_type_map() -> dict[str, str]:
    meta = show_all_field_keys()
    return dict(zip(meta["fieldKey"].astype(str), meta["fieldType"].astype(str)))

@st.cache_data(show_spinner=False)
def load_data_for_windows(windows: tuple[tuple[str, str], ...], fields_mean: tuple[str, ...], fields_last: tuple[str, ...]) -> pd.DataFrame:
    dfs = []
    for start_iso, end_iso in windows:
        df = fetch_aggregated(
            fields_mean=list(fields_mean),
            fields_last=list(fields_last),
            start_utc_iso=start_iso,
            end_utc_iso=end_iso,
            bucket="10s",
        )
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out

# --------------------
# Jours + plage horaire
# --------------------
nav_days = ensure_nav_days()
if not nav_days:
    st.warning("Librairie des jours navigués indisponible. Ouvre d’abord la page d’accueil.")
    st.stop()

nav_labels = [format_date_fr(d) for d in nav_days]
label_to_day = {format_date_fr(d): d for d in nav_days}

st.markdown("**1: sélectionner journées, 2: si 1 seule journée est choisie, régler la plage horaire UTC, 3: appliquer filtres**")
st.markdown("**data lissées sur 10 secondes**")

days_labels = st.multiselect(
    "Journées — coche au moins 1",
    options=nav_labels,
    default=[nav_labels[0]] if nav_labels else [],
    key="custom_days_allchannels",
)

single_day = len(days_labels) == 1

if single_day:
    st.markdown("**Plage horaire UTC (journée unique)**")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        t_start = st.time_input("Heure début UTC",value=dt.time(0, 0, 0),step=60,key="custom_time_start")
    with tcol2:
        t_end = st.time_input("Heure fin UTC",value=dt.time(23, 59, 0),step=60,key="custom_time_end")
else:
    t_start = None
    t_end = None

load_clicked = st.button("Sélectionner journées", type="primary")

def build_windows(selected_labels: list[str], t_start=None, t_end=None) -> list[tuple[str, str]]:
    windows = []
    if len(selected_labels) == 1 and t_start is not None and t_end is not None:
        day = label_to_day[selected_labels[0]]
        day0 = pd.Timestamp(day).floor("D")
        start_dt = pd.Timestamp.combine(day0.date(), t_start).tz_localize("UTC")
        end_dt = pd.Timestamp.combine(day0.date(), t_end).tz_localize("UTC")
        if end_dt <= start_dt:
            raise ValueError("L'heure de fin doit être strictement après l'heure de début.")
        windows.append((
            start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ))
    else:
        for lab in selected_labels:
            day = label_to_day[lab]
            windows.append(day_to_range_utc(day))
    return windows

if load_clicked:
    if not days_labels:
        st.error("Sélectionne au moins une journée.")
        st.stop()

    try:
        windows = build_windows(days_labels, t_start, t_end)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Détection des channels présents sur la période…"):
        present_fields = detect_present_fields_union(tuple(windows))

    st.session_state["custom_windows_allchannels"] = windows
    st.session_state["custom_present_fields"] = present_fields
    st.session_state.pop("custom_df_filtered_allchannels", None)
    st.success(f"{len(present_fields)} channels détectés sur la période.")

windows = st.session_state.get("custom_windows_allchannels")
present_fields = st.session_state.get("custom_present_fields", [])

if not windows or not present_fields:
    st.info("Sélectionne des journées puis clique sur **Sélectionner journées**.")
    st.stop()

# --------------------
# Filtres + carte à droite
# --------------------
st.subheader("Filtres")

left, right = st.columns([1.0, 1.35], vertical_alignment="top")

with left:
    twa_min, twa_max = st.slider("abs(TWA) — degrés", 0, 180, (0, 180), step=1, key="custom_twa_allchannels")
    bsp_min, bsp_max = st.slider("BSP — nds", 0, 30, (0, 30), step=1, key="custom_bsp_allchannels")
    tws_min, tws_max = st.slider("TWS — nds", 0, 40, (0, 40), step=1, key="custom_tws_allchannels")
    pr_min, pr_max = st.slider("BSP_polarRatio", 0, 160, (70, 130), step=1, key="custom_pr_allchannels")

    bin_mode = st.radio("Bin", options=["TWS", "BSP", "abs(TWA)"], horizontal=True, index=0, key="custom_bin_mode_allchannels")

    st.markdown("**Jitter**")
    j1, j2 = st.columns(2)
    with j1:
        jitter_x = st.checkbox("Jitter X", value=False, key="custom_jitter_x")
        jitter_x_amp = st.number_input("Amplitude X", value=0.15, min_value=0.0, step=0.05, key="custom_jitter_x_amp")
    with j2:
        jitter_y = st.checkbox("Jitter Y", value=False, key="custom_jitter_y")
        jitter_y_amp = st.number_input("Amplitude Y", value=0.15, min_value=0.0, step=0.05, key="custom_jitter_y_amp")

# Construire options channels
all_option_labels = list(FRIENDLY_CHANNELS.keys())
extra_raw = [f for f in present_fields if f not in FRIENDLY_CHANNELS.values() and f not in COMBO_SOURCE_COLS]
all_option_labels += extra_raw

# éviter doublons
all_option_labels = list(dict.fromkeys(all_option_labels))

# X/Y/Color
st.subheader("Scatter plot")

colXYC1, colXYC2, colXYC3 = st.columns(3)
with colXYC1:
    x_key = st.selectbox("X", options=all_option_labels, index=all_option_labels.index("abs(TWA)") if "abs(TWA)" in all_option_labels else 0, key="custom_x_allchannels")
with colXYC2:
    y_key = st.selectbox("Y", options=all_option_labels, index=all_option_labels.index("abs(Heel)") if "abs(Heel)" in all_option_labels else 0, key="custom_y_allchannels")
with colXYC3:
    c_key = st.selectbox("Color", options=all_option_labels, index=all_option_labels.index("BSP_polarRatio") if "BSP_polarRatio" in all_option_labels else 0, key="custom_c_allchannels")

apply_filters = st.button("Appliquer filtres", key="custom_apply_allchannels")

# Déduire fields à charger
ftypes = field_type_map()

def resolve_option_to_field(opt: str) -> str:
    if opt in FRIENDLY_CHANNELS:
        return FRIENDLY_CHANNELS[opt]
    return opt

def build_fields_to_load(x_key, y_key, c_key):
    requested = {resolve_option_to_field(x_key), resolve_option_to_field(y_key), resolve_option_to_field(c_key)}
    fields_mean = set(BASE_FIELDS_MEAN)
    fields_last = set(BASE_FIELDS_LAST)

    for f in requested:
        if f in ["__abs_twa__", "__abs_heel__", "__combo_id__"]:
            continue
        ftype = ftypes.get(f, "float")
        if ftype.lower() == "string":
            fields_last.add(f)
        else:
            fields_mean.add(f)

    return tuple(sorted(fields_mean)), tuple(sorted(fields_last))

if apply_filters or ("custom_df_filtered_allchannels" not in st.session_state):
    fields_mean, fields_last = build_fields_to_load(x_key, y_key, c_key)

    with st.spinner("Chargement des données utiles…"):
        df_raw = load_data_for_windows(tuple(windows), fields_mean, fields_last)

    if df_raw.empty:
        st.error("Aucune donnée chargée.")
        st.stop()

    # normalisation strings sailcodes
    for c in COMBO_SOURCE_COLS:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].apply(clean_sail)

    # colonnes dérivées
    df_raw["abs_twa"] = df_raw["SilverData.WIND_TWA"].abs()
    df_raw["abs_heel"] = df_raw["SilverData.AHRS_Heel"].abs()
    df_raw["combo_tuple"] = df_raw.apply(combo_tuple_from_row, axis=1)

    uniq_combos = list(dict.fromkeys(df_raw["combo_tuple"].tolist()))
    combo_id_map = {t: i + 1 for i, t in enumerate(uniq_combos)}
    
    df_raw["combo_id"] = df_raw["combo_tuple"].apply(lambda t: combo_id_map.get(t)).astype(float)

    # filtres communs
    df_f = df_raw.dropna(subset=[
        "time_utc",
        "SilverData.WIND_TWA",
        "SilverData.BSP_BoatSpeed",
        "SilverData.WIND_TWS",
        "SilverData.PERF_BSP_PolarRatio",
    ]).copy()

    df_f = df_f[
        (df_f["abs_twa"] >= twa_min) & (df_f["abs_twa"] <= twa_max) &
        (df_f["SilverData.BSP_BoatSpeed"] >= bsp_min) & (df_f["SilverData.BSP_BoatSpeed"] <= bsp_max) &
        (df_f["SilverData.WIND_TWS"] >= tws_min) & (df_f["SilverData.WIND_TWS"] <= tws_max) &
        (df_f["SilverData.PERF_BSP_PolarRatio"] >= pr_min) & (df_f["SilverData.PERF_BSP_PolarRatio"] <= pr_max)
    ].copy()

    st.session_state["custom_df_raw_allchannels"] = df_raw
    st.session_state["custom_df_filtered_allchannels"] = df_f
    st.session_state["custom_combo_id_map"] = combo_id_map
    st.session_state["custom_uniq_combos"] = uniq_combos

df_raw = st.session_state.get("custom_df_raw_allchannels", pd.DataFrame())
df_f = st.session_state.get("custom_df_filtered_allchannels", pd.DataFrame())
combo_id_map = st.session_state.get("custom_combo_id_map", {})
uniq_combos = st.session_state.get("custom_uniq_combos", [])

if df_f.empty:
    st.warning("Aucun point après filtrage.")
    st.stop()

# palette combos
palette = build_combo_palette(len(uniq_combos))
combo_color_rgba = {combo_id_map[t]: palette[i] for i, t in enumerate(uniq_combos)}
combo_color_css = {cid: rgba_to_css(combo_color_rgba[cid], alpha=0.95) for cid in combo_color_rgba.keys()}

# --------------------
# Carte GPS + colorbar
# --------------------
def pr_to_rgba_clamped(pr: float, alpha: int, cmap) -> list[int]:
    if pr is None or (isinstance(pr, float) and np.isnan(pr)):
        pr = MAP_PR_MIN
    pr_c = float(np.clip(pr, MAP_PR_MIN, MAP_PR_MAX))
    t = (pr_c - MAP_PR_MIN) / (MAP_PR_MAX - MAP_PR_MIN)
    r, g, b, _ = cmap(t)
    return [int(255 * r), int(255 * g), int(255 * b), alpha]

def build_map_frames(df_all: pd.DataFrame, df_filtered: pd.DataFrame):
    lat_col = "SilverData.GPS_Latitude"
    lon_col = "SilverData.GPS_Longitude"
    pr_col = "SilverData.PERF_BSP_PolarRatio"

    all_map = df_all.dropna(subset=["time_utc", lat_col, lon_col, pr_col]).copy()
    fil_map = df_filtered.dropna(subset=["time_utc", lat_col, lon_col, pr_col]).copy()

    fil_times = set(fil_map["time_utc"].astype("int64"))
    all_map["_t"] = all_map["time_utc"].astype("int64")
    bg_map = all_map[~all_map["_t"].isin(fil_times)].copy()

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

            cmap_map = plt.get_cmap()

            if not bg_map.empty:
                bg_map = bg_map.copy()
                bg_map["color"] = bg_map["pr"].apply(lambda _: [0, 0, 0, 90])
            else:
                bg_map = pd.DataFrame(columns=["lat", "lon", "pr", "color"])

            if not fil_map.empty:
                fil_map = fil_map.copy()
                fil_map["color"] = fil_map["pr"].apply(lambda v: pr_to_rgba_clamped(float(v), alpha=210, cmap=cmap_map))
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
        fig_cb, ax_cb = plt.subplots(figsize=(1.0, 3.2))
        cmap_map = plt.get_cmap()
        norm_map = mpl.colors.Normalize(vmin=MAP_PR_MIN, vmax=MAP_PR_MAX)
        fig_cb.colorbar(
            mpl.cm.ScalarMappable(norm=norm_map, cmap=cmap_map),
            cax=ax_cb,
            label="BSP_polarRatio",
        )
        fig_cb.tight_layout()
        st.pyplot(fig_cb)
        plt.close(fig_cb)

# --------------------
# Bins
# --------------------
if bin_mode == "TWS":
    bin_col = "SilverData.WIND_TWS"
    unit = "nds"
    bins = build_bins_2units_with_optional_first_1(
        int(float(df_f[bin_col].min())),
        int(float(df_f[bin_col].max())) + 1
    )
elif bin_mode == "BSP":
    bin_col = "SilverData.BSP_BoatSpeed"
    unit = "nds"
    bins = build_bins_2units_with_optional_first_1(
        int(float(df_f[bin_col].min())),
        int(float(df_f[bin_col].max())) + 1
    )
else:
    bin_col = "abs_twa"
    unit = "deg"
    bins = build_abs_twa_bins_10deg()

bins_non_empty = []
for lo, hi in bins:
    if not df_f[(df_f[bin_col] >= lo) & (df_f[bin_col] < hi)].empty:
        bins_non_empty.append((lo, hi))

if not bins_non_empty:
    st.warning("Aucun bin non vide avec les filtres actuels.")
    st.stop()

# --------------------
# Préparer colonnes X/Y/Color
# --------------------
def resolve_channel(opt_label: str) -> str:
    return FRIENDLY_CHANNELS.get(opt_label, opt_label)

x_col = resolve_channel(x_key)
y_col = resolve_channel(y_key)
c_col = resolve_channel(c_key)

# Encodage catégories éventuelles
mapping_tables = []

def ensure_numeric_plot_col(df: pd.DataFrame, source_col: str, prefix: str):
    """
    Retourne (colonne numérique à utiliser, mapping_df ou None)
    """
    if source_col == "__combo_id__":
        return "combo_id", pd.DataFrame({
            "combo_id": list(combo_id_map.values()),
            "Main": [t[0] for t in combo_id_map.keys()],
            "Stay": [t[1] for t in combo_id_map.keys()],
            "Jib": [t[2] for t in combo_id_map.keys()],
            "Head": [t[3] for t in combo_id_map.keys()],
        }).sort_values("combo_id").reset_index(drop=True)

    if source_col == "__abs_twa__":
        return "abs_twa", None
    if source_col == "__abs_heel__":
        return "abs_heel", None

    if source_col not in df.columns:
        raise KeyError(f"Channel absent du dataframe: {source_col}")

    if pd.api.types.is_numeric_dtype(df[source_col]):
        return source_col, None

    # catégoriel quelconque -> codes
    vals = df[source_col].fillna("none").astype(str).map(clean_sail)
    uniq = list(dict.fromkeys(vals.tolist()))
    mapping = {v: i + 1 for i, v in enumerate(uniq)}
    code_col = f"__{prefix}_code__"
    df[code_col] = vals.map(mapping).astype(float)
    map_df = pd.DataFrame({
        f"{prefix}_id": list(mapping.values()),
        "value": list(mapping.keys()),
    }).sort_values(f"{prefix}_id").reset_index(drop=True)
    return code_col, map_df

df_plot = df_f.copy()

x_plot_col, x_map_df = ensure_numeric_plot_col(df_plot, x_col, "x")
y_plot_col, y_map_df = ensure_numeric_plot_col(df_plot, y_col, "y")
c_plot_col, c_map_df = ensure_numeric_plot_col(df_plot, c_col, "color")

if x_map_df is not None:
    mapping_tables.append(("Mapping X", x_map_df))
if y_map_df is not None:
    mapping_tables.append(("Mapping Y", y_map_df))
if c_map_df is not None and c_key != "Combo":
    mapping_tables.append(("Mapping Color", c_map_df))

# colormap / norm
if c_key == "Combo":
    n = len(uniq_combos)
    cmap = mpl.colors.ListedColormap([combo_color_rgba[i + 1] for i in range(n)])
    bounds = np.arange(0.5, n + 1.5, 1.0)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
else:
    c_min = float(df_plot[c_plot_col].min())
    c_max = float(df_plot[c_plot_col].max())
    norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
    cmap = plt.get_cmap()

# jitter
rng = np.random.default_rng(42)

def add_jitter(arr: np.ndarray, enabled: bool, amp: float):
    if not enabled or amp <= 0:
        return arr
    return arr + rng.uniform(-amp, amp, size=len(arr))

# --------------------
# Plots
# --------------------
for i in range(0, len(bins_non_empty), 2):
    row_bins = bins_non_empty[i:i + 2]
    c1, c2, ccb = st.columns([1, 1, 0.18])
    cols = [c1, c2]
    have_any = False

    for j, (lo, hi) in enumerate(row_bins):
        sub = df_plot[(df_plot[bin_col] >= lo) & (df_plot[bin_col] < hi)].copy()
        if sub.empty:
            continue

        x_vals = sub[x_plot_col].to_numpy(dtype=float)
        y_vals = sub[y_plot_col].to_numpy(dtype=float)
        c_vals = sub[c_plot_col].to_numpy(dtype=float)

        x_vals = add_jitter(x_vals, jitter_x, jitter_x_amp)
        y_vals = add_jitter(y_vals, jitter_y, jitter_y_amp)

        fig = plt.figure()
        plt.scatter(
            x_vals,
            y_vals,
            s=SCATTER_S,
            alpha=SCATTER_ALPHA,
            c=c_vals,
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

# --------------------
# Tableau Combo coloré
# --------------------
st.subheader("Mapping Combo (codes / voiles)")

combo_rows = []
for combo_tup, cid in combo_id_map.items():
    main, stay, jib, head = combo_tup
    combo_rows.append({
        "combo_id": cid,
        "Main": main,
        "Stay": stay,
        "Jib": jib,
        "Head": head,
    })
df_combo = pd.DataFrame(combo_rows).sort_values("combo_id").reset_index(drop=True)

def style_combo_row(row):
    cid = int(row["combo_id"])
    css = combo_color_css.get(cid, "rgba(255,255,255,0)")
    return [f"background-color: {css}; color: black;"] * len(row)

st.dataframe(df_combo.style.apply(style_combo_row, axis=1), width="stretch", hide_index=True)

# autres mappings éventuels
for title, map_df in mapping_tables:
    with st.expander(title, expanded=False):
        st.dataframe(map_df, width="stretch", hide_index=True)

with st.expander("Aperçu données filtrées", expanded=False):
    preview_cols = [
        "time_utc", "day_utc",
        "SilverData.BSP_BoatSpeed",
        "SilverData.WIND_TWS",
        "SilverData.WIND_TWA",
        "abs_twa",
        "SilverData.PERF_BSP_PolarRatio",
        "SilverData.AHRS_Trim",
        "abs_heel",
        "combo_id",
    ]
    keep = [c for c in preview_cols if c in df_plot.columns]
    st.dataframe(df_plot[keep].head(500), width="stretch", hide_index=True)