import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.db_client import fetch_aggregated

st.set_page_config(page_title="Comparaison voiles", layout="wide")
st.title("Comparaison voiles")

# --------------------
# Style helpers
# --------------------
SCATTER_S = 2
SCATTER_ALPHA = 0.4

def h_color(text: str, color: str, level: int = 3):
    st.markdown(
        f"<h{level} style='margin:0; padding:0; color:{color};'>{text}</h{level}>",
        unsafe_allow_html=True,
    )

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

def unique_sorted(series: pd.Series) -> list:
    s = series.dropna()
    if s.empty:
        return []
    return sorted(list(pd.unique(s)))

# --------------------
# SailOnDeck mapping
# --------------------
SAIL_ON_DECK_MAP = {
    1: "J1",
    2: "J2",
    3: "J3",
    4: "J4",
    5: "A1",
    6: "A5",
    7: "GNK",
    8: "A2",
    9: "A4",
    10: "FJT",
}

# 10 markers distincts (tous "fillables")
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', '*']
CODE_TO_MARKER = {code: MARKERS[i] for i, code in enumerate(sorted(SAIL_ON_DECK_MAP.keys()))}

def sail_name(code) -> str:
    try:
        c = int(code)
    except Exception:
        return str(code)
    return SAIL_ON_DECK_MAP.get(c, str(c))

# --------------------
# Champs à charger (10s)
# --------------------
FIELDS_MEAN = [
    "SilverData.WIND_TWA",
    "SilverData.BSP_BoatSpeed",
    "SilverData.WIND_TWS",
    "SilverData.PERF_BSP_PolarRatio",
]
FIELDS_LAST = [
    "SilverData.PERF_MainSail",
    "SilverData.PERF_UpwashTableSelected",
    "SilverData.PERF_SailOnDeck",
]

# --------------------
# UI : sélection des journées A / B
# --------------------
nav_days = ensure_nav_days()
if not nav_days:
    st.warning("Librairie des jours navigués indisponible. Ouvre d’abord la page d’accueil.")
    st.stop()

nav_labels = [format_date_fr(d) for d in nav_days]
label_to_day = {format_date_fr(d): d for d in nav_days}

h_color("Sélection des journées", "#FFD54A", level=3)
st.markdown("**1: sélectionner journées, 2: appliquer filtres communs, 3: appliquer filtres voiles (A/B)**")
st.markdown("**data lissées sur 10 secondes**")

# Defaults : A = dernier jour navigué, B = jour précédent si dispo sinon le même
default_A = [nav_labels[0]] if nav_labels else []
default_B = [nav_labels[1]] if len(nav_labels) > 1 else default_A

colA, colB = st.columns(2)

with colA:
    h_color("Data set A", "#2F6FED", level=3)
    daysA_labels = st.multiselect(
        "Journées (A) — coche au moins 1",
        options=nav_labels,
        default=default_A,
        key="sails_daysA",
    )

with colB:
    h_color("Data set B", "#E53935", level=3)
    daysB_labels = st.multiselect(
        "Journées (B) — coche au moins 1",
        options=nav_labels,
        default=default_B,
        key="sails_daysB",
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
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

if load_clicked:
    if len(daysA_labels) < 1 or len(daysB_labels) < 1:
        st.error("Il faut sélectionner au moins 1 journée pour Data set A et 1 journée pour Data set B.")
    else:
        with st.spinner("Chargement des données (moyenne 10s)…"):
            st.session_state["sails_dfA_raw"] = load_days_10s(daysA_labels)
            st.session_state["sails_dfB_raw"] = load_days_10s(daysB_labels)

        for k in ["sails_dfA_common", "sails_dfB_common", "sails_dfA_final", "sails_dfB_final"]:
            st.session_state.pop(k, None)

        st.success("Données chargées.")

dfA_raw = st.session_state.get("sails_dfA_raw", pd.DataFrame())
dfB_raw = st.session_state.get("sails_dfB_raw", pd.DataFrame())

# --------------------
# Niveau 1 : Filtres communs
# --------------------
h_color("Filtres communs", "#FFD54A", level=3)

colF1, colF2, colF3, colF4 = st.columns(4)
with colF1:
    twa_min, twa_max = st.slider("abs(TWA) — degrés", 0, 180, (0, 180), step=1, key="sails_twa_range")
with colF2:
    bsp_min, bsp_max = st.slider("BSP — nds", 0, 30, (0, 30), step=1, key="sails_bsp_range")
with colF3:
    tws_min, tws_max = st.slider("TWS — nds", 0, 40, (0, 40), step=1, key="sails_tws_range")
with colF4:
    pr_min, pr_max = st.slider("BSP_polarRatio", 0, 160, (70, 130), step=1, key="sails_pr_range")

apply_common = st.button("Appliquer filtres communs")

def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    needed = set(FIELDS_MEAN + FIELDS_LAST + ["time_utc"])
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    out = df.dropna(subset=["time_utc"] + FIELDS_MEAN).copy()
    out["abs_twa"] = out["SilverData.WIND_TWA"].abs()

    out = out[
        (out["abs_twa"] >= twa_min) & (out["abs_twa"] <= twa_max) &
        (out["SilverData.BSP_BoatSpeed"] >= bsp_min) & (out["SilverData.BSP_BoatSpeed"] <= bsp_max) &
        (out["SilverData.WIND_TWS"] >= tws_min) & (out["SilverData.WIND_TWS"] <= tws_max) &
        (out["SilverData.PERF_BSP_PolarRatio"] >= pr_min) & (out["SilverData.PERF_BSP_PolarRatio"] <= pr_max)
    ]
    return out

if apply_common:
    st.session_state["sails_dfA_common"] = apply_common_filters(dfA_raw)
    st.session_state["sails_dfB_common"] = apply_common_filters(dfB_raw)
    st.session_state.pop("sails_dfA_final", None)
    st.session_state.pop("sails_dfB_final", None)

dfA_common = st.session_state.get("sails_dfA_common", pd.DataFrame())
dfB_common = st.session_state.get("sails_dfB_common", pd.DataFrame())

# --------------------
# Niveau 2 : Choix combinaison de voiles (A et B séparés)
# --------------------
h_color("Choix combinaison de voiles", "#FFD54A", level=3)

mainA_vals = unique_sorted(dfA_common.get("SilverData.PERF_MainSail", pd.Series(dtype="object"))) if not dfA_common.empty else []
uwtA_vals  = unique_sorted(dfA_common.get("SilverData.PERF_UpwashTableSelected", pd.Series(dtype="object"))) if not dfA_common.empty else []
sodA_codes = unique_sorted(dfA_common.get("SilverData.PERF_SailOnDeck", pd.Series(dtype="float"))) if not dfA_common.empty else []
sodA_codes = [int(x) for x in sodA_codes if pd.notna(x)]
sodA_names = [sail_name(c) for c in sodA_codes]
name_to_code_A = {sail_name(c): c for c in sodA_codes}

mainB_vals = unique_sorted(dfB_common.get("SilverData.PERF_MainSail", pd.Series(dtype="object"))) if not dfB_common.empty else []
uwtB_vals  = unique_sorted(dfB_common.get("SilverData.PERF_UpwashTableSelected", pd.Series(dtype="object"))) if not dfB_common.empty else []
sodB_codes = unique_sorted(dfB_common.get("SilverData.PERF_SailOnDeck", pd.Series(dtype="float"))) if not dfB_common.empty else []
sodB_codes = [int(x) for x in sodB_codes if pd.notna(x)]
sodB_names = [sail_name(c) for c in sodB_codes]
name_to_code_B = {sail_name(c): c for c in sodB_codes}

colSA, colSB = st.columns(2)

with colSA:
    h_color("Data set A", "#2F6FED", level=4)
    mainA_sel = st.multiselect("GV — mainSail (A)", options=mainA_vals, default=mainA_vals, key="sails_mainA_sel")
    uwtA_sel  = st.multiselect("sail_code_UWT (A)", options=uwtA_vals, default=uwtA_vals, key="sails_uwtA_sel")
    sodA_sel_names = st.multiselect("SailOnDeck (A)", options=sodA_names, default=sodA_names, key="sails_sodA_sel")

with colSB:
    h_color("Data set B", "#E53935", level=4)
    mainB_sel = st.multiselect("GV — mainSail (B)", options=mainB_vals, default=mainB_vals, key="sails_mainB_sel")
    uwtB_sel  = st.multiselect("sail_code_UWT (B)", options=uwtB_vals, default=uwtB_vals, key="sails_uwtB_sel")
    sodB_sel_names = st.multiselect("SailOnDeck (B)", options=sodB_names, default=sodB_names, key="sails_sodB_sel")

apply_sails = st.button("Appliquer filtres voiles (A/B)")

def apply_sail_filters(df: pd.DataFrame, main_sel: list, uwt_sel: list, sod_sel_names: list, name_to_code: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if main_sel is not None:
        if len(main_sel) == 0:
            return out.iloc[0:0]
        out = out[out["SilverData.PERF_MainSail"].isin(main_sel)]

    if uwt_sel is not None:
        if len(uwt_sel) == 0:
            return out.iloc[0:0]
        out = out[out["SilverData.PERF_UpwashTableSelected"].isin(uwt_sel)]

    if sod_sel_names is not None:
        if len(sod_sel_names) == 0:
            return out.iloc[0:0]
        sod_codes = [name_to_code[n] for n in sod_sel_names if n in name_to_code]
        out = out[out["SilverData.PERF_SailOnDeck"].fillna(-999).astype(int).isin(sod_codes)]

    return out

if apply_sails:
    st.session_state["sails_dfA_final"] = apply_sail_filters(dfA_common, mainA_sel, uwtA_sel, sodA_sel_names, name_to_code_A)
    st.session_state["sails_dfB_final"] = apply_sail_filters(dfB_common, mainB_sel, uwtB_sel, sodB_sel_names, name_to_code_B)

dfA = st.session_state.get("sails_dfA_final", dfA_common)
dfB = st.session_state.get("sails_dfB_final", dfB_common)

# --------------------
# Résultats + graphes
# --------------------
h_color("Résultats", "#FFD54A", level=3)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Points A (bruts)", f"{len(dfA_raw):,}")
m2.metric("Points A (filtrés)", f"{len(dfA):,}")
m3.metric("Points B (bruts)", f"{len(dfB_raw):,}")
m4.metric("Points B (filtrés)", f"{len(dfB):,}")

if dfA.empty and dfB.empty:
    st.info("1) Sélectionner journées, 2) appliquer filtres communs, 3) appliquer filtres voiles (A/B).")
    st.stop()

# ---- Plot 1
h_color("BSP_polarRatio vs TWS", "#FFD54A", level=3)
fig1 = plt.figure()
if not dfA.empty:
    plt.scatter(dfA["SilverData.WIND_TWS"], dfA["SilverData.PERF_BSP_PolarRatio"],
                s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Data set A")
if not dfB.empty:
    plt.scatter(dfB["SilverData.WIND_TWS"], dfB["SilverData.PERF_BSP_PolarRatio"],
                s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Data set B")
plt.xlabel("TWS (nds)")
plt.ylabel("BSP_polarRatio")
plt.title("Daguet 5 — BSP_polarRatio vs TWS")
plt.legend()
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ---- Plot 2 + Plot 3 helpers
def polar_base(ax, outer):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_rlim(0, outer)
    ax.grid(False)

    lw_grid = 0.45
    theta_grid = np.linspace(0, 2 * np.pi, 721)

    for r in range(2, outer + 1, 2):
        ls = "-" if (r % 10 == 0) else "--"
        ax.plot(theta_grid, np.full_like(theta_grid, r), color="black", linestyle=ls, linewidth=lw_grid, alpha=0.9)

    for deg in range(0, 360, 10):
        th = np.deg2rad(deg)
        ax.plot([th, th], [0, outer], color="black", linestyle="-", linewidth=0.35, alpha=0.6)

    twa_tick_labels = list(range(-180, 181, 30))
    theta_ticks = [np.deg2rad((t + 360) % 360) for t in twa_tick_labels]
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([str(t) for t in twa_tick_labels])

    yticks = list(range(0, outer + 1, 10))
    ax.set_yticks(yticks)
    ax.set_rlabel_position(0)

# Compute outer
bspA = dfA["SilverData.BSP_BoatSpeed"].to_numpy() if not dfA.empty else np.array([])
bspB = dfB["SilverData.BSP_BoatSpeed"].to_numpy() if not dfB.empty else np.array([])
max_bsp = 0.0
if bspA.size: max_bsp = max(max_bsp, float(np.nanmax(bspA)))
if bspB.size: max_bsp = max(max_bsp, float(np.nanmax(bspB)))
outer = int(max_bsp) + 1 if max_bsp > 0 else 1

# ---- Plot 2 (simple)
h_color("BSP vs TWA (polaire)", "#FFD54A", level=3)
twaA = dfA["SilverData.WIND_TWA"].to_numpy() if not dfA.empty else np.array([])
twaB = dfB["SilverData.WIND_TWA"].to_numpy() if not dfB.empty else np.array([])
thetaA = np.deg2rad((twaA + 360.0) % 360.0) if twaA.size else np.array([])
thetaB = np.deg2rad((twaB + 360.0) % 360.0) if twaB.size else np.array([])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="polar")
polar_base(ax2, outer)
if thetaA.size:
    ax2.scatter(thetaA, bspA, s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Data set A")
if thetaB.size:
    ax2.scatter(thetaB, bspB, s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Data set B")
ax2.set_title("BSP (rayon) vs TWA (angle)", pad=15)
ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ---- Plot 3 (marker = SailOnDeck)
h_color("BSP vs TWA (polaire) — marker = SailOnDeck", "#FFD54A", level=3)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection="polar")
polar_base(ax3, outer)

def plot_by_sail(df, color):
    if df.empty:
        return []
    codes = df["SilverData.PERF_SailOnDeck"].dropna().astype(int).unique().tolist()
    used = []
    for code in sorted(codes):
        sub = df[df["SilverData.PERF_SailOnDeck"].fillna(-999).astype(int) == int(code)]
        if sub.empty:
            continue
        twa = sub["SilverData.WIND_TWA"].to_numpy()
        bsp = sub["SilverData.BSP_BoatSpeed"].to_numpy()
        th = np.deg2rad((twa + 360.0) % 360.0)
        marker = CODE_TO_MARKER.get(int(code), "o")

        # Points pleins
        ax3.scatter(
            th, bsp,
            s=12,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=0.2,
            alpha=SCATTER_ALPHA,
        )
        used.append(int(code))
    return used

usedA = plot_by_sail(dfA, "blue")
usedB = plot_by_sail(dfB, "red")
used_codes = sorted(set(usedA) | set(usedB))

# Légendes compactes
legend_dataset = [
    Line2D([0], [0], marker='o', color='w', label='Data set A', markerfacecolor='blue', markeredgecolor='blue', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Data set B', markerfacecolor='red', markeredgecolor='red', markersize=5),
]
legend_sails = [
    Line2D([0], [0], marker=CODE_TO_MARKER.get(code, 'o'), color='black',
           label=SAIL_ON_DECK_MAP.get(code, str(code)), linestyle='None', markersize=5)
    for code in used_codes
]

ax3.set_title("Couleur = dataset, forme = SailOnDeck", pad=15)
leg1 = ax3.legend(handles=legend_dataset, loc="upper right", bbox_to_anchor=(1.22, 1.10), fontsize=8)
ax3.add_artist(leg1)

# Légende voiles : petite, en 2 colonnes, placée plus à l’extérieur
if legend_sails:
    ax3.legend(handles=legend_sails, loc="lower right", bbox_to_anchor=(1.32, -0.10), ncol=2, fontsize=8)

plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

with st.expander("Aperçu données filtrées (A/B)", expanded=False):
    h_color("Data set A (filtré)", "#2F6FED", level=4)
    if not dfA.empty:
        showA = dfA.copy()
        showA["SailOnDeck"] = showA["SilverData.PERF_SailOnDeck"].apply(sail_name)
        st.dataframe(
            showA[[
                "time_utc",
                "SilverData.WIND_TWS",
                "SilverData.PERF_BSP_PolarRatio",
                "SilverData.BSP_BoatSpeed",
                "SilverData.WIND_TWA",
                "SilverData.PERF_MainSail",
                "SilverData.PERF_UpwashTableSelected",
                "SailOnDeck",
            ]].head(200),
            width="stretch",
        )

    h_color("Data set B (filtré)", "#E53935", level=4)
    if not dfB.empty:
        showB = dfB.copy()
        showB["SailOnDeck"] = showB["SilverData.PERF_SailOnDeck"].apply(sail_name)
        st.dataframe(
            showB[[
                "time_utc",
                "SilverData.WIND_TWS",
                "SilverData.PERF_BSP_PolarRatio",
                "SilverData.BSP_BoatSpeed",
                "SilverData.WIND_TWA",
                "SilverData.PERF_MainSail",
                "SilverData.PERF_UpwashTableSelected",
                "SailOnDeck",
            ]].head(200),
            width="stretch",
        )
