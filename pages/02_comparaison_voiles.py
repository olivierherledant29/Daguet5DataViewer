import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.db_client import fetch_aggregated

st.set_page_config(page_title="Comparaison voiles", layout="wide")
st.title("Comparaison voiles")

SCATTER_S = 2
SCATTER_ALPHA = 0.4
PLOT3_ALPHA = 0.85

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

def clean_sail(x) -> str | None:
    """
    Normalise les sailcodes:
    - None/NaN/"" -> None
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

FIELDS_MEAN = [
    "SilverData.WIND_TWA",
    "SilverData.BSP_BoatSpeed",
    "SilverData.WIND_TWS",
    "SilverData.PERF_BSP_PolarRatio",
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

# --------------------
# Sélection jours A/B
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

default_A = [nav_labels[0]] if nav_labels else []
default_B = [nav_labels[1]] if len(nav_labels) > 1 else default_A

colA, colB = st.columns(2)
with colA:
    h_color("Data set A", "#2F6FED", level=3)
    daysA_labels = st.multiselect("Journées (A) — coche au moins 1", nav_labels, default=default_A, key="sails_daysA")
with colB:
    h_color("Data set B", "#E53935", level=3)
    daysB_labels = st.multiselect("Journées (B) — coche au moins 1", nav_labels, default=default_B, key="sails_daysB")

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
            # nettoyage sailcodes dès le chargement
            for c in POS_COLS.values():
                if c in df.columns:
                    df[c] = df[c].apply(lambda x: clean_sail(x) if clean_sail(x) is not None else "none")
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
# Filtres communs
# --------------------
h_color("Filtres communs", "#FFD54A", level=3)
c1, c2, c3, c4 = st.columns(4)
with c1:
    twa_min, twa_max = st.slider("abs(TWA) — degrés", 0, 180, (0, 180), 1, key="sails_twa_range")
with c2:
    bsp_min, bsp_max = st.slider("BSP — nds", 0, 30, (0, 30), 1, key="sails_bsp_range")
with c3:
    tws_min, tws_max = st.slider("TWS — nds", 0, 40, (0, 40), 1, key="sails_tws_range")
with c4:
    pr_min, pr_max = st.slider("BSP_polarRatio", 0, 160, (70, 130), 1, key="sails_pr_range")

apply_common = st.button("Appliquer filtres communs")

def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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
# Filtres voiles utilisées (4 positions) A/B
# --------------------
h_color("Choix voiles utilisées", "#FFD54A", level=3)

def build_options(df: pd.DataFrame, col: str) -> list[str]:
    if df.empty or col not in df.columns:
        return ["All"]
    vals = sorted(list(dict.fromkeys([str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()])))
    return ["All"] + vals

colSA, colSB = st.columns(2)

with colSA:
    h_color("Data set A", "#2F6FED", level=4)
    mainA_sel = st.multiselect("Main (A)", options=build_options(dfA_common, POS_COLS["Main"]), default=["All"], key="mainA_sel")
    stayA_sel = st.multiselect("StaySail (A)", options=build_options(dfA_common, POS_COLS["Stay"]), default=["All"], key="stayA_sel")
    jibA_sel  = st.multiselect("Jib (A)", options=build_options(dfA_common, POS_COLS["Jib"]), default=["All"], key="jibA_sel")
    headA_sel = st.multiselect("HeadSail (A)", options=build_options(dfA_common, POS_COLS["Head"]), default=["All"], key="headA_sel")

with colSB:
    h_color("Data set B", "#E53935", level=4)
    mainB_sel = st.multiselect("Main (B)", options=build_options(dfB_common, POS_COLS["Main"]), default=["All"], key="mainB_sel")
    stayB_sel = st.multiselect("StaySail (B)", options=build_options(dfB_common, POS_COLS["Stay"]), default=["All"], key="stayB_sel")
    jibB_sel  = st.multiselect("Jib (B)", options=build_options(dfB_common, POS_COLS["Jib"]), default=["All"], key="jibB_sel")
    headB_sel = st.multiselect("HeadSail (B)", options=build_options(dfB_common, POS_COLS["Head"]), default=["All"], key="headB_sel")

apply_sails = st.button("Appliquer filtres voiles (A/B)")

def apply_sail_filters_positions(df: pd.DataFrame, sels: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col, sel in sels.items():
        if sel is None or len(sel) == 0:
            return out.iloc[0:0]
        if "All" in sel:
            continue
        out = out[out[col].isin(sel)]
    return out

if apply_sails:
    selsA = {
        POS_COLS["Main"]: mainA_sel,
        POS_COLS["Stay"]: stayA_sel,
        POS_COLS["Jib"]:  jibA_sel,
        POS_COLS["Head"]: headA_sel,
    }
    selsB = {
        POS_COLS["Main"]: mainB_sel,
        POS_COLS["Stay"]: stayB_sel,
        POS_COLS["Jib"]:  jibB_sel,
        POS_COLS["Head"]: headB_sel,
    }
    st.session_state["sails_dfA_final"] = apply_sail_filters_positions(dfA_common, selsA)
    st.session_state["sails_dfB_final"] = apply_sail_filters_positions(dfB_common, selsB)

dfA = st.session_state.get("sails_dfA_final", dfA_common)
dfB = st.session_state.get("sails_dfB_final", dfB_common)

# --------------------
# Résultats
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

# --------------------
# Plot 1
# --------------------
h_color("BSP_polarRatio vs TWS", "#FFD54A", level=3)
fig1 = plt.figure(figsize=(8, 4.5))
if not dfA.empty:
    plt.scatter(dfA["SilverData.WIND_TWS"], dfA["SilverData.PERF_BSP_PolarRatio"], s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Data set A")
if not dfB.empty:
    plt.scatter(dfB["SilverData.WIND_TWS"], dfB["SilverData.PERF_BSP_PolarRatio"], s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Data set B")
plt.xlabel("TWS (nds)")
plt.ylabel("BSP_polarRatio")
plt.title("Daguet 5 — BSP_polarRatio vs TWS")
plt.legend()
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# --------------------
# Helpers polaires
# --------------------
def polar_base_signed(ax, outer):
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

    # ticks -180..180 sur cercle extérieur
    twa_tick_labels = list(range(-180, 181, 30))
    theta_ticks = [np.deg2rad((t + 360) % 360) for t in twa_tick_labels]
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([str(t) for t in twa_tick_labels])

    yticks = list(range(0, outer + 1, 10))
    ax.set_yticks(yticks)
    ax.set_rlabel_position(0)

def polar_base_abs(ax, outer):
    """Polar pour abs(TWA): angles 0..180"""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(0, outer)
    ax.grid(False)

    lw_grid = 0.45
    theta_grid = np.linspace(0, np.pi, 361)

    for r in range(2, outer + 1, 2):
        ls = "-" if (r % 10 == 0) else "--"
        ax.plot(theta_grid, np.full_like(theta_grid, r), color="black", linestyle=ls, linewidth=lw_grid, alpha=0.9)

    for deg in range(0, 181, 10):
        th = np.deg2rad(deg)
        ax.plot([th, th], [0, outer], color="black", linestyle="-", linewidth=0.35, alpha=0.6)

    tick_labels = list(range(0, 181, 30))
    ax.set_xticks([np.deg2rad(t) for t in tick_labels])
    ax.set_xticklabels([str(t) for t in tick_labels])

    yticks = list(range(0, outer + 1, 10))
    ax.set_yticks(yticks)
    ax.set_rlabel_position(0)

def sail_combo_tuple(row: pd.Series) -> tuple[str, str, str, str]:
    def v(col):
        s = row.get(col, "none")
        s2 = str(s).strip().lower()
        if s2.startswith("nan") or s2.startswith("none") or s2 == "" or s2 == "none":
            return "none"
        return str(s).strip()
    return (
        v(POS_COLS["Main"]),
        v(POS_COLS["Stay"]),
        v(POS_COLS["Jib"]),
        v(POS_COLS["Head"]),
    )

# outer radius
bspA = dfA["SilverData.BSP_BoatSpeed"].to_numpy() if not dfA.empty else np.array([])
bspB = dfB["SilverData.BSP_BoatSpeed"].to_numpy() if not dfB.empty else np.array([])
max_bsp = 0.0
if bspA.size: max_bsp = max(max_bsp, float(np.nanmax(bspA)))
if bspB.size: max_bsp = max(max_bsp, float(np.nanmax(bspB)))
outer = int(max_bsp) + 1 if max_bsp > 0 else 1

# --------------------
# Plot 2 (signed)
# --------------------
h_color("BSP vs TWA (polaire)", "#FFD54A", level=3)
twaA = dfA["SilverData.WIND_TWA"].to_numpy() if not dfA.empty else np.array([])
twaB = dfB["SilverData.WIND_TWA"].to_numpy() if not dfB.empty else np.array([])
thetaA = np.deg2rad((twaA + 360.0) % 360.0) if twaA.size else np.array([])
thetaB = np.deg2rad((twaB + 360.0) % 360.0) if twaB.size else np.array([])

fig2 = plt.figure(figsize=(7.5, 7.5))
ax2 = fig2.add_subplot(111, projection="polar")
polar_base_signed(ax2, outer)
if thetaA.size:
    ax2.scatter(thetaA, bspA, s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Data set A")
if thetaB.size:
    ax2.scatter(thetaB, bspB, s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Data set B")
ax2.set_title("BSP (rayon) vs TWA (angle)", pad=15)
ax2.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10))
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# --------------------
# Plot 2 bis (abs)
# --------------------
h_color("BSP vs abs(TWA) (polaire) — superposition tribord/bâbord", "#FFD54A", level=3)
abs_thetaA = np.deg2rad(np.abs(twaA)) if twaA.size else np.array([])
abs_thetaB = np.deg2rad(np.abs(twaB)) if twaB.size else np.array([])

fig2b = plt.figure(figsize=(7.5, 7.5))
ax2b = fig2b.add_subplot(111, projection="polar")
polar_base_abs(ax2b, outer)
if abs_thetaA.size:
    ax2b.scatter(abs_thetaA, bspA, s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Data set A")
if abs_thetaB.size:
    ax2b.scatter(abs_thetaB, bspB, s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Data set B")
ax2b.set_title("BSP (rayon) vs abs(TWA) (angle)", pad=15)
ax2b.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10))
plt.tight_layout()
st.pyplot(fig2b)
plt.close(fig2b)

# --------------------
# Plot 3 : couleurs = combos + tableau coloré (signed)
# --------------------
h_color("BSP vs TWA (polaire) — couleur = combinaison voiles utilisées", "#FFD54A", level=3)

def add_combo_tuple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["combo_tuple"] = out.apply(sail_combo_tuple, axis=1)
    return out

dfA3 = add_combo_tuple(dfA)
dfB3 = add_combo_tuple(dfB)

mapping_global = {}
seen = set()

def add_tuple_to_mapping(tup):
    if tup in seen:
        return
    seen.add(tup)
    mapping_global[f"combo{len(mapping_global) + 1}"] = tup

if not dfA3.empty:
    for t in dfA3["combo_tuple"].tolist():
        add_tuple_to_mapping(t)
if not dfB3.empty:
    for t in dfB3["combo_tuple"].tolist():
        add_tuple_to_mapping(t)

inv_global = {t: k for k, t in mapping_global.items()}

if not dfA3.empty:
    dfA3["combo_id"] = dfA3["combo_tuple"].apply(lambda t: inv_global.get(t))
if not dfB3.empty:
    dfB3["combo_id"] = dfB3["combo_tuple"].apply(lambda t: inv_global.get(t))

combo_ids_present = []
if not dfA3.empty:
    combo_ids_present += dfA3["combo_id"].dropna().unique().tolist()
if not dfB3.empty:
    combo_ids_present += dfB3["combo_id"].dropna().unique().tolist()
combo_ids_present = sorted(list(dict.fromkeys(combo_ids_present)))

def build_combo_palette(n: int):
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for cm in cmaps:
        colors.extend([cm(i) for i in range(cm.N)])
    if n <= len(colors):
        return colors[:n]
    rep = int(np.ceil(n / len(colors)))
    return (colors * rep)[:n]

palette = build_combo_palette(len(combo_ids_present))
combo_to_color = {cid: palette[i] for i, cid in enumerate(combo_ids_present)}

fig3 = plt.figure(figsize=(8.5, 8.5))
ax3 = fig3.add_subplot(111, projection="polar")
polar_base_signed(ax3, outer)

def scatter_by_combo_signed(df: pd.DataFrame, marker: str):
    if df.empty:
        return
    for cid, sub in df.groupby("combo_id"):
        if cid not in combo_to_color:
            continue
        color = combo_to_color[cid]
        twa = sub["SilverData.WIND_TWA"].to_numpy()
        bsp = sub["SilverData.BSP_BoatSpeed"].to_numpy()
        th = np.deg2rad((twa + 360.0) % 360.0)
        ax3.scatter(
            th, bsp,
            s=12,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=0.2,
            alpha=PLOT3_ALPHA,
        )

scatter_by_combo_signed(dfA3, marker="o")
scatter_by_combo_signed(dfB3, marker="^")

ax3.set_title("Couleur = combo ; A=o ; B=^", pad=15)
legend_dataset = [
    Line2D([0], [0], marker='o', color='black', label='Data set A', linestyle='None', markersize=6),
    Line2D([0], [0], marker='^', color='black', label='Data set B', linestyle='None', markersize=6),
]
ax3.legend(handles=legend_dataset, loc="upper right", bbox_to_anchor=(1.20, 1.10), fontsize=8)

plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# tableau combos (signed)
rows = []
for cid in combo_ids_present:
    tup = mapping_global.get(cid)
    if tup is None:
        continue
    main, stay, jib, head = tup
    rows.append({"combo": cid, "MainSail": main, "StaySail": stay, "Jib": jib, "HeadSail": head})
df_combo = pd.DataFrame(rows)

def rgba_to_css(rgba):
    r, g, b, a = rgba
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"

if not df_combo.empty:
    df_combo = df_combo.sort_values("combo").reset_index(drop=True)
    combo_to_css = {cid: rgba_to_css(combo_to_color[cid]) for cid in combo_ids_present}

    def style_rows(row):
        cid = row["combo"]
        css = combo_to_css.get(cid, "rgba(255,255,255,0)")
        return [f"background-color: {css}; color: black;"] * len(row)

    st.dataframe(df_combo.style.apply(style_rows, axis=1), width="stretch", hide_index=True)
else:
    st.info("Aucune combinaison détectée dans les données filtrées.")

# --------------------
# Plot 3 bis (abs) : même combos, mais angle = abs(TWA)
# --------------------
h_color("BSP vs abs(TWA) (polaire) — couleur = combinaison voiles utilisées", "#FFD54A", level=3)

fig3b = plt.figure(figsize=(8.5, 8.5))
ax3b = fig3b.add_subplot(111, projection="polar")
polar_base_abs(ax3b, outer)

def scatter_by_combo_abs(df: pd.DataFrame, marker: str):
    if df.empty:
        return
    for cid, sub in df.groupby("combo_id"):
        if cid not in combo_to_color:
            continue
        color = combo_to_color[cid]
        twa = sub["SilverData.WIND_TWA"].to_numpy()
        bsp = sub["SilverData.BSP_BoatSpeed"].to_numpy()
        th = np.deg2rad(np.abs(twa))
        ax3b.scatter(
            th, bsp,
            s=12,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=0.2,
            alpha=PLOT3_ALPHA,
        )

scatter_by_combo_abs(dfA3, marker="o")
scatter_by_combo_abs(dfB3, marker="^")

ax3b.set_title("Couleur = combo ; A=o ; B=^ (angle = abs(TWA))", pad=15)
ax3b.legend(handles=legend_dataset, loc="upper right", bbox_to_anchor=(1.20, 1.10), fontsize=8)

plt.tight_layout()
st.pyplot(fig3b)
plt.close(fig3b)