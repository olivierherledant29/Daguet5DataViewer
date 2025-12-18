import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np

from lib.db_client import fetch_fields

st.set_page_config(page_title="Comparaison plages de temps", layout="wide")
st.title("Comparaison plages de temps")

TZ_OPTIONS = {
    "UTC": "UTC",
    "Heure locale Europe Centrale": "Europe/Paris",
    "Heure locale Antigua": "America/Antigua",
    "Heure locale Athènes": "Europe/Athens",
    "Heure locale Sydney": "Australia/Sydney",
}

FIELDS = [
    "SilverData.WIND_TWA",
    "SilverData.BSP_BoatSpeed",
    "SilverData.WIND_TWS",
    "SilverData.PERF_BSP_PolarRatio",
]

SCATTER_S = 2
SCATTER_ALPHA = 0.4

def to_utc_iso(dt_local: pd.Timestamp) -> str:
    dt_utc = dt_local.tz_convert("UTC")
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

def build_local_ts(d, t, tz_name: str) -> pd.Timestamp:
    return pd.Timestamp.combine(d, t).tz_localize(ZoneInfo(tz_name))

@st.cache_data(ttl=60)
def load_window(start_utc_iso: str, end_utc_iso: str) -> pd.DataFrame:
    return fetch_fields(FIELDS, start_utc_iso, end_utc_iso)

# ---- Timezone selector
tz_label = st.selectbox("Timezone d’affichage", list(TZ_OPTIONS.keys()), index=0)
tz_name = TZ_OPTIONS[tz_label]

# ---- Defaults for testing windows where the boat navigated
# A: 2025-12-02 11:00-12:00 UTC
# B: 2025-12-03 11:00-12:00 UTC
if "page1_defaults_set" not in st.session_state:
    st.session_state["A_start_utc"] = pd.Timestamp("2025-12-02T11:00:00Z")
    st.session_state["A_end_utc"]   = pd.Timestamp("2025-12-02T12:00:00Z")
    st.session_state["B_start_utc"] = pd.Timestamp("2025-12-03T11:00:00Z")
    st.session_state["B_end_utc"]   = pd.Timestamp("2025-12-03T12:00:00Z")
    st.session_state["page1_defaults_set"] = True

A_start_disp = st.session_state["A_start_utc"].tz_convert(ZoneInfo(tz_name))
A_end_disp   = st.session_state["A_end_utc"].tz_convert(ZoneInfo(tz_name))
B_start_disp = st.session_state["B_start_utc"].tz_convert(ZoneInfo(tz_name))
B_end_disp   = st.session_state["B_end_utc"].tz_convert(ZoneInfo(tz_name))

# ---- UI in a form
st.subheader("Paramètres")

with st.form("page1_form"):
    st.markdown("### Plages temporelles (DB en UTC, affichage selon timezone choisie)")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Plage A")
        A_start_date = st.date_input("Début A — date", value=A_start_disp.date(), key="A_start_date")
        A_start_time = st.time_input("Début A — heure", value=A_start_disp.time().replace(microsecond=0), key="A_start_time")
        A_end_date   = st.date_input("Fin A — date", value=A_end_disp.date(), key="A_end_date")
        A_end_time   = st.time_input("Fin A — heure", value=A_end_disp.time().replace(microsecond=0), key="A_end_time")

    with colB:
        st.markdown("#### Plage B")
        B_start_date = st.date_input("Début B — date", value=B_start_disp.date(), key="B_start_date")
        B_start_time = st.time_input("Début B — heure", value=B_start_disp.time().replace(microsecond=0), key="B_start_time")
        B_end_date   = st.date_input("Fin B — date", value=B_end_disp.date(), key="B_end_date")
        B_end_time   = st.time_input("Fin B — heure", value=B_end_disp.time().replace(microsecond=0), key="B_end_time")

    st.markdown("### Filtres")
    colF1, colF2, colF3, colF4 = st.columns(4)
    with colF1:
        twa_min, twa_max = st.slider("abs(TWA) — degrés (SilverData.WIND_TWA)", 0, 180, (0, 180), step=1, key="twa_range")
    with colF2:
        bsp_min, bsp_max = st.slider("BSP — nds (SilverData.BSP_BoatSpeed)", 0, 30, (0, 30), step=1, key="bsp_range")
    with colF3:
        tws_min, tws_max = st.slider("TWS — nds (SilverData.WIND_TWS)", 0, 40, (0, 40), step=1, key="tws_range")
    with colF4:
        pr_min, pr_max = st.slider("BSP_polarRatio (SilverData.PERF_BSP_PolarRatio)", 0, 160, (70, 130), step=1, key="pr_range")

    apply = st.form_submit_button("Appliquer")

def validate_range(name: str, s: pd.Timestamp, e: pd.Timestamp) -> bool:
    if e <= s:
        st.error(f"{name}: la fin doit être après le début.")
        return False
    return True

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    needed = set(FIELDS) | {"time_utc"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["time_utc"] + FIELDS).copy()
    df["abs_twa"] = df["SilverData.WIND_TWA"].abs()

    df = df[
        (df["abs_twa"] >= twa_min) & (df["abs_twa"] <= twa_max) &
        (df["SilverData.BSP_BoatSpeed"] >= bsp_min) & (df["SilverData.BSP_BoatSpeed"] <= bsp_max) &
        (df["SilverData.WIND_TWS"] >= tws_min) & (df["SilverData.WIND_TWS"] <= tws_max) &
        (df["SilverData.PERF_BSP_PolarRatio"] >= pr_min) & (df["SilverData.PERF_BSP_PolarRatio"] <= pr_max)
    ]
    return df

# Build timestamps
startA_local = build_local_ts(A_start_date, A_start_time, tz_name)
endA_local   = build_local_ts(A_end_date,   A_end_time,   tz_name)
startB_local = build_local_ts(B_start_date, B_start_time, tz_name)
endB_local   = build_local_ts(B_end_date,   B_end_time,   tz_name)

okA = validate_range("Plage A", startA_local, endA_local)
okB = validate_range("Plage B", startB_local, endB_local)

if "page1_autorun_done" not in st.session_state:
    st.session_state["page1_autorun_done"] = False

should_run = apply or (not st.session_state["page1_autorun_done"])

st.subheader("BSP_polarRatio vs TWS")

if should_run and okA and okB:
    st.session_state["page1_autorun_done"] = True

    st.session_state["A_start_utc"] = startA_local.tz_convert("UTC")
    st.session_state["A_end_utc"] = endA_local.tz_convert("UTC")
    st.session_state["B_start_utc"] = startB_local.tz_convert("UTC")
    st.session_state["B_end_utc"] = endB_local.tz_convert("UTC")

    startA_utc_iso = to_utc_iso(startA_local)
    endA_utc_iso   = to_utc_iso(endA_local)
    startB_utc_iso = to_utc_iso(startB_local)
    endB_utc_iso   = to_utc_iso(endB_local)

    with st.spinner("Chargement des données…"):
        dfA_raw = load_window(startA_utc_iso, endA_utc_iso)
        dfB_raw = load_window(startB_utc_iso, endB_utc_iso)

    dfA = prepare(dfA_raw)
    dfB = prepare(dfB_raw)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points A (bruts)", f"{len(dfA_raw):,}")
    c2.metric("Points A (filtrés)", f"{len(dfA):,}")
    c3.metric("Points B (bruts)", f"{len(dfB_raw):,}")
    c4.metric("Points B (filtrés)", f"{len(dfB):,}")

    if dfA.empty and dfB.empty:
        st.warning("Aucun point à afficher après filtrage.")
    else:
        # Plot 1
        fig1 = plt.figure()
        if not dfA.empty:
            plt.scatter(dfA["SilverData.WIND_TWS"], dfA["SilverData.PERF_BSP_PolarRatio"],
                        s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Plage A")
        if not dfB.empty:
            plt.scatter(dfB["SilverData.WIND_TWS"], dfB["SilverData.PERF_BSP_PolarRatio"],
                        s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Plage B")
        plt.xlabel("TWS (nds)")
        plt.ylabel("BSP_polarRatio")
        plt.title("Daguet 5 — BSP_polarRatio vs TWS")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # Plot 2: polar BSP vs TWA
        st.subheader("BSP vs TWA (polaire)")

        bspA = dfA["SilverData.BSP_BoatSpeed"].to_numpy() if not dfA.empty else np.array([])
        bspB = dfB["SilverData.BSP_BoatSpeed"].to_numpy() if not dfB.empty else np.array([])

        max_bsp = 0.0
        if bspA.size:
            max_bsp = max(max_bsp, float(np.nanmax(bspA)))
        if bspB.size:
            max_bsp = max(max_bsp, float(np.nanmax(bspB)))

        outer = int(max_bsp) + 1 if max_bsp > 0 else 1

        twaA = dfA["SilverData.WIND_TWA"].to_numpy() if not dfA.empty else np.array([])
        twaB = dfB["SilverData.WIND_TWA"].to_numpy() if not dfB.empty else np.array([])

        thetaA = np.deg2rad((twaA + 360.0) % 360.0) if twaA.size else np.array([])
        thetaB = np.deg2rad((twaB + 360.0) % 360.0) if twaB.size else np.array([])

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(360)
        ax.set_rlim(0, outer)
        ax.grid(False)

        # Cercles noirs : pointillés sauf multiples de 10 en plein
        lw_grid = 0.45
        theta_grid = np.linspace(0, 2 * np.pi, 721)
        for r in range(2, outer + 1, 2):
            ls = "-" if (r % 10 == 0) else "--"
            ax.plot(theta_grid, np.full_like(theta_grid, r), color="black", linestyle=ls, linewidth=lw_grid, alpha=0.9)

        # Rayons fins tous les 10°
        for deg in range(0, 360, 10):
            th = np.deg2rad(deg)
            ax.plot([th, th], [0, outer], color="black", linestyle="-", linewidth=0.35, alpha=0.6)

        # Labels angulaires -180..180
        twa_tick_labels = list(range(-180, 181, 30))
        theta_ticks = [np.deg2rad((t + 360) % 360) for t in twa_tick_labels]
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([str(t) for t in twa_tick_labels])

        # Labels radiaux en haut
        yticks = list(range(0, outer + 1, 10))
        ax.set_yticks(yticks)
        ax.set_rlabel_position(0)

        if thetaA.size:
            ax.scatter(thetaA, bspA, s=SCATTER_S, c="blue", alpha=SCATTER_ALPHA, label="Plage A")
        if thetaB.size:
            ax.scatter(thetaB, bspB, s=SCATTER_S, c="red", alpha=SCATTER_ALPHA, label="Plage B")

        ax.set_title("BSP (rayon) vs TWA (angle) — TWA négatif à gauche, positif à droite", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        with st.expander("Aperçu données filtrées (A/B)", expanded=False):
            st.markdown("**Plage A (filtrée)**")
            st.dataframe(
                dfA[["time_utc", "SilverData.WIND_TWS", "SilverData.PERF_BSP_PolarRatio",
                     "abs_twa", "SilverData.BSP_BoatSpeed", "SilverData.WIND_TWA"]].head(200),
                width="stretch",
            )
            st.markdown("**Plage B (filtrée)**")
            st.dataframe(
                dfB[["time_utc", "SilverData.WIND_TWS", "SilverData.PERF_BSP_PolarRatio",
                     "abs_twa", "SilverData.BSP_BoatSpeed", "SilverData.WIND_TWA"]].head(200),
                width="stretch",
            )
else:
    st.info("Clique sur **Appliquer** pour relancer les requêtes (et vérifie que début < fin).")
