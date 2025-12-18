import pandas as pd
import streamlit as st
from datetime import timedelta

from .db_client import run_influxql


@st.cache_data(ttl=6 * 3600)  # 6 heures
def get_nav_days(last_days: int = 200, bsp_threshold: float = 5.0) -> list[pd.Timestamp]:
    """
    Retourne la liste des jours (UTC) où le bateau a navigué
    selon la règle ultra-rapide :
    max(BSP) > bsp_threshold sur au moins un bloc de 5 minutes.
    """

    query = f"""
    SELECT max("SilverData.BSP_BoatSpeed") AS bsp_max
    FROM "C54"
    WHERE time >= now()-{last_days}d AND time < now()
    GROUP BY time(5m) fill(null)
    """

    data = run_influxql(query)

    if not data or "results" not in data or not data["results"]:
        return []

    res = data["results"][0]
    if "series" not in res:
        return []

    series = res["series"][0]
    df = pd.DataFrame(series["values"], columns=series["columns"])

    if df.empty or "bsp_max" not in df:
        return []

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.dropna(subset=["bsp_max"])

    # Détection navigation
    df_nav = df[df["bsp_max"] > bsp_threshold].copy()

    if df_nav.empty:
        return []

    # On garde uniquement les jours UTC
    df_nav["day_utc"] = df_nav["time"].dt.floor("D")

    nav_days = sorted(df_nav["day_utc"].unique())

    return list(nav_days)
