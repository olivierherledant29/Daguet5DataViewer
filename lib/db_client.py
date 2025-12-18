import requests
import pandas as pd

from .config import get_setting

def _query_url() -> str:
    """
    Supporte 2 modes de config :
    - EXOCET_GRAFANA_URL = URL complète finissant par /query
      ex: https://exocet.cloud/grafana/api/datasources/proxy/uid/<UID>/query
    - OU bien EXOCET_GRAFANA_URL = base url (https://exocet.cloud)
      + EXOCET_GRAFANA_DS_UID = <UID>
      => on reconstruit l'URL /grafana/api/datasources/proxy/uid/<UID>/query
    """
    url = get_setting("EXOCET_GRAFANA_URL").rstrip("/")
    if url.endswith("/query"):
        return url
    ds_uid = get_setting("EXOCET_GRAFANA_DS_UID")
    return f"{url}/grafana/api/datasources/proxy/uid/{ds_uid}/query"

def run_influxql(query: str) -> dict:
    url = _query_url()
    token = get_setting("EXOCET_GRAFANA_TOKEN")  # récupère aussi EXOCET_BEARER_TOKEN via aliases
    db = get_setting("EXOCET_DB")

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "db": db,
        "epoch": "ms",
        "precision": "ms",
        "q": query,
    }

    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_series_to_df(data: dict) -> pd.DataFrame:
    """
    Parse InfluxQL JSON -> DataFrame.
    Attend data['results'][0]['series'][0]['columns'/'values'].
    """
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()
    series = results[0].get("series")
    if not series:
        return pd.DataFrame()

    rows = []
    for s in series:
        cols = s.get("columns", [])
        vals = s.get("values", [])
        if cols and vals:
            rows.append(pd.DataFrame(vals, columns=cols))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def fetch_aggregated(fields_mean, fields_last, start_utc_iso: str, end_utc_iso: str, bucket: str = "10s") -> pd.DataFrame:
    """
    fields_mean : champs numériques -> mean()
    fields_last : champs string/int codes -> last()
    bucket      : ex '10s'
    """
    parts = []
    for f in fields_mean:
        parts.append(f'mean("{f}") AS "{f}"')
    for f in fields_last:
        parts.append(f'last("{f}") AS "{f}"')

    select = ", ".join(parts)
    q = (
        f"SELECT {select} FROM \"C54\" "
        f"WHERE time >= '{start_utc_iso}' AND time < '{end_utc_iso}' "
        f"GROUP BY time({bucket}) fill(null)"
    )

    data = run_influxql(q)
    df = parse_series_to_df(data)
    if df.empty:
        return df

    # time en ms -> datetime UTC
    if "time" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    return df
