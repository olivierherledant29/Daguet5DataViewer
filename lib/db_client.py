import requests
import pandas as pd
from .config import get_setting

# Influx measurement (d'après tes tests)
MEASUREMENT = '"C54"'


def run_influxql(query: str) -> dict:
    url = get_setting("EXOCET_GRAFANA_URL")
    token = get_setting("EXOCET_BEARER_TOKEN")
    db = get_setting("EXOCET_DB", "C54")

    headers = {"Authorization": f"Bearer {token}"}
    params = {"db": db, "epoch": "ms", "precision": "ms", "q": query}
    r = requests.get(url, headers=headers, params=params, timeout=120)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} on {r.url}\n{r.text[:500]}")
    return r.json()


def series_to_df(resp_json: dict) -> pd.DataFrame:
    results = resp_json.get("results", [])
    if not results:
        return pd.DataFrame()
    series = results[0].get("series", [])
    if not series:
        return pd.DataFrame()
    s = series[0]
    df = pd.DataFrame(s.get("values", []), columns=s.get("columns", []))
    if "time" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def _q_ident(name: str) -> str:
    # safe quoting for field names containing dots
    return '"' + name.replace('"', '\\"') + '"'


def fetch_fields(fields: list[str], start_utc_iso: str, end_utc_iso: str) -> pd.DataFrame:
    """
    Fetch multiple fields in one InfluxQL query (raw, no aggregation).
    Returns: time, <fields...>, time_utc
    """
    if not fields:
        return pd.DataFrame()

    select = ", ".join(_q_ident(f) for f in fields)
    query = (
        f"SELECT {select} FROM {MEASUREMENT} "
        f"WHERE time >= '{start_utc_iso}' AND time < '{end_utc_iso}'"
    )

    data = run_influxql(query)
    return series_to_df(data)


def fetch_aggregated(
    fields_mean: list[str],
    fields_last: list[str],
    start_utc_iso: str,
    end_utc_iso: str,
    bucket: str = "10s",
) -> pd.DataFrame:
    """
    Fetch aggregated data:
      - mean() for numeric fields
      - last() for categorical/string fields
    All outputs are aliased back to the original field names.
    """
    parts = []
    for f in fields_mean:
        qi = _q_ident(f)
        parts.append(f"mean({qi}) AS {qi}")
    for f in fields_last:
        qi = _q_ident(f)
        parts.append(f"last({qi}) AS {qi}")

    if not parts:
        return pd.DataFrame()

    select = ", ".join(parts)
    query = (
        f"SELECT {select} FROM {MEASUREMENT} "
        f"WHERE time >= '{start_utc_iso}' AND time < '{end_utc_iso}' "
        f"GROUP BY time({bucket}) fill(null)"
    )

    data = run_influxql(query)
    return series_to_df(data)
