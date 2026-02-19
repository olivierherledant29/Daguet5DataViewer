import math
import re
import pandas as pd

from lib.db_client import run_influxql, parse_series_to_df

MEASUREMENT = "C54"

# Regex Python (pas Influx) pour attraper les champs voiles/perf
PY_FIELD_REGEX = re.compile(
    r"^(SilverData\.)?(PERF_.*(Sail|SAIL|Upwash|UWT|Deck|Jib|Head|Code)|.*(MainSail|UpwashTableSelected|SailOnDeck|SailCode).*)$"
)

CHUNK_SIZE = 20


def show_all_field_keys(measurement: str) -> list[str]:
    q = f'SHOW FIELD KEYS FROM "{measurement}"'
    data = run_influxql(q)
    df = parse_series_to_df(data)
    if df.empty or "fieldKey" not in df.columns:
        return []
    return df["fieldKey"].dropna().astype(str).unique().tolist()


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_last_per_hour(fields: list[str], start_iso: str, end_iso: str) -> pd.DataFrame:
    parts = [f'last("{f}") AS "{f}"' for f in fields]
    select = ", ".join(parts)

    q = (
        f'SELECT {select} FROM "{MEASUREMENT}" '
        f"WHERE time >= '{start_iso}' AND time < '{end_iso}' "
        f"GROUP BY time(1h) fill(null)"
    )

    data = run_influxql(q)
    df = parse_series_to_df(data)

    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    df["time_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop(columns=["time"])
    df = df.set_index("time_utc").sort_index()
    return df


def non_null_summary_per_hour(df: pd.DataFrame) -> list[str]:
    lines = []
    for t, row in df.iterrows():
        nn = row.dropna()
        if nn.empty:
            continue
        items = [f"{k}={v}" for k, v in nn.items()]
        lines.append(f"{t.isoformat()}  ->  " + " | ".join(items))
    return lines


def main():
    # Hier UTC (jour complet)
    yesterday = (pd.Timestamp.utcnow().floor("D") - pd.Timedelta(days=1))
    start_iso = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = (yesterday + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Fenêtre UTC: {start_iso} -> {end_iso}")

    # 1) Récupère tous les fields, puis filtre en Python
    all_fields = show_all_field_keys(MEASUREMENT)
    print(f"Nb fields total: {len(all_fields)}")

    fields = sorted([f for f in all_fields if PY_FIELD_REGEX.match(f)])
    print(f"\nChamps retenus (filtrage Python regex): {len(fields)}")
    for f in fields:
        print(" -", f)

    if not fields:
        print("\nAucun champ matché. => élargis PY_FIELD_REGEX ou donne-moi 2-3 noms de fields récents.")
        return

    # 2) Lecture chunkée (last() par heure)
    dfs = []
    n_chunks = math.ceil(len(fields) / CHUNK_SIZE)

    for idx, chunk in enumerate(chunked(fields, CHUNK_SIZE), start=1):
        print(f"\nRequête chunk {idx}/{n_chunks} ({len(chunk)} fields)...")
        dfc = fetch_last_per_hour(chunk, start_iso, end_iso)
        if not dfc.empty:
            dfs.append(dfc)

    if not dfs:
        print("\nAucune donnée retournée sur la fenêtre (sur ces champs).")
        return

    df = pd.concat(dfs, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    print("\n=== Aperçu table (5 premières lignes) ===")
    print(df.head(5))

    print("\n=== Aperçu table (5 dernières lignes) ===")
    print(df.tail(5))

    print("\n=== Valeurs non-null par heure ===")
    lines = non_null_summary_per_hour(df)
    if not lines:
        print("Aucune valeur non-null trouvée (sur ces champs).")
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
