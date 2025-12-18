import os

# Aliases: si ton code demande EXOCET_*, on accepte aussi d'autres clés possibles
ALIASES = {
    "EXOCET_GRAFANA_URL": [
        "EXOCET_GRAFANA_URL",
        "GRAFANA_BASE_URL",
        "GRAFANA_URL",
        "EXOCET_CLOUD_URL",
    ],
    "EXOCET_GRAFANA_DS_UID": [
        "EXOCET_GRAFANA_DS_UID",
        "GRAFANA_DS_UID",
        "DATASOURCE_UID",
    ],
    "EXOCET_DB": [
        "EXOCET_DB",
        "INFLUX_DB",
        "DB",
    ],
    # IMPORTANT: on supporte aussi EXOCET_BEARER_TOKEN (ton naming local)
    "EXOCET_GRAFANA_TOKEN": [
        "EXOCET_GRAFANA_TOKEN",
        "EXOCET_BEARER_TOKEN",
        "GRAFANA_TOKEN",
        "TOKEN",
    ],
}


def _get_from_streamlit_secrets(key: str):
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # d'abord clé directe
            if key in st.secrets and str(st.secrets[key]).strip():
                return str(st.secrets[key])
            # ensuite aliases
            for k in ALIASES.get(key, []):
                if k in st.secrets and str(st.secrets[k]).strip():
                    return str(st.secrets[k])
    except Exception:
        pass
    return None

def _get_from_env(key: str):
    # direct
    v = os.getenv(key)
    if v and v.strip():
        return v.strip()
    # aliases
    for k in ALIASES.get(key, []):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    return None

def get_setting(key: str) -> str:
    v = _get_from_streamlit_secrets(key)
    if v is not None:
        return v
    v = _get_from_env(key)
    if v is not None:
        return v
    raise KeyError(f"Missing setting: {key} (checked st.secrets and env, including aliases)")
