import os

def get_setting(key: str) -> str:
    """
    Récupère une variable de config :
    - Streamlit Cloud: st.secrets[key]
    - Local: variables d'environnement (chargées via .env si python-dotenv est utilisé)
    """

    # 1) Streamlit secrets (si streamlit dispo et secrets configurés)
    try:
        import streamlit as st  # import local pour éviter dépendance dure au runtime
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is None or str(val).strip() == "":
                raise KeyError(f"Empty setting in st.secrets: {key}")
            return str(val)
    except Exception:
        # streamlit pas dispo ou secrets non accessibles -> on continue
        pass

    # 2) Environnement OS
    val = os.getenv(key)
    if val is None or val.strip() == "":
        raise KeyError(f"Missing setting: {key}")
    return val
