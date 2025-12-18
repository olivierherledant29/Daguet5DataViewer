import os
import streamlit as st

def get_setting(key: str, default: str | None = None) -> str:
    """
    Priority:
    1) st.secrets (Streamlit local secrets.toml or Cloud Secrets)
    2) OS environment variables
    3) default (if provided)
    """
    if key in st.secrets:
        return str(st.secrets[key])

    env_val = os.getenv(key)
    if env_val is not None and env_val != "":
        return str(env_val)

    if default is not None:
        return str(default)

    raise KeyError(f"Missing setting: {key}")
