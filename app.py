import streamlit as st

from lib.nav_days import get_nav_days

st.set_page_config(
    page_title="Daguet 5 – Performance Data",
    page_icon="assets/logo_daguet5.svg",
    layout="wide",
)

col_logo, col_title = st.columns([0.15, 0.85], vertical_alignment="center")

with col_logo:
    st.image("assets/logo_daguet5.svg", width=80)

with col_title:
    st.markdown("## Daguet 5 — Performance Data")

st.markdown("---")

try:
    st.session_state["nav_days"] = get_nav_days(last_days=200)
except Exception as e:
    st.error("Erreur pendant la détection des jours navigués.")
    st.exception(e)
    st.session_state["nav_days"] = []

nav_days = st.session_state.get("nav_days", [])

st.subheader("Accueil")

if nav_days:
    st.markdown("### 10 derniers jours navigués détectés")
    for d in sorted(nav_days, reverse=True)[:10]:
        st.write(d)
else:
    st.warning("Aucun jour navigué détecté.")

st.markdown("---")
st.caption("Utilise le menu de gauche pour naviguer entre les pages.")