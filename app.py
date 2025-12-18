import streamlit as st
from pathlib import Path

from lib.nav_days import get_nav_days

st.set_page_config(
    page_title="Daguet 5 – Data App",
    layout="wide",
)

# =========================
# Initialisation globale
# =========================
if "nav_days" not in st.session_state:
    with st.spinner("Initialisation de la librairie des jours de navigation…"):
        st.session_state["nav_days"] = get_nav_days(last_days=200)

nav_days = st.session_state["nav_days"]

# =========================
# Helpers (format dates FR)
# =========================
MOIS_FR = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

def format_date_fr(ts):
    return f"{ts.day} {MOIS_FR[ts.month]} {ts.year}"

# =========================
# Page d'accueil
# =========================
col_left, col_right = st.columns([3, 1])

with col_left:
    st.title("Daguet 5 – Data Explorer")

    st.markdown(f"""
### Librairie des jours de navigation
- Période analysée : **200 derniers jours**
- Jours détectés : **{len(nav_days)}**
""")

    if nav_days:
        st.markdown("**10 derniers jours de navigation (UTC)**")
        nav_days_sorted = sorted(nav_days, reverse=True)[:10]
        for d in nav_days_sorted:
            st.write(f"- {format_date_fr(d)}")
    else:
        st.warning("Aucune journée de navigation détectée sur la période.")

with col_right:
    img_path = Path(__file__).parent / "assets" / "daguet5.jpg"
    if img_path.exists():
        # Pas de use_container_width (deprecated)
        st.image(str(img_path), width=320)
    else:
        st.warning("Image non trouvée (assets/daguet5.jpg)")
