"""Streamlit dashboard.

Run with::

    streamlit run src/rlstress/dashboard.py

Stage 1 shows only the loaded configuration and the planned page layout.
"""

from __future__ import annotations

from dataclasses import asdict

from rlstress.config import load_config

PAGES = (
    "Portfolio & limits",
    "Discovered scenarios",
    "Recommendations",
    "Mitigation lab",
)


def main() -> None:
    """Render the dashboard."""
    import streamlit as st

    st.set_page_config(page_title="RL Stress Testing", layout="wide")
    st.title("RL-Based Adaptive Stress Testing")

    config = load_config()
    page = st.sidebar.radio("Page", PAGES)

    if page == "Portfolio & limits":
        st.subheader("Portfolio")
        st.json(asdict(config.portfolio))
        st.subheader("Risk limits")
        st.json([asdict(l) for l in config.risk.limits])
    else:
        st.info(f"'{page}' is not implemented yet.")


if __name__ == "__main__":
    main()
