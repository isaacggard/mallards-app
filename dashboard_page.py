from __future__ import annotations

import streamlit as st

from fan_dashboard import missing_core_columns, read_master_csv, render_dashboard
from ui_theme import brand_header, section_label


def render_dashboard_page() -> None:
    uploaded_master = st.sidebar.file_uploader(
        "Fan Master CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    source = None
    if uploaded_master is not None:
        source = uploaded_master
    elif "fan_master_csv" in st.session_state:
        source = st.session_state["fan_master_csv"]

    if source is None:
        brand_header(
            "Fan Intelligence Dashboard",
            'Madison <span class="green">Mallards</span> & Night Mares',
            "Load a fan master CSV",
        )
        section_label("Data Source")
        with st.container(border=True):
            st.info("Build the fan master on the first page, or upload full_fan_master.csv in the sidebar.")
        return

    try:
        master = read_master_csv(source)
        missing = missing_core_columns(master)
        if missing:
            st.error("This file does not match the fan master format.")
            st.write(", ".join(missing))
            return
        render_dashboard(master)
    except Exception as exc:
        st.error("The dashboard could not load this file.")
        st.code(str(exc))
