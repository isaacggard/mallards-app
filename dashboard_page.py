from __future__ import annotations

import streamlit as st

from fan_dashboard import missing_core_columns, read_master_csv, render_dashboard
from ui_theme import brand_header, section_label


@st.cache_data(show_spinner=False)
def _load_master_cached(csv_bytes: bytes):
    return read_master_csv(csv_bytes)


def render_dashboard_page() -> None:
    uploaded_master = st.sidebar.file_uploader(
        "Fan Master CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="dashboard_master_upload",
    )

    if uploaded_master is not None:
        st.session_state["dashboard_master_csv"] = uploaded_master.getvalue()
        st.session_state["dashboard_master_name"] = uploaded_master.name

    if st.session_state.get("dashboard_master_csv") is not None:
        source_bytes = st.session_state["dashboard_master_csv"]
        source_name = st.session_state.get("dashboard_master_name", "Uploaded fan master")
    elif "fan_master_csv" in st.session_state:
        source_bytes = st.session_state["fan_master_csv"]
        source_name = "Session fan master"
    else:
        source_bytes = None
        source_name = ""

    with st.sidebar:
        if source_bytes is not None:
            st.caption(f"Loaded: {source_name}")
            if st.button("Clear Dashboard Data", key="clear-dashboard-data", use_container_width=True):
                st.session_state.pop("dashboard_master_csv", None)
                st.session_state.pop("dashboard_master_name", None)
                st.rerun()

    if source_bytes is None:
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
        master = _load_master_cached(source_bytes)
        missing = missing_core_columns(master)
        if missing:
            st.error("This file does not match the fan master format.")
            st.write(", ".join(missing))
            return
        render_dashboard(master)
    except Exception as exc:
        st.error("The dashboard could not load this file.")
        st.code(str(exc))
