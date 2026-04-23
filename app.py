from __future__ import annotations

import traceback

import streamlit as st

from dashboard_page import render_dashboard_page
from fan_dashboard import inject_dashboard_css
from fan_master_cleaner import build_fan_master, classify_file, read_columns, read_csv
from survey_page import render_survey_page
from ui_theme import PALETTE, brand_header, go_to_page, inject_base_css, render_sidebar_nav, section_label, stat_card


TYPE_LABELS = {
    "all_tickets": "Ticket Data",
    "transactions": "Transactions",
}

STATUS_LABELS = {
    "all_tickets": "Ready",
    "transactions": "Ready",
    None: "Needs Review",
}


def inspect_upload(uploaded_file):
    try:
        uploaded_file.seek(0)
        columns = read_columns(uploaded_file)
        uploaded_file.seek(0)
        return classify_file(uploaded_file.name, columns)
    except Exception as exc:
        uploaded_file.seek(0)
        return classify_file(uploaded_file.name, [f"READ_ERROR: {exc}"])


def render_builder_page() -> None:
    brand_header(
        "Fan Data Operations",
        'Madison <span class="green">Mallards</span> & Night Mares',
        "Fan Master Builder",
    )

    section_label("Source Files")
    with st.container(border=True):
        uploads = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

    upload_signature = tuple((upload.name, getattr(upload, "size", None)) for upload in uploads)
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state["upload_signature"] = upload_signature
        st.session_state.pop("fan_master_csv", None)
        st.session_state.pop("fan_master_summary", None)

    classifications = [inspect_upload(upload) for upload in uploads]

    all_ticket_count = sum(item.kind == "all_tickets" for item in classifications)
    transaction_count = sum(item.kind == "transactions" for item in classifications)
    invalid_count = sum(item.kind is None for item in classifications)

    metric_cols = st.columns(3)
    metric_cols[0].markdown(
        stat_card("Ticket Files", f"{all_ticket_count:,}", PALETTE["green_soft"]),
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        stat_card("Transaction Files", f"{transaction_count:,}", PALETTE["gold"]),
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        stat_card("Needs Review", f"{invalid_count:,}", PALETTE["danger"] if invalid_count else PALETTE["blue"]),
        unsafe_allow_html=True,
    )

    if classifications:
        section_label("File Check")
        status_rows = [
            {
                "File": item.filename,
                "Detected Type": TYPE_LABELS.get(item.kind, "Invalid"),
                "Columns": item.column_count,
                "Status": STATUS_LABELS[item.kind],
                "Detail": item.message,
            }
            for item in classifications
        ]
        st.dataframe(
            status_rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "File": st.column_config.TextColumn(width="large"),
                "Detected Type": st.column_config.TextColumn(width="medium"),
                "Columns": st.column_config.NumberColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Detail": st.column_config.TextColumn(width="large"),
            },
        )

    done = st.button("Done uploading", disabled=not uploads, type="primary")
    if done:
        invalid_files = [item.filename for item in classifications if item.kind is None]
        has_ticket_file = any(item.kind == "all_tickets" for item in classifications)

        if invalid_files:
            st.error("Remove invalid files before building the fan master.")
            st.write(", ".join(invalid_files))
        elif not has_ticket_file:
            st.error("At least one ticket data file is required.")
        else:
            progress = st.progress(0, text="Reading files")
            ticket_frames = []
            transaction_frames = []

            try:
                for index, (upload, item) in enumerate(zip(uploads, classifications), start=1):
                    upload.seek(0)
                    df = read_csv(upload)
                    if item.kind == "all_tickets":
                        ticket_frames.append(df)
                    elif item.kind == "transactions":
                        transaction_frames.append(df)
                    progress.progress(index / max(len(uploads), 1), text=f"Reading {upload.name}")

                progress.progress(1.0, text="Building fan master")
                with st.spinner("Cleaning tickets and linking merch transactions"):
                    fan_master = build_fan_master(ticket_frames, transaction_frames)

                st.session_state["fan_master_csv"] = fan_master.to_csv(index=False).encode("utf-8")
                st.session_state["fan_master_summary"] = {
                    "fans": len(fan_master),
                    "columns": len(fan_master.columns),
                    "merch_buyers": int(fan_master["IS_MERCH_BUYER"].sum()),
                }
                progress.empty()
                st.success("full_fan_master.csv is ready.")
            except Exception:
                progress.empty()
                st.error("The fan master could not be built.")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    if "fan_master_csv" in st.session_state:
        summary = st.session_state.get("fan_master_summary", {})
        section_label("Export")
        st.markdown(
            """
            <div class="result-panel">
                <div class="result-title">full_fan_master.csv is ready</div>
                <div class="result-copy">The dashboard view can use this file immediately from the current session.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_cols = st.columns(3)
        summary_cols[0].markdown(
            stat_card("Fans", f"{summary.get('fans', 0):,}", PALETTE["green_soft"]),
            unsafe_allow_html=True,
        )
        summary_cols[1].markdown(
            stat_card("Columns", f"{summary.get('columns', 0):,}", PALETTE["gold"]),
            unsafe_allow_html=True,
        )
        summary_cols[2].markdown(
            stat_card("Merch Buyers", f"{summary.get('merch_buyers', 0):,}", PALETTE["blue"]),
            unsafe_allow_html=True,
        )

        action_cols = st.columns([1, 1, 3])
        with action_cols[0]:
            st.download_button(
                "Download CSV",
                data=st.session_state["fan_master_csv"],
                file_name="full_fan_master.csv",
                mime="text/csv",
                type="primary",
            )
        with action_cols[1]:
            if st.button("Open Dashboard", key="open-dashboard", use_container_width=True):
                go_to_page("dashboard")


st.set_page_config(page_title="Mallards Fan Intelligence", layout="wide")

current_page = st.session_state.get("active_page", "builder")
if current_page == "dashboard":
    inject_dashboard_css()
else:
    inject_base_css()

requested_page = render_sidebar_nav(current_page)
if requested_page != current_page:
    st.session_state["active_page"] = requested_page
    st.rerun()

current_page = st.session_state.get("active_page", current_page)

if current_page == "builder":
    render_builder_page()
elif current_page == "dashboard":
    render_dashboard_page()
elif current_page == "survey":
    render_survey_page()
else:
    st.session_state["active_page"] = "builder"
    st.rerun()
