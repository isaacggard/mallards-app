from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from survey_analysis import (
    DEFAULT_RATING_COLUMNS,
    DEFAULT_REVIEW_COLUMNS,
    SurveyFileClassification,
    build_survey_long,
    classify_survey_file,
    load_survey_workbook,
    parse_configured_columns,
    resolve_sentiment_columns,
    score_response_sentiment,
    score_sentiment,
    summarize_surveys,
    top_quotes,
)
from ui_theme import PALETTE, brand_header, section_label, stat_card


def _plot_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="#272d29",
        plot_bgcolor="#272d29",
        font=dict(color="#f4f1e7", family="Aptos, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)", zeroline=False),
    )
    return fig


def _bar_chart(categories: list[str], values: list[float], *, color: str, height: int = 320) -> go.Figure:
    fig = go.Figure(go.Bar(x=categories, y=values, marker_color=color))
    return _plot_layout(fig, height)


def _sentiment_color(compound: float) -> str:
    if compound <= -0.2:
        return PALETTE["danger"]
    if compound >= 0.2:
        return PALETTE["green"]
    return PALETTE["gold"]


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_sentiment(value: float) -> str:
    return f"{value:+.2f}"


def _load_local_surveys() -> list[tuple[str, bytes]]:
    base = Path.cwd()
    sources: list[tuple[str, bytes]] = []
    for folder in [base / "post_game_surveys", base / "post_season_surveys"]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.xlsx")):
            sources.append((path.name, path.read_bytes()))
    return sources


@st.cache_data(show_spinner=False)
def _run_survey_analysis_cached(
    survey_files: tuple[tuple[str, bytes, str | None, str | None, int | None], ...],
    review_columns: tuple[str, ...],
    rating_columns: tuple[str, ...],
) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for filename, payload, kind, team, year in survey_files:
        frame = load_survey_workbook(
            filename,
            payload,
            kind=kind,
            team=team,
            year=year,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return {
            "long_df": pd.DataFrame(),
            "response_df": pd.DataFrame(),
            "summaries": summarize_surveys(pd.DataFrame()),
            "matched_review": [],
            "matched_rating": [],
        }

    combined_wide = pd.concat(frames, ignore_index=True, sort=False)
    matched_review, matched_rating = resolve_sentiment_columns(
        combined_wide,
        review_columns=list(review_columns),
        rating_columns=list(rating_columns),
    )
    response_df = score_response_sentiment(
        combined_wide,
        review_columns=list(review_columns),
        rating_columns=list(rating_columns),
    )
    long_df = build_survey_long(
        frames,
        review_columns=list(review_columns),
        rating_columns=list(rating_columns),
    )
    long_df = score_sentiment(long_df)

    return {
        "long_df": long_df,
        "response_df": response_df,
        "summaries": summarize_surveys(long_df),
        "matched_review": matched_review,
        "matched_rating": matched_rating,
    }


def render_survey_page() -> None:
    brand_header(
        "Survey Insights",
        'Madison <span class="green">Mallards</span> & Night Mares',
        "Postgame and postseason survey sentiment, topics, and opportunities",
    )

    if "survey_review_columns_text" not in st.session_state:
        st.session_state["survey_review_columns_text"] = "\n".join(DEFAULT_REVIEW_COLUMNS)
    if "survey_rating_columns_text" not in st.session_state:
        st.session_state["survey_rating_columns_text"] = "\n".join(DEFAULT_RATING_COLUMNS)
    if "survey_uploaded_sources" not in st.session_state:
        st.session_state["survey_uploaded_sources"] = []

    section_label("Survey Workbooks")
    use_local = False
    with st.container(border=True):
        uploads = st.file_uploader(
            "Upload survey workbooks (.xlsx)",
            type=["xlsx"],
            accept_multiple_files=True,
            key="survey_uploads",
        )

        with st.expander("Load bundled survey files (local project)"):
            st.caption("Uses the files under post_game_surveys/ and post_season_surveys/ if available.")
            use_local = st.button("Use bundled surveys", type="secondary")

        if st.button("Clear Survey Data", key="clear-survey-data", use_container_width=True):
            for key in [
                "survey_uploaded_sources",
                "survey_long",
                "survey_response_df",
                "survey_summaries",
                "survey_column_config",
                "survey_upload_signature",
                "survey_uploads",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
    st.markdown(
        """
        <style>
        /* Target ONLY the two specific text areas using their keys */
        textarea[data-testid="stTextArea"][id="survey_review_columns_text"] {
            color: white !important;
            background-color: #1e1e1e !important;
        }
    
        textarea[data-testid="stTextArea"][id="survey_rating_columns_text"] {
            color: white !important;
            background-color: #1e1e1e !important;
        }
    
        /* Also make their labels white */
        label[for="survey_review_columns_text"],
        label[for="survey_rating_columns_text"] {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    section_label("Sentiment Configuration")
    with st.container(border=True):
        config_cols = st.columns(2)
        with config_cols[0]:
            review_columns_text = st.text_area(
                "Review Columns",
                help="List one review-style survey question per line.",
                height=180,
                key="survey_review_columns_text",
            )
        with config_cols[1]:
            rating_columns_text = st.text_area(
                "Rating Columns",
                help="List one rating-style survey question per line.",
                height=180,
                key="survey_rating_columns_text",
            )

    if use_local:
        st.session_state["survey_uploaded_sources"] = _load_local_surveys()
    elif uploads:
        st.session_state["survey_uploaded_sources"] = [(upload.name, upload.getvalue()) for upload in uploads]

    sources: list[tuple[str, bytes]] = st.session_state.get("survey_uploaded_sources", [])

    config_signature = (
        tuple((name, len(payload)) for name, payload in sources),
        review_columns_text,
        rating_columns_text,
    )
    if st.session_state.get("survey_upload_signature") != config_signature:
        st.session_state["survey_upload_signature"] = config_signature
        st.session_state.pop("survey_long", None)
        st.session_state.pop("survey_response_df", None)
        st.session_state.pop("survey_summaries", None)
        st.session_state.pop("survey_column_config", None)

    classifications: list[SurveyFileClassification] = [
        classify_survey_file(filename, payload) for filename, payload in sources
    ]

    valid = [classification for classification in classifications if classification.kind in ("postgame", "postseason")]
    invalid = [classification for classification in classifications if classification.kind is None]

    metrics = st.columns(3)
    metrics[0].markdown(stat_card("Survey Files", f"{len(classifications):,}", PALETTE["green_soft"]), unsafe_allow_html=True)
    metrics[1].markdown(stat_card("Recognized", f"{len(valid):,}", PALETTE["gold"]), unsafe_allow_html=True)
    metrics[2].markdown(
        stat_card("Needs Review", f"{len(invalid):,}", PALETTE["danger"] if invalid else PALETTE["blue"]),
        unsafe_allow_html=True,
    )

    if classifications:
        section_label("File Check")
        rows = []
        for classification in classifications:
            rows.append(
                {
                    "File": classification.filename,
                    "Type": "Post-game" if classification.kind == "postgame" else ("Post-season" if classification.kind == "postseason" else "Unknown"),
                    "Team": classification.team or "",
                    "Year": classification.year or "",
                    "Sheets": classification.sheet_count,
                    "Status": "Ready" if classification.kind else "Needs Review",
                    "Detail": classification.message,
                }
            )

        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "File": st.column_config.TextColumn(width="large"),
                "Type": st.column_config.TextColumn(width="medium"),
                "Team": st.column_config.TextColumn(width="small"),
                "Year": st.column_config.TextColumn(width="small"),
                "Sheets": st.column_config.NumberColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Detail": st.column_config.TextColumn(width="large"),
            },
        )

    analyze = st.button("Analyze Surveys", disabled=not valid, type="primary")
    if analyze:
        review_columns = parse_configured_columns(review_columns_text)
        rating_columns = parse_configured_columns(rating_columns_text)

        survey_files = tuple(
            (filename, payload, classification.kind, classification.team, classification.year)
            for (filename, payload), classification in zip(sources, classifications)
            if classification.kind is not None
        )
        analysis_result = _run_survey_analysis_cached(
            survey_files,
            tuple(review_columns),
            tuple(rating_columns),
        )

        long_df = analysis_result["long_df"]
        response_df = analysis_result["response_df"]
        if long_df.empty or response_df.empty:
            st.warning("No usable survey responses were found in the uploaded workbooks.")
        else:
            st.session_state["survey_long"] = long_df
            st.session_state["survey_response_df"] = response_df
            st.session_state["survey_summaries"] = analysis_result["summaries"]
            st.session_state["survey_column_config"] = {
                "review": analysis_result["matched_review"],
                "rating": analysis_result["matched_rating"],
            }

    if "survey_long" not in st.session_state or "survey_response_df" not in st.session_state:
        return

    long_df: pd.DataFrame = st.session_state["survey_long"]
    response_df: pd.DataFrame = st.session_state["survey_response_df"]
    summaries: dict[str, pd.DataFrame] = st.session_state.get("survey_summaries", {})
    column_config = st.session_state.get("survey_column_config", {"review": [], "rating": []})

    section_label("Filters")
    with st.container(border=True):
        filter_cols = st.columns([1.0, 1.0, 1.0, 1.8, 1.2])
        survey_types = sorted([value for value in long_df["survey_type"].dropna().unique().tolist() if value])
        teams = sorted([value for value in long_df["team"].dropna().unique().tolist() if value])
        years = sorted([int(value) for value in long_df["survey_year"].dropna().unique().tolist() if str(value).isdigit()])

        with filter_cols[0]:
            selected_types = st.multiselect("Survey Type", options=survey_types, default=survey_types)
        with filter_cols[1]:
            selected_teams = st.multiselect("Team", options=teams, default=teams)
        with filter_cols[2]:
            selected_years = st.multiselect("Year", options=years, default=years)
        with filter_cols[3]:
            selected_topics = st.multiselect(
                "Topic",
                options=sorted(long_df["topic"].dropna().unique().tolist()),
                default=[],
            )
        with filter_cols[4]:
            sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])

    filtered = long_df.copy()
    if selected_types:
        filtered = filtered[filtered["survey_type"].isin(selected_types)]
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]
    if selected_years:
        filtered = filtered[filtered["survey_year"].isin(selected_years)]
    if selected_topics:
        filtered = filtered[filtered["topic"].isin(selected_topics)]

    response_filtered = response_df.copy()
    if selected_types and "survey_type" in response_filtered.columns:
        response_filtered = response_filtered[response_filtered["survey_type"].isin(selected_types)]
    if selected_teams and "team" in response_filtered.columns:
        response_filtered = response_filtered[response_filtered["team"].isin(selected_teams)]
    if selected_years and "survey_year" in response_filtered.columns:
        response_filtered = response_filtered[response_filtered["survey_year"].isin(selected_years)]
    if sentiment_filter != "All" and "Sentiment" in response_filtered.columns:
        response_filtered = response_filtered[response_filtered["Sentiment"] == sentiment_filter]

    text_rows = filtered[filtered["question_kind"].eq("text") & filtered["sentiment_compound"].notna()].copy()
    rating_rows = filtered[filtered["question_kind"].eq("rating") & filtered["value_num"].notna()].copy()

    response_count = int(len(response_filtered))
    avg_sent = float(response_filtered["Sentiment Score"].mean()) if not response_filtered.empty else 0.0
    neg_rate = float((response_filtered["Sentiment"] == "Negative").mean()) if not response_filtered.empty else 0.0
    avg_rating = float(rating_rows["value_num"].mean()) if not rating_rows.empty else float("nan")

    section_label("Overview")
    kpi = st.columns(4)
    kpi[0].markdown(stat_card("Responses", f"{response_count:,}", PALETTE["green_soft"]), unsafe_allow_html=True)
    kpi[1].markdown(stat_card("Avg Sentiment", _format_sentiment(avg_sent), _sentiment_color(avg_sent)), unsafe_allow_html=True)
    kpi[2].markdown(
        stat_card("Negative Rate", _format_pct(neg_rate), PALETTE["danger"] if neg_rate >= 0.35 else PALETTE["gold"]),
        unsafe_allow_html=True,
    )
    kpi[3].markdown(
        stat_card("Avg Rating", f"{avg_rating:.2f}" if not math.isnan(avg_rating) else "-", PALETTE["blue"]),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.caption(
            f"Detected {len(column_config.get('review', []))} review columns and "
            f"{len(column_config.get('rating', []))} rating columns across the uploaded surveys."
        )

    section_label("Response Sentiment")
    response_cols = st.columns([1.0, 1.3])
    with response_cols[0].container(border=True):
        counts = response_filtered["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        fig = _bar_chart(counts.index.tolist(), counts.astype(float).tolist(), color=PALETTE["gold"], height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with response_cols[1].container(border=True):
        preview_columns = [
            column
            for column in [
                "survey_type",
                "team",
                "survey_year",
                "_source_file",
                "timestamp",
                "email",
                "Sentiment",
                "Sentiment Score",
                "Sentiment Source",
                "Sentiment Driver",
            ]
            if column in response_filtered.columns
        ]
        st.dataframe(
            response_filtered[preview_columns].head(250),
            hide_index=True,
            use_container_width=True,
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            response_filtered.to_excel(writer, index=False, sheet_name="Sentiment Results")
        output.seek(0)

        st.download_button(
            "Download Sentiment Workbook",
            data=output,
            file_name="survey_with_sentiment.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

    section_label("Topic Sentiment")
    topic_view = (
        text_rows.groupby("topic")
        .agg(responses=("response_text", "count"), avg=("sentiment_compound", "mean"))
        .reset_index()
        .sort_values("avg")
    )

    chart_cols = st.columns([1.2, 1.0])
    with chart_cols[0].container(border=True):
        if topic_view.empty:
            st.info("No open-text questions available for this filter selection.")
        else:
            colors = [_sentiment_color(value) for value in topic_view["avg"].tolist()]
            fig = go.Figure(go.Bar(x=topic_view["topic"], y=topic_view["avg"], marker_color=colors))
            fig = _plot_layout(fig, 330)
            fig.update_yaxes(range=[-1, 1], tickformat=".2f")
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_cols[1].container(border=True):
        if rating_rows.empty:
            st.info("No rating questions available for this filter selection.")
        else:
            rating_view = (
                rating_rows.groupby("topic")
                .agg(responses=("value_num", "count"), avg=("value_num", "mean"))
                .reset_index()
                .sort_values("avg")
            )
            fig = _bar_chart(rating_view["topic"].tolist(), rating_view["avg"].tolist(), color=PALETTE["blue"], height=330)
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section_label("Opportunities")
    with st.container(border=True):
        opportunities = summaries.get("opportunities", pd.DataFrame())
        if opportunities.empty:
            st.info("No opportunities found. Upload additional surveys or widen filters.")
        else:
            opp_view = opportunities.copy()
            if selected_types:
                opp_view = opp_view[opp_view["survey_type"].isin(selected_types)]
            if selected_teams:
                opp_view = opp_view[opp_view["team"].isin(selected_teams)]
            if selected_years:
                opp_view = opp_view[opp_view["survey_year"].isin(selected_years)]
            if selected_topics:
                opp_view = opp_view[opp_view["topic"].isin(selected_topics)]

            opp_view = opp_view.sort_values("opportunity_score", ascending=False).head(8)
            opp_view = opp_view.rename(
                columns={
                    "survey_type": "Survey",
                    "team": "Team",
                    "survey_year": "Year",
                    "topic": "Topic",
                    "responses": "Responses",
                    "avg_sentiment": "Avg Sentiment",
                    "negative_rate": "Negative Rate",
                    "top_terms": "Top Terms",
                }
            )
            opp_view["Survey"] = opp_view["Survey"].map({"postgame": "Post-game", "postseason": "Post-season"}).fillna(opp_view["Survey"])
            opp_view["Negative Rate"] = opp_view["Negative Rate"].map(_format_pct)
            opp_view["Avg Sentiment"] = opp_view["Avg Sentiment"].map(_format_sentiment)

            st.dataframe(
                opp_view[["Survey", "Team", "Year", "Topic", "Responses", "Avg Sentiment", "Negative Rate", "Top Terms"]],
                hide_index=True,
                use_container_width=True,
            )

            topics = opp_view["Topic"].dropna().unique().tolist()
            if topics:
                selected_topic = st.selectbox("Drill into a topic", options=topics)
                quote_cols = st.columns(2)
                with quote_cols[0]:
                    st.subheader("Most Negative")
                    for quote in top_quotes(filtered, topic=selected_topic, sentiment="Negative", limit=4):
                        st.write(f"- {quote}")
                with quote_cols[1]:
                    st.subheader("Most Positive")
                    for quote in top_quotes(filtered, topic=selected_topic, sentiment="Positive", limit=4):
                        st.write(f"- {quote}")

    section_label("Question Detail")
    with st.container(border=True):
        question_text = summaries.get("question_text", pd.DataFrame())
        if question_text.empty:
            st.info("No open-text questions for this filter selection.")
        else:
            view = question_text.copy()
            if selected_types:
                view = view[view["survey_type"].isin(selected_types)]
            if selected_teams:
                view = view[view["team"].isin(selected_teams)]
            if selected_years:
                view = view[view["survey_year"].isin(selected_years)]
            if selected_topics:
                view = view[view["topic"].isin(selected_topics)]

            view = view.sort_values(["negative_rate", "responses"], ascending=[False, False]).head(20)
            view = view.rename(
                columns={
                    "survey_type": "Survey",
                    "team": "Team",
                    "survey_year": "Year",
                    "topic": "Topic",
                    "question": "Question",
                    "responses": "Responses",
                    "avg_sentiment": "Avg Sentiment",
                    "negative_rate": "Negative Rate",
                }
            )
            view["Survey"] = view["Survey"].map({"postgame": "Post-game", "postseason": "Post-season"}).fillna(view["Survey"])
            view["Negative Rate"] = view["Negative Rate"].map(_format_pct)
            view["Avg Sentiment"] = view["Avg Sentiment"].map(_format_sentiment)
            st.dataframe(
                view[["Survey", "Team", "Year", "Topic", "Question", "Responses", "Avg Sentiment", "Negative Rate"]],
                hide_index=True,
                use_container_width=True,
                column_config={"Question": st.column_config.TextColumn(width="large")},
            )
