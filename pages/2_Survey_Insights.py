from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from survey_analysis import (
    SurveyFileClassification,
    build_survey_long,
    classify_survey_file,
    load_survey_workbook,
    score_sentiment,
    summarize_surveys,
    top_quotes,
)
from ui_theme import PALETTE, brand_header, inject_base_css, render_sidebar_nav, section_label, stat_card


st.set_page_config(page_title="Survey Insights", layout="wide")
inject_base_css()
render_sidebar_nav("survey")


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


brand_header(
    "Survey Insights",
    'Madison <span class="green">Mallards</span> & Night Mares',
    "Postgame and postseason survey sentiment, topics, and opportunities",
)

section_label("Survey Workbooks")

with st.container(border=True):
    uploads = st.file_uploader(
        "Upload survey workbooks (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    with st.expander("Load bundled survey files (local project)"):
        st.caption("Uses the files under post_game_surveys/ and post_season_surveys/ if available.")
        use_local = st.button("Use bundled surveys", type="secondary")


sources: list[tuple[str, bytes]] = []
if use_local:
    sources = _load_local_surveys()
else:
    sources = [(upload.name, upload.getvalue()) for upload in uploads]


upload_signature = tuple((name, len(payload)) for name, payload in sources)
if st.session_state.get("survey_upload_signature") != upload_signature:
    st.session_state["survey_upload_signature"] = upload_signature
    st.session_state.pop("survey_long", None)
    st.session_state.pop("survey_summaries", None)


classifications: list[SurveyFileClassification] = [
    classify_survey_file(filename, payload) for filename, payload in sources
]

valid = [c for c in classifications if c.kind in ("postgame", "postseason")]
invalid = [c for c in classifications if c.kind is None]

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
    for c in classifications:
        status = "Ready" if c.kind else "Needs Review"
        rows.append(
            {
                "File": c.filename,
                "Type": "Post-game" if c.kind == "postgame" else ("Post-season" if c.kind == "postseason" else "Unknown"),
                "Team": c.team or "",
                "Year": c.year or "",
                "Sheets": c.sheet_count,
                "Status": status,
                "Detail": c.message,
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
            "Status": st.column_config.Column(width="small"),
            "Detail": st.column_config.TextColumn(width="large"),
        },
    )


analyze = st.button("Analyze Surveys", disabled=not valid, type="primary")
if analyze:
    frames: list[pd.DataFrame] = []
    for (filename, payload), classification in zip(sources, classifications):
        if classification.kind is None:
            continue
        frame = load_survey_workbook(
            filename,
            payload,
            kind=classification.kind,
            team=classification.team,
            year=classification.year,
        )
        if not frame.empty:
            frames.append(frame)

    long_df = build_survey_long(frames)
    long_df = score_sentiment(long_df)
    st.session_state["survey_long"] = long_df
    st.session_state["survey_summaries"] = summarize_surveys(long_df)


if "survey_long" not in st.session_state:
    st.stop()


long_df: pd.DataFrame = st.session_state["survey_long"]
summaries: dict[str, pd.DataFrame] = st.session_state.get("survey_summaries", {})

section_label("Filters")
with st.container(border=True):
    filter_cols = st.columns([1.1, 1.1, 1.1, 2.2])
    survey_types = sorted([t for t in long_df["survey_type"].dropna().unique().tolist() if t])
    teams = sorted([t for t in long_df["team"].dropna().unique().tolist() if t])
    years = sorted([int(y) for y in long_df["survey_year"].dropna().unique().tolist() if str(y).isdigit()])

    with filter_cols[0]:
        selected_types = st.multiselect(
            "Survey Type",
            options=survey_types,
            default=survey_types,
        )
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


filtered = long_df.copy()
if selected_types:
    filtered = filtered[filtered["survey_type"].isin(selected_types)]
if selected_teams:
    filtered = filtered[filtered["team"].isin(selected_teams)]
if selected_years:
    filtered = filtered[filtered["survey_year"].isin(selected_years)]
if selected_topics:
    filtered = filtered[filtered["topic"].isin(selected_topics)]


text_rows = filtered[filtered["question_kind"].eq("text") & filtered["sentiment_compound"].notna()].copy()
rating_rows = filtered[filtered["question_kind"].eq("rating") & filtered["value_num"].notna()].copy()

id_cols = [c for c in ["timestamp", "email", "_source_file", "_source_sheet"] if c in filtered.columns]
response_count = filtered.drop_duplicates(id_cols).shape[0] if id_cols else 0

avg_sent = float(text_rows["sentiment_compound"].mean()) if not text_rows.empty else 0.0
neg_rate = float((text_rows["sentiment_label"] == "Negative").mean()) if not text_rows.empty else 0.0
avg_rating = float(rating_rows["value_num"].mean()) if not rating_rows.empty else float("nan")

kpi = st.columns(4)
kpi[0].markdown(stat_card("Responses", f"{response_count:,}", PALETTE["green_soft"]), unsafe_allow_html=True)
kpi[1].markdown(stat_card("Avg Sentiment", _format_sentiment(avg_sent), _sentiment_color(avg_sent)), unsafe_allow_html=True)
kpi[2].markdown(stat_card("Negative Rate", _format_pct(neg_rate), PALETTE["danger"] if neg_rate >= 0.35 else PALETTE["gold"]), unsafe_allow_html=True)
kpi[3].markdown(
    stat_card("Avg Rating", f"{avg_rating:.2f}" if not math.isnan(avg_rating) else "—", PALETTE["blue"]),
    unsafe_allow_html=True,
)


section_label("Topic Sentiment")
topic_text = summaries.get("topic_text", pd.DataFrame())
topic_rating = summaries.get("topic_rating", pd.DataFrame())

topic_view = text_rows.groupby("topic").agg(responses=("response_text", "count"), avg=("sentiment_compound", "mean")).reset_index()
topic_view = topic_view.sort_values("avg")

chart_cols = st.columns([1.2, 1.0])
with chart_cols[0].container(border=True):
    if topic_view.empty:
        st.info("No open-text questions available for this filter selection.")
    else:
        colors = [_sentiment_color(v) for v in topic_view["avg"].tolist()]
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

        st.dataframe(opp_view[["Survey", "Team", "Year", "Topic", "Responses", "Avg Sentiment", "Negative Rate", "Top Terms"]], hide_index=True, use_container_width=True)

        topics = opp_view["Topic"].dropna().unique().tolist()
        if topics:
            selected_topic = st.selectbox("Drill into a topic", options=topics)
            quote_cols = st.columns(2)
            with quote_cols[0]:
                st.subheader("Most Negative")
                for quote in top_quotes(filtered, topic=selected_topic, sentiment="Negative", limit=4):
                    st.write(f"• {quote}")
            with quote_cols[1]:
                st.subheader("Most Positive")
                for quote in top_quotes(filtered, topic=selected_topic, sentiment="Positive", limit=4):
                    st.write(f"• {quote}")


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
