from __future__ import annotations

import html

import streamlit as st


PALETTE = {
    "ink": "#11140f",
    "panel": "#20251f",
    "panel_2": "#272d29",
    "panel_3": "#30363d",
    "line": "#4f6155",
    "green": "#58b85a",
    "green_soft": "#94d38d",
    "gold": "#f4cc52",
    "blue": "#66baf2",
    "paper": "#f4f1e7",
    "text": "#f8f8f1",
    "muted": "#b8c0b4",
    "danger": "#e16c5b",
}


def inject_base_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --mallards-ink: {PALETTE["ink"]};
            --mallards-panel: {PALETTE["panel"]};
            --mallards-panel-2: {PALETTE["panel_2"]};
            --mallards-panel-3: {PALETTE["panel_3"]};
            --mallards-line: {PALETTE["line"]};
            --mallards-green: {PALETTE["green"]};
            --mallards-green-soft: {PALETTE["green_soft"]};
            --mallards-gold: {PALETTE["gold"]};
            --mallards-blue: {PALETTE["blue"]};
            --mallards-paper: {PALETTE["paper"]};
            --mallards-text: {PALETTE["text"]};
            --mallards-muted: {PALETTE["muted"]};
            --mallards-danger: {PALETTE["danger"]};
        }}

        .stApp {{
            background:
                linear-gradient(180deg, rgba(244, 204, 82, .035), rgba(17, 20, 15, 0) 280px),
                repeating-linear-gradient(90deg, rgba(255,255,255,.018) 0, rgba(255,255,255,.018) 1px, transparent 1px, transparent 72px),
                #11140f;
            color: var(--mallards-text);
        }}

        .block-container {{
            max-width: 1480px;
            padding: .9rem 1.85rem 3.2rem;
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
            height: 0;
        }}

        [data-testid="stToolbar"],
        #MainMenu,
        footer {{
            visibility: hidden;
            height: 0;
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015)),
                #171c17;
            border-right: 1px solid rgba(102, 186, 242, .22);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background:
                linear-gradient(180deg, rgba(244,204,82,.045), rgba(17,20,15,0) 240px),
                linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01)),
                #171c17;
        }}

        [data-testid="stSidebarNav"] {{
            display: none;
        }}

        [data-testid="stSidebarUserContent"] {{
            padding: .8rem .9rem 1.2rem;
        }}

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div:not([data-testid="stFileUploader"]) {{
            color: var(--mallards-paper);
        }}

        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: var(--mallards-muted) !important;
        }}

        .sidebar-brand {{
            margin: 0 0 14px;
            padding: 16px 10px 14px;
            border: 1px solid rgba(102, 186, 242, .16);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015)),
                rgba(39, 45, 41, .92);
            box-shadow: 0 14px 30px rgba(0,0,0,.18);
        }}

        .sidebar-kicker {{
            color: #b8c0b4 !important;
            font-size: .72rem;
            font-weight: 900;
            text-transform: uppercase;
        }}

        .sidebar-title {{
            color: #eef2e7 !important;
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-weight: 800;
            font-size: 1.22rem;
            line-height: 1.05;
            margin-top: 4px;
        }}

        .sidebar-title span {{
            color: var(--mallards-green) !important;
        }}

        .sidebar-nav-active {{
            display: flex;
            align-items: center;
            min-height: 44px;
            margin-bottom: 8px;
            padding: 0 14px;
            border-radius: 6px;
            border: 1px solid rgba(244, 204, 82, .82);
            background: linear-gradient(180deg, #f6d66c, #dcb342);
            color: #17170e;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(0,0,0,.24);
        }}

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {{
            padding-top: .7rem;
            padding-left: .65rem;
            background: transparent;
        }}

        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button {{
            border-radius: 10px;
            border: 1px solid rgba(102, 186, 242, .26);
            background:
                linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.015)),
                #20251f;
            color: var(--mallards-paper);
            box-shadow: 0 10px 24px rgba(0,0,0,.18);
        }}

        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover {{
            border-color: rgba(244, 204, 82, .7);
            color: var(--mallards-gold);
        }}

        h1, h2, h3, h4, h5, h6, p, span, label, div {{
            letter-spacing: 0;
        }}

        h1, h2, h3 {{
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
        }}

        p, span, label, div, button {{
            font-family: "Aptos", "Segoe UI", sans-serif;
        }}

        div[data-testid="stFileUploader"] {{
            padding: 1.1rem;
            border: 1px dashed rgba(148, 211, 141, .56);
            border-radius: 8px;
            background: rgba(244, 241, 231, .035);
        }}

        div[data-testid="stFileUploader"] section {{
            background: rgba(255, 255, 255, .03);
            border-color: rgba(255, 255, 255, .14);
            border-radius: 8px;
        }}

        div[data-testid="stFileUploader"] small {{
            color: var(--mallards-muted);
        }}

        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] p,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small {{
            color: var(--mallards-paper) !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stFileUploader"] {{
            background: #20251f;
            border-color: rgba(88,184,90,.6);
            margin-top: .35rem;
        }}

        [data-testid="stSidebar"] div[data-testid="stFileUploader"] button {{
            color: #11140f !important;
        }}

        .stButton > button,
        .stDownloadButton > button,
        a[data-testid="stPageLink-NavLink"] {{
            border-radius: 6px;
            border: 1px solid rgba(244, 204, 82, .75);
            background: linear-gradient(180deg, #f6d66c, #dcb342);
            color: #17170e;
            font-weight: 800;
            min-height: 44px;
            box-shadow: 0 12px 28px rgba(0,0,0,.24);
            margin-bottom: 8px;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        a[data-testid="stPageLink-NavLink"]:hover {{
            border-color: #ffe08a;
            color: #11140f;
            transform: translateY(-1px);
        }}

        .stButton > button:disabled {{
            background: #3b403b;
            color: rgba(255,255,255,.45);
            border-color: rgba(255,255,255,.12);
            box-shadow: none;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(32, 37, 31, .92);
            border: 1px solid rgba(102, 186, 242, .48);
            border-radius: 8px;
            box-shadow: 0 16px 36px rgba(0,0,0,.24);
            padding: 1rem 1.1rem 1.05rem;
            margin: .35rem 0 1rem;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {{
            gap: .75rem;
        }}

        [data-testid="column"] > div {{
            height: 100%;
        }}

        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stAlert"],
        div[data-testid="stMetric"] {{
            margin-top: .2rem;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,.08);
        }}

        .brand-header {{
            position: relative;
            min-height: 176px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            padding: 28px 30px;
            border: 1px solid rgba(148, 211, 141, .45);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(88, 184, 90, .18), rgba(102, 186, 242, .05) 42%, rgba(244, 204, 82, .12)),
                linear-gradient(180deg, rgba(255,255,255,.065), rgba(255,255,255,.015)),
                #20251f;
            overflow: hidden;
            box-shadow: 0 22px 54px rgba(0,0,0,.28);
        }}

        .brand-header::before {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0 46%, rgba(255,255,255,.06) 46% 47%, transparent 47%),
                repeating-linear-gradient(0deg, rgba(255,255,255,.028) 0, rgba(255,255,255,.028) 1px, transparent 1px, transparent 18px);
            pointer-events: none;
        }}

        .brand-copy {{
            position: relative;
            z-index: 1;
        }}

        .brand-kicker {{
            color: var(--mallards-gold);
            font-size: .85rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .brand-title {{
            margin: 0;
            color: #fffdf5;
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-size: clamp(2.15rem, 4.2vw, 4.5rem);
            line-height: .98;
            font-weight: 700;
        }}

        .brand-title .green {{
            color: var(--mallards-green-soft);
        }}

        .brand-subline {{
            margin-top: 12px;
            color: var(--mallards-muted);
            font-size: clamp(.95rem, 1.2vw, 1.08rem);
            max-width: 720px;
        }}

        .brand-monogram {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(2, 74px);
            grid-template-rows: repeat(2, 52px);
            gap: 8px;
            transform: skewX(-8deg);
        }}

        .brand-monogram span {{
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,.18);
            color: #10140f;
            font-weight: 900;
            font-size: 1.15rem;
        }}

        .brand-monogram span:nth-child(1),
        .brand-monogram span:nth-child(4) {{
            background: var(--mallards-green-soft);
        }}

        .brand-monogram span:nth-child(2),
        .brand-monogram span:nth-child(3) {{
            background: var(--mallards-gold);
        }}

        .section-label {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 1.15rem 0 .75rem;
            color: var(--mallards-paper);
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
        }}

        .section-label::after {{
            content: "";
            height: 1px;
            flex: 1;
            background: linear-gradient(90deg, rgba(244,204,82,.8), rgba(148,211,141,.38), transparent);
        }}

        .stat-card {{
            border: 1px solid var(--accent);
            background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015)), #272d29;
            border-radius: 8px;
            padding: 16px 18px;
            min-height: 104px;
            box-shadow: 0 14px 32px rgba(0,0,0,.22);
        }}

        .stat-label {{
            color: var(--mallards-muted);
            font-size: .86rem;
            font-weight: 800;
            text-transform: uppercase;
        }}

        .stat-value {{
            color: var(--accent);
            font-size: 2rem;
            line-height: 1.05;
            margin-top: 8px;
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0 10px;
            border-radius: 999px;
            font-size: .82rem;
            font-weight: 800;
            border: 1px solid var(--pill-border);
            color: var(--pill-color);
            background: var(--pill-bg);
        }}

        .result-panel {{
            margin-top: 1rem;
            padding: 18px;
            border-radius: 8px;
            border: 1px solid rgba(148, 211, 141, .65);
            background: linear-gradient(135deg, rgba(88,184,90,.14), rgba(102,186,242,.08)), #20251f;
        }}

        .result-title {{
            color: var(--mallards-green-soft);
            font-weight: 900;
            font-size: 1.1rem;
        }}

        .result-copy {{
            color: var(--mallards-muted);
            margin-top: 4px;
            font-size: .95rem;
        }}

        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: .8rem;
            }}
            .brand-header {{
                align-items: flex-start;
                flex-direction: column;
            }}
            .brand-monogram {{
                grid-template-columns: repeat(4, 52px);
                grid-template-rows: 42px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand_header(kicker: str, title_html: str, subline: str) -> None:
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-copy">
                <div class="brand-kicker">{html.escape(kicker)}</div>
                <h1 class="brand-title">{title_html}</h1>
                <div class="brand-subline">{html.escape(subline)}</div>
            </div>
            <div class="brand-monogram" aria-hidden="true">
                <span>MM</span><span>NM</span><span>FI</span><span>26</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_to_page(page_key: str) -> None:
    st.session_state["active_page"] = page_key
    st.rerun()


def render_sidebar_nav(current_page: str) -> str:
    nav_items = [
        ("builder", "Fan Master Builder"),
        ("dashboard", "Fan Dashboard"),
        ("survey", "Survey Insights"),
    ]

    requested_page = current_page

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-kicker">Madison</div>
                <div class="sidebar-title"><span>Mallards</span> Fan Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for key, label in nav_items:
            if key == current_page:
                st.markdown(
                    f'<div class="sidebar-nav-active">{html.escape(label)}</div>',
                    unsafe_allow_html=True,
                )
            elif st.button(label, key=f"nav-{key}", use_container_width=True):
                requested_page = key

    return requested_page


def section_label(label: str) -> None:
    st.markdown(f'<div class="section-label">{html.escape(label)}</div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, accent: str) -> str:
    return f"""
    <div class="stat-card" style="--accent:{accent};">
        <div class="stat-label">{html.escape(label)}</div>
        <div class="stat-value">{html.escape(value)}</div>
    </div>
    """


def status_pill(label: str, tone: str) -> str:
    tones = {
        "good": ("rgba(148,211,141,.14)", "rgba(148,211,141,.78)", "#94d38d"),
        "warn": ("rgba(244,204,82,.14)", "rgba(244,204,82,.78)", "#f4cc52"),
        "bad": ("rgba(225,108,91,.14)", "rgba(225,108,91,.78)", "#e16c5b"),
    }
    bg, border, color = tones.get(tone, tones["warn"])
    return (
        f'<span class="status-pill" style="--pill-bg:{bg};--pill-border:{border};'
        f'--pill-color:{color};">{html.escape(label)}</span>'
    )
