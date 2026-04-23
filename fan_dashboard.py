from __future__ import annotations

import html
import io
from typing import BinaryIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_theme import PALETTE, brand_header, inject_base_css, section_label


CORE_COLUMNS = {
    "CLEAN_CITY",
    "CLEAN_STATE",
    "CLEAN_ZIP",
    "FAN_KEY",
    "FIRST_GAME",
    "TOTAL_TICKETS",
    "TOTAL_TICKET_PAID",
    "TICKETS_SCANNED",
    "MERCH_NET_TOTAL",
    "GAMES_ATTENDED",
    "SPORTS_ATTENDED",
    "MOST_COMMON_SECTION",
    "IS_MERCH_BUYER",
}

NUMERIC_COLUMNS = [
    "TOTAL_TICKETS",
    "TOTAL_TICKET_SPEND",
    "TOTAL_TICKET_PAID",
    "GAMES_ATTENDED",
    "TICKETS_SCANNED",
    "SCAN_RATE",
    "MERCH_GROSS_SALES",
    "MERCH_DISCOUNTS",
    "MERCH_SERVICE_CHARGES",
    "MERCH_PARTIAL_REFUNDS",
    "MERCH_NET_SALES",
    "MERCH_CARD",
    "MERCH_CASH",
    "MERCH_SQUARE_GIFT_CARD",
    "MERCH_OTHER_TENDER",
    "MERCH_FEES",
    "MERCH_TOTAL_COLLECTED",
    "MERCH_NET_TOTAL",
    "MERCH_TAX",
    "MERCH_TIP",
    "MERCH_GIFT_CARD_SALES",
    "MERCH_CASH_APP",
    "IS_MERCH_BUYER",
]

STATE_NAME_TO_ABBR = {
    "WISCONSIN": "WI",
    "ILLINOIS": "IL",
    "MINNESOTA": "MN",
    "IOWA": "IA",
    "MICHIGAN": "MI",
}

ZIP_COORDS = {
    "53703": (43.0747, -89.3841),
    "53704": (43.1192, -89.3553),
    "53705": (43.0723, -89.4590),
    "53711": (43.0350, -89.4512),
    "53713": (43.0364, -89.3846),
    "53714": (43.1009, -89.3129),
    "53715": (43.0674, -89.4012),
    "53716": (43.0713, -89.3145),
    "53717": (43.0709, -89.5240),
    "53718": (43.0976, -89.2832),
    "53719": (43.0345, -89.5011),
    "53726": (43.0719, -89.4232),
    "53508": (42.8594, -89.5337),
    "53523": (43.0042, -89.0167),
    "53527": (43.0761, -89.1996),
    "53528": (43.1142, -89.6557),
    "53531": (43.0517, -89.0757),
    "53532": (43.2477, -89.3437),
    "53533": (42.9603, -90.1301),
    "53534": (42.8367, -89.0726),
    "53536": (42.7789, -89.2990),
    "53551": (43.0814, -88.9118),
    "53555": (43.3149, -89.5260),
    "53558": (43.0125, -89.2898),
    "53559": (43.1686, -89.0665),
    "53562": (43.0980, -89.5043),
    "53566": (42.6011, -89.6385),
    "53572": (43.0086, -89.7385),
    "53575": (42.9261, -89.3846),
    "53578": (43.2864, -89.7240),
    "53589": (42.9169, -89.2179),
    "53590": (43.1836, -89.2137),
    "53593": (42.9908, -89.5332),
    "53597": (43.1919, -89.4557),
    "53598": (43.2187, -89.3421),
    "53901": (43.5391, -89.4626),
    "53913": (43.4711, -89.7443),
    "53916": (43.4578, -88.8373),
    "53925": (43.3387, -89.0154),
    "53955": (43.3903, -89.4112),
}

CITY_COORDS = {
    "MADISON": (43.0747, -89.3841),
    "SUN PRAIRIE": (43.1836, -89.2137),
    "WAUNAKEE": (43.1919, -89.4557),
    "MIDDLETON": (43.0980, -89.5043),
    "VERONA": (42.9908, -89.5332),
    "DEFOREST": (43.2477, -89.3437),
    "MCFARLAND": (43.0125, -89.2898),
    "COTTAGE GROVE": (43.0761, -89.1996),
    "STOUGHTON": (42.9169, -89.2179),
    "OREGON": (42.9261, -89.3846),
    "MOUNT HOREB": (43.0086, -89.7385),
    "BARABOO": (43.4711, -89.7443),
    "WINDSOR": (43.2187, -89.3421),
    "LODI": (43.3149, -89.5260),
    "COLUMBUS": (43.3387, -89.0154),
    "MARSHALL": (43.1686, -89.0665),
    "BELLEVILLE": (42.8594, -89.5337),
    "JANESVILLE": (42.6828, -89.0187),
    "CROSS PLAINS": (43.1142, -89.6557),
    "EDGERTON": (42.8367, -89.0726),
    "DEERFIELD": (43.0517, -89.0757),
    "MILWAUKEE": (43.0389, -87.9065),
    "PORTAGE": (43.5391, -89.4626),
    "LAKE MILLS": (43.0814, -88.9118),
    "DODGEVILLE": (42.9603, -90.1301),
    "POYNETTE": (43.3903, -89.4112),
    "CAMBRIDGE": (43.0042, -89.0167),
    "PRAIRIE DU SAC": (43.2864, -89.7240),
    "BEAVER DAM": (43.4578, -88.8373),
    "MONROE": (42.6011, -89.6385),
    "EVANSVILLE": (42.7789, -89.2990),
}


def read_master_csv(source: bytes | BinaryIO) -> pd.DataFrame:
    if isinstance(source, bytes):
        buffer = io.BytesIO(source)
    else:
        source.seek(0)
        buffer = source

    return pd.read_csv(
        buffer,
        dtype={"CLEAN_ZIP": "string"},
        low_memory=False,
        keep_default_na=True,
    )


def missing_core_columns(df: pd.DataFrame) -> list[str]:
    return sorted(CORE_COLUMNS.difference(df.columns))


def prepare_master(df: pd.DataFrame) -> pd.DataFrame:
    master = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)

    for col in ["CLEAN_CITY", "CLEAN_STATE", "SPORTS_ATTENDED", "MOST_COMMON_SECTION", "TICKET_TYPE"]:
        if col in master.columns:
            master[col] = master[col].astype("string").str.strip()

    if "CLEAN_CITY" in master.columns:
        master["CLEAN_CITY"] = master["CLEAN_CITY"].str.upper()

    if "CLEAN_STATE" in master.columns:
        states = master["CLEAN_STATE"].str.upper()
        master["CLEAN_STATE"] = states.replace(STATE_NAME_TO_ABBR)

    if "CLEAN_ZIP" in master.columns:
        zips = master["CLEAN_ZIP"].astype("string").str.strip()
        zips = zips.str.replace(r"\.0$", "", regex=True)
        zips = zips.str.extract(r"(\d{5})", expand=False)
        master["CLEAN_ZIP"] = zips

    for col in ["FIRST_GAME", "LAST_GAME", "MERCH_FIRST_PURCHASE", "MERCH_LAST_PURCHASE"]:
        if col in master.columns:
            master[col] = pd.to_datetime(master[col], errors="coerce")

    master["TOTAL_REVENUE"] = master.get("TOTAL_TICKET_PAID", 0) + master.get("MERCH_NET_TOTAL", 0)
    master["MERCH_BUYER_FLAG"] = (master.get("IS_MERCH_BUYER", 0) > 0).astype(int)
    master["SECTION_GROUP"] = master.get("MOST_COMMON_SECTION", pd.Series(index=master.index, dtype="string")).apply(
        classify_section
    )
    return master


def classify_section(value) -> str:
    if pd.isna(value):
        return "Unknown"
    section = str(value).strip().upper()
    if not section:
        return "Unknown"
    if "DUCK BLIND" in section:
        return "Duck Blind"
    if "SUITE" in section or "ARCH SOLAR" in section or "LEFT FIELD" in section:
        return "Suites"
    if section[:1] == "1" and section[:3].isdigit():
        return "100 Level"
    if section[:1] == "2" and section[:3].isdigit():
        return "200 Level"
    if "GENERAL" in section or "GRANDSTAND" in section or "ROOFTOP" in section or section.endswith(" GA"):
        return "General Admission"
    return "Other"


def inject_dashboard_css() -> None:
    inject_base_css()
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1540px;
            padding-bottom: 3rem;
        }
        .section-rule {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 1.15rem 0 .65rem;
            color: var(--mallards-paper);
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .section-rule::after {
            content: "";
            height: 2px;
            flex: 1;
            background: linear-gradient(90deg, rgba(244,204,82,.8), rgba(148,211,141,.38), transparent);
        }
        .kpi-card {
            border: 1px solid var(--accent);
            border-top-width: 4px;
            background:
                linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.012)),
                #272d29;
            border-radius: 8px;
            padding: 15px 18px 16px;
            min-height: 108px;
            box-shadow: 0 16px 34px rgba(0,0,0,.24);
        }
        .kpi-label {
            color: #dfe7d8;
            font-size: .88rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .kpi-value {
            color: var(--accent);
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-size: 2.35rem;
            line-height: 1.05;
            margin-top: 8px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.014)),
                #272d29;
            border-color: rgba(102, 186, 242, .42);
            border-radius: 8px;
            box-shadow: 0 16px 36px rgba(0,0,0,.24);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] h3 {
            color: #f4f1e7;
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-size: 1.2rem;
            font-weight: 650;
        }
        .sport-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
            padding: 4px 2px 8px;
        }
        .sport-tile {
            border: 1px solid var(--mallards-blue);
            border-radius: 8px;
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: rgba(102, 186, 242, .06);
        }
        .sport-number {
            color: var(--mallards-blue);
            font-family: "Bahnschrift", "Aptos Display", "Segoe UI", sans-serif;
            font-size: 2.55rem;
            line-height: 1;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }
        .sport-label {
            color: var(--mallards-muted);
            font-size: 1rem;
            margin-top: 8px;
        }
        .js-plotly-plot .plotly .modebar {
            display: none;
        }
        @media (max-width: 1100px) {
            .kpi-card {
                min-height: 96px;
            }
            .kpi-value {
                font-size: 1.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def compact_number(value: float) -> str:
    value = float(value or 0)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def compact_money(value: float) -> str:
    return f"${compact_number(value)}"


def percent(value: float) -> str:
    return f"{value:.2%}"


def kpi_card(label: str, value: str, color: str) -> str:
    return f"""
    <div class="kpi-card" style="--accent:{color};">
        <div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-value">{html.escape(value)}</div>
    </div>
    """


def section_rule(label: str) -> str:
    return f'<div class="section-rule"><span>{html.escape(label)}</span></div>'


def base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor="#272d29",
        plot_bgcolor="#272d29",
        font=dict(color="#f4f1e7", family="Aptos, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def bar_figure(x, y, color: str, height: int = 320, orientation: str = "v", text=None) -> go.Figure:
    if orientation == "h":
        fig = go.Figure(go.Bar(x=x, y=y, orientation="h", marker_color=color, text=text, textposition="outside"))
    else:
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color, text=text, textposition="outside"))
    return base_layout(fig, height)


def games_attended_counts(master: pd.DataFrame) -> pd.Series:
    games = master["GAMES_ATTENDED"].fillna(0)
    bins = [0, 2, 3, 5, 10, np.inf]
    labels = ["1 to 2", "2 to 3", "3 to 5", "5 to 10", "10+"]
    bucketed = pd.cut(games[games > 0], bins=bins, labels=labels, include_lowest=True, right=True)
    return bucketed.value_counts().reindex(labels, fill_value=0)


def day_of_week_counts(master: pd.DataFrame) -> pd.Series:
    dates = master.loc[master["FIRST_GAME"].notna()].copy()
    if "TICKET_TYPE" in dates.columns:
        dates = dates[dates["TICKET_TYPE"].astype("string").str.upper().eq("SINGLE")]
    counts = dates["FIRST_GAME"].dt.day_name().value_counts()
    return counts.sort_values(ascending=False)


def sport_counts(master: pd.DataFrame) -> dict[str, int]:
    sports = master["SPORTS_ATTENDED"].astype("string").str.lower().fillna("")
    has_baseball = sports.str.contains("baseball", regex=False)
    has_softball = sports.str.contains("softball", regex=False)
    return {
        "Baseball Only": int((has_baseball & ~has_softball).sum()),
        "Softball Only": int((has_softball & ~has_baseball).sum()),
        "Both Sports": int((has_baseball & has_softball).sum()),
        "Unknown": int((~has_baseball & ~has_softball).sum()),
    }


def top_city_counts(master: pd.DataFrame, limit: int = 10) -> pd.Series:
    cities = master["CLEAN_CITY"].dropna()
    cities = cities[cities.astype(str).str.len() > 0]
    return cities.value_counts().head(limit).sort_values()


def zip_map_data(master: pd.DataFrame, limit: int = 40) -> pd.DataFrame:
    working = master.dropna(subset=["CLEAN_ZIP"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["CLEAN_ZIP", "fans", "city", "lat", "lon"])

    counts = working["CLEAN_ZIP"].value_counts().head(limit).rename_axis("CLEAN_ZIP").reset_index(name="fans")
    city_lookup = (
        working.dropna(subset=["CLEAN_CITY"])
        .groupby("CLEAN_ZIP")["CLEAN_CITY"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
        .reset_index(name="city")
    )
    counts = counts.merge(city_lookup, on="CLEAN_ZIP", how="left")

    def coords(row):
        zip_code = str(row["CLEAN_ZIP"])
        city = str(row.get("city", "")).upper()
        return ZIP_COORDS.get(zip_code) or CITY_COORDS.get(city) or (np.nan, np.nan)

    points = counts.apply(coords, axis=1, result_type="expand")
    counts["lat"] = points[0]
    counts["lon"] = points[1]
    return counts.dropna(subset=["lat", "lon"])


def zip_map_figure(data: pd.DataFrame) -> go.Figure:
    if data.empty:
        return base_layout(go.Figure(), 330)

    max_fans = max(data["fans"].max(), 1)
    sizes = 12 + (data["fans"] / max_fans * 34)

    fig = go.Figure(
        go.Scattermapbox(
            lat=data["lat"],
            lon=data["lon"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=data["fans"],
                colorscale=[[0, "#66baf2"], [0.5, "#ffd552"], [1, "#4caf50"]],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Fans", font=dict(color="#f4f1e7")),
                    tickfont=dict(color="#f4f1e7"),
                ),
                opacity=0.82,
            ),
            text=data["CLEAN_ZIP"],
            customdata=np.stack([data["city"].fillna(""), data["fans"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]:,} fans<extra></extra>",
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=43.09, lon=-89.39),
            zoom=8.35,
        ),
        height=330,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#272d29",
        font=dict(color="#f4f1e7", family="Aptos, Segoe UI, sans-serif"),
    )
    return fig


def average_spend_by_city(master: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    working = master[
        master["CLEAN_STATE"].astype("string").str.upper().eq("WI") & master["CLEAN_CITY"].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=["City", "Average Total Spend", "Fans"])

    grouped = (
        working.groupby("CLEAN_CITY")
        .agg(**{"Average Total Spend": ("TOTAL_REVENUE", "mean"), "Fans": ("FAN_KEY", "count")})
        .query("Fans >= 10")
        .sort_values("Average Total Spend", ascending=False)
        .head(limit)
        .reset_index()
        .rename(columns={"CLEAN_CITY": "City"})
    )
    grouped["Average Total Spend"] = grouped["Average Total Spend"].round(2)
    return grouped


def spend_bucket_counts(master: pd.DataFrame) -> pd.Series:
    bins = [-0.01, 20, 50, 100, 200, 500, np.inf]
    labels = ["0 to 20", "20 to 50", "50 to 100", "100 to 200", "200 to 500", "500+"]
    bucketed = pd.cut(master["TOTAL_REVENUE"].fillna(0), bins=bins, labels=labels)
    return bucketed.value_counts().reindex(labels, fill_value=0)


def section_merch_rates(master: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        master.groupby("SECTION_GROUP")
        .agg(rate=("MERCH_BUYER_FLAG", "mean"), fans=("FAN_KEY", "count"))
        .query("fans >= 50")
        .sort_values("rate", ascending=False)
        .head(6)
        .reset_index()
    )
    grouped["rate_pct"] = grouped["rate"] * 100
    return grouped


def table_figure(df: pd.DataFrame, height: int = 330) -> go.Figure:
    if df.empty:
        return base_layout(go.Figure(), height)

    avg = df["Average Total Spend"]
    span = max(avg.max() - avg.min(), 1)
    normalized = (avg - avg.min()) / span
    blue_cells = [f"rgba(42, {145 + int(v * 45)}, 226, .95)" for v in normalized]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[1.15, 1.5, .85],
                header=dict(
                    values=["<b>City</b>", "<b>Average Total Spend</b>", "<b>Fans</b>"],
                    align=["left", "center", "right"],
                    fill_color="#20251f",
                    font=dict(color="#f4f1e7", size=14),
                    line_color="rgba(255,255,255,.08)",
                    height=34,
                ),
                cells=dict(
                    values=[
                        df["City"],
                        df["Average Total Spend"].map(lambda value: f"{value:,.2f}"),
                        df["Fans"].map(lambda value: f"{value:,}"),
                    ],
                    align=["left", "center", "right"],
                    fill_color=[["#272d29"] * len(df), blue_cells, ["#272d29"] * len(df)],
                    font=dict(color="#f5f7fb", size=13),
                    line_color="rgba(255,255,255,.05)",
                    height=34,
                ),
            )
        ]
    )
    return base_layout(fig, height)


def donut_figure(master: pd.DataFrame) -> go.Figure:
    buyers = int(master["MERCH_BUYER_FLAG"].sum())
    non_buyers = max(len(master) - buyers, 0)
    fig = go.Figure(
        go.Pie(
            labels=["Merch Buyers", "No Merch"],
            values=[buyers, non_buyers],
            hole=.58,
            marker=dict(colors=[PALETTE["green"], "#a7aca2"], line=dict(color="#272d29", width=4)),
            textinfo="percent",
            textfont=dict(color="#ffffff", size=16),
            sort=False,
        )
    )
    fig.update_traces(hovertemplate="%{label}<br>%{value:,} fans<extra></extra>")
    return base_layout(fig, 330)


def render_dashboard(master: pd.DataFrame) -> None:
    import streamlit as st

    master = prepare_master(master)

    brand_header(
        "Fan Intelligence Dashboard",
        'Madison <span class="green">Mallards</span> & Night Mares',
        "Attendance, revenue, geography, merchandise, and seating intelligence",
    )

    ticket_revenue = master["TOTAL_TICKET_PAID"].sum()
    merch_revenue = master["MERCH_NET_TOTAL"].sum()
    total_revenue = ticket_revenue + merch_revenue
    buyer_rate = master["MERCH_BUYER_FLAG"].mean() if len(master) else 0
    total_tickets = master["TOTAL_TICKETS"].sum()
    scan_rate = master["TICKETS_SCANNED"].sum() / total_tickets if total_tickets else 0

    kpi_columns = st.columns(6)
    kpis = [
        ("Total Fans", f"{len(master):,}", PALETTE["green_soft"]),
        ("Total Revenue", compact_money(total_revenue), PALETTE["green_soft"]),
        ("Ticket Revenue", compact_money(ticket_revenue), PALETTE["gold"]),
        ("Merch Revenue", compact_money(merch_revenue), PALETTE["gold"]),
        ("Merch Buyer Rate", percent(buyer_rate), PALETTE["blue"]),
        ("Ticket Scan Rate", percent(scan_rate), PALETTE["blue"]),
    ]
    for column, (label, value, color) in zip(kpi_columns, kpis):
        column.markdown(kpi_card(label, value, color), unsafe_allow_html=True)

    section_label("Attendance & Engagement")
    left, middle, right = st.columns([1.05, 1.05, 1.05])
    with left.container(border=True):
        st.subheader("Games Attended per Fan")
        games = games_attended_counts(master)
        fig = bar_figure(games.index.astype(str), games.values, PALETTE["green"], height=300)
        fig.update_yaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with middle.container(border=True):
        st.subheader("Ticket Sales by Day of Week")
        dow = day_of_week_counts(master)
        fig = bar_figure(dow.index.astype(str), dow.values, PALETTE["gold"], height=300)
        fig.update_xaxes(tickangle=-25)
        fig.update_yaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right.container(border=True):
        st.subheader("Sport Attended")
        sports = sport_counts(master)
        st.markdown(
            f"""
            <div class="sport-grid">
                <div class="sport-tile"><div class="sport-number">{sports["Baseball Only"]:,}</div><div class="sport-label">Baseball Only</div></div>
                <div class="sport-tile"><div class="sport-number">{sports["Softball Only"]:,}</div><div class="sport-label">Softball Only</div></div>
                <div class="sport-tile"><div class="sport-number">{sports["Both Sports"]:,}</div><div class="sport-label">Both Sports</div></div>
                <div class="sport-tile"><div class="sport-number">{sports["Unknown"]:,}</div><div class="sport-label">Unknown</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_label("Geography")
    geo_left, geo_mid, geo_right = st.columns([1.05, 1.05, 1.05])
    with geo_left.container(border=True):
        st.subheader("Top Cities by Total Fans")
        cities = top_city_counts(master)
        fig = bar_figure(cities.values, cities.index.astype(str), PALETTE["green"], height=330, orientation="h")
        fig.update_xaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with geo_mid.container(border=True):
        st.subheader("Top ZIP Codes by Total Fans")
        fig = zip_map_figure(zip_map_data(master))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with geo_right.container(border=True):
        st.subheader("Average Total Spend by City - WI Only")
        fig = table_figure(average_spend_by_city(master))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section_label("Merchandise & Seating")
    merch_left, merch_mid, merch_right = st.columns([1.05, 1.05, 1.05])
    with merch_left.container(border=True):
        st.subheader("Merchandise Buyers")
        st.plotly_chart(donut_figure(master), use_container_width=True, config={"displayModeBar": False})

    with merch_mid.container(border=True):
        st.subheader("Total Spend per Fan")
        spend = spend_bucket_counts(master)
        fig = bar_figure(spend.index.astype(str), spend.values, PALETTE["gold"], height=330, text=[compact_number(v) for v in spend.values])
        fig.update_xaxes(tickangle=0)
        fig.update_yaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with merch_right.container(border=True):
        st.subheader("Merchandise Purchase Rate by Seating Section")
        rates = section_merch_rates(master)
        fig = bar_figure(rates["SECTION_GROUP"], rates["rate_pct"], PALETTE["blue"], height=330)
        fig.update_yaxes(ticksuffix="%", range=[0, max(40, rates["rate_pct"].max() * 1.18 if not rates.empty else 40)])
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
