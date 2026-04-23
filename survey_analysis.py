from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[assignment]


POSTGAME_HINTS = [
    "rank your experience",
    "date of the game you attended",
    "order food and beverages",
    "wait in line",
    "on-field presentation",
    "promotional ideas",
]

POSTSEASON_HINTS = [
    "post season",
    "cleanliness of the stadium",
    "ballpark cleanliness issues",
    "annual household income",
    "do you own your home",
    "support a business because they support",
]


TOPIC_RULES: list[tuple[str, list[str]]] = [
    ("Food & Beverage", ["food", "beverage", "concession", "stand", "drink", "beer", "soda", "fries", "menu", "line"]),
    ("Checkout & Tech", ["mashgin", "automated checkout", "checkout device", "device", "kiosk"]),
    ("Entertainment & On-Field", ["on-field", "presentation", "music", "promotion", "promotional", "announcer", "between innings"]),
    ("Cleanliness & Facilities", ["clean", "cleanliness", "restroom", "bathroom", "trash", "bins", "facility"]),
    ("Staff & Service", ["staff", "employee", "security", "service", "usher"]),
    ("Value & Pricing", ["price", "cost", "value", "expensive", "affordable"]),
    ("Seating & Comfort", ["seat", "seating", "view", "umbrella", "shade", "comfort"]),
    ("Parking & Access", ["parking", "gate", "entry", "entrance", "traffic", "walk"]),
    ("Brand & Marketing", ["hear about", "facebook", "instagram", "email", "promotion (if any)"]),
    ("Overall Experience", ["overall", "rank your experience", "fan experience", "better fan experience"]),
]


IGNORE_QUESTION_SUBSTRINGS = [
    "agree to share your info",
    "by checking this box",
]


META_ALIASES = {
    "timestamp": ["Timestamp", "timestamp", "Submitted At", "submitted_at"],
    "email": ["Email", "Email Address", "email", "Email address"],
    "first_name": ["First Name", "first name", "FirstName"],
    "last_name": ["Last Name", "last name", "LastName"],
    "zip_code": ["Your ZIP Code", "What is your Zip Code?", "Zip Code", "ZIP", "Zip"],
    "phone": ["Phone Number", "Phone", "phone"],
    "gender": ["Gender", "gender"],
    "age": ["How old are you?", "Age", "age"],
    "game_date": ["What was the date of the game you attended?", "Game Date", "game date"],
}


@dataclass(frozen=True)
class SurveyFileClassification:
    filename: str
    kind: str | None
    team: str | None
    year: int | None
    message: str
    sheet_count: int


def _to_bytes(source: bytes | BinaryIO) -> bytes:
    if isinstance(source, bytes):
        return source
    source.seek(0)
    return source.read()


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(20\d{2})", filename)
    return int(match.group(1)) if match else None


def _extract_team(filename: str) -> str | None:
    lowered = filename.lower()
    if "night mares" in lowered or "nightmares" in lowered or "nm" in lowered:
        return "Night Mares"
    if "mallard" in lowered:
        return "Mallards"
    return None


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def _topic_for_question(question: str) -> str:
    lowered = question.lower()
    for topic, keywords in TOPIC_RULES:
        for keyword in keywords:
            if keyword in lowered:
                return topic
    return "Other"


def _question_is_ignored(question: str) -> bool:
    lowered = question.lower()
    return any(fragment in lowered for fragment in IGNORE_QUESTION_SUBSTRINGS)


def _guess_kind(all_columns: Iterable[str], filename: str) -> str | None:
    columns = [str(col).strip().lower() for col in all_columns if str(col).strip()]
    filename_lower = filename.lower()

    pg = sum(any(hint in col for col in columns) for hint in POSTGAME_HINTS)
    ps = sum(any(hint in col for col in columns) for hint in POSTSEASON_HINTS)

    # Some postseason files are Google Forms exports with a "Form Responses 1" sheet.
    if "post season" in filename_lower or "postseason" in filename_lower:
        ps += 2
    if "post game" in filename_lower or "postgame" in filename_lower:
        pg += 2

    if max(pg, ps) == 0:
        return None
    if pg >= ps:
        return "postgame"
    return "postseason"


def classify_survey_file(filename: str, source: bytes | BinaryIO) -> SurveyFileClassification:
    try:
        payload = _to_bytes(source)
        xl = pd.ExcelFile(io.BytesIO(payload))
        sheet_count = len(xl.sheet_names)
        columns_union: set[str] = set()
        for sheet_name in xl.sheet_names:
            try:
                cols = list(xl.parse(sheet_name, nrows=0).columns)
            except Exception:
                continue
            columns_union.update(str(col).strip() for col in cols if str(col).strip())

        kind = _guess_kind(columns_union, filename)
        team = _extract_team(filename)
        year = _extract_year(filename)

        if kind is None:
            return SurveyFileClassification(
                filename=filename,
                kind=None,
                team=team,
                year=year,
                message="Could not recognize this workbook as a postgame or postseason survey export.",
                sheet_count=sheet_count,
            )

        label = "Post-game survey" if kind == "postgame" else "Post-season survey"
        return SurveyFileClassification(
            filename=filename,
            kind=kind,
            team=team,
            year=year,
            message=label,
            sheet_count=sheet_count,
        )
    except Exception as exc:
        return SurveyFileClassification(
            filename=filename,
            kind=None,
            team=_extract_team(filename),
            year=_extract_year(filename),
            message=f"Unreadable workbook: {exc}",
            sheet_count=0,
        )


def load_survey_workbook(
    filename: str,
    source: bytes | BinaryIO,
    *,
    kind: str | None = None,
    team: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    payload = _to_bytes(source)
    xl = pd.ExcelFile(io.BytesIO(payload))

    frames: list[pd.DataFrame] = []
    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name)
        except Exception:
            continue
        if df.empty:
            continue

        df.columns = [str(col).strip() for col in df.columns]

        # Skip summary sheets.
        sheet_lower = str(sheet_name).lower().strip()
        if "average" in sheet_lower or "summary" in sheet_lower:
            continue

        # Heuristic: keep sheets that look like response tables.
        has_timestamp = any(col in df.columns for col in META_ALIASES["timestamp"])
        has_email = any(col in df.columns for col in META_ALIASES["email"])
        if not has_timestamp and not has_email:
            continue

        df["_source_sheet"] = str(sheet_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_source_file"] = filename
    combined["survey_type"] = kind or _guess_kind(combined.columns, filename)
    combined["team"] = team or _extract_team(filename)
    combined["survey_year"] = year or _extract_year(filename)
    return combined


def _rename_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, aliases in META_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def _guess_question_kind(question: str, responses: pd.Series) -> str:
    lowered = question.lower()

    if any(fragment in lowered for fragment in ["rank your experience", "rate", "from 1", "scale of 1", "1-10", "1 to 10"]):
        return "rating"
    if "how long did you wait" in lowered or "wait in line" in lowered:
        return "duration"
    if lowered.startswith("did you") or lowered.startswith("do you") or lowered.startswith("are you"):
        sample = responses.dropna().astype(str).str.strip().str.lower().head(50)
        if not sample.empty:
            yn = sample.isin(["yes", "no", "y", "n", "true", "false"])
            if yn.mean() >= 0.85:
                return "choice"

    avg_len = responses.dropna().astype(str).map(len).mean() if responses.notna().any() else 0
    if any(fragment in lowered for fragment in ["comment", "improve", "please explain", "what are some", "ideas"]) or avg_len >= 28:
        return "text"

    return "other"


def _parse_duration_minutes(value: str) -> float | None:
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(min|mins|minute|minutes)?", text)
    if not match:
        return None
    return float(match.group(1))


def build_survey_long(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, ignore_index=True, sort=False)
    wide.columns = [str(col).strip() for col in wide.columns]
    wide = _rename_meta_columns(wide)

    meta_columns = [
        "survey_type",
        "team",
        "survey_year",
        "_source_file",
        "_source_sheet",
        "timestamp",
        "email",
        "first_name",
        "last_name",
        "zip_code",
        "phone",
        "gender",
        "age",
        "game_date",
    ]
    meta_columns = [col for col in meta_columns if col in wide.columns]

    # Question columns are anything not in metadata or internal columns.
    question_columns = [
        col
        for col in wide.columns
        if col not in meta_columns and not col.startswith("_") and not _question_is_ignored(col)
    ]

    melted = wide.melt(
        id_vars=meta_columns,
        value_vars=question_columns,
        var_name="question",
        value_name="response_raw",
    )

    melted["response_raw"] = melted["response_raw"].map(_normalize_text)
    melted = melted[melted["response_raw"] != ""].copy()

    if "timestamp" in melted.columns:
        melted["timestamp"] = pd.to_datetime(melted["timestamp"], errors="coerce")

    # Build per-question kind mapping using the wide table for stability.
    kind_map: dict[str, str] = {}
    for question in question_columns:
        kind_map[question] = _guess_question_kind(question, wide[question] if question in wide.columns else pd.Series(dtype=object))

    melted["question_kind"] = melted["question"].map(kind_map).fillna("other")
    melted["topic"] = melted["question"].map(_topic_for_question)

    melted["response_text"] = np.where(melted["question_kind"] == "text", melted["response_raw"], "")
    melted["value_num"] = np.nan

    rating_mask = melted["question_kind"].eq("rating")
    if rating_mask.any():
        melted.loc[rating_mask, "value_num"] = pd.to_numeric(melted.loc[rating_mask, "response_raw"], errors="coerce")

    duration_mask = melted["question_kind"].eq("duration")
    if duration_mask.any():
        melted.loc[duration_mask, "value_num"] = (
            melted.loc[duration_mask, "response_raw"].map(_parse_duration_minutes).astype(float)
        )

    choice_mask = melted["question_kind"].eq("choice")
    if choice_mask.any():
        normalized = melted.loc[choice_mask, "response_raw"].str.strip().str.lower()
        melted.loc[choice_mask, "value_num"] = normalized.isin(["yes", "y", "true"]).astype(float)

    return melted.reset_index(drop=True)


def _sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def score_sentiment(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    df = long_df.copy()
    text_mask = df["question_kind"].eq("text") & df["response_text"].astype(str).str.strip().ne("")
    if not text_mask.any():
        df["sentiment_compound"] = np.nan
        df["sentiment_label"] = ""
        return df

    analyzer = SentimentIntensityAnalyzer()

    def score_one(text: str) -> float:
        return analyzer.polarity_scores(text).get("compound", 0.0)

    df.loc[text_mask, "sentiment_compound"] = df.loc[text_mask, "response_text"].astype(str).map(score_one)
    df["sentiment_label"] = df["sentiment_compound"].map(lambda x: _sentiment_label(float(x)) if pd.notna(x) else "")
    df["sentiment_index"] = np.where(
        df["sentiment_compound"].notna(),
        (df["sentiment_compound"].astype(float) + 1.0) * 50.0,
        np.nan,
    )
    return df


def _top_terms(texts: list[str], limit: int = 8) -> list[str]:
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 2:
        return []

    if TfidfVectorizer is None:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    try:
        matrix = vectorizer.fit_transform(texts)
    except Exception:
        return []

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    if scores.size == 0:
        return []

    top_indices = scores.argsort()[-limit:][::-1]
    top_terms = [terms[i] for i in top_indices if scores[i] > 0]
    return top_terms[:limit]


def _short_quote(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def summarize_surveys(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if long_df.empty:
        return {
            "topic_text": pd.DataFrame(),
            "topic_rating": pd.DataFrame(),
            "question_text": pd.DataFrame(),
            "question_rating": pd.DataFrame(),
            "opportunities": pd.DataFrame(),
            "delighters": pd.DataFrame(),
        }

    df = long_df.copy()

    # Text sentiment summaries
    text = df[df["question_kind"].eq("text") & df["sentiment_compound"].notna()].copy()
    if not text.empty:
        topic_text = (
            text.groupby(["survey_type", "team", "survey_year", "topic"], dropna=False)
            .agg(
                responses=("response_text", "count"),
                avg_sentiment=("sentiment_compound", "mean"),
                negative_rate=("sentiment_label", lambda s: (s == "Negative").mean()),
                positive_rate=("sentiment_label", lambda s: (s == "Positive").mean()),
            )
            .reset_index()
        )
        question_text = (
            text.groupby(["survey_type", "team", "survey_year", "topic", "question"], dropna=False)
            .agg(
                responses=("response_text", "count"),
                avg_sentiment=("sentiment_compound", "mean"),
                negative_rate=("sentiment_label", lambda s: (s == "Negative").mean()),
            )
            .reset_index()
        )
    else:
        topic_text = pd.DataFrame()
        question_text = pd.DataFrame()

    # Numeric summaries (ratings, durations, choices)
    numeric = df[df["value_num"].notna()].copy()
    if not numeric.empty:
        topic_rating = (
            numeric.groupby(["survey_type", "team", "survey_year", "topic", "question_kind"], dropna=False)
            .agg(
                responses=("value_num", "count"),
                avg_value=("value_num", "mean"),
                p25=("value_num", lambda s: s.quantile(0.25)),
                p75=("value_num", lambda s: s.quantile(0.75)),
            )
            .reset_index()
        )
        question_rating = (
            numeric.groupby(["survey_type", "team", "survey_year", "topic", "question_kind", "question"], dropna=False)
            .agg(
                responses=("value_num", "count"),
                avg_value=("value_num", "mean"),
            )
            .reset_index()
        )
    else:
        topic_rating = pd.DataFrame()
        question_rating = pd.DataFrame()

    # Opportunities and delighters (text only)
    opportunities = pd.DataFrame()
    delighters = pd.DataFrame()
    if not text.empty and not topic_text.empty:
        enriched = topic_text.copy()
        enriched["opportunity_score"] = (
            enriched["negative_rate"].fillna(0) * np.log1p(enriched["responses"].fillna(0).astype(float))
        )
        enriched["delight_score"] = (
            enriched["positive_rate"].fillna(0) * np.log1p(enriched["responses"].fillna(0).astype(float))
        )

        def attach_terms(frame: pd.DataFrame, label: str) -> pd.DataFrame:
            def eq_or_isna(series: pd.Series, value) -> pd.Series:
                if pd.isna(value):
                    return series.isna()
                return series == value

            rows = []
            for _, row in frame.iterrows():
                subset = text[
                    eq_or_isna(text["survey_type"], row["survey_type"])
                    & eq_or_isna(text["team"], row["team"])
                    & eq_or_isna(text["survey_year"], row["survey_year"])
                    & eq_or_isna(text["topic"], row["topic"])
                ]
                if subset.empty:
                    rows.append([])
                    continue
                if label == "Negative":
                    subset = subset[subset["sentiment_label"] == "Negative"]
                else:
                    subset = subset[subset["sentiment_label"] == "Positive"]

                terms = _top_terms(subset["response_text"].astype(str).tolist())
                rows.append(", ".join(terms))
            frame = frame.copy()
            frame["top_terms"] = rows
            return frame

        opp = enriched.sort_values("opportunity_score", ascending=False).head(8)
        opp = attach_terms(opp, "Negative")
        opportunities = opp

        dl = enriched.sort_values("delight_score", ascending=False).head(6)
        dl = attach_terms(dl, "Positive")
        delighters = dl

    return {
        "topic_text": topic_text,
        "topic_rating": topic_rating,
        "question_text": question_text,
        "question_rating": question_rating,
        "opportunities": opportunities,
        "delighters": delighters,
    }


def top_quotes(long_df: pd.DataFrame, *, topic: str, sentiment: str, limit: int = 4) -> list[str]:
    df = long_df[
        long_df["question_kind"].eq("text")
        & long_df["topic"].eq(topic)
        & long_df["sentiment_label"].eq(sentiment)
        & long_df["response_text"].astype(str).str.strip().ne("")
    ].copy()
    if df.empty:
        return []

    df = df.sort_values("sentiment_compound", ascending=(sentiment == "Negative"))
    return [_short_quote(t) for t in df["response_text"].astype(str).head(limit).tolist()]
