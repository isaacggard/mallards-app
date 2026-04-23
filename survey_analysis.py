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


DEFAULT_REVIEW_COLUMNS = [
    "What improvements can be made to the food and beverage experience from a fan perspective?",
    "Please provide any additional comments on on-field promotions you enjoyed, did not enjoy, or would like to see in the future.",
    "What was your favorite part of coming to the Mallards game?",
]


DEFAULT_RATING_COLUMNS = [
    "Overall, how was your ticketing experience?",
    "Overall, what was your food and beverage experience?",
    "Overall, what was your experience in the Paul Davis Team Store?",
    "Overall, please rate the On-Field Promotions on a scale of 1 to 10.",
    "On a scale of 1-10, how good of a value would you say your Mallards game experience was?",
]


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
    if "mallard" in lowered or "mallards" in lowered:
        return "Mallards"
    return None


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def parse_configured_columns(raw_text: str | None) -> list[str]:
    if not raw_text:
        return []

    parts: list[str] = []
    for line in str(raw_text).replace("\r", "\n").split("\n"):
        item = line.strip()
        if item:
            parts.append(item)

    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = _normalize_key(part)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(part)
    return ordered


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

        sheet_lower = str(sheet_name).lower().strip()
        if "average" in sheet_lower or "summary" in sheet_lower:
            continue

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


def _guess_question_kind(
    question: str,
    responses: pd.Series,
    review_keys: set[str] | None = None,
    rating_keys: set[str] | None = None,
) -> str:
    lowered = question.lower()
    normalized = _normalize_key(question)
    review_keys = review_keys or set()
    rating_keys = rating_keys or set()

    if normalized in rating_keys:
        return "rating"
    if normalized in review_keys:
        return "text"

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
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes)?", text)
    if not match:
        return None
    return float(match.group(1))


def _infer_rating_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 10.0

    max_value = float(numeric.max())
    if max_value <= 5.0:
        return 5.0
    if max_value <= 10.0:
        return 10.0
    return max(10.0, max_value)


def resolve_sentiment_columns(
    wide_df: pd.DataFrame,
    *,
    review_columns: list[str] | None = None,
    rating_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    if wide_df.empty:
        return [], []

    df = wide_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = _rename_meta_columns(df)

    review_keys = {_normalize_key(col) for col in (review_columns or []) if _normalize_key(col)}
    rating_keys = {_normalize_key(col) for col in (rating_columns or []) if _normalize_key(col)}

    meta_columns = {
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
    }
    question_columns = [
        col for col in df.columns if col not in meta_columns and not col.startswith("_") and not _question_is_ignored(col)
    ]

    resolved_review: list[str] = []
    resolved_rating: list[str] = []

    for question in question_columns:
        normalized = _normalize_key(question)
        if normalized in rating_keys:
            resolved_rating.append(question)
        elif normalized in review_keys:
            resolved_review.append(question)

    for question in question_columns:
        if question in resolved_review or question in resolved_rating:
            continue
        kind = _guess_question_kind(question, df[question], set(), set())
        if kind == "rating":
            resolved_rating.append(question)
        elif kind == "text":
            resolved_review.append(question)

    return resolved_review, resolved_rating


def build_survey_long(
    frames: list[pd.DataFrame],
    *,
    review_columns: list[str] | None = None,
    rating_columns: list[str] | None = None,
) -> pd.DataFrame:
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

    question_columns = [
        col
        for col in wide.columns
        if col not in meta_columns and not col.startswith("_") and not _question_is_ignored(col)
    ]

    review_keys = {_normalize_key(col) for col in (review_columns or []) if _normalize_key(col)}
    rating_keys = {_normalize_key(col) for col in (rating_columns or []) if _normalize_key(col)}

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

    kind_map: dict[str, str] = {}
    rating_scale_map: dict[str, float] = {}
    for question in question_columns:
        responses = wide[question] if question in wide.columns else pd.Series(dtype=object)
        kind = _guess_question_kind(question, responses, review_keys, rating_keys)
        kind_map[question] = kind
        if kind == "rating" and question in wide.columns:
            rating_scale_map[question] = _infer_rating_scale(wide[question])

    melted["question_kind"] = melted["question"].map(kind_map).fillna("other")
    melted["topic"] = melted["question"].map(_topic_for_question)
    melted["rating_scale_max"] = melted["question"].map(rating_scale_map).astype(float)

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


def _rating_thresholds(scale_max: float | None) -> tuple[float, float]:
    scale = 10.0 if pd.isna(scale_max) else float(scale_max)
    if scale <= 5.0:
        return 4.0, 2.0
    return 6.0, 4.0


def rating_to_compound(rating: float, scale_max: float | None = None) -> float:
    scale = 10.0 if pd.isna(scale_max) else float(scale_max)
    value = float(rating)
    if scale <= 5.0:
        return float(np.clip((value - 3.0) / 2.0, -1.0, 1.0))
    return float(np.clip((value - 5.0) / 5.0, -1.0, 1.0))


def get_sentiment(review: str | None, rating: float | None, scale_max: float | None = None) -> str:
    if pd.notna(rating):
        positive_floor, negative_ceiling = _rating_thresholds(scale_max)
        if float(rating) >= positive_floor:
            return "Positive"
        if float(rating) <= negative_ceiling:
            return "Negative"
        return "Neutral"

    if isinstance(review, str) and review.strip():
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(review).get("compound", 0.0)
        if score > 0.05:
            return "Positive"
        if score < -0.05:
            return "Negative"

    return "Neutral"


def score_sentiment(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    df = long_df.copy()
    df["sentiment_compound"] = np.nan
    df["sentiment_label"] = ""

    text_mask = df["question_kind"].eq("text") & df["response_text"].astype(str).str.strip().ne("")
    if text_mask.any():
        analyzer = SentimentIntensityAnalyzer()
        df.loc[text_mask, "sentiment_compound"] = df.loc[text_mask, "response_text"].astype(str).map(
            lambda text: analyzer.polarity_scores(text).get("compound", 0.0)
        )
        df.loc[text_mask, "sentiment_label"] = df.loc[text_mask, "sentiment_compound"].map(
            lambda value: _sentiment_label(float(value)) if pd.notna(value) else ""
        )

    rating_mask = df["question_kind"].eq("rating") & df["value_num"].notna()
    if rating_mask.any():
        df.loc[rating_mask, "sentiment_compound"] = df.loc[rating_mask].apply(
            lambda row: rating_to_compound(row["value_num"], row.get("rating_scale_max")),
            axis=1,
        )
        df.loc[rating_mask, "sentiment_label"] = df.loc[rating_mask].apply(
            lambda row: get_sentiment(None, row["value_num"], row.get("rating_scale_max")),
            axis=1,
        )

    df["sentiment_index"] = np.where(
        df["sentiment_compound"].notna(),
        (df["sentiment_compound"].astype(float) + 1.0) * 50.0,
        np.nan,
    )
    return df


def score_response_sentiment(
    wide_df: pd.DataFrame,
    *,
    review_columns: list[str] | None = None,
    rating_columns: list[str] | None = None,
) -> pd.DataFrame:
    if wide_df.empty:
        return wide_df.copy()

    df = wide_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = _rename_meta_columns(df)

    resolved_review, resolved_rating = resolve_sentiment_columns(
        df,
        review_columns=review_columns,
        rating_columns=rating_columns,
    )

    for column in resolved_rating:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    rating_scales = {
        column: _infer_rating_scale(df[column])
        for column in resolved_rating
        if column in df.columns
    }

    analyzer = SentimentIntensityAnalyzer()
    sentiments: list[str] = []
    scores: list[float] = []
    sources: list[str] = []
    drivers: list[str] = []

    for _, row in df.iterrows():
        sentiment = "Neutral"
        score = 0.0
        source = ""
        driver = ""

        for column in resolved_rating:
            if column not in df.columns or pd.isna(row.get(column)):
                continue
            rating = float(row[column])
            scale_max = rating_scales.get(column, 10.0)
            candidate_sentiment = get_sentiment(None, rating, scale_max)
            candidate_score = rating_to_compound(rating, scale_max)

            sentiment = candidate_sentiment
            score = candidate_score
            source = "Rating"
            driver = column

            if candidate_sentiment != "Neutral":
                break

        if sentiment == "Neutral":
            for column in resolved_review:
                if column not in df.columns or pd.isna(row.get(column)):
                    continue
                review = str(row[column]).strip()
                if not review:
                    continue

                candidate_score = analyzer.polarity_scores(review).get("compound", 0.0)
                candidate_sentiment = get_sentiment(review, None)

                sentiment = candidate_sentiment
                score = candidate_score
                source = "Review"
                driver = column

                if candidate_sentiment != "Neutral":
                    break

        sentiments.append(sentiment)
        scores.append(float(score))
        sources.append(source)
        drivers.append(driver)

    df["Sentiment"] = sentiments
    df["Sentiment Score"] = scores
    df["Sentiment Source"] = sources
    df["Sentiment Driver"] = drivers
    return df


def _top_terms(texts: list[str], limit: int = 8) -> list[str]:
    cleaned = [text for text in texts if text and text.strip()]
    if len(cleaned) < 2 or TfidfVectorizer is None:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except Exception:
        return []

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    if scores.size == 0:
        return []

    terms = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[-limit:][::-1]
    return [terms[index] for index in top_indices if scores[index] > 0][:limit]


def _short_quote(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


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

    text = df[df["question_kind"].eq("text") & df["sentiment_compound"].notna()].copy()
    if not text.empty:
        topic_text = (
            text.groupby(["survey_type", "team", "survey_year", "topic"], dropna=False)
            .agg(
                responses=("response_text", "count"),
                avg_sentiment=("sentiment_compound", "mean"),
                negative_rate=("sentiment_label", lambda values: (values == "Negative").mean()),
                positive_rate=("sentiment_label", lambda values: (values == "Positive").mean()),
            )
            .reset_index()
        )
        question_text = (
            text.groupby(["survey_type", "team", "survey_year", "topic", "question"], dropna=False)
            .agg(
                responses=("response_text", "count"),
                avg_sentiment=("sentiment_compound", "mean"),
                negative_rate=("sentiment_label", lambda values: (values == "Negative").mean()),
            )
            .reset_index()
        )
    else:
        topic_text = pd.DataFrame()
        question_text = pd.DataFrame()

    numeric = df[df["value_num"].notna()].copy()
    if not numeric.empty:
        topic_rating = (
            numeric.groupby(["survey_type", "team", "survey_year", "topic", "question_kind"], dropna=False)
            .agg(
                responses=("value_num", "count"),
                avg_value=("value_num", "mean"),
                p25=("value_num", lambda values: values.quantile(0.25)),
                p75=("value_num", lambda values: values.quantile(0.75)),
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

        def eq_or_isna(series: pd.Series, value) -> pd.Series:
            if pd.isna(value):
                return series.isna()
            return series == value

        def attach_terms(frame: pd.DataFrame, sentiment_label: str) -> pd.DataFrame:
            rows: list[str] = []
            for _, row in frame.iterrows():
                subset = text[
                    eq_or_isna(text["survey_type"], row["survey_type"])
                    & eq_or_isna(text["team"], row["team"])
                    & eq_or_isna(text["survey_year"], row["survey_year"])
                    & eq_or_isna(text["topic"], row["topic"])
                ]
                subset = subset[subset["sentiment_label"] == sentiment_label]
                rows.append(", ".join(_top_terms(subset["response_text"].astype(str).tolist())))

            enriched_frame = frame.copy()
            enriched_frame["top_terms"] = rows
            return enriched_frame

        opportunities = attach_terms(enriched.sort_values("opportunity_score", ascending=False).head(8), "Negative")
        delighters = attach_terms(enriched.sort_values("delight_score", ascending=False).head(6), "Positive")

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
    return [_short_quote(text) for text in df["response_text"].astype(str).head(limit).tolist()]
