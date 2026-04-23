from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd


TICKET_COLUMNS = [
    " ",
    "Row",
    "Seat Number",
    "Home Team",
    "Opponent",
    "Event Name",
    "Game Date",
    "Game Time",
    "Account Name",
    "Account Number",
    "Customer Phone",
    "Customer Company",
    "Customer First",
    "Customer Last",
    "Customer Address",
    "Customer City",
    "Customer State",
    "Customer Zip Code",
    "Customer Email",
    "Billing First",
    "Billing Last",
    "Billing Address",
    "Billing City",
    "Billing State",
    "Billing Zip",
    "Confirmation Number",
    "Price",
    "L-Fee",
    "T-Fee",
    "Mail Fee",
    "Tax",
    "Cart Discount",
    "Total",
    "Barcode",
    "Transaction Date",
    "Scanned?",
    "Released By",
    "Donation",
    "Promo Name",
    "Package Name",
    "Transaction Time",
    "Ticket Type",
    "Seat Type",
    "Split Orders",
    "Last Updated",
    "FP Redemption",
]


NIGHT_MARES_TICKET_COLUMNS = ["Section", *TICKET_COLUMNS[1:]]


TRANSACTION_COLUMNS = [
    "Date",
    "Time",
    "Time Zone",
    "Gross Sales",
    "Discounts",
    "Service Charges",
    "Net Sales",
    "Gift Card Sales",
    "Tax",
    "Tip",
    "Partial Refunds",
    "Total Collected",
    "Source",
    "Card",
    "Card Entry Methods",
    "Cash",
    "Square Gift Card",
    "Other Tender",
    "Other Tender Type",
    "Tender Note",
    "Fees",
    "Net Total",
    "Transaction ID",
    "Payment ID",
    "Card Brand",
    "PAN Suffix",
    "Device Name",
    "Staff Name",
    "Staff ID",
    "Details",
    "Description",
    "Event Type",
    "Location",
    "Dining Option",
    "Customer ID",
    "Customer Name",
    "Customer Reference ID",
    "Device Nickname",
    "Third Party Fees",
    "Deposit ID",
    "Deposit Date",
    "Deposit Details",
    "Fee Percentage Rate",
    "Fee Fixed Rate",
    "Refund Reason",
    "Discount Name",
    "Transaction Status",
    "Cash App",
    "Order Reference ID",
    "Fulfillment Note",
    "Free Processing Applied",
    "Channel",
    "Unattributed Tips",
    "Table Info",
    "International Fee",
]


OUTPUT_COLUMNS = [
    "FAN_KEY",
    "FIRST_NAME",
    "LAST_NAME",
    "EMAIL",
    "PHONE",
    "COMPANY",
    "ADDRESS",
    "CLEAN_CITY",
    "CLEAN_STATE",
    "CLEAN_ZIP",
    "ACCOUNT_NUM",
    "ACCOUNT_NAME",
    "TOTAL_TICKETS",
    "TOTAL_TICKET_SPEND",
    "TOTAL_TICKET_PAID",
    "GAMES_ATTENDED",
    "FIRST_GAME",
    "LAST_GAME",
    "OPPONENTS_SEEN",
    "SPORTS_ATTENDED",
    "MOST_COMMON_SECTION",
    "TICKET_TYPE",
    "PACKAGE",
    "PROMO",
    "TICKETS_SCANNED",
    "SCAN_RATE",
    "T_NAME_KEY_x",
    "T_NAME_KEY_y",
    "T_NAME_KEY",
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
    "MERCH_FIRST_PURCHASE",
    "MERCH_LAST_PURCHASE",
    "IS_MERCH_BUYER",
]


TICKET_RENAME_MAP = {
    " ": "section",
    "Section": "section",
    "Home Team": "HOME_TEAM",
    "Opponent": "OPPONENT",
    "Event Name": "EVENT_NAME",
    "Game Date": "GAME_DATE",
    "Account Name": "ACCOUNT_NAME",
    "Account Number": "ACCOUNT_NUM",
    "Customer Phone": "PHONE",
    "Customer Company": "COMPANY",
    "Customer First": "FIRST_NAME",
    "Customer Last": "LAST_NAME",
    "Customer Address": "ADDRESS",
    "Customer City": "CITY",
    "Customer State": "STATE",
    "Customer Zip Code": "ZIP",
    "Customer Email": "EMAIL",
    "Price": "PRICE",
    "Total": "TOTAL",
    "Transaction Date": "TRANS_DATE",
    "Scanned?": "SCANNED",
    "Promo Name": "PROMO",
    "Package Name": "PACKAGE",
    "Ticket Type": "TICKET_TYPE",
}


MERCH_MONEY_COLUMNS = [
    "Gross Sales",
    "Discounts",
    "Service Charges",
    "Partial Refunds",
    "Net Sales",
    "Card",
    "Cash",
    "Square Gift Card",
    "Other Tender",
    "Fees",
    "Total Collected",
    "Net Total",
    "Tax",
    "Tip",
    "Gift Card Sales",
    "Cash App",
]


MERCH_OUTPUT_NUMERIC_COLUMNS = [
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
]


BLANK_VALUES = {"", "nan", "none", "null", "nat", "<na>"}
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


@dataclass(frozen=True)
class FileClassification:
    filename: str
    kind: str | None
    message: str
    column_count: int


def read_csv(source: str | BinaryIO, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        source,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=True,
        low_memory=False,
        nrows=nrows,
    )


def read_columns(source: str | BinaryIO) -> list[str]:
    return list(read_csv(source, nrows=0).columns)


def classify_columns(columns: Iterable[str]) -> str | None:
    columns = list(columns)
    if columns in (TICKET_COLUMNS, NIGHT_MARES_TICKET_COLUMNS):
        return "all_tickets"
    if columns == TRANSACTION_COLUMNS:
        return "transactions"
    return None


def classify_file(filename: str, columns: Iterable[str]) -> FileClassification:
    columns = list(columns)
    kind = classify_columns(columns)
    if kind == "all_tickets":
        return FileClassification(filename, kind, "All Tickets data", len(columns))
    if kind == "transactions":
        return FileClassification(filename, kind, "Transaction-level data", len(columns))
    return FileClassification(
        filename,
        None,
        "Invalid header. File must exactly match the All Tickets or transaction export format.",
        len(columns),
    )


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.lower().isin(BLANK_VALUES))


def _clean_money(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.replace(r"[\$, ]", "", regex=True)
    )
    cleaned = cleaned.mask(cleaned.str.lower().isin(BLANK_VALUES | {"-"}), "0")
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _clean_zip(series: pd.Series) -> pd.Series:
    text = _clean_text(series)
    digits = text.str.extract(r"(\d{5})", expand=False)
    return digits.mask(digits.isin(["00000"]))


def _name_key(first: pd.Series, last: pd.Series) -> pd.Series:
    first_clean = _clean_text(first).fillna("")
    last_clean = _clean_text(last).fillna("")
    key = (
        first_clean.str.cat(last_clean, na_rep="")
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .str.strip()
    )
    return key.mask(key == "")


def _normalize_key_text(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series).str.lower()
    return cleaned.mask(cleaned.str.lower().isin(BLANK_VALUES))


def _mode_or_none(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    mode = values.mode()
    if mode.empty:
        return values.iloc[0]
    return mode.iloc[0]


def _join_unique(series: pd.Series):
    values = series.dropna().astype(str)
    if values.empty:
        return None
    return ", ".join(pd.unique(values))


def _format_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    coalesced = {}
    ordered_columns = []
    for col in df.columns:
        if col in coalesced:
            continue

        matching = df.loc[:, df.columns == col]
        if matching.shape[1] == 1:
            coalesced[col] = matching.iloc[:, 0]
        else:
            coalesced[col] = matching.bfill(axis=1).iloc[:, 0]
        ordered_columns.append(col)

    return pd.DataFrame(coalesced, index=df.index)[ordered_columns]


def _prepare_tickets(df_tickets: pd.DataFrame) -> pd.DataFrame:
    tickets = _coalesce_duplicate_columns(df_tickets.rename(columns=TICKET_RENAME_MAP)).copy()

    text_columns = [
        "section",
        "HOME_TEAM",
        "OPPONENT",
        "FIRST_NAME",
        "LAST_NAME",
        "EMAIL",
        "PHONE",
        "COMPANY",
        "ADDRESS",
        "CITY",
        "STATE",
        "ZIP",
        "ACCOUNT_NUM",
        "ACCOUNT_NAME",
        "SCANNED",
        "PROMO",
        "PACKAGE",
        "TICKET_TYPE",
    ]
    for col in text_columns:
        tickets[col] = _clean_text(tickets[col])

    tickets["T_EMAIL_KEY"] = tickets["EMAIL"].str.lower()
    tickets["T_NAME_KEY"] = _name_key(tickets["FIRST_NAME"], tickets["LAST_NAME"])
    tickets["FAN_KEY"] = tickets["T_EMAIL_KEY"].combine_first(tickets["T_NAME_KEY"])

    tickets["PRICE"] = _clean_money(tickets["PRICE"])
    tickets["TOTAL"] = _clean_money(tickets["TOTAL"])
    tickets["GAME_DATE"] = pd.to_datetime(tickets["GAME_DATE"], errors="coerce")
    tickets["TRANS_DATE"] = pd.to_datetime(tickets["TRANS_DATE"], errors="coerce")

    tickets["CLEAN_CITY"] = tickets["CITY"].str.upper()
    tickets["CLEAN_STATE"] = tickets["STATE"].str.upper()
    tickets["CLEAN_ZIP"] = _clean_zip(tickets["ZIP"])

    tickets["SCANNED"] = tickets["SCANNED"].str.upper()
    home_team = tickets["HOME_TEAM"].str.lower()
    opponent = tickets["OPPONENT"]
    tickets["SPORT"] = pd.NA
    tickets.loc[home_team == "madison night mares", "SPORT"] = "softball"
    tickets.loc[tickets["SPORT"].isna() & opponent.notna(), "SPORT"] = "baseball"

    return tickets[tickets["FAN_KEY"].notna()].copy()


def _aggregate_tickets(df_tickets_keyed: pd.DataFrame) -> pd.DataFrame:
    tickets = df_tickets_keyed.copy()
    tickets["_SCANNED_FLAG"] = (tickets["SCANNED"] == "Y").astype(int)
    grouped = tickets.groupby("FAN_KEY", sort=False, dropna=True)

    first_cols = [
        "FIRST_NAME",
        "LAST_NAME",
        "T_EMAIL_KEY",
        "PHONE",
        "COMPANY",
        "ADDRESS",
        "CLEAN_CITY",
        "CLEAN_STATE",
        "CLEAN_ZIP",
        "ACCOUNT_NUM",
        "ACCOUNT_NAME",
        "PACKAGE",
        "PROMO",
        "T_NAME_KEY",
    ]
    fan_agg = grouped[first_cols].first().rename(columns={"T_EMAIL_KEY": "EMAIL"})

    fan_agg["TOTAL_TICKETS"] = grouped.size()
    fan_agg["TOTAL_TICKET_SPEND"] = grouped["PRICE"].sum()
    fan_agg["TOTAL_TICKET_PAID"] = grouped["TOTAL"].sum()
    fan_agg["GAMES_ATTENDED"] = grouped["GAME_DATE"].nunique()
    fan_agg["FIRST_GAME"] = grouped["GAME_DATE"].min()
    fan_agg["LAST_GAME"] = grouped["GAME_DATE"].max()
    fan_agg["TICKETS_SCANNED"] = grouped["_SCANNED_FLAG"].sum()

    def unique_join(column: str) -> pd.Series:
        values = tickets[["FAN_KEY", column]].dropna().drop_duplicates()
        if values.empty:
            return pd.Series(dtype="object")
        return values.groupby("FAN_KEY", sort=False)[column].agg(", ".join)

    def most_common(column: str) -> pd.Series:
        values = tickets[["FAN_KEY", column]].dropna()
        if values.empty:
            return pd.Series(dtype="object")
        counts = values.groupby(["FAN_KEY", column], sort=True).size().reset_index(name="_count")
        counts = counts.sort_values(["FAN_KEY", "_count", column], ascending=[True, False, True])
        return counts.drop_duplicates("FAN_KEY").set_index("FAN_KEY")[column]

    fan_agg["OPPONENTS_SEEN"] = unique_join("OPPONENT")
    fan_agg["SPORTS_ATTENDED"] = unique_join("SPORT")
    fan_agg["MOST_COMMON_SECTION"] = most_common("section")
    fan_agg["TICKET_TYPE"] = most_common("TICKET_TYPE")

    fan_agg = fan_agg.reset_index()

    fan_agg["SCAN_RATE"] = np.where(
        fan_agg["TOTAL_TICKETS"] > 0,
        (fan_agg["TICKETS_SCANNED"] / fan_agg["TOTAL_TICKETS"]).round(3),
        0,
    )

    for col in ["FIRST_GAME", "LAST_GAME"]:
        fan_agg[col] = _format_date(fan_agg[col])

    return fan_agg


def _parse_customer_name(value):
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    if not text or text.lower() in BLANK_VALUES:
        return None, None
    if EMAIL_RE.match(text):
        return text.lower(), None
    normalized = re.sub(r"[^a-z0-9]", "", text.lower())
    return None, normalized or None


def _prepare_merch(df_merch: pd.DataFrame) -> pd.DataFrame:
    merch = df_merch.copy()
    for col in MERCH_MONEY_COLUMNS:
        merch[col] = _clean_money(merch[col])

    parsed = merch["Customer Name"].apply(_parse_customer_name)
    merch["MERCH_EMAIL"] = parsed.apply(lambda item: item[0])
    merch["MERCH_JOIN_NAME"] = parsed.apply(lambda item: item[1])

    merch["Date"] = _format_date(merch["Date"])
    return merch


def _empty_merch_by_key() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["JOIN_KEY", *MERCH_OUTPUT_NUMERIC_COLUMNS, "MERCH_FIRST_PURCHASE", "MERCH_LAST_PURCHASE"]
    )


def _aggregate_merch_by(df_merch_keyed: pd.DataFrame, key_col: str) -> pd.DataFrame:
    subset = df_merch_keyed[df_merch_keyed[key_col].notna()].copy()
    if subset.empty:
        return _empty_merch_by_key()

    result = subset.groupby(key_col, dropna=True)[MERCH_MONEY_COLUMNS].sum().reset_index()

    date_agg = subset.groupby(key_col, dropna=True)["Date"].agg(["min", "max"]).reset_index()
    date_agg.columns = [key_col, "MERCH_FIRST_PURCHASE", "MERCH_LAST_PURCHASE"]
    result = result.merge(date_agg, on=key_col, how="left")

    rename = {col: f"MERCH_{col.upper().replace(' ', '_')}" for col in MERCH_MONEY_COLUMNS}
    rename[key_col] = "JOIN_KEY"
    return result.rename(columns=rename)


def _join_merch(fan_agg: pd.DataFrame, df_merch: pd.DataFrame) -> pd.DataFrame:
    if df_merch.empty:
        merch_by_email = _empty_merch_by_key()
        merch_by_name = _empty_merch_by_key()
    else:
        df_merch_keyed = _prepare_merch(df_merch)
        merch_by_email = _aggregate_merch_by(df_merch_keyed, "MERCH_EMAIL")
        merch_by_name = _aggregate_merch_by(df_merch_keyed, "MERCH_JOIN_NAME")

    fan_agg = fan_agg.copy()
    fan_agg["JOIN_KEY"] = fan_agg["EMAIL"]

    pass1 = pd.merge(
        fan_agg,
        merch_by_email,
        on="JOIN_KEY",
        how="left",
        indicator="_merge_p1",
    )
    matched_p1 = pass1["_merge_p1"] == "both"
    pass1 = pass1.drop(columns=["_merge_p1"])

    merch_cols_to_drop = [col for col in merch_by_email.columns if col != "JOIN_KEY"]
    unmatched = pass1.loc[~matched_p1].drop(columns=merch_cols_to_drop, errors="ignore").copy()
    unmatched["JOIN_KEY"] = unmatched["T_NAME_KEY"]

    pass2 = pd.merge(unmatched, merch_by_name, on="JOIN_KEY", how="left")

    master = pd.concat([pass1.loc[matched_p1], pass2], ignore_index=True)
    master = master.drop(columns=["JOIN_KEY"], errors="ignore")

    if "MERCH_NET_TOTAL" in master.columns:
        master = master.sort_values("MERCH_NET_TOTAL", ascending=False, na_position="last")
    master = master.drop_duplicates("FAN_KEY").reset_index(drop=True)

    return master


def _finalize_output(df_master: pd.DataFrame) -> pd.DataFrame:
    master = df_master.copy()

    if "T_NAME_KEY" in master.columns:
        master["T_NAME_KEY_x"] = master["T_NAME_KEY"]
        master["T_NAME_KEY_y"] = master["T_NAME_KEY"]

    for col in MERCH_OUTPUT_NUMERIC_COLUMNS:
        if col not in master.columns:
            master[col] = 0.0
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0.0)

    if "MERCH_NET_TOTAL" in master.columns:
        master["IS_MERCH_BUYER"] = (master["MERCH_NET_TOTAL"] > 0).astype(int)
    else:
        master["IS_MERCH_BUYER"] = 0

    for col in OUTPUT_COLUMNS:
        if col not in master.columns:
            master[col] = pd.NA

    return master[OUTPUT_COLUMNS]


def build_fan_master(
    ticket_frames: Iterable[pd.DataFrame],
    transaction_frames: Iterable[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    ticket_frames = list(ticket_frames)
    if not ticket_frames:
        raise ValueError("At least one All Tickets file is required to build the fan master.")

    transaction_frames = list(transaction_frames or [])
    df_tickets = pd.concat(ticket_frames, ignore_index=True)
    df_merch = (
        pd.concat(transaction_frames, ignore_index=True)
        if transaction_frames
        else pd.DataFrame(columns=TRANSACTION_COLUMNS)
    )

    tickets_keyed = _prepare_tickets(df_tickets)
    fan_agg = _aggregate_tickets(tickets_keyed)
    master = _join_merch(fan_agg, df_merch)
    return _finalize_output(master)
