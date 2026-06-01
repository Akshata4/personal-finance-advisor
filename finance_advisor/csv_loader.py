import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import logfire
import pandas as pd
from pydantic_ai import Agent

from .config import CATEGORIES, CATEGORIZER_BATCH_SIZE, DB_PATH
from .database import ensure_budgets_table
from .models import CategorizedItem, CategorizationResult, ColumnMapping

_column_mapper = Agent(
    "google:gemini-3.1-flash-lite",
    output_type=ColumnMapping,
    system_prompt=(
        "You identify which CSV columns correspond to date, description, and amount. "
        "Return the exact column names as they appear in the CSV header. "
        "If the CSV has separate debit and credit columns instead of a single amount, "
        "set amount_col=None and fill debit_col and credit_col."
    ),
)

_categorizer = Agent(
    "google:gemini-3.1-flash-lite",
    output_type=CategorizationResult,
    system_prompt=(
        f"Categorize each transaction description into exactly one of: {', '.join(CATEGORIES)}.\n"
        "Guidelines:\n"
        "- Food: Zomato, Swiggy, restaurants, groceries, cafes\n"
        "- Transport: Uber, Ola, fuel, metro, train, flight\n"
        "- Shopping: Amazon, Flipkart, retail, clothing\n"
        "- Entertainment: Netflix, Spotify, movies, games\n"
        "- Utilities: electricity, water, gas, internet, phone recharge\n"
        "- Rent: rent, housing society\n"
        "- Health: pharmacy, hospital, doctor, gym\n"
        "- Subscriptions: recurring SaaS, apps, memberships\n"
        "- Other: anything that doesn't fit clearly\n\n"
        "Set is_ambiguous=True when you are not confident."
    ),
)


def _categorize_batch(batch: list[str]) -> list[CategorizedItem]:
    """Single agent run for one batch of merchant names. Runs in its own thread."""
    desc_list = "\n".join(f"- {d}" for d in batch)
    result = _categorizer.run_sync(
        f"Categorize these {len(batch)} transaction descriptions:\n{desc_list}"
    )
    return result.output.items


def _categorize_parallel(unique_descs: list[str]) -> list[CategorizedItem]:
    """Split merchants into batches and spawn one agent per batch in parallel."""
    batches = [
        unique_descs[i : i + CATEGORIZER_BATCH_SIZE]
        for i in range(0, len(unique_descs), CATEGORIZER_BATCH_SIZE)
    ]
    num_agents = len(batches)
    logfire.info(f"Spawning {num_agents} categorisation agent(s) for {len(unique_descs)} merchants")

    all_items: list[CategorizedItem] = []
    with ThreadPoolExecutor(max_workers=num_agents) as pool:
        futures = {pool.submit(_categorize_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            all_items.extend(future.result())

    return all_items


def load_csv_to_db(csv_content: str) -> tuple[int, list[str]]:
    """
    Parse a bank CSV, normalise columns, categorise merchants, write to SQLite.
    Returns (row_count, ambiguous_descriptions).
    """
    df = pd.read_csv(StringIO(csv_content))
    df.columns = df.columns.str.strip()

    # Ask LLM to identify which columns map to date/description/amount
    preview = f"Columns: {list(df.columns)}\n\nFirst 3 rows:\n{df.head(3).to_string()}"
    mapping = _column_mapper.run_sync(f"Identify the columns:\n{preview}").output

    # Normalise to standard shape
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(
        df[mapping.date_col], format="mixed"
    ).dt.strftime("%Y-%m-%d")
    out["description"] = df[mapping.description_col].astype(str).str.strip()

    if mapping.amount_col:
        out["amount"] = pd.to_numeric(
            df[mapping.amount_col].astype(str).str.replace(",", ""), errors="coerce"
        ).abs()
    else:
        debit = pd.to_numeric(
            df[mapping.debit_col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
        credit = pd.to_numeric(
            df[mapping.credit_col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
        # debit = money out (spending), credit = money in; keep spending positive
        out["amount"] = (debit - credit).abs()

    out["month"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m")
    out = out.dropna(subset=["date", "amount"])

    # Categorise merchants using parallel agents (one per batch)
    unique_descs = out["description"].unique().tolist()
    items = _categorize_parallel(unique_descs)

    category_map = {item.description: item.category for item in items}
    ambiguous = [item.description for item in items if item.is_ambiguous]
    out["category"] = out["description"].map(category_map).fillna("Other")

    # Write to SQLite (replace on each load)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS transactions")
    conn.execute("""
        CREATE TABLE transactions (
            date        TEXT,
            description TEXT,
            amount      REAL,
            category    TEXT,
            month       TEXT
        )
    """)
    out[["date", "description", "amount", "category", "month"]].to_sql(
        "transactions", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()
    ensure_budgets_table()

    return len(out), ambiguous
