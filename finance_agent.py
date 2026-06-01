import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import httpx
import logfire
import pandas as pd
import sqlglot
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Logfire — explicitly pass token from .env, falls back to console if missing
_logfire_token = os.getenv("LOGFIRE_TOKEN")
try:
    logfire.configure(token=_logfire_token)
    logfire.instrument_pydantic_ai()
except Exception:
    logfire.configure(send_to_logfire=False)
    logfire.instrument_pydantic_ai()

DB_PATH = Path(__file__).parent / "transactions.db"
WIKI_HEADERS = {"User-Agent": "FinanceAgent/1.0 (learning project; contact@example.com)"}
CATEGORIZER_BATCH_SIZE = 50  # merchants per agent; tune based on model context limits

CATEGORIES = [
    "Food", "Transport", "Shopping", "Entertainment",
    "Utilities", "Rent", "Health", "Subscriptions", "Other",
]


# ── Structured output models ──────────────────────────────────────────────────

class ColumnMapping(BaseModel):
    date_col: str
    description_col: str
    amount_col: str | None = None   # single column, e.g. "Amount"
    debit_col: str | None = None    # separate debit column
    credit_col: str | None = None   # separate credit column


class CategorizedItem(BaseModel):
    description: str
    category: str
    is_ambiguous: bool


class CategorizationResult(BaseModel):
    items: list[CategorizedItem]


# ── Helper agents (run only during CSV load) ──────────────────────────────────

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


# ── DB utilities ──────────────────────────────────────────────────────────────

def db_exists() -> bool:
    if not DB_PATH.exists():
        return False
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
    ).fetchone()
    conn.close()
    return row is not None


def get_transaction_count() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    conn.close()
    return count


def clear_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()


def ensure_budgets_table() -> None:
    """Create the budgets table if it doesn't exist yet."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            category TEXT PRIMARY KEY,
            amount   REAL
        )
    """)
    conn.commit()
    conn.close()


# ── Multi-agent categorisation ────────────────────────────────────────────────

def _categorize_batch(batch: list[str]) -> list[CategorizedItem]:
    """Single agent run for one batch of merchant names. Runs in its own thread."""
    desc_list = "\n".join(f"- {d}" for d in batch)
    result = _categorizer.run_sync(
        f"Categorize these {len(batch)} transaction descriptions:\n{desc_list}"
    )
    return result.output.items


def _categorize_parallel(unique_descs: list[str]) -> list[CategorizedItem]:
    """
    Split merchants into batches and spawn one agent per batch in parallel.
    Each ThreadPoolExecutor worker is an independent agent run.
    """
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


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv_to_db(csv_content: str) -> tuple[int, list[str]]:
    """
    Parse a bank CSV, normalise columns, categorise merchants, write to SQLite.
    Returns (row_count, ambiguous_descriptions).
    """
    df = pd.read_csv(StringIO(csv_content))
    df.columns = df.columns.str.strip()

    # Step 1 — ask LLM to identify which columns map to date/description/amount
    preview = f"Columns: {list(df.columns)}\n\nFirst 3 rows:\n{df.head(3).to_string()}"
    mapping = _column_mapper.run_sync(f"Identify the columns:\n{preview}").output

    # Step 2 — normalise to standard shape
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

    # Step 3 — categorise merchants using parallel agents (one per batch)
    unique_descs = out["description"].unique().tolist()
    items = _categorize_parallel(unique_descs)

    category_map = {item.description: item.category for item in items}
    ambiguous = [item.description for item in items if item.is_ambiguous]
    out["category"] = out["description"].map(category_map).fillna("Other")

    # Step 4 — write to SQLite (replace on each load)
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


# ── Schema string injected into the agent's system prompt ────────────────────

_SCHEMA = f"""Table: transactions
  date        TEXT    — format YYYY-MM-DD
  description TEXT    — original merchant / narration
  amount      REAL    — positive = money spent
  category    TEXT    — one of: {', '.join(CATEGORIES)}
  month       TEXT    — format YYYY-MM (e.g. '2026-05')

Table: budgets
  category    TEXT PRIMARY KEY — same categories as above
  amount      REAL             — monthly budget target for that category"""


# ── Main interactive agent ────────────────────────────────────────────────────

finance_agent = Agent(
    "google:gemini-3.1-flash-lite",
    system_prompt=(
        "You are a personal finance assistant. Answer questions about the user's transactions.\n\n"
        f"Database schema:\n{_SCHEMA}\n\n"
        "Tools available:\n"
        "- execute_query(sql)                            — SELECT queries on transactions/budgets\n"
        "- find_recurring_charges()                      — detect subscriptions and recurring bills\n"
        "- set_budget(category, amount)                  — set a monthly budget for a category\n"
        "- get_budget_status()                           — actual spending vs budgets for latest month\n"
        "- explain_merchant(name)                        — look up what a merchant/company is\n"
        "- convert_currency(amount, from_currency, to_currency) — live exchange rate conversion\n\n"
        "Rules:\n"
        "- Only SELECT is allowed in execute_query\n"
        "- Filter by month using: WHERE month = 'YYYY-MM'\n"
        "- Round amounts to 2 decimal places in your response\n"
        "- Always interpret results in plain English, not raw numbers"
    ),
)


@finance_agent.tool_plain
def execute_query(sql: str) -> str:
    """Execute a read-only SQL SELECT query on the transactions table."""
    # Layer 2: parse and validate with sqlglot
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception as e:
        return f"SQL parse error: {e}. Please fix and retry."

    is_select = isinstance(parsed, sqlglot.expressions.Select)
    is_cte_select = (
        isinstance(parsed, sqlglot.expressions.With)
        and isinstance(parsed.this, sqlglot.expressions.Select)
    )
    if not (is_select or is_cte_select):
        return "Error: only SELECT queries are allowed."

    # Layer 3: read-only SQLite connection
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        conn.close()
    except Exception as e:
        return f"Query failed: {e}"

    if not rows:
        return "No results found."

    # Format as a readable table (cap at 50 rows)
    lines = [" | ".join(cols), "-" * max(len(" | ".join(cols)), 20)]
    for row in rows[:50]:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    if len(rows) > 50:
        lines.append(f"... and {len(rows) - 50} more rows")
    return "\n".join(lines)


@finance_agent.tool_plain
def find_recurring_charges() -> str:
    """Find transactions that appear across multiple months — likely subscriptions or bills."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT
            description,
            category,
            COUNT(DISTINCT month)   AS months_seen,
            ROUND(AVG(amount), 2)   AS avg_amount,
            ROUND(MIN(amount), 2)   AS min_amount,
            ROUND(MAX(amount), 2)   AS max_amount
        FROM transactions
        GROUP BY description
        HAVING COUNT(DISTINCT month) >= 2
        ORDER BY months_seen DESC, avg_amount DESC
    """).fetchall()
    conn.close()

    if not rows:
        return "No recurring charges found."

    lines = ["description | category | months_seen | avg_amount | amount_range"]
    lines.append("-" * 75)
    for desc, cat, months, avg, mn, mx in rows:
        rng = str(mn) if mn == mx else f"{mn}–{mx}"
        lines.append(f"{desc} | {cat} | {months} | {avg} | {rng}")
    return "\n".join(lines)


@finance_agent.tool_plain
def set_budget(category: str, amount: float) -> str:
    """Set a monthly budget target for a spending category."""
    if category not in CATEGORIES:
        return f"Unknown category '{category}'. Valid categories: {', '.join(CATEGORIES)}"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO budgets (category, amount) VALUES (?, ?)",
        (category, amount),
    )
    conn.commit()
    conn.close()
    return f"Budget set: {category} = {amount:.2f} per month."


@finance_agent.tool_plain
def get_budget_status() -> str:
    """Compare actual spending against set budgets for the latest month in the data."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    latest = conn.execute("SELECT MAX(month) FROM transactions").fetchone()[0]
    if not latest:
        conn.close()
        return "No transactions found."

    budgets = dict(conn.execute("SELECT category, amount FROM budgets").fetchall())
    if not budgets:
        conn.close()
        return "No budgets set yet. Use set_budget(category, amount) to add one."

    actuals = dict(conn.execute("""
        SELECT category, ROUND(SUM(amount), 2)
        FROM transactions WHERE month = ?
        GROUP BY category
    """, (latest,)).fetchall())
    conn.close()

    lines = [f"Budget status for {latest}", "-" * 55]
    lines.append("category | budget | spent | remaining | status")
    lines.append("-" * 55)
    for cat, budget in sorted(budgets.items()):
        spent = actuals.get(cat, 0.0)
        remaining = round(budget - spent, 2)
        pct = (spent / budget * 100) if budget > 0 else 0
        status = "OVER" if pct >= 100 else ("WARNING" if pct >= 80 else "OK")
        lines.append(f"{cat} | {budget} | {spent} | {remaining} | {status} ({pct:.0f}%)")
    return "\n".join(lines)


@finance_agent.tool_plain
def explain_merchant(name: str) -> str:
    """Search Wikipedia to find out what a merchant or company is."""
    resp = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": name,
            "format": "json",
            "srlimit": 1,
        },
        headers=WIKI_HEADERS,
        timeout=10,
    )
    if resp.status_code != 200:
        return f"Search failed (HTTP {resp.status_code})."
    hits = resp.json().get("query", {}).get("search", [])
    if not hits:
        return f"No information found for '{name}' on Wikipedia."
    snippet = hits[0]["snippet"].replace('<span class="searchmatch">', "").replace("</span>", "")
    return f"{hits[0]['title']}: {snippet}..."


@finance_agent.tool_plain
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using live exchange rates from frankfurter.app."""
    resp = httpx.get(
        "https://api.frankfurter.app/latest",
        params={
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "amount": amount,
        },
        follow_redirects=True,
        timeout=10,
    )
    if resp.status_code != 200:
        return f"Currency conversion failed (HTTP {resp.status_code})."
    data = resp.json()
    converted = data.get("rates", {}).get(to_currency.upper())
    if converted is None:
        return f"Currency '{to_currency}' not supported."
    rate = converted / amount
    return (
        f"{amount:,.2f} {from_currency.upper()} = {converted:,.2f} {to_currency.upper()} "
        f"(rate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}, "
        f"date: {data.get('date')})"
    )


# ── Ensure budgets table exists for any pre-existing DB ───────────────────────
ensure_budgets_table()
