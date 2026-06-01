import sqlite3

import httpx
import sqlglot
from pydantic_ai import Agent

from . import observability  # noqa: F401 — must be imported before agents are defined
from .config import CATEGORIES, DB_PATH, WIKI_HEADERS

_SCHEMA = f"""Table: transactions
  date        TEXT    — format YYYY-MM-DD
  description TEXT    — original merchant / narration
  amount      REAL    — positive = money spent
  category    TEXT    — one of: {', '.join(CATEGORIES)}
  month       TEXT    — format YYYY-MM (e.g. '2026-05')

Table: budgets
  category    TEXT PRIMARY KEY — same categories as above
  amount      REAL             — monthly budget target for that category"""

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
