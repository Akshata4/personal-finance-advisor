# Personal Finance Advisor

An interactive AI-powered finance assistant. Upload your bank statement CSV and ask anything about your spending in plain English. The agent writes SQL against a local SQLite database to answer each question, keeping conversation history across turns.

---

## Features

- Upload any bank CSV — the agent figures out the column structure automatically
- Ask ad-hoc questions: *"Compare food spending April vs May"*, *"What subscriptions am I paying?"*
- Set monthly budgets and track how you are tracking against them
- Detects recurring charges and subscriptions automatically
- Looks up unknown merchants on Wikipedia
- Converts spending to any currency using live exchange rates
- Full observability via Logfire — see every LLM call, tool call, and token usage

---

## Architecture

```
CSV upload
    │
    ├── LLM maps CSV columns → date, description, amount
    ├── Amounts normalised (single amount or debit/credit columns)
    ├── Merchants categorised in parallel (one agent per batch of 50)
    └── Written to local SQLite (transactions.db)

User question (plain English)
    │
    └── Agent writes SQL → execute_query → SQLite → plain English answer
```

### Application-level sandboxing

LLM agents are capable of generating and executing arbitrary SQL. Without constraints, a misbehaving or prompt-injected agent could run `DROP TABLE transactions`, `DELETE FROM budgets`, or an `UPDATE` that silently corrupts your data. Trusting the model to "just not do that" is not a safe default.

This app enforces a read-only sandbox at two independent layers so that even if one fails, the other still holds:

- **Layer 1 — sqlglot (parse-time check):** Every SQL string the agent produces is parsed by sqlglot before it reaches the database. If the statement is anything other than a `SELECT` (including CTEs that resolve to a SELECT), it is rejected outright and the agent receives an error. This catches malicious or accidental writes at the application level, before any I/O happens.

- **Layer 2 — read-only SQLite connection:** The database connection itself is opened with `sqlite3.connect("file:transactions.db?mode=ro", uri=True)`. This is an OS-level flag — SQLite will refuse any write operation on this connection regardless of what SQL is sent. Even if Layer 1 were bypassed, the database file cannot be modified.

The two layers defend against different failure modes: Layer 1 catches the intent (bad SQL), Layer 2 catches the execution (bad connection). Together they ensure your transaction history is never modified by the agent under any circumstances.

### Multi-agent CSV loading

When you upload a bank statement, every unique merchant name needs to be categorised (Food, Transport, Shopping, etc.). Sending all merchants to a single LLM call has two problems: it hits context-length limits on large statements, and it processes everything sequentially — the second half waits for the first half to finish.

Instead, the app splits merchants into batches of 50 and spawns one independent agent per batch, all running in parallel via `ThreadPoolExecutor`. Each agent only sees its own slice of merchants and returns results concurrently.

**Why this matters in practice:**

| Statement size | Sequential (1 agent) | Parallel (N agents) |
|---|---|---|
| 50 merchants | 1 call, ~3s | 1 call, ~3s |
| 200 merchants | 4 calls in series, ~12s | 4 calls in parallel, ~3s |
| 500 merchants | 10 calls in series, ~30s | 10 calls in parallel, ~3s |

Wall-clock time stays roughly constant regardless of statement size — the slowest single batch determines the total wait, not the sum of all batches. The batch size (50) is tunable via `CATEGORIZER_BATCH_SIZE` in `finance_advisor/config.py`.

---

## Tools

| Tool | What it does |
|---|---|
| `execute_query(sql)` | Runs any SELECT — the agent writes the SQL itself |
| `find_recurring_charges()` | Detects subscriptions and bills across multiple months |
| `set_budget(category, amount)` | Sets a monthly budget target |
| `get_budget_status()` | Actual spending vs budgets for the latest month |
| `explain_merchant(name)` | Looks up an unknown merchant on Wikipedia |
| `convert_currency(amount, from, to)` | Live exchange rates via frankfurter.app |

---

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/Akshata4/personal-finance-advisor
cd personal-finance-advisor
uv sync
```

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com).
Get a Logfire token (optional) at [logfire.pydantic.dev](https://logfire.pydantic.dev).

**Run:**
```bash
uv run streamlit run main.py
```

---

## CSV format

Any standard bank export works. The agent automatically detects columns for date, description, and amount. Supports both single amount columns and separate debit/credit columns.

**Spending categories:** Food · Transport · Shopping · Entertainment · Utilities · Rent · Health · Subscriptions · Other

---

## Stack

| Library | Purpose |
|---|---|
| [PydanticAI](https://ai.pydantic.dev/) | Agent framework |
| [Streamlit](https://streamlit.io/) | UI |
| [SQLite](https://www.sqlite.org/) | Local transaction store |
| [sqlglot](https://sqlglot.com/) | SQL parsing for read-only guardrail |
| [pandas](https://pandas.pydata.org/) | CSV normalisation |
| [Logfire](https://logfire.pydantic.dev/) | LLM observability |
| [httpx](https://www.python-httpx.org/) | Wikipedia and currency API calls |
