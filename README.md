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

### Read-only guardrail

The agent can never modify your data. Two enforced layers:
- **sqlglot** — rejects any SQL that is not a SELECT before it reaches the database
- **Read-only SQLite connection** — the connection itself is opened read-only at the OS level

### Multi-agent CSV loading

Merchant categorisation uses parallel agents — one per batch of 50 unique merchants. On a 500-merchant statement, 10 agents run simultaneously instead of one large sequential call.

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
uv run streamlit run finance_app.py
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
