import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

DB_PATH = Path(__file__).parent.parent / "transactions.db"

WIKI_HEADERS = {"User-Agent": "FinanceAgent/1.0 (learning project; contact@example.com)"}

CATEGORIZER_BATCH_SIZE = 50  # merchants per agent; tune based on model context limits

CATEGORIES = [
    "Food", "Transport", "Shopping", "Entertainment",
    "Utilities", "Rent", "Health", "Subscriptions", "Other",
]
