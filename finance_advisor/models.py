from pydantic import BaseModel


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
