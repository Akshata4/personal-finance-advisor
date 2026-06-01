import os

import logfire

_token = os.getenv("LOGFIRE_TOKEN")
try:
    logfire.configure(token=_token)
    logfire.instrument_pydantic_ai()
except Exception:
    logfire.configure(send_to_logfire=False)
    logfire.instrument_pydantic_ai()
