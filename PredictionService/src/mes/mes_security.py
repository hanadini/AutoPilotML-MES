from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from config.settings import MES_API_KEY

api_key_header = APIKeyHeader(
    name="x-api-key",
    auto_error=False,
)



def verify_mes_api_key(
    x_api_key: str = Security(api_key_header),
) -> None:
    if x_api_key != MES_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing MES API key.",
        )