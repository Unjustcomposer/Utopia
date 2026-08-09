from slowapi import Limiter
from fastapi import Request

def get_tenant_id(request: Request) -> str:
    """Extract tenant_id from the request state for rate limiting."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"
    import logging
    logging.getLogger(__name__).warning("No tenant_id found, falling back to IP-based rate limiting")
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=get_tenant_id)
