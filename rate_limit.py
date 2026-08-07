from slowapi import Limiter
from fastapi import Request

def get_tenant_id(request: Request) -> str:
    """Extract tenant_id from the request state for rate limiting."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=get_tenant_id)
