import os
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
JWT_SECRET = os.getenv("JWT_SECRET")

# Dev mode: AUTH0_DOMAIN is unset or explicitly "dev"
_DEV_MODE = (not AUTH0_DOMAIN) or AUTH0_DOMAIN == "dev"

if not _DEV_MODE:
    from jwt import PyJWKClient
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://api.utopia.com")
    AUTH0_ALGORITHMS = ["RS256"]
else:
    logger.info("Running in dev/test mode — using symmetric JWT validation (HS256).")
    AUTH0_AUDIENCE = None
    AUTH0_ALGORITHMS = ["HS256"]
    if not JWT_SECRET:
        JWT_SECRET = "dev-secret"

security = HTTPBearer(auto_error=not _DEV_MODE)

class User(BaseModel):
    username: str
    tenant_id: str
    role: str = "analyst"

def verify_jwt_token(token: str) -> dict:
    if _DEV_MODE:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=AUTH0_ALGORITHMS,
            options={"verify_exp": False, "verify_aud": False},
        )
    else:
        jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
        jwks_client = PyJWKClient(jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=AUTH0_ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    # Dev mode without token → return a default dev user
    if _DEV_MODE and credentials is None:
        return User(username="dev_user", tenant_id="dev_tenant")

    token = credentials.credentials
    try:
        payload = verify_jwt_token(token)

        username = payload.get("sub", "dev_user")
        tenant_id = (
            payload.get("https://utopia.com/tenant_id")
            or payload.get("tenant_id", "dev_tenant")
        )
        role = (
            payload.get("https://utopia.com/role")
            or payload.get("role", "analyst")
        )

        if not username or not tenant_id:
            raise HTTPException(
                status_code=401,
                detail="Missing user or tenant claim in token",
            )

        return User(username=username, tenant_id=tenant_id, role=role)
    except jwt.PyJWTError as e:
        logger.error(f"JWT Validation Error: {e}")
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )

def get_admin_user(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=403, detail="Forbidden: Admin access required"
        )
    return user
