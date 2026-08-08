import os
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
if not AUTH0_DOMAIN:
    raise RuntimeError("AUTH0_DOMAIN must be set. For local dev, set AUTH0_DOMAIN=dev and JWT_SECRET to a strong secret.")

AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://api.utopia.com")
AUTH0_ALGORITHMS = ["RS256"]

DEV_MODE = os.getenv("NEXUSAI_DEV_MODE", "false").lower() == "true"
JWT_SECRET = os.getenv("JWT_SECRET")
if DEV_MODE and not JWT_SECRET:
    raise RuntimeError("JWT_SECRET must be set when NEXUSAI_DEV_MODE=true.")

security = HTTPBearer()

class User(BaseModel):
    username: str
    tenant_id: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if token == "mock-token":
        if not DEV_MODE:
            raise HTTPException(status_code=401, detail="Mock tokens are only accepted in dev mode")
        logger.warning("SECURITY: Mock token used for authentication — dev mode only")
        return User(username="local_dev", tenant_id="local_tenant")
        
    try:
        if DEV_MODE and AUTH0_DOMAIN == "dev":
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        else:
            jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
            jwks_client = PyJWKClient(jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=AUTH0_ALGORITHMS,
                audience=AUTH0_AUDIENCE,
                issuer=f"https://{AUTH0_DOMAIN}/"
            )

        username = payload.get("sub")
        tenant_id = payload.get("https://utopia.com/tenant_id") or payload.get("tenant_id")
        
        if not username or not tenant_id:
            raise HTTPException(status_code=401, detail="Missing user or tenant claim in token")
            
        return User(username=username, tenant_id=tenant_id)
    except jwt.PyJWTError as e:
        logger.error(f"JWT Validation Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
