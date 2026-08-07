import os
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "utopia-mock.us.auth0.com")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://api.utopia.com")
AUTH0_ALGORITHMS = ["RS256"]
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

security = HTTPBearer()

class User(BaseModel):
    username: str
    tenant_id: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        # Fallback to HS256 for testing if domain is mock
        if AUTH0_DOMAIN == "utopia-mock.us.auth0.com":
            try:
                # Try RS256 without verification just to parse, or fallback to HS256
                unverified_header = jwt.get_unverified_header(token)
                if unverified_header.get("alg") == "HS256":
                    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                else:
                    payload = jwt.decode(token, options={"verify_signature": False})
            except Exception:
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
