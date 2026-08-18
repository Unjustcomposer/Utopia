import uvicorn
import logging
import os

# Mock required environment variables before importing the production app
os.environ.setdefault("AUTH0_DOMAIN", "dev-local.auth0.com")

from server import app
from utopia.enterprise.auth import get_current_user, User

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# WARNING: DEVELOPMENT SERVER ENTRYPOINT
# -----------------------------------------------------------------------------
# This entrypoint completely bypasses Auth0 and uses a mocked admin user.
# NEVER run this in production or expose this server to the internet.
# -----------------------------------------------------------------------------

def override_get_current_user():
    return User(username="dev_user", tenant_id="dev_tenant", role="admin")

if __name__ == "__main__":
    # Apply the test dependency override globally for the dev server
    app.dependency_overrides[get_current_user] = override_get_current_user

    banner = """
    ======================================================================
    [WARNING] RUNNING IN LOCAL DEVELOPMENT MODE WITH MOCKED AUTHENTICATION
    ======================================================================
    Auth0 verification is completely disabled. All requests will be 
    authorized automatically as tenant: 'dev_tenant', role: 'admin'.
    
    DO NOT DEPLOY THIS ENTRYPOINT.
    ======================================================================
    """
    print(banner)
    logger.warning(banner)
    
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
