"""Shared test configuration and fixtures."""
import os

# Set environment variables BEFORE any application imports
os.environ.setdefault("AUTH0_DOMAIN", "dev")
os.environ.setdefault("JWT_SECRET", "testsecret")
os.environ.setdefault("UTOPIA_DEV_MODE", "true")
