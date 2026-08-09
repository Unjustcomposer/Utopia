# Utopia Security & Quality Audit Log

This ledger tracks findings from VC technical due diligence and security audits to prevent regression and ensure a single source of truth across codebase iterations.

| Date | Finding | Component | Status | Notes |
|---|---|---|---|---|
| **2026-08-08** | **Hardcoded / Missing Auth** | `utopia/enterprise/auth.py` | `FIXED` | Hardcoded mock-token auth bypassed. Replaced with proper token decoding middleware. |
| **2026-08-08** | **CI Build Failures** | `.github/workflows/ci.yml` | `FIXED` | Upgraded deprecated Node actions (v4/v5), added missing `pytest-timeout` to `requirements-dev.txt`. |
| **2026-08-09** | **Live API Ignores Trained LMM Checkpoints** | `server.py`, `app.py` | `FIXED` | API endpoints originally generated random weights unless the server was booted *after* training. Fixed by dynamically calling `load_lmm_checkpoint()` at request-time instead of module-load time. |
| **2026-08-09** | **Silent Failover in ERP Connectors** | `utopia/connectors/erp_connectors.py` | `OPEN` | Connectors attempt live SAP/Oracle calls but silently catch `RequestException` and return HTTP 200 with hardcoded mock data. This deceptive fallback is worse than failing loudly. |
| **2026-08-09** | **Spoofable Tenant Rate Limiting** | `utopia/enterprise/rate_limit.py` | `OPEN` | Extracts `tenant_id` from JWT but explicitly sets `verify_signature=False`. Any user can forge a tenant ID to bypass `slowapi` rate limits and disrupt cost attribution. |
| **2026-08-09** | **Cosmetic SOC2 Audit Logger** | `utopia/enterprise/audit_logger.py` | `OPEN` | "Secure" logger hashes payloads but writes to `stdout`. Lacks WORM (Write Once Read Many) storage or an immutable append-only ledger, rendering non-repudiation claims invalid. |
| **2026-08-09** | **Missing Licensing & Packaging** | Repository Root | `OPEN` | No `LICENSE` file and no `pyproject.toml`. Codebase cannot be legally used or cleanly installed via `pip`. |

---
*Note: Always verify the current state of a finding in this ledger before making claims in external documentation or investment memos.*
