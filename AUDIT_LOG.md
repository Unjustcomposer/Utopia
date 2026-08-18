# Utopia Security & Quality Audit Log

This ledger tracks findings from VC technical due diligence and security audits to prevent regression and ensure a single source of truth across codebase iterations.

| Date | Finding | Component | Status | Notes |
|---|---|---|---|---|
| **2026-08-08** | **Hardcoded / Missing Auth** | `utopia/enterprise/auth.py` | `FIXED` | Hardcoded mock-token auth bypassed. Replaced with proper token decoding middleware. |
| **2026-08-08** | **CI Build Failures** | `.github/workflows/ci.yml` | `FIXED` | Upgraded deprecated Node actions (v4/v5), added missing `pytest-timeout` to `requirements-dev.txt`. |
| **2026-08-09** | **Live API Ignores Trained LMM Checkpoints** | `server.py`, `app.py` | `FIXED` | API endpoints originally generated random weights unless the server was booted *after* training. Fixed by dynamically calling `load_lmm_checkpoint()` at request-time instead of module-load time. |
| **2026-08-09** | **Silent Failover in ERP Connectors** | `utopia/connectors/erp_connectors.py` | `FIXED` | Hardcoded fallback removed; added explicit failure logging and metrics. |
| **2026-08-09** | **Spoofable Tenant Rate Limiting** | `utopia/enterprise/rate_limit.py`, `server.py` | `FIXED` | Extracts `tenant_id` safely, and `server.py` securely verifies the `RS256` Auth0 signature via `PyJWKClient` before applying limits. |
| **2026-08-09** | **Cosmetic SOC2 Audit Logger** | `utopia/enterprise/audit_logger.py` | `FIXED` | Implemented secure JSON-L logging with **cryptographic SHA-256 hash chaining** across entries (spanning restarts) to mathematically guarantee WORM immutability. |
| **2026-08-09** | **Missing Licensing & Packaging** | Repository Root | `FIXED` | Added `LICENSE` (MIT) and `pyproject.toml` for standard `pip` installation. |
| **2026-08-17** | **Phantom Inventory Flooding** | `utopia/core/simulation_jax.py` | `FIXED` | Resolved bug allocating full initial stock across all goods instead of only the firm's `good_produced`. |
| **2026-08-17** | **Capital Capacity Hard-cap Leak** | `utopia/core/engine/firm_logic.py` | `FIXED` | Fixed SFC float leak and allowed dynamic capital investment up to 20% of cash flows. |
| **2026-08-17** | **Tracking Error vs Historical Data** | `backtest_historical.py` | `FIXED` | Resolved underlying model structural issues to bring simulation trajectory back into alignment with reality. |

---
*Note: Always verify the current state of a finding in this ledger before making claims in external documentation or investment memos.*
