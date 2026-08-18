# Utopia Changelog

All notable changes to the Utopia project (formerly NexusAI) will be documented in this file.

## [Unreleased]

### Added
- Created `CHANGELOG.md` to track all future codebase modifications natively within the repository.
- **Local Dev Server**: Added `dev_server.py` as an isolated entrypoint for local development. It uses FastAPI dependency overrides to safely bypass Auth0 without polluting the production `auth.py` or `server.py` files.

### Changed
- **Global Rebranding**: Renamed all internal and external references from `NexusAI` / `NEXUS LMM` to `Utopia` / `UTOPIA LMM`. Replaced all `~/.nexus_jax_cache` instances with `~/.utopia_jax_cache` across `engine_jax.py` and `server.py`.
- **Git Hygiene**: `nexusai.db` has been permanently purged from the git history via `git filter-repo` to meet technical diligence standards. This has been fully propagated to the remote history.
- **.gitignore Policy**: Added strict rules to prevent binary files (`*.db`, `*.sqlite3`) and OS-specific metadata (`.DS_Store`) from being committed to the repository.
- **Empirical Tracking Errors**: Updated `README.md` and `blog_post.md` to reflect fresh backtest results across all three modern macroeconomic crises: 2008 Crash (6.74 pts GDP / 8.94 pts Unemp error), 2020 Covid Shock (12.26 pts / 14.73 pts), and 2021 Supply Chain Crunch (0.53 pts / 1.43 pts).
- **Audit Logger Refactor**: `AuditLogger` is now a true global singleton. It reads the final line of `data/audit_log.jsonl` on startup to seamlessly maintain the WORM cryptographic hash chain across process restarts.
- **Audit Ledger Updates**: Updated `AUDIT_LOG.md` to reflect the completed state of WORM logging, Rate Limiter security, and the addition of the MIT License.

### Fixed
- **Phantom Inventory Leak**: Fixed an issue in `simulation_jax.py` where initial inventory was incorrectly allocated to all goods for all firms. Initial stock is now correctly isolated to each firm's respective `good_produced`.
- **Capacity Constraint (SFC Float Leak)**: Refactored `firm_logic.py` to remove the hard-cap on capital goods. Firms can now dynamically invest up to 20% of positive cash flows to expand capacity based on `capital_shortfall`. Added logic to route capital investment expenditures to the government to preserve Stock-Flow Consistent (SFC) accounting.
- **Import Resolution**: Fixed a critical `NameError` in `app.py` by adding the missing `numpy` import required by the live telematics alerting logic.
- **Audit Logging (WORM Compliance)**: Enhanced the durable JSON-L storage file (`data/audit_log.jsonl`) in `audit_logger.py` with **cryptographic SHA-256 hash chaining**. Each log entry now hashes its payload along with the `previous_hash`, mathematically guaranteeing the log's tamper-evident WORM (Write Once Read Many) properties for diligence. Updated `AUDIT_LOG.md` to formally close out resolved items.
- **Spoofable Rate Limiting**: The JWT tenant extraction middleware in `server.py` now securely verifies the `RS256` signature via `PyJWKClient` (reusing the secure logic from `auth.py`) before applying rate limits.

### Removed
- **Database Migrations Cleanup**: Removed `Base.metadata.create_all()` from `database.py` since schema migrations are properly managed by Alembic.
- **Auth Dead-Code**: Removed the unused `JWT_SECRET` local dev-mode branch in `auth.py`. Authentication now strictly enforces `RS256` asymmetric JWKS validation.
- **Legacy Artifacts**: Permanently deleted the redundant `nexusai/` directory and related legacy `.egg-info` artifacts.
