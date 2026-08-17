# Utopia Changelog

All notable changes to the Utopia project (formerly NexusAI) will be documented in this file.

## [Unreleased]

### Added
- Created `CHANGELOG.md` to track all future codebase modifications natively within the repository.

### Changed
- **Global Rebranding**: Renamed all internal and external references from `NexusAI` / `NEXUS LMM` to `Utopia` / `UTOPIA LMM`. Replaced all `~/.nexus_jax_cache` instances with `~/.utopia_jax_cache` across `engine_jax.py` and `server.py`.
- **Git Hygiene**: `nexusai.db` has been permanently purged from the git history via `git filter-repo` to meet technical diligence standards. 
- **.gitignore Policy**: Added strict rules to prevent binary files (`*.db`, `*.sqlite3`) and OS-specific metadata (`.DS_Store`) from being committed to the repository.

### Fixed
- **Phantom Inventory Leak**: Fixed an issue in `simulation_jax.py` where initial inventory was incorrectly allocated to all goods for all firms. Initial stock is now correctly isolated to each firm's respective `good_produced`.
- **Capacity Constraint (SFC Float Leak)**: Refactored `firm_logic.py` to remove the hard-cap on capital goods. Firms can now dynamically invest up to 20% of positive cash flows to expand capacity based on `capital_shortfall`. Added logic to route capital investment expenditures to the government to preserve Stock-Flow Consistent (SFC) accounting.
- **Import Resolution**: Fixed a critical `NameError` in `app.py` by adding the missing `numpy` import required by the live telematics alerting logic.
- **Audit Logging**: Updated `AUDIT_LOG.md` to formally close out resolved diligence items (Silent Failovers, Rate Limiting, Audit Logging).

### Removed
- **Legacy Artifacts**: Permanently deleted the redundant `nexusai/` directory and related legacy `.egg-info` artifacts.
