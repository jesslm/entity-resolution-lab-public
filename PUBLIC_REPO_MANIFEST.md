# PUBLIC Repo Manifest (Curated Subset of Private Authoring Repo)

This manifest defines **exactly what to copy into the PUBLIC demo repo** (and what to exclude), without deleting anything from the private repo.

Primary UX: **notebooks first**. Authority: the five v3 notebooks + `NEW_USERS_START_HERE.md`.

## Scope + invariants

- **Do not delete anything** from the private repo.
- PUBLIC repo is a curated subset for a blog series.
- **Blog 2 promise**: all dataset tiers (Tier 0–5) exist in a “comprehensive evaluation” directory.
- **Datasets are safe to redistribute** (LLM-generated), but Tier‑5 has multiple variants; canonical is per `CANONICAL_TIER5_DATASET.md`.

## ✅ Explicit INCLUDE list (PUBLIC repo)

### A) Notebooks (authoritative)

- `notebooks/` (PUBLIC repo top-level; copied from private `entity_resolution_demo_package/notebooks/`):
  - `01_entity_preparation_v3.ipynb`
  - `02_article_processing_v3.ipynb`
  - `03_entity_matching_v3.ipynb`
  - `04_function_calling_optimization_v3.ipynb`
  - `05_ultimate_challenge_v3.ipynb`

**Rationale**: Primary user experience and final authority for paths/dataset usage.

### B) Minimal datasets used by notebooks 1–2

- `notebooks/minimal_entities.json` (source: `entity_resolution_demo_package/notebooks/minimal_entities.json`)
- `notebooks/minimal_articles.json` (source: `entity_resolution_demo_package/notebooks/minimal_articles.json`)
- **Required**: `pipeline_state/golden_standard.json`
  - Source: `entity_resolution_demo_package/notebooks/pipeline_state/golden_standard.json` (canonical public location is repo-root `pipeline_state/`)

**Rationale**: Required for notebook execution; golden standard is required for notebook 4 quality comparison.

### C) Canonical tiered datasets (Blog 2 promise)

- `comprehensive_evaluation/data/` (PUBLIC repo top-level)
  - Source: `use_cases/comprehensive_evaluation/data/**`
  - Includes all tiers 0–5 and metadata:
    - `tier0_*`, `tier1_*`, `tier2_*`, `tier3_*`, `tier4_*`
    - Tier‑5 canonical:
      - `tier5_test_articles_v2.json`
      - `tier5_watch_list_cleaned.json`
    - `tiered_metadata.json`

**Rationale**: Blog promise + used by notebook 5 after path-only updates.

### D) Installable demo package (minimum needed for notebooks to run)

- `entity_resolution_demo_package/` (subset):
  - `pyproject.toml`, `setup.py` (install entrypoints)
  - `setup_environment.py`, `setup_kernel.py` (onboarding scripts referenced by `NEW_USERS_START_HERE.md`)
  - `src/entity_resolution_demo/**` (core implementation used by notebooks)

**Rationale**: Notebooks import `entity_resolution_demo.*` and rely on real implementations.

### E) Onboarding docs (authoritative)

- `NEW_USERS_START_HERE.md` (repo root)

**Rationale**: Defines intended onboarding flow; should remain authoritative in PUBLIC.

## ❌ Explicit EXCLUDE list (PUBLIC repo)

### A) Private authoring / hackathon / internal tooling

- `agent_builder_hackathon/**`
  - **Rationale**: Not required for notebooks-first blog demo; contains alternate dataset copies and hackathon-specific GUI/tools.
- Root-level one-off debug scripts and working notes (examples):
  - `debug_*.py`, `diagnose_*.md`, `NOTEBOOKS_1-3_ANALYSIS.md`, internal planning docs, etc.
  - **Rationale**: Authoring artifacts; distract from curated public narrative.

### B) Duplicate Tier‑5 dataset copies outside canonical location

Exclude these *copies* from PUBLIC (keep only the canonical dataset under `comprehensive_evaluation/data/`):

- `entity_resolution_demo_package/notebooks/data/tier5_*`
- `agent_builder_hackathon/data/tier5_*`
- `use_cases/comprehensive_evaluation/data/tier5_test_articles.json` and `tier5_watch_list.json` (legacy pair)
- `use_cases/comprehensive_evaluation/external_package/sample_data/tier5_sample/**` (unless explicitly included as optional samples)

**Rationale**: Reduce confusion; avoid multiple “Tier‑5” definitions in public.

### C) Generated outputs / evaluation artifacts (default exclude)

- `use_cases/comprehensive_evaluation/results/**`
- `use_cases/comprehensive_evaluation/visualizations/**`
- `use_cases/comprehensive_evaluation/external_package/**` (zip, reports, extra README variants)

**Rationale**: Large artifacts; not required for notebooks to run. Can be added later if blog narrative needs them.

### D) Backups

- Any `*.backup_*`

**Rationale**: Redundant historical snapshots.

## MUST FIX (blocking public release)

- **Notebook paths after relocation**: notebooks must reference data/state relative to PUBLIC layout (paths only).
- **`.env` template availability**: PUBLIC repo must ship an env template alongside the installable package. In this repo we provide `entity_resolution_demo_package/env.template` and onboarding uses `cp env.template .env`.
- **Notebook `.env` loading**: notebooks 3 and 5 call `load_dotenv()` without a path; PUBLIC layout should load `entity_resolution_demo_package/.env` explicitly (path-only change).

## SHOULD FIX (polish)

- Add a minimal `comprehensive_evaluation/README.md` explaining tiers, provenance, and canonical Tier‑5.
- Add a short PUBLIC `README.md` that links to `NEW_USERS_START_HERE.md` and clarifies the notebooks-first flow.
- Decide whether `pipeline_state/` in PUBLIC should ship only `golden_standard.json` (recommended) and treat all other state files as generated.

## OPTIONAL (can defer)

- Include evaluation scripts under `comprehensive_evaluation/scripts/` if the blog series expects readers to reproduce figures.
- Include a tiny `tier5_sample/` subset for quick download/testing (keeping canonical dataset as the source of truth).

