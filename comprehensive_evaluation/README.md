# Comprehensive Evaluation (Tiered Datasets)

This directory contains the **Tier 0–5** datasets used in the blog series and referenced by the notebooks.

## Synthetic demo data (LLM-generated)

All datasets under `comprehensive_evaluation/data/` are **LLM-generated synthetic demo data** created for evaluation and teaching. They are **not derived from real individuals, customers, or proprietary screening lists**.

## Directory structure

- `comprehensive_evaluation/data/`
  - `tier0_test_articles.json`, `tier0_watch_list.json`
  - `tier1_test_articles.json`, `tier1_watch_list.json`
  - `tier2_test_articles.json`, `tier2_watch_list.json`
  - `tier3_test_articles.json`, `tier3_watch_list.json`
  - `tier4_test_articles.json`, `tier4_watch_list.json`
  - **Tier 5 (Ultimate Challenge)**:
    - `tier5_test_articles_v2.json`
    - `tier5_watch_list_cleaned.json`
  - `tiered_metadata.json` (tier definitions + dataset metadata)

## Canonical Tier‑5 (and why)

Tier‑5 has multiple historical variants in the private authoring repo. The PUBLIC repo uses the **canonical Tier‑5 filenames referenced by the authoritative v4 notebook**:

- `data/tier5_test_articles_v2.json`
- `data/tier5_watch_list_cleaned.json`

This avoids ambiguity and ensures `notebooks/05_ultimate_challenge_v4.ipynb` runs against the intended dataset. (Full audit: `../CANONICAL_TIER5_DATASET.md`.)

## Practical note for notebook users

- Notebooks 1–4 use the small starter datasets under `notebooks/`.
- Notebook 5 reads Tier‑5 from `comprehensive_evaluation/data/`.

## Disclaimer (demo/testing only)

These datasets and notebooks are for **demo/testing/education only**. Do not use them for real screening, compliance, or production decisions.

