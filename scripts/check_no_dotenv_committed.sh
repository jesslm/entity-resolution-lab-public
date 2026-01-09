#!/usr/bin/env bash
set -euo pipefail

# Fail the commit if any staged file is a .env (or .env.*) anywhere in the repo.
staged="$(git diff --cached --name-only)"

if echo "${staged}" | grep -Eq '(^|/)\.env($|\.)'; then
  echo "ERROR: Refusing to commit a .env file."
  echo "Remove it from the commit (recommended):"
  echo "  git restore --staged path/to/.env"
  echo "Or, if you somehow tracked it already:"
  echo "  git rm --cached path/to/.env"
  exit 1
fi

