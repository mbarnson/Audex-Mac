#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="python"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT_DIR}/.venv/bin/python"
fi
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Missing Python executable: ${PYTHON}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON}" -m black --check audex_mac tests
"${PYTHON}" -m ruff check audex_mac tests
bash -n start.sh scripts/*.sh .githooks/pre-commit
