#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
git config core.hooksPath .githooks
echo "Installed Audex-Mac git hooks from .githooks"
echo "Do not use git commit --no-verify; fix hook failures instead."
