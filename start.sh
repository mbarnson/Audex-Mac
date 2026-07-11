#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VLLM_METAL_VENDOR_DIR="${ROOT_DIR}/.audex/vendor/vllm-metal"
VLLM_METAL_VENV_DIR="${ROOT_DIR}/.audex/vendor/vllm-metal/.venv-vllm-metal"
STATE_DIR="${ROOT_DIR}/.audex/state"
DEPS_STAMP="${STATE_DIR}/deps.stamp"
VLLM_METAL_DEPS_STAMP="${STATE_DIR}/vllm-metal-deps.stamp"
PYTHON_BIN="${PYTHON_BIN:-}"
REFRESH_DEPS=0
VLLM_METAL_INSTALL_REQUIRED=0
NEEDS_AUDIO_EVAL_DEPS=0
ARGS=()
ARGS_COUNT=0
export AUDEX_VLLM_TTS_CFG="${AUDEX_VLLM_TTS_CFG:-0}"
export AUDEX_VLLM_DIRECT_AUDIO_RESPONSE="${AUDEX_VLLM_DIRECT_AUDIO_RESPONSE:-1}"
export AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL="${AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL:-0}"
export AUDEX_VLLM_EAGER_AUDIO_COMPONENTS="${AUDEX_VLLM_EAGER_AUDIO_COMPONENTS:-1}"

require_metal_env() {
  local name="$1"
  local expected="$2"
  local current="${!name:-${expected}}"
  if [[ "${current}" != "${expected}" ]]; then
    echo "Audex-Mac requires ${name}=${expected} for native Metal/MLX inference." >&2
    echo "Current ${name}=${current}; refusing to run a CPU-mode vLLM Metal path." >&2
    exit 1
  fi
  export "${name}=${expected}"
}

select_python_bin() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi
  if command -v python3.13 >/dev/null 2>&1; then
    command -v python3.13
    return
  fi
  echo "python3.12"
}

python_runtime_ok() {
  local candidate="$1"
  "${candidate}" -c 'import platform, sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) and platform.machine() == "arm64" else 1)' >/dev/null 2>&1
}

repair_hidden_pth_files() {
  local venv_dir="$1"
  local site_packages
  if ! command -v chflags >/dev/null 2>&1; then
    return
  fi
  site_packages="${venv_dir}/lib"
  if [[ ! -d "${site_packages}" ]]; then
    return
  fi
  find "${site_packages}" -name '*.pth' -exec chflags nohidden {} + 2>/dev/null || true
}

vllm_metal_pythonpath() {
  local vendor_dir="${VLLM_METAL_VENDOR_DIR}"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    echo "${ROOT_DIR}:${vendor_dir}:${PYTHONPATH}"
  else
    echo "${ROOT_DIR}:${vendor_dir}"
  fi
}

vllm_nonpaged_kv_capacity_seqs() {
  if [[ -n "${AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS:-}" ]]; then
    echo "${AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS}"
    return
  fi
  case "${AUDEX_VLLM_TTS_CFG:-}" in
    1|true|TRUE|yes|YES|on|ON)
      echo "2"
      ;;
  esac
}

vllm_metal_pin_field() {
  local field="$1"
  "${PYTHON_BIN}" -c 'import json, sys; print(json.load(open("vendor_pins.json"))["vllm_metal"][sys.argv[1]])' "${field}"
}

move_unmanaged_vllm_metal_runtime() {
  local backup_dir
  backup_dir="${STATE_DIR}/runtime-backups/vllm-metal-$(date +%Y%m%d%H%M%S)-$$"
  echo "Moving unmanaged vLLM Metal runtime to ${backup_dir}"
  mkdir -p "$(dirname "${backup_dir}")"
  mv "${VLLM_METAL_VENDOR_DIR}" "${backup_dir}"
  VLLM_METAL_INSTALL_REQUIRED=1
}

ensure_vllm_metal_checkout() {
  local repo pinned_commit installed_commit origin_url checkout_created=0
  repo="$(vllm_metal_pin_field repo)"
  pinned_commit="$(vllm_metal_pin_field pinned_commit)"

  mkdir -p "$(dirname "${VLLM_METAL_VENDOR_DIR}")"
  if [[ -e "${VLLM_METAL_VENDOR_DIR}" ]]; then
    origin_url="$(
      git -C "${VLLM_METAL_VENDOR_DIR}" remote get-url origin 2>/dev/null || true
    )"
    if [[ ! -d "${VLLM_METAL_VENDOR_DIR}/.git" || "${origin_url}" != "${repo}" ]]; then
      move_unmanaged_vllm_metal_runtime
    fi
  fi

  if [[ ! -d "${VLLM_METAL_VENDOR_DIR}/.git" ]]; then
    echo "Cloning pinned vLLM Metal runtime into ${VLLM_METAL_VENDOR_DIR}"
    git clone "${repo}" "${VLLM_METAL_VENDOR_DIR}"
    VLLM_METAL_INSTALL_REQUIRED=1
    checkout_created=1
  fi

  installed_commit="$(git -C "${VLLM_METAL_VENDOR_DIR}" rev-parse HEAD)"
  if [[ "${checkout_created}" == "1" || "${installed_commit}" != "${pinned_commit}" ]]; then
    echo "Checking out pinned vLLM Metal commit ${pinned_commit}"
    git -C "${VLLM_METAL_VENDOR_DIR}" fetch "${repo}" "${pinned_commit}" >/dev/null
    git -C "${VLLM_METAL_VENDOR_DIR}" checkout --detach "${pinned_commit}" >/dev/null
    VLLM_METAL_INSTALL_REQUIRED=1
  fi
}

ensure_vllm_metal_runtime() {
  ensure_vllm_metal_checkout

  echo "Installing pinned vLLM Metal runtime. This can take a while on first run."
  (cd "${VLLM_METAL_VENDOR_DIR}" && ./install.sh)
}

ensure_vllm_metal_audex_deps() {
  local python_bin="${VLLM_METAL_VENV_DIR}/bin/python"
  local deps_imports="import audex_mac, huggingface_hub, prompt_toolkit, sounddevice"
  local install_target="${ROOT_DIR}"
  local deps_ready=0
  if [[ "${NEEDS_AUDIO_EVAL_DEPS}" == "1" ]]; then
    deps_imports="import audex_mac, huggingface_hub, prompt_toolkit, scipy, sounddevice, soundfile, torch, transformers"
    install_target="${ROOT_DIR}[audio-eval]"
  fi
  mkdir -p "${STATE_DIR}"
  if PYTHONPATH="${VLLM_METAL_PYTHONPATH}" "${python_bin}" -c "${deps_imports}" >/dev/null 2>&1; then
    deps_ready=1
  fi
  if [[ "${VLLM_METAL_INSTALL_REQUIRED}" == "1" || "${deps_ready}" != "1" || ! -f "${VLLM_METAL_DEPS_STAMP}" || "${ROOT_DIR}/pyproject.toml" -nt "${VLLM_METAL_DEPS_STAMP}" ]]; then
    echo "Installing Audex-Mac dependencies into pinned vLLM Metal runtime"
    "${python_bin}" -m pip install -e "${install_target}" >/dev/null
    touch "${VLLM_METAL_DEPS_STAMP}"
  fi
}

run_vllm_metal_patch_guards() {
  local python_bin="${VLLM_METAL_VENV_DIR}/bin/python"
  local pinned_commit installed_commit upstream_head
  local guard_args
  pinned_commit="$(vllm_metal_pin_field pinned_commit)"
  installed_commit="$(git -C "${VLLM_METAL_VENDOR_DIR}" rev-parse HEAD)"
  upstream_head="$(git -C "${VLLM_METAL_VENDOR_DIR}" rev-parse refs/remotes/origin/main 2>/dev/null || true)"
  guard_args=(
    --installed-commit "${installed_commit}"
    --pinned-commit "${pinned_commit}"
    --update-prompt-path "${STATE_DIR}/vllm-metal-update-prompt.md"
  )
  if [[ -n "${upstream_head}" ]]; then
    guard_args+=(--upstream-head "${upstream_head}")
  fi
  mkdir -p "${STATE_DIR}"
  PYTHONPATH="${VLLM_METAL_PYTHONPATH}" \
    "${python_bin}" -m audex_mac.patch_guards "${guard_args[@]}"
}

exec_vllm_metal_cli() {
  local python_bin="${VLLM_METAL_VENV_DIR}/bin/python"
  local nonpaged_kv_capacity_seqs
  nonpaged_kv_capacity_seqs="$(vllm_nonpaged_kv_capacity_seqs)"
  if [[ "${ARGS_COUNT}" -gt 0 ]]; then
    exec env \
      AUDEX_MAC_AUTO_PATCHES=1 \
      AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="${nonpaged_kv_capacity_seqs}" \
      PYTHONPATH="${VLLM_METAL_PYTHONPATH}" \
      VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}" \
      TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
      "${python_bin}" -m audex_mac.cli "${ARGS[@]}"
  fi
  exec env \
    AUDEX_MAC_AUTO_PATCHES=1 \
    AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="${nonpaged_kv_capacity_seqs}" \
    PYTHONPATH="${VLLM_METAL_PYTHONPATH}" \
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}" \
    TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
    "${python_bin}" -m audex_mac.cli
}

exec_project_cli() {
  local python_bin="${VENV_DIR}/bin/python"
  if [[ "${ARGS_COUNT}" -gt 0 ]]; then
    exec "${python_bin}" -m audex_mac.cli "${ARGS[@]}"
  fi
  exec "${python_bin}" -m audex_mac.cli
}

main() {
PYTHON_BIN="$(select_python_bin)"

require_metal_env VLLM_METAL_USE_MLX 1
require_metal_env VLLM_MLX_DEVICE gpu
require_metal_env VLLM_METAL_USE_PAGED_ATTENTION 0
require_metal_env VLLM_METAL_MEMORY_FRACTION auto

for arg in "$@"; do
  case "${arg}" in
    --refresh-deps)
      REFRESH_DEPS=1
      ;;
    eval-audio-capabilities)
      NEEDS_AUDIO_EVAL_DEPS=1
      ARGS+=("${arg}")
      ARGS_COUNT=$((ARGS_COUNT + 1))
      ;;
    sound-lab)
      NEEDS_AUDIO_EVAL_DEPS=1
      ARGS+=("${arg}")
      ARGS_COUNT=$((ARGS_COUNT + 1))
      ;;
    *)
      ARGS+=("${arg}")
      ARGS_COUNT=$((ARGS_COUNT + 1))
      ;;
  esac
done

ensure_vllm_metal_checkout

if [[ "${REFRESH_DEPS}" == "0" && "${VLLM_METAL_INSTALL_REQUIRED}" == "0" && -x "${VLLM_METAL_VENV_DIR}/bin/python" ]]; then
  repair_hidden_pth_files "${VLLM_METAL_VENV_DIR}"
  VLLM_METAL_PYTHONPATH="$(vllm_metal_pythonpath)"
  if PYTHONPATH="${VLLM_METAL_PYTHONPATH}" "${VLLM_METAL_VENV_DIR}/bin/python" -c "import audex_mac, vllm, vllm_metal" >/dev/null 2>&1; then
    ensure_vllm_metal_audex_deps
    run_vllm_metal_patch_guards
    PYTHONPATH="${VLLM_METAL_PYTHONPATH}" "${VLLM_METAL_VENV_DIR}/bin/python" -m audex_mac.patches.install >/dev/null
    exec_vllm_metal_cli
  fi
  echo "Existing vLLM Metal environment is incomplete; reinstalling it."
  VLLM_METAL_INSTALL_REQUIRED=1
fi

if [[ "${REFRESH_DEPS}" == "1" || "${VLLM_METAL_INSTALL_REQUIRED}" == "1" || ! -x "${VLLM_METAL_VENV_DIR}/bin/python" ]]; then
  ensure_vllm_metal_runtime
  repair_hidden_pth_files "${VLLM_METAL_VENV_DIR}"
  VLLM_METAL_PYTHONPATH="$(vllm_metal_pythonpath)"
  ensure_vllm_metal_audex_deps
  run_vllm_metal_patch_guards
  PYTHONPATH="${VLLM_METAL_PYTHONPATH}" "${VLLM_METAL_VENV_DIR}/bin/python" -m audex_mac.patches.install >/dev/null
  exec_vllm_metal_cli
fi

mkdir -p "${STATE_DIR}"

if [[ -x "${VENV_DIR}/bin/python" ]] && ! python_runtime_ok "${VENV_DIR}/bin/python"; then
  BACKUP_DIR="${STATE_DIR}/venv-backups/$(date +%Y%m%d%H%M%S)"
  echo "Existing .venv uses an unsupported Python for vLLM Metal; moving it to ${BACKUP_DIR}" >&2
  mkdir -p "$(dirname "${BACKUP_DIR}")"
  mv "${VENV_DIR}" "${BACKUP_DIR}"
  REFRESH_DEPS=1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Audex-Mac needs native arm64 Python 3.12 or 3.13 available as ${PYTHON_BIN}." >&2
    echo "Install Python 3.12/3.13, or set PYTHON_BIN=/path/to/python3.13." >&2
    exit 1
  fi
  if ! python_runtime_ok "${PYTHON_BIN}"; then
    echo "Audex-Mac needs native arm64 Python >=3.12,<3.14 for vLLM Metal." >&2
    echo "Selected interpreter is not compatible: ${PYTHON_BIN}" >&2
    exit 1
  fi
  echo "Creating local virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  REFRESH_DEPS=1
fi

DEPS_READY=0
if "${VENV_DIR}/bin/python" -c "import audex_mac, huggingface_hub" >/dev/null 2>&1; then
  DEPS_READY=1
fi
if [[ "${DEPS_READY}" == "1" && ! -f "${DEPS_STAMP}" ]]; then
  touch "${DEPS_STAMP}"
fi
if [[ "${REFRESH_DEPS}" == "1" || "${DEPS_READY}" != "1" || "${ROOT_DIR}/pyproject.toml" -nt "${DEPS_STAMP}" ]]; then
  echo "Installing pinned Audex-Mac dependencies"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel >/dev/null
  "${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}" >/dev/null
  touch "${DEPS_STAMP}"
fi

exec_project_cli
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
