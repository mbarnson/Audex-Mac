"""Install generated-venv shims for Audex runtime patches."""

from __future__ import annotations

import sysconfig
from pathlib import Path

MLX_LM_DENSE_SHIM = '''"""Generated Audex-Mac shim for MLX-LM Nemotron Dense/Audex."""

from audex_mac.patches.mlx_lm_nemotron_dense import *  # noqa: F401,F403
'''

MLX_LM_H_AUDEX_SHIM = '''"""Generated Audex-Mac shim for MLX-LM Nemotron-H Audex."""

from audex_mac.patches.mlx_lm_nemotron_h_audex import *  # noqa: F401,F403
'''

MLX_LM_SHIMS = {
    "nemotron_dense.py": MLX_LM_DENSE_SHIM,
    "nemotron_dense_audex.py": MLX_LM_DENSE_SHIM,
    "nemotron_h_audex.py": MLX_LM_H_AUDEX_SHIM,
}

SITECUSTOMIZE_SHIM = '''"""Generated Audex-Mac runtime patch hook for spawned workers."""

from __future__ import annotations

import os
import sys

if os.environ.get("AUDEX_MAC_AUTO_PATCHES") == "1":
    try:
        from audex_mac.patches.runtime import apply_audex_runtime_patches

        apply_audex_runtime_patches()
    except Exception as exc:  # pragma: no cover - startup failure path
        print(
            f"Audex-Mac startup patch hook failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
'''


def install_generated_venv_patches() -> tuple[Path, ...]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    mlx_models_dir = purelib / "mlx_lm" / "models"
    if not mlx_models_dir.exists():
        raise FileNotFoundError(
            f"Cannot find MLX-LM models directory: {mlx_models_dir}"
        )

    installed: list[Path] = []
    for shim_name, shim_content in MLX_LM_SHIMS.items():
        shim_path = mlx_models_dir / shim_name
        if (
            not shim_path.exists()
            or shim_path.read_text(encoding="utf-8") != shim_content
        ):
            shim_path.write_text(shim_content, encoding="utf-8")
        installed.append(shim_path)
    sitecustomize_path = purelib / "sitecustomize.py"
    if (
        not sitecustomize_path.exists()
        or sitecustomize_path.read_text(encoding="utf-8") != SITECUSTOMIZE_SHIM
    ):
        sitecustomize_path.write_text(SITECUSTOMIZE_SHIM, encoding="utf-8")
    installed.append(sitecustomize_path)
    return tuple(installed)


def main() -> int:
    for path in install_generated_venv_patches():
        print(f"Installed Audex-Mac patch shim: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
