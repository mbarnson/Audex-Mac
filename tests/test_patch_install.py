from __future__ import annotations

import sysconfig
from pathlib import Path

import pytest

from audex_mac.patches import install

pytestmark = pytest.mark.fast


def test_install_generated_venv_patches_writes_mlx_lm_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    purelib = tmp_path / "site-packages"
    models_dir = purelib / "mlx_lm" / "models"
    models_dir.mkdir(parents=True)
    monkeypatch.setattr(
        sysconfig,
        "get_paths",
        lambda: {"purelib": str(purelib)},
    )

    paths = install.install_generated_venv_patches()

    assert paths == (
        models_dir / "nemotron_dense.py",
        models_dir / "nemotron_dense_audex.py",
        models_dir / "nemotron_h_audex.py",
        purelib / "sitecustomize.py",
    )
    assert paths[0].read_text(encoding="utf-8") == install.MLX_LM_DENSE_SHIM
    assert paths[1].read_text(encoding="utf-8") == install.MLX_LM_DENSE_SHIM
    assert paths[2].read_text(encoding="utf-8") == install.MLX_LM_H_AUDEX_SHIM
    assert paths[3].read_text(encoding="utf-8") == install.SITECUSTOMIZE_SHIM
    assert "AUDEX_MAC_AUTO_PATCHES" in install.SITECUSTOMIZE_SHIM
