"""Hugging Face dataset acquisition for autonomous audio evaluation."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .audio_evaluation_datasets import MaterializedAudio

HF_API_BASE = "https://huggingface.co/api/datasets"
HF_DATASET_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"
EVALUATION_AUDIO_SAMPLE_RATE = 16_000


class HttpTransport(Protocol):
    def get_json(
        self, url: str, *, headers: Mapping[str, str]
    ) -> Mapping[str, Any]: ...

    def get_bytes(self, url: str, *, headers: Mapping[str, str]) -> bytes: ...


@dataclass(frozen=True, slots=True)
class DatasetPin:
    """One exact dataset view used by an evaluation run."""

    repo_id: str
    revision: str
    config: str
    split: str
    license: str
    expected_rows: int | None = None

    def __post_init__(self) -> None:
        required = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "license": self.license,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"dataset pin has empty fields: {missing}")
        if self.expected_rows is not None and self.expected_rows <= 0:
            raise ValueError("expected_rows must be positive")


class UrlLibTransport:
    """Small urllib transport so tests can replace network access cleanly."""

    def get_json(self, url: str, *, headers: Mapping[str, str]) -> Mapping[str, Any]:
        request = Request(url, headers=dict(headers))
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"expected JSON object from {url}")
        return payload

    def get_bytes(self, url: str, *, headers: Mapping[str, str]) -> bytes:
        request = Request(url, headers=dict(headers))
        with urlopen(request, timeout=60) as response:
            return response.read()


class HfDatasetClient:
    """Read pinned Hugging Face dataset metadata, rows, and audio assets."""

    def __init__(
        self,
        *,
        token: str | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self._transport = transport or UrlLibTransport()
        self._headers = {"User-Agent": "audex-mac-audio-eval/0.1"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def dataset_info(self, pin: DatasetPin) -> Mapping[str, Any]:
        repo = quote(pin.repo_id, safe="/")
        return self._transport.get_json(f"{HF_API_BASE}/{repo}", headers=self._headers)

    def rows_page(
        self,
        pin: DatasetPin,
        *,
        offset: int,
        length: int,
    ) -> Mapping[str, Any]:
        query = urlencode(
            {
                "dataset": pin.repo_id,
                "config": pin.config,
                "split": pin.split,
                "offset": int(offset),
                "length": int(length),
            }
        )
        return self._transport.get_json(
            f"{HF_DATASET_SERVER_ROWS}?{query}",
            headers=self._headers,
        )

    def download_bytes(self, url: str) -> bytes:
        return self._transport.get_bytes(url, headers=self._headers)


def fetch_verified_rows(
    pin: DatasetPin,
    *,
    client: HfDatasetClient,
    page_size: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Fetch all rows for a pinned dataset split after validating revision drift."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    info = client.dataset_info(pin)
    actual_revision = str(info.get("sha", "")).strip()
    if actual_revision != pin.revision:
        raise ValueError(
            "dataset revision drift: "
            f"{pin.repo_id} expected {pin.revision}, got {actual_revision or '<empty>'}"
        )

    rows: list[dict[str, Any]] = []
    expected_total: int | None = None
    for offset in range(0, pin.expected_rows or page_size, page_size):
        page = client.rows_page(pin, offset=offset, length=page_size)
        total = _positive_int(page.get("num_rows_total"), "num_rows_total")
        if expected_total is None:
            expected_total = total
            if pin.expected_rows is not None and total != pin.expected_rows:
                raise ValueError(
                    f"dataset row-count drift: {pin.repo_id}/{pin.split} "
                    f"expected {pin.expected_rows}, got {total}"
                )
        elif expected_total != total:
            raise ValueError("dataset-server returned inconsistent total row counts")

        raw_rows = page.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError("dataset-server page is missing rows")
        for entry in raw_rows:
            if not isinstance(entry, Mapping):
                raise ValueError("dataset-server row entry is not an object")
            truncated = entry.get("truncated_cells") or []
            if truncated:
                raise ValueError(
                    f"dataset-server returned truncated cells: {truncated}"
                )
            row = entry.get("row")
            if not isinstance(row, Mapping):
                raise ValueError("dataset-server row entry is missing row")
            rows.append(dict(row))

        if len(rows) >= total:
            break
        if not raw_rows:
            raise ValueError("dataset-server returned an empty page before completion")

    if expected_total is None:
        raise ValueError("dataset-server returned no pages")
    if len(rows) != expected_total:
        raise ValueError(f"expected {expected_total} rows, fetched {len(rows)}")
    return tuple(rows)


def select_stratified_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    master_seed: int,
    row_id: Callable[[Mapping[str, Any]], str],
    stratum: Callable[[Mapping[str, Any]], str],
) -> tuple[Mapping[str, Any], ...]:
    """Select a deterministic equal share from each row stratum."""

    if count <= 0:
        raise ValueError("selection count must be positive")
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    seen: set[str] = set()
    for row in rows:
        current_id = row_id(row).strip()
        current_stratum = stratum(row).strip().lower()
        if not current_id:
            raise ValueError("row id function returned an empty id")
        if not current_stratum:
            raise ValueError(f"row {current_id} has an empty stratum")
        if current_id in seen:
            raise ValueError(f"duplicate row id: {current_id}")
        seen.add(current_id)
        by_stratum.setdefault(current_stratum, []).append(row)
    if not by_stratum:
        raise ValueError("cannot select from an empty row set")
    if count % len(by_stratum):
        raise ValueError(
            f"count {count} cannot be balanced across {len(by_stratum)} strata"
        )

    share = count // len(by_stratum)
    selected: list[Mapping[str, Any]] = []
    for name in sorted(by_stratum):
        available = by_stratum[name]
        if len(available) < share:
            raise ValueError(
                f"stratum {name!r} requires {share} rows but has {len(available)}"
            )
        selected.extend(
            sorted(
                available,
                key=lambda row: (
                    hashlib.sha256(
                        f"{int(master_seed)}\0{row_id(row)}".encode()
                    ).hexdigest(),
                    row_id(row),
                ),
            )[:share]
        )
    return tuple(sorted(selected, key=row_id))


class HfAudioMaterializer:
    """Download one selected dataset-server audio asset into evaluator WAV form."""

    def __init__(
        self,
        *,
        client: HfDatasetClient,
        cache_dir: Path,
        decoder: Callable[[bytes, Path], MaterializedAudio] | None = None,
    ) -> None:
        self._client = client
        self._cache_dir = cache_dir
        self._decoder = decoder or decode_audio_to_16k_wav

    def materialize(self, row: Mapping[str, Any]) -> MaterializedAudio:
        source_url = _extract_audio_asset_url(row)
        raw_bytes = self._client.download_bytes(source_url)
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        destination = self._cache_dir / f"{source_hash}.wav"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            return inspect_materialized_audio(destination)
        return self._decoder(raw_bytes, destination)


def decode_audio_to_16k_wav(raw_bytes: bytes, destination: Path) -> MaterializedAudio:
    """Decode arbitrary audio bytes, mono-mix, resample, and write PCM16 WAV."""

    try:
        np = importlib.import_module("numpy")
        sf = importlib.import_module("soundfile")
        scipy_signal = importlib.import_module("scipy.signal")
    except ImportError as exc:
        raise RuntimeError(
            "audio evaluation materialization requires the audio-eval optional "
            "dependencies: numpy, scipy, and soundfile"
        ) from exc

    samples, sample_rate = sf.read(BytesIO(raw_bytes), dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("decoded audio is empty")
    mono = np.mean(samples, axis=1)
    if sample_rate != EVALUATION_AUDIO_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), EVALUATION_AUDIO_SAMPLE_RATE)
        mono = scipy_signal.resample_poly(
            mono,
            EVALUATION_AUDIO_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype("float32")
        sample_rate = EVALUATION_AUDIO_SAMPLE_RATE
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, mono, sample_rate, subtype="PCM_16")
    return inspect_materialized_audio(destination)


def inspect_materialized_audio(path: Path) -> MaterializedAudio:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "audio evaluation materialization requires soundfile"
        ) from exc

    info = sf.info(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return MaterializedAudio(
        path=str(path),
        sha256=digest,
        sample_rate=int(info.samplerate),
        duration_seconds=float(info.frames) / float(info.samplerate),
    )


def _extract_audio_asset_url(row: Mapping[str, Any]) -> str:
    audio = row.get("audio")
    first = audio[0] if isinstance(audio, list) and audio else audio
    if not isinstance(first, Mapping):
        raise ValueError("row does not contain a dataset-server audio asset")
    source_url = str(first.get("src", "")).strip()
    if not source_url:
        raise ValueError("dataset-server audio asset is missing src")
    return source_url


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
