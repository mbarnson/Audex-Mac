"""vLLM Metal CFG sampling patches for Audex TTS."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from logging import getLogger
from math import ceil
from typing import Any

from audex_mac.vllm_sts_requests import (
    AUDEX_TEXT_STATE_APPEND_MODE,
    AUDEX_TEXT_STATE_BOUNDARY_ARG,
    AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY,
    AUDEX_TEXT_STATE_KEY_ARG,
    AUDEX_TEXT_STATE_MODE_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG,
    AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG,
)

try:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except ModuleNotFoundError:
    SamplingParams = Any  # type: ignore[misc, assignment]
    BatchUpdate = Any  # type: ignore[misc, assignment]

    class LogitsProcessor:  # type: ignore[no-redef]
        pass


PATCH_SENTINEL = "_audex_mac_cfg_sampling_patch"
SCHEDULER_PATCH_SENTINEL = "_audex_mac_cfg_scheduler_patch"
KV_ALLOC_DEBUG_PATCH_SENTINEL = "_audex_mac_cfg_kv_alloc_debug_patch"
TIMING_PATCH_SENTINEL = "_audex_mac_paged_timing_patch"
CACHE_TIMING_PATCH_SENTINEL = "_audex_mac_cache_timing_patch"
PERSISTENT_CACHE_PATCH_SENTINEL = "_audex_mac_persistent_cache_patch"
TEXT_STATE_PATCH_SENTINEL = "_audex_mac_text_state_snapshot_patch"
MPS_CLEANUP_PATCH_SENTINEL = "_audex_mac_mps_cleanup_patch"
TTS_WINDOW_DECODE_PATCH_SENTINEL = "_audex_mac_tts_window_decode_patch"
TTS_WINDOW_BATCH_DECODE_PATCH_SENTINEL = "_audex_mac_tts_window_batch_decode_patch"
CFG_TTS_WINDOW_DECODE_ENV = "AUDEX_VLLM_CFG_TTS_WINDOW_DECODE"
DEBUG_SYNC_TTS_WINDOW_STAGES_ENV = "AUDEX_VLLM_DEBUG_SYNC_TTS_WINDOW_STAGES"
NONPAGED_ASYNC_EVAL_ENV = "AUDEX_VLLM_NONPAGED_ASYNC_EVAL"
NONPAGED_ASYNC_EVAL_TARGET_ENV = "AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET"
NONPAGED_ASYNC_EVAL_TARGET_LOGITS = "logits"
NONPAGED_ASYNC_EVAL_TARGET_NONE = "none"
NONPAGED_ASYNC_EVAL_TARGET_SAMPLE_LOGITS = "sample_logits"
NONPAGED_PERSISTENT_BATCH_CACHE_ENV = "AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE"
SPEECH_FIRST_SCHEDULING_ENV = "AUDEX_VLLM_SPEECH_FIRST_SCHEDULING"
LAST_ERROR: str | None = None
NATIVE_SAMPLE_COUNT = 0
NATIVE_SAMPLE_SECONDS = 0.0
NATIVE_SAMPLED_ROWS = 0
NATIVE_OUTPUT_ROWS = 0
NATIVE_DETAIL_SECONDS_BY_CATEGORY: dict[str, float] = {}
NATIVE_DETAIL_COUNT_BY_CATEGORY: dict[str, int] = {}
NATIVE_REJECTION_REASONS_LOGGED: set[str] = set()
TTS_WINDOW_DECODE_COUNT = 0
TTS_WINDOW_WEIGHT_CACHE_HITS = 0
TTS_WINDOW_WEIGHT_CACHE_MISSES = 0
NONPAGED_PERSISTENT_BATCH_CACHE_HITS = 0
NONPAGED_PERSISTENT_BATCH_CACHE_MISSES = 0
NONPAGED_PERSISTENT_BATCH_CACHE_FLUSHES = 0
TEXT_STATE_SNAPSHOT_CAPTURES = 0
CFG_SCHEDULER_DEBUG_COUNT = 0
CFG_KV_ALLOC_REJECTION_DEBUG_COUNT = 0
PAGED_SAMPLE_DEPTH = 0
SKIP_NEXT_PAGED_LOGITS_EVAL = False
PAGED_LOGITS_EVAL_SKIP_COUNT = 0
MX_EVAL_SECONDS_BY_CATEGORY: dict[str, float] = {}
MX_EVAL_COUNT_BY_CATEGORY: dict[str, int] = {}
MX_EVAL_SHAPE_COUNT_BY_CATEGORY: dict[str, dict[tuple[int, ...], int]] = {}
LOGGER = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VllmMetalCfgPatchReport:
    sample_from_logits: bool
    sample_prefill_tokens: bool
    scheduler: bool
    model_runner_symbols: bool
    mps_cleanup: bool
    error: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.sample_from_logits
            and self.sample_prefill_tokens
            and self.scheduler
            and self.model_runner_symbols
            and self.mps_cleanup
            and self.error is None
        )


class AudexMetalCFGTokenSyncInstaller(LogitsProcessor):
    """Install vLLM Metal post-sampling token sync inside each worker process.

    NVIDIA's ``CFGLogitsProcessor`` patches CUDA ``GPUModelRunner._sample`` from
    its constructor because spawned vLLM workers do not inherit parent-process
    monkey patches.  vLLM Metal does not use that CUDA runner, so this no-op
    logits processor mirrors the worker-local install point and patches the
    Metal sampler symbols instead.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        return None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_vllm_metal_cfg_patches()

    def is_argmax_invariant(self) -> bool:
        return True

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        return None

    def apply(self, logits: Any) -> Any:
        return logits


def apply_vllm_metal_cfg_patches() -> VllmMetalCfgPatchReport:
    """Patch vLLM Metal sampling so Audex CFG pairs remain token-synchronized."""

    global LAST_ERROR
    try:
        import vllm_metal.v1.model_runner as model_runner
        import vllm_metal.v1.sampling_batch as sampling_batch
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return VllmMetalCfgPatchReport(
            sample_from_logits=False,
            sample_prefill_tokens=False,
            scheduler=False,
            model_runner_symbols=False,
            mps_cleanup=False,
            error=LAST_ERROR,
        )

    _patch_sample_from_logits(sampling_batch)
    _patch_sample_prefill_tokens(sampling_batch)
    scheduler = _patch_scheduler_for_cfg()
    _patch_model_runner_paged_timing(model_runner)
    _patch_model_runner_non_paged_timing(model_runner)
    _patch_model_runner_cache_timing(model_runner)
    _patch_model_runner_persistent_batch_cache_cleanup(model_runner)
    _patch_mx_eval_timing(model_runner)
    _patch_model_runner_tts_window_decode(model_runner)
    mps_cleanup = _patch_mps_allocator_cleanup_assert()

    model_runner.sample_from_logits = sampling_batch.sample_from_logits
    model_runner.sample_prefill_tokens = sampling_batch.sample_prefill_tokens

    return VllmMetalCfgPatchReport(
        sample_from_logits=bool(
            getattr(sampling_batch.sample_from_logits, PATCH_SENTINEL, False)
        ),
        sample_prefill_tokens=bool(
            getattr(sampling_batch.sample_prefill_tokens, PATCH_SENTINEL, False)
        ),
        scheduler=scheduler,
        model_runner_symbols=(
            model_runner.sample_from_logits is sampling_batch.sample_from_logits
            and model_runner.sample_prefill_tokens
            is sampling_batch.sample_prefill_tokens
        ),
        mps_cleanup=mps_cleanup,
    )


def sync_cfg_token_ids(
    token_ids: list[int],
    sampling_params_list: Sequence[Any],
) -> int:
    """Copy each conditional CFG token into its unconditional partner slot."""

    pairs: dict[str, dict[str, int]] = {}
    for index, sampling_params in enumerate(sampling_params_list):
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args:
            continue
        role = extra_args.get("cfg_role")
        pair_id = extra_args.get("cfg_pair_id")
        if role not in {"cond", "uncond"} or not pair_id:
            continue
        pairs.setdefault(str(pair_id), {})[str(role)] = index

    synced = 0
    for roles in pairs.values():
        cond_index = roles.get("cond")
        uncond_index = roles.get("uncond")
        if cond_index is None or uncond_index is None:
            continue
        if cond_index >= len(token_ids) or uncond_index >= len(token_ids):
            continue
        token_ids[uncond_index] = token_ids[cond_index]
        synced += 1
    return synced


def _speech_first_scheduling_enabled() -> bool:
    value = os.environ.get(SPEECH_FIRST_SCHEDULING_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_audex_tts_request(request: Any) -> bool:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, Mapping):
        return False
    return "audex_tts_speechgen_end_id" in extra_args or extra_args.get("cfg_role") in {
        "cond",
        "uncond",
    }


def _scheduler_has_tts_work(scheduler: Any) -> bool:
    for source_name in ("running", "waiting", "skipped_waiting"):
        source = getattr(scheduler, source_name, ())
        if any(_is_audex_tts_request(request) for request in source):
            return True
    return False


def _hold_non_tts_running_requests(scheduler: Any) -> list[Any]:
    if not _speech_first_scheduling_enabled() or not _scheduler_has_tts_work(scheduler):
        return []
    held = [
        request for request in scheduler.running if not _is_audex_tts_request(request)
    ]
    if held:
        scheduler.running[:] = [
            request for request in scheduler.running if _is_audex_tts_request(request)
        ]
    return held


def _restore_non_tts_running_requests(scheduler: Any, held: list[Any]) -> None:
    requests = getattr(scheduler, "requests", {})
    running_ids = {request.request_id for request in scheduler.running}
    for request in held:
        if request.request_id in requests and request.request_id not in running_ids:
            scheduler.running.append(request)
            running_ids.add(request.request_id)


def _schedule_with_tts_priority(
    scheduler: Any,
    schedule: Any,
) -> Any:
    held = _hold_non_tts_running_requests(scheduler)
    try:
        return schedule()
    finally:
        _restore_non_tts_running_requests(scheduler, held)


def _patch_scheduler_for_cfg() -> bool:
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:
        global LAST_ERROR
        LAST_ERROR = f"CFG scheduler patch unavailable: {type(exc).__name__}: {exc}"
        return False

    current_schedule = Scheduler.schedule
    if getattr(current_schedule, SCHEDULER_PATCH_SENTINEL, False):
        return True

    original_init = Scheduler.__init__
    original_add = Scheduler.add_request
    original_finish = Scheduler.finish_requests
    original_schedule = Scheduler.schedule

    def init_with_cfg_tracking(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._audex_cfg_pairs = {}
        self._audex_cfg_req_to_pair = {}
        _patch_kv_allocate_slots_for_cfg_debug(getattr(self, "kv_cache_manager", None))
        _record_cfg_kv_capacity_once(self)

    def add_request_with_cfg_tracking(self: Any, request: Any) -> Any:
        result = original_add(self, request)
        pair_id = _cfg_pair_id_from_request(request)
        role = _cfg_role_from_request(request)
        if pair_id and role:
            self._audex_cfg_pairs.setdefault(pair_id, {})[role] = request.request_id
            self._audex_cfg_req_to_pair[request.request_id] = pair_id
        return result

    def finish_requests_with_cfg_tracking(
        self: Any,
        request_ids: Any,
        finished_status: Any,
    ) -> Any:
        if request_ids is None:
            result = original_finish(self, request_ids, finished_status)
            _cfg_pairs(self).clear()
            _cfg_req_to_pair(self).clear()
            return result

        ids = {request_ids} if isinstance(request_ids, str) else set(request_ids)
        partner_ids: set[str] = set()
        for request_id in ids:
            pair_id = _cfg_req_to_pair(self).get(request_id)
            if pair_id is None:
                continue
            for _role, partner_id in _cfg_pairs(self).get(pair_id, {}).items():
                if partner_id != request_id and partner_id in self.requests:
                    partner_ids.add(partner_id)

        all_ids = ids | partner_ids
        result = original_finish(self, all_ids, finished_status)
        for request_id in all_ids:
            pair_id = _cfg_req_to_pair(self).pop(request_id, None)
            if pair_id:
                _cfg_pairs(self).pop(pair_id, None)
        return result

    def schedule_with_cfg_pairs(self: Any, *args: Any, **kwargs: Any) -> Any:
        def schedule_cfg_pairs() -> Any:
            if not _cfg_pairs(self):
                return original_schedule(self, *args, **kwargs)

            _reorder_waiting_for_cfg(self)
            held = _hold_incomplete_cfg_pairs(self)
            scheduler_config = getattr(self, "scheduler_config", None)
            original_threshold = (
                getattr(scheduler_config, "long_prefill_token_threshold", None)
                if scheduler_config is not None
                else None
            )
            if scheduler_config is not None and original_threshold is not None:
                scheduler_config.long_prefill_token_threshold = (
                    int(getattr(self, "max_num_scheduled_tokens", 0) or 0) // 2
                )
            try:
                scheduler_output = original_schedule(self, *args, **kwargs)
            finally:
                if scheduler_config is not None and original_threshold is not None:
                    scheduler_config.long_prefill_token_threshold = original_threshold

            for request in reversed(held):
                _waiting_prepend(self.waiting, request)

            _equalize_cfg_pair_progress(self, scheduler_output)
            _log_split_cfg_pairs(self)
            _record_cfg_scheduler_admission(
                self,
                scheduler_output,
                held_incomplete_count=len(held),
            )
            return scheduler_output

        return _schedule_with_tts_priority(self, schedule_cfg_pairs)

    setattr(schedule_with_cfg_pairs, SCHEDULER_PATCH_SENTINEL, True)
    schedule_with_cfg_pairs.__wrapped__ = original_schedule  # type: ignore[attr-defined]
    Scheduler.__init__ = init_with_cfg_tracking
    Scheduler.add_request = add_request_with_cfg_tracking
    Scheduler.finish_requests = finish_requests_with_cfg_tracking
    Scheduler.schedule = schedule_with_cfg_pairs
    return True


def _cfg_pair_id_from_request(request: Any) -> str | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    pair_id = extra_args.get("cfg_pair_id")
    return str(pair_id) if pair_id else None


def _cfg_role_from_request(request: Any) -> str | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    role = extra_args.get("cfg_role")
    return str(role) if role in {"cond", "uncond"} else None


def _request_has_cfg_pair_metadata(request: Any) -> bool:
    return _cfg_pair_id_from_request(request) is not None


def _cfg_pairs(scheduler: Any) -> dict[str, dict[str, str]]:
    return getattr(scheduler, "_audex_cfg_pairs", {})


def _cfg_req_to_pair(scheduler: Any) -> dict[str, str]:
    return getattr(scheduler, "_audex_cfg_req_to_pair", {})


def _hold_incomplete_cfg_pairs(scheduler: Any) -> list[Any]:
    held: list[Any] = []
    keep: list[Any] = []
    for request in list(scheduler.waiting):
        pair_id = _cfg_req_to_pair(scheduler).get(request.request_id)
        complete = True
        if pair_id is not None:
            roles = _cfg_pairs(scheduler).get(pair_id, {})
            complete = len(roles) == 2 and all(
                request_id in scheduler.requests for request_id in roles.values()
            )
        (keep if complete else held).append(request)
    if held:
        scheduler.waiting.clear()
        scheduler.waiting.extend(keep)
    return held


def _reorder_waiting_for_cfg(scheduler: Any) -> None:
    waiting = scheduler.waiting
    if len(waiting) < 2:
        return

    requests = list(waiting)
    waiting.clear()
    seen: set[str] = set()
    result: list[Any] = []
    for request in requests:
        request_id = request.request_id
        if request_id in seen:
            continue
        seen.add(request_id)
        result.append(request)

        pair_id = _cfg_req_to_pair(scheduler).get(request_id)
        if pair_id is None:
            continue
        roles = _cfg_pairs(scheduler).get(pair_id, {})
        for partner_id in roles.values():
            if partner_id == request_id or partner_id in seen:
                continue
            for candidate in requests:
                if candidate.request_id == partner_id:
                    seen.add(partner_id)
                    result.append(candidate)
                    break

    waiting.extend(result)


def _waiting_prepend(waiting: Any, request: Any) -> None:
    prepend = getattr(waiting, "prepend_request", None)
    if callable(prepend):
        prepend(request)
        return
    insert = getattr(waiting, "insert", None)
    if callable(insert):
        insert(0, request)
        return
    waiting.appendleft(request)


def _equalize_cfg_pair_progress(scheduler: Any, scheduler_output: Any) -> None:
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return

    for roles in _cfg_pairs(scheduler).values():
        cond_id = roles.get("cond")
        uncond_id = roles.get("uncond")
        if not cond_id or not uncond_id:
            continue

        cond_sched = int(scheduled.get(cond_id, 0) or 0)
        uncond_sched = int(scheduled.get(uncond_id, 0) or 0)
        if cond_sched == 0 or uncond_sched == 0:
            continue

        cond_req = scheduler.requests.get(cond_id)
        uncond_req = scheduler.requests.get(uncond_id)
        if cond_req is None or uncond_req is None:
            continue
        if cond_req.num_computed_tokens == uncond_req.num_computed_tokens:
            continue

        target = min(cond_req.num_computed_tokens, uncond_req.num_computed_tokens)
        feasible = True
        for request, sched in ((cond_req, cond_sched), (uncond_req, uncond_sched)):
            if request.num_computed_tokens - target >= sched:
                feasible = False
                break
        if not feasible:
            continue

        for request_id, request, original_sched in (
            (cond_id, cond_req, cond_sched),
            (uncond_id, uncond_req, uncond_sched),
        ):
            diff = request.num_computed_tokens - target
            if diff <= 0:
                continue
            request.num_computed_tokens = target
            scheduled[request_id] = original_sched - diff
            scheduler_output.total_num_scheduled_tokens -= diff


def _log_split_cfg_pairs(scheduler: Any) -> None:
    if not _cfg_pairs(scheduler):
        return
    running_ids = {request.request_id for request in getattr(scheduler, "running", ())}
    split: list[str] = []
    for roles in _cfg_pairs(scheduler).values():
        cond_id = roles.get("cond")
        uncond_id = roles.get("uncond")
        if cond_id is None or uncond_id is None:
            continue
        if cond_id not in scheduler.requests or uncond_id not in scheduler.requests:
            continue
        if (cond_id in running_ids) != (uncond_id in running_ids):
            split.append(cond_id if cond_id in running_ids else uncond_id)
            continue
        cond_req = scheduler.requests.get(cond_id)
        uncond_req = scheduler.requests.get(uncond_id)
        if (
            cond_req is not None
            and uncond_req is not None
            and cond_req.num_computed_tokens != uncond_req.num_computed_tokens
        ):
            split.append(
                cond_id
                if cond_req.num_computed_tokens > uncond_req.num_computed_tokens
                else uncond_id
            )
    if split:
        LOGGER.warning("Audex CFG pair split detected in scheduler: %s", split)


def _patch_sample_from_logits(sampling_batch: Any) -> None:
    current = sampling_batch.sample_from_logits
    if getattr(current, PATCH_SENTINEL, False):
        return

    def sample_from_logits_with_cfg_sync(
        logits_2d: Any,
        batch: Any,
        sampler: Any,
        device: Any,
    ) -> Any:
        sample_started = time.perf_counter()
        guarded_logits = _apply_disallowed_token_mask_mlx(
            logits_2d,
            batch.sampling_params_list,
        )
        native_result = _sample_native_mlx_if_supported(
            guarded_logits, batch, sampling_batch
        )
        if native_result is not None:
            _record_native_sample_seconds(time.perf_counter() - sample_started)
            return native_result
        result = current(guarded_logits, batch, sampler, device)
        sync_cfg_token_ids(result.token_ids, batch.sampling_params_list)
        return result

    setattr(sample_from_logits_with_cfg_sync, PATCH_SENTINEL, True)
    sample_from_logits_with_cfg_sync.__wrapped__ = current  # type: ignore[attr-defined]
    sampling_batch.sample_from_logits = sample_from_logits_with_cfg_sync


def _patch_model_runner_paged_timing(model_runner: Any) -> None:
    runner_cls = getattr(model_runner, "MetalModelRunner", None)
    if runner_cls is None:
        return
    current = runner_cls._sample_paged_batch
    if getattr(current, TIMING_PATCH_SENTINEL, False):
        return

    def sample_paged_batch_with_timing(self: Any, grammar_output: Any | None = None):
        global PAGED_SAMPLE_DEPTH, SKIP_NEXT_PAGED_LOGITS_EVAL
        state = getattr(self, "_execute_model_state", None)
        decode_count = len(getattr(state, "decode_reqs", ()) or ())
        prefill_count = len(getattr(state, "prefill_reqs", ()) or ())
        token_count = int(getattr(state, "num_decode_tokens", 0) or 0)
        started = time.perf_counter()
        PAGED_SAMPLE_DEPTH += 1
        SKIP_NEXT_PAGED_LOGITS_EVAL = (
            _skip_paged_logits_eval_enabled()
            and _can_skip_paged_logits_eval(
                state,
                grammar_output,
            )
        )
        try:
            result = current(self, grammar_output)
        finally:
            SKIP_NEXT_PAGED_LOGITS_EVAL = False
            PAGED_SAMPLE_DEPTH -= 1
        _record_paged_sample_timing(
            self,
            time.perf_counter() - started,
            decode_count=decode_count,
            prefill_count=prefill_count,
            token_count=token_count,
        )
        return result

    setattr(sample_paged_batch_with_timing, TIMING_PATCH_SENTINEL, True)
    sample_paged_batch_with_timing.__wrapped__ = current  # type: ignore[attr-defined]
    runner_cls._sample_paged_batch = sample_paged_batch_with_timing


def _patch_model_runner_non_paged_timing(model_runner: Any) -> None:
    runner_cls = getattr(model_runner, "MetalModelRunner", None)
    if runner_cls is None:
        return
    current = getattr(runner_cls, "_run_non_paged_decode_batch", None)
    if current is None or getattr(current, TIMING_PATCH_SENTINEL, False):
        return

    def run_non_paged_decode_batch_with_timing(self: Any, batch: Any) -> Any:
        decode_reqs = tuple(getattr(batch, "valid_decode_reqs", ()) or ())
        cached_req_ids = tuple(getattr(batch, "scheduled_cached_req_ids", ()) or ())
        cfg_counts = _cfg_counts_for_decode_reqs(decode_reqs)
        started = time.perf_counter()
        result = current(self, batch)
        _record_non_paged_decode_timing(
            self,
            time.perf_counter() - started,
            decode_count=len(decode_reqs),
            cached_count=len(cached_req_ids),
            batched=len(decode_reqs) >= 2,
            cfg_cond_count=cfg_counts["cond"],
            cfg_uncond_count=cfg_counts["uncond"],
            cfg_complete_pair_count=cfg_counts["complete_pairs"],
        )
        return result

    setattr(run_non_paged_decode_batch_with_timing, TIMING_PATCH_SENTINEL, True)
    run_non_paged_decode_batch_with_timing.__wrapped__ = current  # type: ignore[attr-defined]
    runner_cls._run_non_paged_decode_batch = run_non_paged_decode_batch_with_timing


def _cfg_counts_for_decode_reqs(
    decode_reqs: Sequence[tuple[str, Any]],
) -> dict[str, int]:
    return _cfg_counts_for_requests(state for _req_id, state in decode_reqs)


def _cfg_counts_for_requests(requests: Any) -> dict[str, int]:
    pairs: dict[str, set[str]] = {}
    cond_count = 0
    uncond_count = 0
    request_count = 0
    for request in requests:
        request_count += 1
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args:
            continue
        role = extra_args.get("cfg_role")
        pair_id = extra_args.get("cfg_pair_id")
        if role == "cond":
            cond_count += 1
        elif role == "uncond":
            uncond_count += 1
        else:
            continue
        if pair_id:
            pairs.setdefault(str(pair_id), set()).add(str(role))
    complete_pairs = sum(1 for roles in pairs.values() if {"cond", "uncond"} <= roles)
    return {
        "requests": request_count,
        "cond": cond_count,
        "uncond": uncond_count,
        "complete_pairs": complete_pairs,
    }


def _record_cfg_scheduler_admission(
    scheduler: Any,
    scheduler_output: Any,
    *,
    held_incomplete_count: int,
) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return
    scheduled_requests = tuple(
        scheduler.requests[request_id]
        for request_id in scheduled
        if request_id in scheduler.requests
    )
    scheduled_counts = _cfg_counts_for_requests(scheduled_requests)
    running_counts = _cfg_counts_for_requests(
        tuple(getattr(scheduler, "running", ()) or ())
    )
    waiting_counts = _cfg_counts_for_requests(
        tuple(getattr(scheduler, "waiting", ()) or ())
        + tuple(getattr(scheduler, "skipped_waiting", ()) or ())
    )
    if (
        not scheduled_counts["complete_pairs"]
        and not running_counts["complete_pairs"]
        and not waiting_counts["complete_pairs"]
    ):
        return

    global CFG_SCHEDULER_DEBUG_COUNT
    CFG_SCHEDULER_DEBUG_COUNT += 1
    should_log = (
        CFG_SCHEDULER_DEBUG_COUNT <= 10
        or CFG_SCHEDULER_DEBUG_COUNT % 50 == 0
        or scheduled_counts["complete_pairs"] > 2
        or running_counts["complete_pairs"] > 2
    )
    if not should_log:
        return

    scheduler_config = getattr(scheduler, "scheduler_config", None)
    reserve_full_isl = (
        getattr(scheduler_config, "scheduler_reserve_full_isl", None)
        if scheduler_config is not None
        else None
    )
    print(
        "Audex vLLM Metal: cfg scheduler admission "
        f"count={CFG_SCHEDULER_DEBUG_COUNT} "
        f"scheduled_reqs={scheduled_counts['requests']} "
        f"scheduled_complete_pairs={scheduled_counts['complete_pairs']} "
        f"scheduled_cond_reqs={scheduled_counts['cond']} "
        f"scheduled_uncond_reqs={scheduled_counts['uncond']} "
        f"running_reqs={running_counts['requests']} "
        f"running_complete_pairs={running_counts['complete_pairs']} "
        f"waiting_reqs={waiting_counts['requests']} "
        f"waiting_complete_pairs={waiting_counts['complete_pairs']} "
        f"held_incomplete_reqs={held_incomplete_count} "
        f"max_running_reqs={getattr(scheduler, 'max_num_running_reqs', None)} "
        f"max_scheduled_tokens={getattr(scheduler, 'max_num_scheduled_tokens', None)} "
        f"total_scheduled_tokens={getattr(scheduler_output, 'total_num_scheduled_tokens', None)} "
        f"scheduler_reserve_full_isl={reserve_full_isl}",
        file=sys.stderr,
        flush=True,
    )


def _patch_kv_allocate_slots_for_cfg_debug(kv_cache_manager: Any) -> None:
    if kv_cache_manager is None:
        return
    current = getattr(kv_cache_manager, "allocate_slots", None)
    if not callable(current) or getattr(current, KV_ALLOC_DEBUG_PATCH_SENTINEL, False):
        return

    def allocate_slots_with_cfg_debug(request: Any, *args: Any, **kwargs: Any) -> Any:
        result = current(request, *args, **kwargs)
        if result is None and _request_has_cfg_pair_metadata(request):
            _record_cfg_kv_alloc_rejection(kv_cache_manager, request, args, kwargs)
        return result

    setattr(allocate_slots_with_cfg_debug, KV_ALLOC_DEBUG_PATCH_SENTINEL, True)
    allocate_slots_with_cfg_debug.__wrapped__ = current  # type: ignore[attr-defined]
    kv_cache_manager.allocate_slots = allocate_slots_with_cfg_debug


def _record_cfg_kv_alloc_rejection(
    kv_cache_manager: Any,
    request: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    global CFG_KV_ALLOC_REJECTION_DEBUG_COUNT
    CFG_KV_ALLOC_REJECTION_DEBUG_COUNT += 1
    if CFG_KV_ALLOC_REJECTION_DEBUG_COUNT > 20 and (
        CFG_KV_ALLOC_REJECTION_DEBUG_COUNT % 50 != 0
    ):
        return

    block_pool = getattr(kv_cache_manager, "block_pool", None)
    get_num_free_blocks = getattr(block_pool, "get_num_free_blocks", None)
    free_blocks = get_num_free_blocks() if callable(get_num_free_blocks) else None
    role = _cfg_role_from_request(request)
    pair_id = _cfg_pair_id_from_request(request)
    num_new_tokens = args[0] if args else kwargs.get("num_new_tokens")
    print(
        "Audex vLLM Metal: cfg kv allocation rejected "
        f"count={CFG_KV_ALLOC_REJECTION_DEBUG_COUNT} "
        f"request_id={getattr(request, 'request_id', None)} "
        f"cfg_role={role} cfg_pair_id={pair_id} "
        f"status={getattr(request, 'status', None)} "
        f"num_tokens={getattr(request, 'num_tokens', None)} "
        f"num_prompt_tokens={getattr(request, 'num_prompt_tokens', None)} "
        f"num_computed_tokens={getattr(request, 'num_computed_tokens', None)} "
        f"num_new_tokens={num_new_tokens} "
        f"free_blocks={free_blocks} "
        f"watermark_blocks={getattr(kv_cache_manager, 'watermark_blocks', None)} "
        f"full_sequence_must_fit={kwargs.get('full_sequence_must_fit')} "
        f"reserved_blocks={kwargs.get('reserved_blocks')} "
        f"has_scheduled_reqs={kwargs.get('has_scheduled_reqs')}",
        file=sys.stderr,
        flush=True,
    )


def _record_cfg_kv_capacity_once(scheduler: Any) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    if getattr(scheduler, "_audex_cfg_capacity_logged", False):
        return
    scheduler._audex_cfg_capacity_logged = True

    kv_cache_config = getattr(scheduler, "kv_cache_config", None)
    groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    group_summaries: list[str] = []
    total_layers = 0
    page_size_values: list[int] = []
    for index, group in enumerate(groups):
        layer_names = tuple(getattr(group, "layer_names", ()) or ())
        total_layers += len(layer_names)
        spec = getattr(group, "kv_cache_spec", None)
        page_size = getattr(spec, "page_size_bytes", None)
        if isinstance(page_size, int):
            page_size_values.append(page_size)
        block_size = getattr(spec, "block_size", None)
        spec_kind = getattr(type(spec), "__name__", None)
        group_summaries.append(
            f"{index}:{spec_kind}:layers={len(layer_names)}:"
            f"block_size={block_size}:page_bytes={page_size}"
        )

    num_blocks = getattr(kv_cache_config, "num_blocks", None)
    max_concurrency = _estimate_cfg_kv_max_concurrency(scheduler, kv_cache_config)
    pool_bytes_per_block = sum(page_size_values) if page_size_values else None
    block_size = getattr(scheduler, "block_size", None)
    max_model_len = getattr(scheduler, "max_model_len", None)
    blocks_per_request = _max_length_blocks_per_request(max_model_len, block_size)
    bytes_per_request = (
        blocks_per_request * pool_bytes_per_block
        if blocks_per_request is not None and pool_bytes_per_block is not None
        else None
    )
    inferred_request_capacity = (
        round(num_blocks / blocks_per_request, 3)
        if isinstance(num_blocks, int)
        and blocks_per_request is not None
        and blocks_per_request > 0
        else None
    )
    print(
        "Audex vLLM Metal: cfg kv capacity "
        f"num_blocks={num_blocks} "
        f"block_size={block_size} "
        f"max_model_len={max_model_len} "
        f"max_length_blocks_per_request={blocks_per_request} "
        f"max_length_bytes_per_request={bytes_per_request} "
        f"max_running_reqs={getattr(scheduler, 'max_num_running_reqs', None)} "
        f"max_scheduled_tokens={getattr(scheduler, 'max_num_scheduled_tokens', None)} "
        f"group_count={len(groups)} "
        f"total_layers={total_layers} "
        f"pool_bytes_per_block={pool_bytes_per_block} "
        f"inferred_request_capacity={inferred_request_capacity} "
        f"estimated_max_concurrency={max_concurrency} "
        f"groups={';'.join(group_summaries)}",
        file=sys.stderr,
        flush=True,
    )


def _max_length_blocks_per_request(max_model_len: Any, block_size: Any) -> int | None:
    if not isinstance(max_model_len, int) or not isinstance(block_size, int):
        return None
    if max_model_len <= 0 or block_size <= 0:
        return None
    return ceil(max_model_len / block_size)


def _estimate_cfg_kv_max_concurrency(
    scheduler: Any,
    kv_cache_config: Any,
) -> float | None:
    if kv_cache_config is None:
        return None
    try:
        from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

        return round(
            float(
                get_max_concurrency_for_kv_cache_config(
                    scheduler.vllm_config,
                    kv_cache_config,
                )
            ),
            3,
        )
    except Exception:
        return None


def _patch_model_runner_cache_timing(model_runner: Any) -> None:
    merge_current = getattr(model_runner, "_merge_kv_caches", None)
    if callable(merge_current) and not getattr(
        merge_current,
        CACHE_TIMING_PATCH_SENTINEL,
        False,
    ):

        def merge_kv_caches_with_timing(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = merge_current(*args, **kwargs)
            _record_native_detail_timing(
                "nonpaged_kv_cache_merge",
                time.perf_counter() - started,
            )
            return result

        setattr(
            merge_kv_caches_with_timing,
            CACHE_TIMING_PATCH_SENTINEL,
            True,
        )
        merge_kv_caches_with_timing.__wrapped__ = merge_current  # type: ignore[attr-defined]
        model_runner._merge_kv_caches = merge_kv_caches_with_timing

    extract_current = getattr(model_runner, "_extract_kv_cache", None)
    if callable(extract_current) and not getattr(
        extract_current,
        CACHE_TIMING_PATCH_SENTINEL,
        False,
    ):

        def extract_kv_cache_with_timing(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = extract_current(*args, **kwargs)
            _record_native_detail_timing(
                "nonpaged_kv_cache_extract",
                time.perf_counter() - started,
            )
            return result

        setattr(
            extract_kv_cache_with_timing,
            CACHE_TIMING_PATCH_SENTINEL,
            True,
        )
        extract_kv_cache_with_timing.__wrapped__ = extract_current  # type: ignore[attr-defined]
        model_runner._extract_kv_cache = extract_kv_cache_with_timing


def _patch_model_runner_persistent_batch_cache_cleanup(model_runner: Any) -> None:
    runner_cls = getattr(model_runner, "MetalModelRunner", None)
    if runner_cls is None:
        return
    current = getattr(runner_cls, "_cleanup_finished_requests", None)
    if current is None or getattr(current, PERSISTENT_CACHE_PATCH_SENTINEL, False):
        return

    def cleanup_finished_requests_with_persistent_cache_flush(
        self: Any,
        evicted_req_ids: set[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if evicted_req_ids:
            _record_text_state_snapshots_before_cleanup(self, evicted_req_ids)
            _flush_persistent_nonpaged_batch_cache(
                self,
                model_runner,
                reason="finished_requests",
            )
        return current(self, evicted_req_ids, *args, **kwargs)

    setattr(
        cleanup_finished_requests_with_persistent_cache_flush,
        PERSISTENT_CACHE_PATCH_SENTINEL,
        True,
    )
    cleanup_finished_requests_with_persistent_cache_flush.__wrapped__ = current  # type: ignore[attr-defined]
    runner_cls._cleanup_finished_requests = (
        cleanup_finished_requests_with_persistent_cache_flush
    )


def _record_text_state_snapshots_before_cleanup(
    runner: Any,
    evicted_req_ids: set[str],
) -> None:
    request_states = getattr(runner, "_request_states", None)
    if not isinstance(request_states, dict):
        return
    for req_id in evicted_req_ids:
        state = request_states.get(req_id)
        metadata = _text_state_snapshot_metadata(req_id, state)
        if metadata is None:
            continue
        if os.environ.get("AUDEX_VLLM_MM_DEBUG") == "1":
            print(
                "Audex vLLM MM debug: text-state snapshot "
                f"key={metadata['state_key']!r} "
                f"eligible={metadata['reuse_eligible']} "
                f"boundary={metadata['boundary']!r} "
                f"tokens={metadata['prefix_token_count']}",
                flush=True,
            )
        snapshots = getattr(runner, "_audex_text_state_snapshots", None)
        if not isinstance(snapshots, dict):
            snapshots = {}
            runner._audex_text_state_snapshots = snapshots
        previous = snapshots.get(metadata["state_key"])
        if (
            isinstance(previous, dict)
            and previous.get("reuse_eligible")
            and not metadata.get("reuse_eligible")
        ):
            continue
        snapshots[metadata["state_key"]] = metadata
        global TEXT_STATE_SNAPSHOT_CAPTURES
        TEXT_STATE_SNAPSHOT_CAPTURES += 1


def _text_state_snapshot_metadata(req_id: str, state: Any) -> dict[str, Any] | None:
    sampling_params = getattr(state, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return None
    state_key = extra_args.get(AUDEX_TEXT_STATE_KEY_ARG)
    if not isinstance(state_key, str) or not state_key:
        return None
    mode = extra_args.get(AUDEX_TEXT_STATE_MODE_ARG)
    if mode != AUDEX_TEXT_STATE_APPEND_MODE:
        return None
    prefix_token_count = _safe_nonnegative_int(
        extra_args.get(AUDEX_TEXT_STATE_PREFIX_TOKEN_COUNT_ARG)
    )
    prefix_token_hash = extra_args.get(AUDEX_TEXT_STATE_PREFIX_TOKEN_HASH_ARG)
    if prefix_token_count is None or not isinstance(prefix_token_hash, str):
        return None
    token_ids = list(getattr(state, "token_ids", ()) or ())
    prompt_len = _safe_nonnegative_int(getattr(state, "prompt_len", None)) or 0
    generated_tokens = (
        _safe_nonnegative_int(getattr(state, "generated_tokens", None)) or 0
    )
    boundary = extra_args.get(AUDEX_TEXT_STATE_BOUNDARY_ARG) or "raw_generation_state"
    boundary_tokens = token_ids
    committed_boundary_verified = False
    reuse_eligible = False
    reuse_blocked_reason = "raw_generation_state_may_differ_from_committed_history"
    if boundary == AUDEX_TEXT_STATE_COMMITTED_HISTORY_BOUNDARY:
        boundary_tokens = token_ids[:prompt_len]
        committed_boundary_verified = (
            prompt_len == prefix_token_count
            and _token_hash(boundary_tokens) == prefix_token_hash
        )
        safe_generated_tokens = generated_tokens <= 1
        has_cache = bool(getattr(state, "cache", None))
        reuse_eligible = (
            committed_boundary_verified and safe_generated_tokens and has_cache
        )
        reuse_blocked_reason = ""
        if not committed_boundary_verified:
            reuse_blocked_reason = "committed_history_prefix_mismatch"
        elif not safe_generated_tokens:
            reuse_blocked_reason = "committed_history_request_decoded_extra_tokens"
        elif not has_cache:
            reuse_blocked_reason = "committed_history_missing_cache"
    return {
        "state_key": state_key,
        "request_id": req_id,
        "mode": mode,
        "boundary": boundary,
        "committed_boundary_verified": committed_boundary_verified,
        "reuse_eligible": reuse_eligible,
        "reuse_blocked_reason": reuse_blocked_reason,
        "prefix_token_count": prefix_token_count,
        "prefix_token_hash": prefix_token_hash,
        "prompt_len": prompt_len,
        "token_count": len(token_ids),
        "token_hash": _token_hash(token_ids),
        "boundary_token_count": len(boundary_tokens),
        "boundary_token_hash": _token_hash(boundary_tokens),
        "generated_tokens": generated_tokens,
        "has_cache": bool(getattr(state, "cache", None)),
        "cache": getattr(state, "cache", None) if reuse_eligible else None,
        "boundary_tokens": tuple(boundary_tokens) if reuse_eligible else (),
    }


def _safe_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _token_hash(tokens: Sequence[int]) -> str:
    digest = sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _patch_model_runner_tts_window_decode(model_runner: Any) -> None:
    runner_cls = getattr(model_runner, "MetalModelRunner", None)
    if runner_cls is None:
        return
    current_sequential = getattr(runner_cls, "_sequential_decode", None)
    if current_sequential is not None and not getattr(
        current_sequential,
        TTS_WINDOW_DECODE_PATCH_SENTINEL,
        False,
    ):
        _patch_model_runner_sequential_tts_window_decode(
            runner_cls,
            current_sequential,
            model_runner,
        )

    current_batched = getattr(runner_cls, "_batched_decode", None)
    if current_batched is not None and not getattr(
        current_batched,
        TTS_WINDOW_BATCH_DECODE_PATCH_SENTINEL,
        False,
    ):
        _patch_model_runner_batched_tts_window_decode(
            runner_cls,
            current_batched,
            model_runner,
        )


def _patch_model_runner_sequential_tts_window_decode(
    runner_cls: Any,
    current: Any,
    model_runner: Any,
) -> None:
    def sequential_decode_with_tts_window_logits(
        self: Any,
        decode_reqs: list[tuple[str, Any]],
    ) -> Any:
        if len(decode_reqs) != 1:
            return current(self, decode_reqs)
        fast_result = _try_sequential_tts_window_decode(
            self,
            decode_reqs[0],
            model_runner,
        )
        if fast_result is None:
            sampling_params = getattr(decode_reqs[0][1], "sampling_params", None)
            if _requires_compact_tts_window_decode(sampling_params):
                raise RuntimeError(
                    "Audex required compact TTS-window decode, but the "
                    "sequential fast path rejected this request."
                )
            return current(self, decode_reqs)
        return fast_result

    setattr(
        sequential_decode_with_tts_window_logits,
        TTS_WINDOW_DECODE_PATCH_SENTINEL,
        True,
    )
    sequential_decode_with_tts_window_logits.__wrapped__ = current  # type: ignore[attr-defined]
    runner_cls._sequential_decode = sequential_decode_with_tts_window_logits


def _patch_model_runner_batched_tts_window_decode(
    runner_cls: Any,
    current: Any,
    model_runner: Any,
) -> None:
    def batched_decode_with_tts_window_logits(
        self: Any,
        decode_reqs: list[tuple[str, Any]],
    ) -> Any:
        if len(decode_reqs) < 2:
            return current(self, decode_reqs)
        fast_result = _try_batched_tts_window_decode(self, decode_reqs, model_runner)
        if fast_result is None:
            if any(
                _requires_compact_tts_window_decode(
                    getattr(state, "sampling_params", None)
                )
                for _req_id, state in decode_reqs
            ):
                raise RuntimeError(
                    "Audex required compact TTS-window decode, but the "
                    "batched fast path rejected this request batch."
                )
            async_result = _try_batched_decode_with_async_eval(
                self,
                decode_reqs,
                model_runner,
            )
            if async_result is not None:
                return async_result
            return current(self, decode_reqs)
        return fast_result

    setattr(
        batched_decode_with_tts_window_logits,
        TTS_WINDOW_BATCH_DECODE_PATCH_SENTINEL,
        True,
    )
    batched_decode_with_tts_window_logits.__wrapped__ = current  # type: ignore[attr-defined]
    runner_cls._batched_decode = batched_decode_with_tts_window_logits


def _try_sequential_tts_window_decode(
    runner: Any,
    decode_req: tuple[str, Any],
    model_runner: Any,
) -> Any | None:
    _req_id, state = decode_req
    sampling_params = getattr(state, "sampling_params", None)
    allowed_window = _allowed_tts_token_window(sampling_params)
    if allowed_window is None or _is_cfg_sampling_params(sampling_params):
        return None
    try:
        import mlx.core as mx

        random_keys = None
        if getattr(state, "generator", None) is not None:
            random_keys = _seeded_random_keys_for_states(
                [sampling_params],
                [state],
                mx,
            )
            if random_keys is None:
                return None

        sampling_result_cls = model_runner._SamplingResult
        last_token = state.token_ids[-1] if state.token_ids else 0
        input_ids = mx.array([[last_token]], dtype=mx.int32)
        debug_timing = _native_sampling_debug_enabled()
        forward_started = time.perf_counter() if debug_timing else 0.0
        hidden_states = _forward_hidden_states_for_tts_window(runner, input_ids, state)
        if debug_timing:
            _record_native_detail_timing(
                "tts_window_forward",
                time.perf_counter() - forward_started,
            )
        if hidden_states is None:
            return None
        if debug_timing and _debug_sync_tts_window_stages_enabled():
            forward_eval_started = time.perf_counter()
            mx.eval(hidden_states)
            _record_native_detail_timing(
                "tts_window_forward_eval",
                time.perf_counter() - forward_eval_started,
            )
        project_started = time.perf_counter() if debug_timing else 0.0
        logits = _project_tts_window_logits(
            runner,
            hidden_states[:, -1, :],
            allowed_window,
            mx,
        )
        if debug_timing:
            _record_native_detail_timing(
                "tts_window_project",
                time.perf_counter() - project_started,
            )
        if logits is None:
            return None
        if debug_timing and _debug_sync_tts_window_stages_enabled():
            project_eval_started = time.perf_counter()
            mx.eval(logits)
            _record_native_detail_timing(
                "tts_window_project_eval",
                time.perf_counter() - project_eval_started,
            )
        sample_started = time.perf_counter() if debug_timing else 0.0
        sample_args = (
            logits,
            [sampling_params],
            runner._logitsprocs,
            sampling_result_cls,
            mx,
        )
        if random_keys is None:
            result = _sample_tts_window_logits_from_params_if_supported(*sample_args)
        else:
            result = _sample_tts_window_logits_from_params_if_supported(
                *sample_args,
                random_keys=random_keys,
            )
        if debug_timing:
            _record_native_detail_timing(
                "tts_window_sample",
                time.perf_counter() - sample_started,
            )
        if result is None:
            return None
        global TTS_WINDOW_DECODE_COUNT
        TTS_WINDOW_DECODE_COUNT += 1
        [next_token] = result.token_ids
        state.token_ids.append(next_token)
        state.generated_tokens += 1
        return sampling_result_cls([next_token], result.logprobs)
    except Exception as exc:
        _log_native_sampling_rejection_once(
            f"tts window decode exception: {type(exc).__name__}: {exc}"
        )
        return None


def _try_batched_tts_window_decode(
    runner: Any,
    decode_reqs: list[tuple[str, Any]],
    model_runner: Any,
) -> Any | None:
    allowed_windows = []
    has_cfg = False
    for _req_id, state in decode_reqs:
        sampling_params = getattr(state, "sampling_params", None)
        allowed_window = _allowed_tts_token_window(sampling_params)
        if allowed_window is None:
            return None
        has_cfg = has_cfg or _is_cfg_sampling_params(sampling_params)
        allowed_windows.append(allowed_window)
    if has_cfg and not _cfg_tts_window_decode_enabled():
        return None
    allowed_window = _uniform_allowed_tts_token_window(allowed_windows)
    if allowed_window is None:
        return None

    try:
        import mlx.core as mx

        states = [state for _req_id, state in decode_reqs]
        has_generators = any(
            getattr(state, "generator", None) is not None for state in states
        )
        random_keys = None
        if has_generators:
            random_keys = _seeded_random_keys_for_states(
                [state.sampling_params for state in states],
                states,
                mx,
            )
            if random_keys is None:
                return None

        sampling_result_cls = model_runner._SamplingResult
        merge_kv_caches = getattr(model_runner, "_merge_kv_caches", None)
        extract_kv_cache = getattr(model_runner, "_extract_kv_cache", None)
        if not callable(merge_kv_caches) or not callable(extract_kv_cache):
            return None

        debug_timing = _native_sampling_debug_enabled()
        last_tokens = [
            state.token_ids[-1] if state.token_ids else 0
            for _req_id, state in decode_reqs
        ]
        batch_cache = _nonpaged_batch_cache_for_decode(
            runner,
            decode_reqs,
            model_runner,
            merge_kv_caches,
        )
        input_ids = mx.array(last_tokens, dtype=mx.int32)[:, None]
        forward_started = time.perf_counter() if debug_timing else 0.0
        hidden_states = _forward_hidden_states_for_tts_window_cache(
            runner,
            input_ids,
            batch_cache,
        )
        if debug_timing:
            _record_native_detail_timing(
                "tts_window_batch_forward",
                time.perf_counter() - forward_started,
            )
        if hidden_states is None:
            return None
        if debug_timing and _debug_sync_tts_window_stages_enabled():
            forward_eval_started = time.perf_counter()
            mx.eval(hidden_states)
            _record_native_detail_timing(
                "tts_window_batch_forward_eval",
                time.perf_counter() - forward_eval_started,
            )

        project_started = time.perf_counter() if debug_timing else 0.0
        logits = _project_tts_window_logits(
            runner,
            hidden_states[:, -1, :],
            allowed_window,
            mx,
        )
        if debug_timing:
            _record_native_detail_timing(
                "tts_window_batch_project",
                time.perf_counter() - project_started,
            )
        if logits is None:
            return None
        if debug_timing and _debug_sync_tts_window_stages_enabled():
            project_eval_started = time.perf_counter()
            mx.eval(logits)
            _record_native_detail_timing(
                "tts_window_batch_project_eval",
                time.perf_counter() - project_eval_started,
            )

        sampling_params_list = [state.sampling_params for _req_id, state in decode_reqs]
        sample_started = time.perf_counter() if debug_timing else 0.0
        sample_args = (
            logits,
            sampling_params_list,
            runner._logitsprocs,
            sampling_result_cls,
            mx,
        )
        if random_keys is None:
            result = _sample_tts_window_logits_from_params_if_supported(*sample_args)
        else:
            result = _sample_tts_window_logits_from_params_if_supported(
                *sample_args,
                random_keys=random_keys,
            )
        if debug_timing:
            _record_native_detail_timing(
                "tts_window_batch_sample",
                time.perf_counter() - sample_started,
            )
        if result is None:
            return None

        next_tokens = [int(token_id) for token_id in result.token_ids]
        if len(next_tokens) != len(decode_reqs):
            return None
        global TTS_WINDOW_DECODE_COUNT
        TTS_WINDOW_DECODE_COUNT += len(next_tokens)
        persistent_cache_enabled = _nonpaged_persistent_batch_cache_enabled()
        for index, (_req_id, state) in enumerate(decode_reqs):
            if not persistent_cache_enabled:
                state.cache = extract_kv_cache(batch_cache, index)
            state.token_ids.append(next_tokens[index])
            state.generated_tokens += 1
        return sampling_result_cls(next_tokens, result.logprobs)
    except Exception as exc:
        _log_native_sampling_rejection_once(
            f"batched tts window decode exception: {type(exc).__name__}: {exc}"
        )
        return None


def _try_batched_decode_with_async_eval(
    runner: Any,
    decode_reqs: list[tuple[str, Any]],
    model_runner: Any,
) -> Any | None:
    if not _nonpaged_async_eval_enabled():
        return None
    try:
        import mlx.core as mx

        sampling_batch_cls = getattr(model_runner, "SamplingBatch", None)
        sampling_result_cls = getattr(model_runner, "_SamplingResult", None)
        sample_from_logits = getattr(model_runner, "sample_from_logits", None)
        merge_kv_caches = getattr(model_runner, "_merge_kv_caches", None)
        extract_kv_cache = getattr(model_runner, "_extract_kv_cache", None)
        if (
            sampling_batch_cls is None
            or sampling_result_cls is None
            or not callable(sample_from_logits)
            or not callable(merge_kv_caches)
            or not callable(extract_kv_cache)
        ):
            return None

        last_tokens = [
            state.token_ids[-1] if state.token_ids else 0
            for _req_id, state in decode_reqs
        ]
        batch_cache = _nonpaged_batch_cache_for_decode(
            runner,
            decode_reqs,
            model_runner,
            merge_kv_caches,
        )
        input_ids = mx.array(last_tokens, dtype=mx.int32)[:, None]
        model_output = runner._forward_model(input_ids, cache=batch_cache)
        logits = runner._extract_logits(model_output)
        next_token_logits = logits[:, -1, :]
        if _nonpaged_async_eval_target() == NONPAGED_ASYNC_EVAL_TARGET_LOGITS:
            _submit_nonpaged_async_eval(
                mx,
                "nonpaged_decode_logits_async_submit",
                next_token_logits,
            )

        sampling_params_list = [state.sampling_params for _req_id, state in decode_reqs]
        prompt_token_ids_list = [
            state.token_ids[: state.prompt_len] for _req_id, state in decode_reqs
        ]
        output_tokens_list = [
            state.token_ids[state.prompt_len :] for _req_id, state in decode_reqs
        ]
        generators = {
            index: state.generator
            for index, (_req_id, state) in enumerate(decode_reqs)
            if getattr(state, "generator", None) is not None
        }
        batch = sampling_batch_cls(
            sampling_params_list,
            prompt_token_ids_list,
            output_tokens_list,
            vocab_size=runner._vocab_size,
            device=runner.device,
            logitsprocs=runner._logitsprocs,
            generators=generators,
        )
        result = sample_from_logits(
            next_token_logits,
            batch,
            runner._sampler,
            runner.device,
        )
        next_tokens = result.token_ids
        if len(next_tokens) != len(decode_reqs):
            return None

        persistent_cache_enabled = _nonpaged_persistent_batch_cache_enabled()
        for index, (_req_id, state) in enumerate(decode_reqs):
            if not persistent_cache_enabled:
                state.cache = extract_kv_cache(batch_cache, index)
            state.token_ids.append(next_tokens[index])
            state.generated_tokens += 1
        return sampling_result_cls(next_tokens, result.logprobs)
    except Exception as exc:
        _log_native_sampling_rejection_once(
            f"async nonpaged batched decode exception: {type(exc).__name__}: {exc}"
        )
        return None


def _nonpaged_batch_cache_for_decode(
    runner: Any,
    decode_reqs: list[tuple[str, Any]],
    model_runner: Any,
    merge_kv_caches: Any,
) -> Any:
    if not _nonpaged_persistent_batch_cache_enabled():
        return merge_kv_caches([state.cache for _req_id, state in decode_reqs])

    req_ids = tuple(req_id for req_id, _state in decode_reqs)
    states = tuple(state for _req_id, state in decode_reqs)
    cached = getattr(runner, "_audex_nonpaged_batch_cache", None)
    if (
        isinstance(cached, dict)
        and cached.get("req_ids") == req_ids
        and _same_state_objects(cached.get("states"), states)
    ):
        global NONPAGED_PERSISTENT_BATCH_CACHE_HITS
        NONPAGED_PERSISTENT_BATCH_CACHE_HITS += 1
        return cached["batch_cache"]

    _flush_persistent_nonpaged_batch_cache(
        runner,
        model_runner,
        reason="batch_membership_changed",
    )
    global NONPAGED_PERSISTENT_BATCH_CACHE_MISSES
    NONPAGED_PERSISTENT_BATCH_CACHE_MISSES += 1
    batch_cache = merge_kv_caches([state.cache for _req_id, state in decode_reqs])
    runner._audex_nonpaged_batch_cache = {
        "req_ids": req_ids,
        "states": states,
        "batch_cache": batch_cache,
    }
    return batch_cache


def _same_state_objects(left: Any, right: tuple[Any, ...]) -> bool:
    if not isinstance(left, tuple) or len(left) != len(right):
        return False
    return all(
        left_state is right_state
        for left_state, right_state in zip(left, right, strict=True)
    )


def _flush_persistent_nonpaged_batch_cache(
    runner: Any,
    model_runner: Any,
    *,
    reason: str,
) -> None:
    cached = getattr(runner, "_audex_nonpaged_batch_cache", None)
    if not isinstance(cached, dict):
        return
    if hasattr(runner, "_audex_nonpaged_batch_cache"):
        delattr(runner, "_audex_nonpaged_batch_cache")
    extract_kv_cache = getattr(model_runner, "_extract_kv_cache", None)
    if not callable(extract_kv_cache):
        return

    states = tuple(cached.get("states") or ())
    batch_cache = cached.get("batch_cache")
    if not states or batch_cache is None:
        return

    started = time.perf_counter() if _native_sampling_debug_enabled() else 0.0
    for index, state in enumerate(states):
        state.cache = extract_kv_cache(batch_cache, index)
    if started:
        _record_native_detail_timing(
            f"nonpaged_persistent_cache_flush_{reason}",
            time.perf_counter() - started,
        )
    global NONPAGED_PERSISTENT_BATCH_CACHE_FLUSHES
    NONPAGED_PERSISTENT_BATCH_CACHE_FLUSHES += 1


def _forward_hidden_states_for_tts_window(
    runner: Any,
    input_ids: Any,
    state: Any,
) -> Any | None:
    return _forward_hidden_states_for_tts_window_cache(runner, input_ids, state.cache)


def _forward_hidden_states_for_tts_window_cache(
    runner: Any,
    input_ids: Any,
    cache: Any,
) -> Any | None:
    model = getattr(runner, "_forward_model", None)
    backbone = getattr(model, "model", None)
    if backbone is None or not callable(backbone):
        return None
    return backbone(input_ids, cache=cache)


def _project_tts_window_logits(
    runner: Any,
    hidden_states: Any,
    allowed_window: tuple[int, int, int],
    mx: Any,
) -> Any | None:
    model = getattr(runner, "_forward_model", None)
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return None
    projection_weight, window_bias = _cached_tts_window_head(
        runner,
        lm_head,
        allowed_window,
        mx,
    )
    logits = hidden_states.astype(mx.float32) @ projection_weight
    if window_bias is not None:
        logits = logits + window_bias
    softcap = getattr(model, "final_logit_softcapping", None)
    if softcap is not None:
        logits = mx.tanh(logits / softcap) * softcap
    return logits


def _cached_tts_window_head(
    runner: Any,
    lm_head: Any,
    allowed_window: tuple[int, int, int],
    mx: Any,
) -> tuple[Any, Any | None]:
    global TTS_WINDOW_WEIGHT_CACHE_HITS, TTS_WINDOW_WEIGHT_CACHE_MISSES
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    cache_key = (id(weight), id(bias), allowed_window)
    cached = getattr(runner, "_audex_mac_tts_window_head_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        TTS_WINDOW_WEIGHT_CACHE_HITS += 1
        return cached["weight"], cached["bias"]

    codec_min_id, codec_max_id, speechgen_end_id = allowed_window
    end_weight = weight[speechgen_end_id : speechgen_end_id + 1]
    codec_weight = weight[codec_min_id : codec_max_id + 1]
    window_weight = mx.concatenate((end_weight, codec_weight), axis=0)
    projection_weight = mx.transpose(window_weight).astype(mx.float32)
    window_bias = None
    if bias is not None:
        end_bias = bias[speechgen_end_id : speechgen_end_id + 1]
        codec_bias = bias[codec_min_id : codec_max_id + 1]
        window_bias = mx.concatenate((end_bias, codec_bias), axis=0).astype(mx.float32)
    runner._audex_mac_tts_window_head_cache = {
        "key": cache_key,
        "weight": projection_weight,
        "bias": window_bias,
    }
    TTS_WINDOW_WEIGHT_CACHE_MISSES += 1
    return projection_weight, window_bias


def _sample_tts_window_logits_if_supported(
    logits: Any,
    batch: Any,
    sampling_result_cls: Any,
    mx: Any,
) -> Any | None:
    if not getattr(batch, "no_top_p", False):
        return None
    if not getattr(batch, "no_top_k", False) and not _has_uniform_positive_top_k(
        batch.sampling_params_list
    ):
        return None
    if _has_disallowed_sampling_constraints(batch):
        return None
    if _unsupported_native_logits_processors(batch):
        return None
    if not getattr(batch, "all_random", False) and not getattr(
        batch,
        "all_greedy",
        False,
    ):
        return None

    plan = _build_cfg_pair_sampling_plan(batch.sampling_params_list)
    if any(len(output_slots) != 1 for output_slots in plan.output_slots):
        return None
    allowed_window = _uniform_allowed_tts_token_window(plan.allowed_token_windows)
    if allowed_window is None:
        return None
    if getattr(batch, "all_greedy", False):
        sampled = mx.argmax(logits, axis=-1)
        tokens = _map_allowed_window_token_ids(sampled, allowed_window, mx)
    else:
        tokens = _sample_random_tokens_mlx(
            logits,
            plan,
            mx,
            logits_are_allowed_window=True,
        )
    mx.eval(tokens)
    token_ids = tokens.tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    return sampling_result_cls([int(token_id) for token_id in token_ids])


def _sample_tts_window_logits_from_params_if_supported(
    logits: Any,
    sampling_params_list: Sequence[Any],
    logitsprocs: Any,
    sampling_result_cls: Any,
    mx: Any,
    *,
    random_keys: Sequence[Any] | None = None,
) -> Any | None:
    if not _no_logprobs_requested(sampling_params_list):
        return None
    if not _no_penalties_in_sampling_params(sampling_params_list):
        return None
    if not _no_top_p_in_sampling_params(sampling_params_list):
        return None
    if not _no_top_k_in_sampling_params(sampling_params_list) and not (
        _has_uniform_positive_top_k(sampling_params_list)
    ):
        return None
    if _has_disallowed_sampling_constraints_for_params(sampling_params_list):
        return None
    if _unsupported_native_logits_processors_for_params(
        logitsprocs,
        sampling_params_list,
    ):
        return None

    all_greedy = _all_greedy_sampling_params(sampling_params_list)
    if not all_greedy and not _all_random_sampling_params(sampling_params_list):
        return None

    plan = _build_cfg_pair_sampling_plan(sampling_params_list)
    allowed_window = _uniform_allowed_tts_token_window(plan.allowed_token_windows)
    if allowed_window is None:
        return None
    sample_logits = _build_native_sample_logits(
        logits,
        sampling_params_list,
        plan,
        mx,
        allowed_window=None,
    )
    if all_greedy:
        sampled = mx.argmax(sample_logits, axis=-1)
        tokens = _map_allowed_window_token_ids(sampled, allowed_window, mx)
    else:
        tokens = _sample_random_tokens_mlx(
            sample_logits,
            plan,
            mx,
            logits_are_allowed_window=True,
            random_keys=random_keys,
        )
    mx.eval(tokens)
    sampled_token_ids = tokens.tolist()
    token_ids = _expand_sampled_token_ids(sampled_token_ids, plan.output_slots)
    return sampling_result_cls([int(token_id) for token_id in token_ids])


def _is_cfg_sampling_params(sampling_params: Any) -> bool:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return False
    return bool(extra_args.get("cfg_role") or extra_args.get("cfg_pair_id"))


def _cfg_tts_window_decode_enabled() -> bool:
    value = os.environ.get(CFG_TTS_WINDOW_DECODE_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _debug_sync_tts_window_stages_enabled() -> bool:
    value = os.environ.get(DEBUG_SYNC_TTS_WINDOW_STAGES_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _nonpaged_async_eval_enabled() -> bool:
    value = os.environ.get(NONPAGED_ASYNC_EVAL_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _nonpaged_async_eval_target() -> str:
    if not _nonpaged_async_eval_enabled():
        return NONPAGED_ASYNC_EVAL_TARGET_NONE
    value = os.environ.get(NONPAGED_ASYNC_EVAL_TARGET_ENV)
    if value is None:
        return NONPAGED_ASYNC_EVAL_TARGET_LOGITS
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "0", "false", "no", "none", "off", "disabled"}:
        return NONPAGED_ASYNC_EVAL_TARGET_NONE
    if normalized in {"sample", "sample_logits", "sample_window"}:
        return NONPAGED_ASYNC_EVAL_TARGET_SAMPLE_LOGITS
    return NONPAGED_ASYNC_EVAL_TARGET_LOGITS


def _nonpaged_persistent_batch_cache_enabled() -> bool:
    value = os.environ.get(NONPAGED_PERSISTENT_BATCH_CACHE_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _patch_mx_eval_timing(model_runner: Any) -> None:
    mx_module = getattr(model_runner, "mx", None)
    if mx_module is None or not hasattr(mx_module, "eval"):
        return
    current = mx_module.eval
    if getattr(current, TIMING_PATCH_SENTINEL, False):
        return

    def eval_with_paged_timing(*args: Any, **kwargs: Any) -> Any:
        if not PAGED_SAMPLE_DEPTH:
            return current(*args, **kwargs)
        category = _classify_mx_eval_args(args)
        if _should_skip_paged_logits_eval(category, args):
            _record_paged_logits_eval_skip()
            return None
        if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
            return current(*args, **kwargs)
        started = time.perf_counter()
        try:
            return current(*args, **kwargs)
        finally:
            _record_mx_eval_timing(category, time.perf_counter() - started, args)

    setattr(eval_with_paged_timing, TIMING_PATCH_SENTINEL, True)
    eval_with_paged_timing.__wrapped__ = current  # type: ignore[attr-defined]
    mx_module.eval = eval_with_paged_timing


def _patch_mps_allocator_cleanup_assert() -> bool:
    """Suppress PyTorch's MPS allocator assert during intentional vLLM shutdown."""

    patched_any = False
    try:
        import vllm.distributed.parallel_state as parallel_state

        patched_any = _patch_cleanup_symbol(parallel_state) or patched_any
    except Exception:
        pass

    try:
        import vllm.v1.engine.core as engine_core

        patched_any = _patch_cleanup_symbol(engine_core) or patched_any
    except Exception:
        pass

    return patched_any


def _patch_cleanup_symbol(module: Any) -> bool:
    current = getattr(module, "cleanup_dist_env_and_memory", None)
    if current is None:
        return False
    if getattr(current, MPS_CLEANUP_PATCH_SENTINEL, False):
        return True

    def cleanup_dist_env_and_memory_with_mps_guard(*args: Any, **kwargs: Any) -> Any:
        try:
            return current(*args, **kwargs)
        except RuntimeError as exc:
            if _is_mps_allocator_cleanup_assert(exc):
                return None
            raise

    setattr(
        cleanup_dist_env_and_memory_with_mps_guard,
        MPS_CLEANUP_PATCH_SENTINEL,
        True,
    )
    cleanup_dist_env_and_memory_with_mps_guard.__wrapped__ = current  # type: ignore[attr-defined]
    module.cleanup_dist_env_and_memory = cleanup_dist_env_and_memory_with_mps_guard
    return True


def _is_mps_allocator_cleanup_assert(exc: RuntimeError) -> bool:
    text = str(exc)
    return (
        "Allocator for mps is not a DeviceAllocator" in text
        or "device_allocator INTERNAL ASSERT FAILED" in text
    )


def _record_paged_sample_timing(
    runner: Any,
    elapsed: float,
    *,
    decode_count: int,
    prefill_count: int,
    token_count: int,
) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    count = int(getattr(runner, "_audex_mac_paged_sample_count", 0)) + 1
    total = float(getattr(runner, "_audex_mac_paged_sample_seconds", 0.0)) + elapsed
    runner._audex_mac_paged_sample_count = count
    runner._audex_mac_paged_sample_seconds = total
    if count not in {1, 10, 50, 100, 200, 500, 1000}:
        return
    avg_ms = total * 1000.0 / count
    native_ms = NATIVE_SAMPLE_SECONDS * 1000.0
    eval_summary = _mx_eval_timing_summary()
    eval_shape_summary = _mx_eval_shape_summary()
    native_detail_summary = _native_detail_timing_summary()
    print(
        "Audex vLLM Metal: paged sample timing "
        f"count={count} avg_ms={avg_ms:.1f} last_ms={elapsed * 1000.0:.1f} "
        f"decode_reqs={decode_count} prefill_reqs={prefill_count} "
        f"decode_tokens={token_count} native_sample_ms={native_ms:.1f} "
        f"native_sampled_rows={NATIVE_SAMPLED_ROWS} "
        f"native_output_rows={NATIVE_OUTPUT_ROWS} "
        f"skipped_logits_eval={PAGED_LOGITS_EVAL_SKIP_COUNT} "
        f"native_detail_ms={native_detail_summary} "
        f"mx_eval_ms={eval_summary} "
        f"mx_eval_shapes={eval_shape_summary}",
        file=sys.stderr,
        flush=True,
    )


def _record_non_paged_decode_timing(
    runner: Any,
    elapsed: float,
    *,
    decode_count: int,
    cached_count: int,
    batched: bool,
    cfg_cond_count: int,
    cfg_uncond_count: int,
    cfg_complete_pair_count: int,
) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    count = int(getattr(runner, "_audex_mac_non_paged_decode_count", 0)) + 1
    total = float(getattr(runner, "_audex_mac_non_paged_decode_seconds", 0.0)) + elapsed
    runner._audex_mac_non_paged_decode_count = count
    runner._audex_mac_non_paged_decode_seconds = total
    if count not in {1, 10, 50, 100, 200, 500, 1000}:
        return
    avg_ms = total * 1000.0 / count
    native_ms = NATIVE_SAMPLE_SECONDS * 1000.0
    native_detail_summary = _native_detail_timing_summary()
    print(
        "Audex vLLM Metal: nonpaged decode timing "
        f"count={count} avg_ms={avg_ms:.1f} last_ms={elapsed * 1000.0:.1f} "
        f"decode_reqs={decode_count} cached_reqs={cached_count} "
        f"batched={int(batched)} native_sample_ms={native_ms:.1f} "
        f"cfg_cond_reqs={cfg_cond_count} "
        f"cfg_uncond_reqs={cfg_uncond_count} "
        f"cfg_complete_pairs={cfg_complete_pair_count} "
        f"native_sampled_rows={NATIVE_SAMPLED_ROWS} "
        f"native_output_rows={NATIVE_OUTPUT_ROWS} "
        f"tts_window_decode_count={TTS_WINDOW_DECODE_COUNT} "
        f"tts_window_weight_cache_hits={TTS_WINDOW_WEIGHT_CACHE_HITS} "
        f"tts_window_weight_cache_misses={TTS_WINDOW_WEIGHT_CACHE_MISSES} "
        f"nonpaged_persistent_cache_hits={NONPAGED_PERSISTENT_BATCH_CACHE_HITS} "
        f"nonpaged_persistent_cache_misses={NONPAGED_PERSISTENT_BATCH_CACHE_MISSES} "
        f"nonpaged_persistent_cache_flushes={NONPAGED_PERSISTENT_BATCH_CACHE_FLUSHES} "
        f"native_detail_ms={native_detail_summary}",
        file=sys.stderr,
        flush=True,
    )


def _sample_native_mlx_if_supported(
    logits_2d: Any,
    batch: Any,
    sampling_batch: Any,
) -> Any | None:
    """Use MLX sampling for simple Audex batches instead of the Torch bridge."""

    if logits_2d is None:
        _log_native_sampling_rejection_once("missing logits")
        return None
    if getattr(batch, "needs_logprobs", False):
        _log_native_sampling_rejection_once("logprobs requested")
        return None
    if getattr(batch, "generators", None):
        _log_native_sampling_rejection_once("custom generators present")
        return None
    if not getattr(batch, "no_penalties", False):
        _log_native_sampling_rejection_once("penalties present")
        return None
    if not getattr(batch, "no_top_p", False):
        _log_native_sampling_rejection_once("top-p filtering requested")
        return None
    if not getattr(batch, "no_top_k", False) and not _has_uniform_positive_top_k(
        batch.sampling_params_list
    ):
        _log_native_sampling_rejection_once(
            "mixed or unsupported top-k filtering requested"
        )
        return None
    if _has_disallowed_sampling_constraints(batch):
        _log_native_sampling_rejection_once("token constraints present")
        return None
    unsupported_processors = _unsupported_native_logits_processors(batch)
    if unsupported_processors:
        _log_native_sampling_rejection_once(
            "unsupported logits processor present: " + ", ".join(unsupported_processors)
        )
        return None
    if not (getattr(batch, "all_greedy", False) or getattr(batch, "all_random", False)):
        _log_native_sampling_rejection_once("mixed greedy/random batch")
        return None

    try:
        import mlx.core as mx

        global NATIVE_SAMPLE_COUNT
        debug_timing = _native_sampling_debug_enabled()
        plan = _build_cfg_pair_sampling_plan(batch.sampling_params_list)
        if _materialize_decode_logits_before_native_sampling_enabled():
            materialize_started = time.perf_counter() if debug_timing else 0.0
            mx.eval(logits_2d)
            if debug_timing:
                _record_native_detail_timing(
                    "materialize_decode_logits",
                    time.perf_counter() - materialize_started,
                )
        build_started = time.perf_counter() if debug_timing else 0.0
        allowed_window = _uniform_allowed_tts_token_window(plan.allowed_token_windows)
        sample_logits = _build_native_sample_logits(
            logits_2d,
            batch.sampling_params_list,
            plan,
            mx,
            allowed_window=allowed_window,
        )
        if debug_timing:
            _record_native_detail_timing(
                "build_sample_logits", time.perf_counter() - build_started
            )
        if _nonpaged_async_eval_target() == NONPAGED_ASYNC_EVAL_TARGET_SAMPLE_LOGITS:
            _submit_nonpaged_async_eval(
                mx,
                "native_sample_logits_async_submit",
                sample_logits,
            )
        sample_started = time.perf_counter() if debug_timing else 0.0
        if getattr(batch, "all_greedy", False):
            tokens = mx.argmax(sample_logits, axis=-1)
        else:
            tokens = _sample_random_tokens_mlx(
                sample_logits,
                plan,
                mx,
                logits_are_allowed_window=allowed_window is not None,
            )
        mx.eval(tokens)
        if debug_timing:
            _record_native_detail_timing(
                "sample_eval", time.perf_counter() - sample_started
            )
        tolist_started = time.perf_counter() if debug_timing else 0.0
        sampled_token_ids = tokens.tolist()
        if debug_timing:
            _record_native_detail_timing("tolist", time.perf_counter() - tolist_started)
        token_ids = _expand_sampled_token_ids(sampled_token_ids, plan.output_slots)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        token_ids = [int(token_id) for token_id in token_ids]
        _record_native_sample_rows(
            sampled_rows=len(plan.sample_row_indices),
            output_rows=len(token_ids),
        )
        NATIVE_SAMPLE_COUNT += 1
        _log_native_sampling_debug_once(NATIVE_SAMPLE_COUNT)
        return sampling_batch._SamplingResult(token_ids)
    except Exception as exc:
        _log_native_sampling_rejection_once(
            f"native sampler exception: {type(exc).__name__}: {exc}"
        )
        return None


def _record_native_sample_seconds(elapsed: float) -> None:
    global NATIVE_SAMPLE_SECONDS
    NATIVE_SAMPLE_SECONDS += elapsed


def _scale_logits_by_temperature(
    logits: Any,
    temperatures: Sequence[float],
    mx: Any,
) -> Any:
    if not temperatures:
        return logits
    if all(float(temperature) == 1.0 for temperature in temperatures):
        return logits
    if len(set(float(temperature) for temperature in temperatures)) == 1:
        return logits * (1.0 / float(temperatures[0]))
    inverse_temperatures = mx.array(
        [1.0 / float(temperature) for temperature in temperatures],
        dtype=mx.float32,
    )
    return logits * inverse_temperatures[:, None]


def _sample_random_tokens_mlx(
    logits: Any,
    plan: NativeSamplingPlan,
    mx: Any,
    *,
    logits_are_allowed_window: bool = False,
    random_keys: Sequence[Any] | None = None,
) -> Any:
    scaled_logits = _scale_logits_by_temperature(logits, plan.temperatures, mx)
    allowed_window = _uniform_allowed_tts_token_window(plan.allowed_token_windows)
    if allowed_window is not None and not logits_are_allowed_window:
        scaled_logits = _restrict_to_allowed_tts_tokens(
            scaled_logits,
            allowed_window,
            mx,
        )

    top_k = _uniform_positive_top_k(plan.top_ks)
    if top_k is None:
        sampled = _categorical_with_optional_keys(scaled_logits, random_keys, mx)
        return _map_allowed_window_token_ids(sampled, allowed_window, mx)

    vocab_size = int(scaled_logits.shape[-1])
    if top_k >= vocab_size:
        sampled = _categorical_with_optional_keys(scaled_logits, random_keys, mx)
        return _map_allowed_window_token_ids(sampled, allowed_window, mx)

    partitioned_indices = mx.argpartition(scaled_logits, vocab_size - top_k, axis=-1)
    top_indices = partitioned_indices[:, -top_k:]
    top_logits = mx.take_along_axis(scaled_logits, top_indices, axis=-1)
    local_tokens = _categorical_with_optional_keys(top_logits, random_keys, mx)
    sampled = mx.take_along_axis(top_indices, local_tokens[:, None], axis=-1)[:, 0]
    return _map_allowed_window_token_ids(sampled, allowed_window, mx)


def _categorical_with_optional_keys(
    logits: Any,
    random_keys: Sequence[Any] | None,
    mx: Any,
) -> Any:
    if random_keys is None:
        return mx.random.categorical(logits)
    if len(random_keys) != int(logits.shape[0]):
        raise ValueError("seeded MLX sampling key count does not match logits rows")
    return mx.stack(
        [
            mx.random.categorical(logits[index], key=key)
            for index, key in enumerate(random_keys)
        ]
    )


def _seeded_random_keys_for_states(
    sampling_params_list: Sequence[Any],
    states: Sequence[Any],
    mx: Any,
) -> tuple[Any, ...] | None:
    """Derive common-random-number keys without leaving compact MLX sampling."""

    plan = _build_cfg_pair_sampling_plan(sampling_params_list)
    keys: list[Any] = []
    for row_index in plan.sample_row_indices:
        sampling_params = sampling_params_list[row_index]
        state = states[row_index]
        seed = getattr(sampling_params, "seed", None)
        if not isinstance(seed, int):
            return None
        step = int(getattr(state, "generated_tokens", 0))
        token_seed = (seed ^ ((step + 1) * 0x9E3779B9)) & 0xFFFFFFFF
        keys.append(mx.random.key(token_seed))
    return tuple(keys)


def _negative_mask_value(logits: Any) -> Any:
    dtype = getattr(logits, "dtype", None)
    if dtype is None:
        return -1.0e30
    return -1.0e30


def _apply_disallowed_token_mask_mlx(
    logits_2d: Any,
    sampling_params_list: Sequence[Any],
) -> Any:
    if logits_2d is None:
        return logits_2d
    vocab_size = _logit_vocab_size(logits_2d)
    if vocab_size <= 0:
        return logits_2d

    row_ranges = [
        _disallowed_token_ranges_from_sampling_params(sampling_params, vocab_size)
        for sampling_params in sampling_params_list
    ]
    if not any(row_ranges):
        return logits_2d

    import mlx.core as mx

    rows = []
    for row_index in range(int(logits_2d.shape[0])):
        row = logits_2d[row_index]
        ranges = row_ranges[row_index] if row_index < len(row_ranges) else ()
        rows.append(_mask_logit_row_mlx(row, ranges, mx))
    return mx.stack(rows)


def _logit_vocab_size(logits_2d: Any) -> int:
    shape = getattr(logits_2d, "shape", None)
    if not shape:
        return 0
    try:
        return int(shape[-1])
    except (TypeError, ValueError):
        return 0


def _mask_logit_row_mlx(
    row: Any,
    ranges: Sequence[tuple[int, int]],
    mx: Any,
) -> Any:
    if not ranges:
        return row
    vocab_size = _logit_vocab_size(row)
    if vocab_size <= 0:
        vocab_size = int(getattr(row, "shape", (0,))[-1])
    parts: list[Any] = []
    cursor = 0
    dtype = getattr(row, "dtype", getattr(mx, "float32", None))
    for start, end in ranges:
        if cursor < start:
            parts.append(row[cursor:start])
        mask_len = end - start + 1
        parts.append(mx.full((mask_len,), _negative_mask_value(row), dtype=dtype))
        cursor = end + 1
    if cursor < vocab_size:
        parts.append(row[cursor:vocab_size])
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=0)


def _disallowed_token_ranges_from_sampling_params(
    sampling_params: Any,
    vocab_size: int,
) -> tuple[tuple[int, int], ...]:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return ()

    ranges: list[tuple[int, int]] = []
    for raw_range in extra_args.get("audex_disallow_token_ranges", ()) or ():
        try:
            start, end = raw_range
            ranges.append((int(start), int(end)))
        except (TypeError, ValueError):
            continue
    for raw_token_id in extra_args.get("audex_disallow_token_ids", ()) or ():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError):
            continue
        ranges.append((token_id, token_id))
    return _sanitize_token_ranges(ranges, vocab_size)


def _sanitize_token_ranges(
    ranges: Sequence[tuple[int, int]],
    vocab_size: int,
) -> tuple[tuple[int, int], ...]:
    if vocab_size <= 0:
        return ()
    clamped: list[tuple[int, int]] = []
    for start, end in ranges:
        start = max(0, int(start))
        end = min(vocab_size - 1, int(end))
        if end < start:
            continue
        clamped.append((start, end))
    if not clamped:
        return ()

    clamped.sort()
    merged: list[tuple[int, int]] = []
    current_start, current_end = clamped[0]
    for start, end in clamped[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return tuple(merged)


def _restrict_to_allowed_tts_tokens(
    logits: Any,
    allowed_window: tuple[int, int, int],
    mx: Any,
) -> Any:
    codec_min_id, codec_max_id, speechgen_end_id = allowed_window
    end_logits = logits[:, speechgen_end_id : speechgen_end_id + 1]
    codec_logits = logits[:, codec_min_id : codec_max_id + 1]
    return mx.concatenate((end_logits, codec_logits), axis=-1)


def _restrict_row_to_allowed_tts_tokens(
    row: Any,
    allowed_window: tuple[int, int, int],
    mx: Any,
) -> Any:
    codec_min_id, codec_max_id, speechgen_end_id = allowed_window
    end_logits = row[speechgen_end_id : speechgen_end_id + 1]
    codec_logits = row[codec_min_id : codec_max_id + 1]
    return mx.concatenate((end_logits, codec_logits), axis=-1)


def _map_allowed_window_token_ids(
    sampled: Any,
    allowed_window: tuple[int, int, int] | None,
    mx: Any,
) -> Any:
    if allowed_window is None:
        return sampled
    codec_min_id, _codec_max_id, speechgen_end_id = allowed_window
    codec_ids = sampled + int(codec_min_id - 1)
    end_ids = mx.full(sampled.shape, speechgen_end_id, dtype=mx.int32)
    return mx.where(sampled == 0, end_ids, codec_ids)


def _uniform_positive_top_k(top_ks: Sequence[int]) -> int | None:
    positive = {int(top_k) for top_k in top_ks if int(top_k) > 0}
    if len(positive) != 1:
        return None
    return positive.pop()


def _allowed_tts_token_window(sampling_params: Any) -> tuple[int, int, int] | None:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    try:
        codec_min_id = int(extra_args["audex_tts_codec_min_id"])
        codec_max_id = int(extra_args["audex_tts_codec_max_id"])
        speechgen_end_id = int(extra_args["audex_tts_speechgen_end_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if codec_min_id < 0 or codec_max_id < codec_min_id or speechgen_end_id < 0:
        return None
    return (codec_min_id, codec_max_id, speechgen_end_id)


def _requires_compact_tts_window_decode(sampling_params: Any) -> bool:
    extra_args = getattr(sampling_params, "extra_args", None)
    return bool(
        extra_args and extra_args.get("audex_tts_require_compact_window_decode")
    )


def _uniform_allowed_tts_token_window(
    windows: Sequence[tuple[int, int, int] | None],
) -> tuple[int, int, int] | None:
    concrete = {window for window in windows if window is not None}
    if len(concrete) != 1:
        return None
    return concrete.pop()


def _has_uniform_positive_top_k(sampling_params_list: Sequence[Any]) -> bool:
    top_ks = [int(getattr(params, "top_k", 0) or 0) for params in sampling_params_list]
    return _uniform_positive_top_k(top_ks) is not None


def _record_native_sample_rows(*, sampled_rows: int, output_rows: int) -> None:
    global NATIVE_OUTPUT_ROWS, NATIVE_SAMPLED_ROWS
    NATIVE_SAMPLED_ROWS += sampled_rows
    NATIVE_OUTPUT_ROWS += output_rows


def _record_mx_eval_timing(
    category: str,
    elapsed: float,
    args: tuple[Any, ...],
) -> None:
    MX_EVAL_SECONDS_BY_CATEGORY[category] = (
        MX_EVAL_SECONDS_BY_CATEGORY.get(category, 0.0) + elapsed
    )
    MX_EVAL_COUNT_BY_CATEGORY[category] = MX_EVAL_COUNT_BY_CATEGORY.get(category, 0) + 1
    category_shapes = MX_EVAL_SHAPE_COUNT_BY_CATEGORY.setdefault(category, {})
    for shape in _shape_tuples(args):
        category_shapes[shape] = category_shapes.get(shape, 0) + 1


def _record_paged_logits_eval_skip() -> None:
    global PAGED_LOGITS_EVAL_SKIP_COUNT, SKIP_NEXT_PAGED_LOGITS_EVAL
    PAGED_LOGITS_EVAL_SKIP_COUNT += 1
    SKIP_NEXT_PAGED_LOGITS_EVAL = False


def _submit_nonpaged_async_eval(mx: Any, category: str, *arrays: Any) -> None:
    async_eval = getattr(mx, "async_eval", None)
    if not callable(async_eval):
        return
    started = time.perf_counter() if _native_sampling_debug_enabled() else 0.0
    async_eval(*arrays)
    if started:
        _record_native_detail_timing(category, time.perf_counter() - started)


def _record_native_detail_timing(category: str, elapsed: float) -> None:
    if not _native_sampling_debug_enabled():
        return
    NATIVE_DETAIL_SECONDS_BY_CATEGORY[category] = (
        NATIVE_DETAIL_SECONDS_BY_CATEGORY.get(category, 0.0) + elapsed
    )
    NATIVE_DETAIL_COUNT_BY_CATEGORY[category] = (
        NATIVE_DETAIL_COUNT_BY_CATEGORY.get(category, 0) + 1
    )


def _mx_eval_timing_summary() -> str:
    if not MX_EVAL_SECONDS_BY_CATEGORY:
        return "none"
    parts = []
    for category in sorted(MX_EVAL_SECONDS_BY_CATEGORY):
        seconds = MX_EVAL_SECONDS_BY_CATEGORY[category]
        count = MX_EVAL_COUNT_BY_CATEGORY.get(category, 0)
        parts.append(f"{category}:{seconds * 1000.0:.1f}/{count}")
    return ",".join(parts)


def _mx_eval_shape_summary() -> str:
    if not MX_EVAL_SHAPE_COUNT_BY_CATEGORY:
        return "none"
    parts = []
    for category in sorted(MX_EVAL_SHAPE_COUNT_BY_CATEGORY):
        shapes = MX_EVAL_SHAPE_COUNT_BY_CATEGORY[category]
        shape_parts = []
        for shape, count in sorted(shapes.items(), key=lambda item: item[0]):
            shape_text = "scalar" if not shape else "x".join(str(dim) for dim in shape)
            shape_parts.append(f"{shape_text}x{count}")
        parts.append(f"{category}:{'+'.join(shape_parts)}")
    return ",".join(parts)


def _native_detail_timing_summary() -> str:
    if not NATIVE_DETAIL_SECONDS_BY_CATEGORY:
        return "none"
    parts = []
    for category in sorted(NATIVE_DETAIL_SECONDS_BY_CATEGORY):
        seconds = NATIVE_DETAIL_SECONDS_BY_CATEGORY[category]
        count = NATIVE_DETAIL_COUNT_BY_CATEGORY.get(category, 0)
        parts.append(f"{category}:{seconds * 1000.0:.1f}/{count}")
    return ",".join(parts)


def _classify_mx_eval_args(args: tuple[Any, ...]) -> str:
    shapes = _shape_tuples(args)
    if any(shape and shape[-1] >= 100_000 for shape in shapes):
        return "logits"
    if shapes and all(_numel_from_shape(shape) <= 256 for shape in shapes if shape):
        return "sample_tokens"
    return "other"


def _should_skip_paged_logits_eval(category: str, args: tuple[Any, ...]) -> bool:
    if not SKIP_NEXT_PAGED_LOGITS_EVAL:
        return False
    if category != "logits" or len(args) != 1:
        return False
    shape = _shape_tuple(args[0])
    return len(shape) == 3 and shape[-1] >= 100_000


def _shape_tuples(args: tuple[Any, ...]) -> list[tuple[int, ...]]:
    return [_shape_tuple(arg) for arg in args]


def _skip_paged_logits_eval_enabled() -> bool:
    return os.environ.get("AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL") != "0"


def _materialize_decode_logits_before_native_sampling_enabled() -> bool:
    return os.environ.get("AUDEX_VLLM_MATERIALIZE_DECODE_LOGITS") == "1"


def _native_sampling_debug_enabled() -> bool:
    return bool(os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"))


def _can_skip_paged_logits_eval(state: Any, grammar_output: Any | None) -> bool:
    if grammar_output is not None or state is None:
        return False
    if getattr(state, "target_hidden_states", None) is not None:
        return False
    if getattr(state, "pooling_hidden_states", None) is not None:
        return False
    if getattr(state, "prefill_reqs", None):
        return False
    decode_reqs = tuple(getattr(state, "decode_reqs", ()) or ())
    if not decode_reqs:
        return False
    sampling_params_list = []
    for _req_id, req_state in decode_reqs:
        sampling_params = getattr(req_state, "sampling_params", None)
        if sampling_params is None:
            return False
        sampling_params_list.append(sampling_params)
    return (
        _has_cfg_sampling_pair(sampling_params_list)
        or _all_have_tts_token_window(sampling_params_list)
        or _all_have_tts_lazy_logits_hint(sampling_params_list)
    )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return ()


def _numel_from_shape(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    total = 1
    for dim in shape:
        total *= max(1, int(dim))
    return total


def _log_native_sampling_debug_once(sample_count: int) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    if sample_count not in {1, 50, 100, 200, 500, 1000}:
        return
    print(
        "Audex vLLM Metal: native MLX sampling fast path used "
        f"{sample_count} time(s)",
        file=sys.stderr,
        flush=True,
    )


def _log_native_sampling_rejection_once(reason: str) -> None:
    if not os.environ.get("AUDEX_VLLM_NATIVE_SAMPLING_DEBUG"):
        return
    if reason in NATIVE_REJECTION_REASONS_LOGGED:
        return
    NATIVE_REJECTION_REASONS_LOGGED.add(reason)
    print(
        "Audex vLLM Metal: native MLX sampling fast path skipped: " f"{reason}",
        file=sys.stderr,
        flush=True,
    )


def _has_disallowed_sampling_constraints(batch: Any) -> bool:
    return _has_disallowed_sampling_constraints_for_params(batch.sampling_params_list)


def _has_disallowed_sampling_constraints_for_params(
    sampling_params_list: Sequence[Any],
) -> bool:
    return any(
        getattr(sampling_params, "allowed_token_ids", None)
        or getattr(sampling_params, "bad_words_token_ids", None)
        for sampling_params in sampling_params_list
    )


def _no_logprobs_requested(sampling_params_list: Sequence[Any]) -> bool:
    return all(
        getattr(sampling_params, "logprobs", None) is None
        for sampling_params in sampling_params_list
    )


def _no_top_p_in_sampling_params(sampling_params_list: Sequence[Any]) -> bool:
    return all(
        _sampling_float_attr(sampling_params, "top_p", 1.0) == 1.0
        for sampling_params in sampling_params_list
    )


def _no_top_k_in_sampling_params(sampling_params_list: Sequence[Any]) -> bool:
    return all(
        int(getattr(sampling_params, "top_k", 0) or 0) <= 0
        for sampling_params in sampling_params_list
    )


def _no_penalties_in_sampling_params(sampling_params_list: Sequence[Any]) -> bool:
    return all(
        _sampling_float_attr(sampling_params, "frequency_penalty", 0.0) == 0.0
        and _sampling_float_attr(sampling_params, "presence_penalty", 0.0) == 0.0
        and _sampling_float_attr(sampling_params, "repetition_penalty", 1.0) == 1.0
        for sampling_params in sampling_params_list
    )


def _all_greedy_sampling_params(sampling_params_list: Sequence[Any]) -> bool:
    return all(
        _sampling_float_attr(sampling_params, "temperature", 1.0) < 1.0e-5
        for sampling_params in sampling_params_list
    )


def _all_random_sampling_params(sampling_params_list: Sequence[Any]) -> bool:
    return bool(sampling_params_list) and all(
        _sampling_float_attr(sampling_params, "temperature", 1.0) >= 1.0e-5
        for sampling_params in sampling_params_list
    )


def _sampling_float_attr(
    sampling_params: Any,
    name: str,
    default: float,
) -> float:
    value = getattr(sampling_params, name, default)
    if value is None:
        value = default
    return float(value)


def _has_cfg_sampling_pair(sampling_params_list: Sequence[Any]) -> bool:
    pairs: dict[str, set[str]] = {}
    for sampling_params in sampling_params_list:
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args:
            continue
        role = extra_args.get("cfg_role")
        pair_id = extra_args.get("cfg_pair_id")
        if role in {"cond", "uncond"} and pair_id:
            pairs.setdefault(str(pair_id), set()).add(str(role))
    return any({"cond", "uncond"}.issubset(roles) for roles in pairs.values())


def _all_have_tts_token_window(sampling_params_list: Sequence[Any]) -> bool:
    return bool(sampling_params_list) and all(
        _allowed_tts_token_window(sampling_params) is not None
        for sampling_params in sampling_params_list
    )


def _all_have_tts_lazy_logits_hint(sampling_params_list: Sequence[Any]) -> bool:
    if not sampling_params_list:
        return False
    for sampling_params in sampling_params_list:
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args or not extra_args.get("audex_tts_skip_paged_logits_eval"):
            return False
    return True


def _has_uniform_positive_top_k(sampling_params_list: Sequence[Any]) -> bool:
    top_ks = {
        int(getattr(sampling_params, "top_k", 0) or 0)
        for sampling_params in sampling_params_list
    }
    return len(top_ks) == 1 and next(iter(top_ks)) > 0


@dataclass(frozen=True, slots=True)
class NativeSamplingPlan:
    sample_row_indices: tuple[int, ...]
    output_slots: tuple[tuple[int, ...], ...]
    temperatures: tuple[float, ...]
    top_ks: tuple[int, ...]
    allowed_token_windows: tuple[tuple[int, int, int] | None, ...]


def _build_cfg_pair_sampling_plan(
    sampling_params_list: Sequence[Any],
) -> NativeSamplingPlan:
    pairs: dict[str, dict[str, int]] = {}
    for index, sampling_params in enumerate(sampling_params_list):
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args:
            continue
        role = extra_args.get("cfg_role")
        pair_id = extra_args.get("cfg_pair_id")
        if role in {"cond", "uncond"} and pair_id:
            pairs.setdefault(str(pair_id), {})[str(role)] = index

    paired_indices: set[int] = set()
    sample_row_indices: list[int] = []
    output_slots: list[tuple[int, ...]] = []
    temperatures: list[float] = []
    top_ks: list[int] = []
    allowed_token_windows: list[tuple[int, int, int] | None] = []
    for roles in pairs.values():
        cond_index = roles.get("cond")
        uncond_index = roles.get("uncond")
        if cond_index is None or uncond_index is None:
            continue
        paired_indices.update((cond_index, uncond_index))
        sample_row_indices.append(cond_index)
        output_slots.append((cond_index, uncond_index))
        temperatures.append(float(sampling_params_list[cond_index].temperature))
        top_ks.append(int(getattr(sampling_params_list[cond_index], "top_k", 0) or 0))
        allowed_token_windows.append(
            _allowed_tts_token_window(sampling_params_list[cond_index])
        )

    for index, sampling_params in enumerate(sampling_params_list):
        if index in paired_indices:
            continue
        sample_row_indices.append(index)
        output_slots.append((index,))
        temperatures.append(float(sampling_params.temperature))
        top_ks.append(int(getattr(sampling_params, "top_k", 0) or 0))
        allowed_token_windows.append(_allowed_tts_token_window(sampling_params))

    return NativeSamplingPlan(
        sample_row_indices=tuple(sample_row_indices),
        output_slots=tuple(output_slots),
        temperatures=tuple(temperatures),
        top_ks=tuple(top_ks),
        allowed_token_windows=tuple(allowed_token_windows),
    )


def _expand_sampled_token_ids(
    sampled_token_ids: int | Sequence[int],
    output_slots: Sequence[Sequence[int]],
) -> list[int]:
    if isinstance(sampled_token_ids, int):
        sampled = [sampled_token_ids]
    else:
        sampled = [int(token_id) for token_id in sampled_token_ids]
    output_len = max((max(slots) for slots in output_slots if slots), default=-1) + 1
    token_ids = [0] * output_len
    for sampled_token_id, slots in zip(sampled, output_slots, strict=False):
        for slot in slots:
            token_ids[int(slot)] = int(sampled_token_id)
    return token_ids


def _unsupported_native_logits_processors(batch: Any) -> tuple[str, ...]:
    logitsprocs = getattr(batch, "logitsprocs", None)
    return _unsupported_native_logits_processors_for_params(
        logitsprocs,
        batch.sampling_params_list,
    )


def _unsupported_native_logits_processors_for_params(
    logitsprocs: Any,
    sampling_params_list: Sequence[Any],
) -> tuple[str, ...]:
    processors = tuple(getattr(logitsprocs, "all", ()) or ())
    if not processors:
        return ()

    unsupported: list[str] = []
    for processor in processors:
        name = f"{type(processor).__module__}.{type(processor).__name__}"
        if name.endswith(".CFGLogitsProcessor"):
            continue
        if type(processor) is AudexMetalCFGTokenSyncInstaller:
            continue
        if _is_inert_builtin_logits_processor(name, sampling_params_list):
            continue
        unsupported.append(name)
    return tuple(unsupported)


def _is_inert_builtin_logits_processor(
    name: str,
    sampling_params_list: Sequence[Any],
) -> bool:
    if name.endswith(".MinPLogitsProcessor"):
        return all(
            float(getattr(sampling_params, "min_p", 0.0) or 0.0) <= 0.0
            for sampling_params in sampling_params_list
        )
    if name.endswith(".MinTokensLogitsProcessor"):
        return all(
            int(getattr(sampling_params, "min_tokens", 0) or 0) <= 0
            for sampling_params in sampling_params_list
        )
    if name.endswith(".LogitBiasLogitsProcessor"):
        return all(
            not getattr(sampling_params, "logit_bias", None)
            for sampling_params in sampling_params_list
        )
    return False


def _apply_cfg_blend_mlx(
    logits: Any,
    sampling_params_list: Sequence[Any],
    mx: Any,
) -> Any:
    pairs: dict[str, dict[str, tuple[int, float]]] = {}
    for index, sampling_params in enumerate(sampling_params_list):
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args:
            continue
        role = extra_args.get("cfg_role")
        pair_id = extra_args.get("cfg_pair_id")
        if role not in {"cond", "uncond"} or not pair_id:
            continue
        pairs.setdefault(str(pair_id), {})[str(role)] = (
            index,
            float(extra_args.get("cfg_scale", 1.0)),
        )
    if not pairs:
        return logits

    rows = [logits[index] for index in range(int(logits.shape[0]))]
    for roles in pairs.values():
        cond = roles.get("cond")
        uncond = roles.get("uncond")
        if cond is None or uncond is None:
            continue
        cond_index, cfg_scale = cond
        uncond_index, _ = uncond
        blended = rows[uncond_index] + cfg_scale * (
            rows[cond_index] - rows[uncond_index]
        )
        rows[cond_index] = blended
        rows[uncond_index] = blended
    return mx.stack(rows)


def _build_native_sample_logits(
    logits: Any,
    sampling_params_list: Sequence[Any],
    plan: NativeSamplingPlan,
    mx: Any,
    *,
    allowed_window: tuple[int, int, int] | None = None,
) -> Any:
    """Build only the rows the native sampler will draw from."""

    blend_specs = _cfg_conditional_blend_specs(sampling_params_list)
    rows: list[Any] = []
    for sample_index in plan.sample_row_indices:
        blend = blend_specs.get(sample_index)
        if blend is None:
            rows.append(_as_sample_logits_row(logits[sample_index], mx, allowed_window))
            continue
        uncond_index, cfg_scale = blend
        cond_row = _as_sample_logits_row(logits[sample_index], mx, allowed_window)
        uncond_row = _as_sample_logits_row(logits[uncond_index], mx, allowed_window)
        rows.append(uncond_row + cfg_scale * (cond_row - uncond_row))
    return mx.stack(rows)


def _cfg_conditional_blend_specs(
    sampling_params_list: Sequence[Any],
) -> dict[int, tuple[int, float]]:
    pairs: dict[str, dict[str, tuple[int, float]]] = {}
    for index, sampling_params in enumerate(sampling_params_list):
        extra_args = getattr(sampling_params, "extra_args", None)
        if not extra_args:
            continue
        role = extra_args.get("cfg_role")
        pair_id = extra_args.get("cfg_pair_id")
        if role not in {"cond", "uncond"} or not pair_id:
            continue
        pairs.setdefault(str(pair_id), {})[str(role)] = (
            index,
            float(extra_args.get("cfg_scale", 1.0)),
        )

    specs: dict[int, tuple[int, float]] = {}
    for roles in pairs.values():
        cond = roles.get("cond")
        uncond = roles.get("uncond")
        if cond is None or uncond is None:
            continue
        cond_index, cfg_scale = cond
        uncond_index, _ = uncond
        specs[cond_index] = (uncond_index, cfg_scale)
    return specs


def _as_float32_row(row: Any, mx: Any) -> Any:
    if hasattr(row, "astype"):
        return row.astype(mx.float32)
    return row


def _as_sample_logits_row(
    row: Any,
    mx: Any,
    allowed_window: tuple[int, int, int] | None,
) -> Any:
    if allowed_window is not None:
        row = _restrict_row_to_allowed_tts_tokens(row, allowed_window, mx)
    return _as_float32_row(row, mx)


def _patch_sample_prefill_tokens(sampling_batch: Any) -> None:
    current = sampling_batch.sample_prefill_tokens
    if getattr(current, PATCH_SENTINEL, False):
        return

    def sample_prefill_tokens_batched(
        logits: Any,
        prefill_reqs: list[Any],
        cu_seqlens: list[int],
        num_decode: int,
        sampler: Any,
        device: Any,
        *,
        vocab_size: int,
        logitsprocs: Any | None = None,
    ) -> Any:
        if not prefill_reqs:
            return sampling_batch._SamplingResult([])

        import mlx.core as mx

        last_logits_rows: list[Any] = []
        sampling_params_list: list[Any] = []
        prompt_token_ids_list: list[list[int]] = []
        output_token_ids_list: list[list[int]] = []
        generators: dict[int, Any] = {}

        for index, prefill in enumerate(prefill_reqs):
            last_idx = cu_seqlens[num_decode + index + 1] - 1
            last_logits_rows.append(logits[0, last_idx, :])

            if prefill.full_prompt_token_ids is not None:
                prompt_len = len(prefill.full_prompt_token_ids)
            elif prefill.prompt_len is not None:
                prompt_len = prefill.prompt_len
            else:
                prompt_len = len(prefill.token_ids)

            prompt_for_meta = (
                prefill.full_prompt_token_ids
                if prefill.full_prompt_token_ids is not None
                else prefill.token_ids
            )
            sampling_params_list.append(prefill.sampling_params)
            prompt_token_ids_list.append(prompt_for_meta[:prompt_len])
            output_token_ids_list.append(prompt_for_meta[prompt_len:])
            if prefill.generator is not None:
                generators[index] = prefill.generator

        batch = sampling_batch.SamplingBatch(
            sampling_params_list,
            prompt_token_ids_list,
            output_token_ids_list,
            vocab_size=vocab_size,
            device=device,
            logitsprocs=logitsprocs,
            generators=generators,
        )
        return sampling_batch.sample_from_logits(
            mx.stack(last_logits_rows),
            batch,
            sampler,
            device,
        )

    setattr(sample_prefill_tokens_batched, PATCH_SENTINEL, True)
    sample_prefill_tokens_batched.__wrapped__ = current  # type: ignore[attr-defined]
    sampling_batch.sample_prefill_tokens = sample_prefill_tokens_batched
