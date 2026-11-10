#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema, coerce_list_int
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector, resolve_batch_req_ids
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager, normalize_token_ids
from vllm_ascend.observability.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_REPEAT, Incident, RequestGuardStore

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


def _ids_already_ints(token_ids: Iterable[int]) -> list[int]:
    """Skip ``normalize_token_ids`` when Store already handed ``list[int]``."""
    if isinstance(token_ids, list):
        if not token_ids or isinstance(token_ids[0], int):
            return token_ids
    return normalize_token_ids(token_ids)


@dataclass
class TokenRepeatState:
    """Per-request sliding window for local token re-reading detection."""

    content: deque[int] = field(default_factory=deque)
    scores: deque[int] = field(default_factory=deque)
    freq: dict[int, int] = field(default_factory=dict)
    repeat_sum: int = 0
    content_tokens_seen: int = 0
    consecutive_hits: int = 0


def push_token_repeat(
    state: TokenRepeatState,
    tid: int,
    *,
    window: int,
    ignore: frozenset[int] | set[int],
) -> int:
    """Fold one token into the sliding window; return this step's score.

    Score = how many times ``tid`` already appears in the prior content window
    (ignored tokens score 0 and are not stored). ``repeat_sum`` is the sum of
    the last ``window`` scores.
    """
    if tid in ignore:
        score = 0
    else:
        score = int(state.freq.get(tid, 0))
        state.content.append(tid)
        state.freq[tid] = score + 1
        state.content_tokens_seen += 1
        while len(state.content) > window:
            old = state.content.popleft()
            left = state.freq.get(old, 0) - 1
            if left <= 0:
                state.freq.pop(old, None)
            else:
                state.freq[old] = left

    state.scores.append(score)
    state.repeat_sum += score
    while len(state.scores) > window:
        state.repeat_sum -= state.scores.popleft()
    return score


class TokenRepeatDetector(ConfigBackedDetector):
    """Detect local token re-reading via sliding-window ``repeat_sum``.

    Consumes the cumulative output stream from ``RequestIoSnapshotManager``
    (IO is appended once in after-sample, not in ``check_after_spec``). A
    per-req cursor tracks how far the stream has been folded into the repeat
    window so each id is pushed once.
    """

    incident_type = "token_repeat"
    section_key = "token_repeat"
    schema = DetectorSchema(
        section_key="token_repeat",
        stage="after_sample",
        help="Sliding-window local token re-read (no logprobs).",
        fields=(
            ConfigField("enabled", False, "bool", help="Master switch"),
            ConfigField("algorithm", "window", "str", help="window or m4"),
            ConfigField("m4_tables", "", "str", help="Tokenizer-specific NPZ"),
            ConfigField("m4_tokenizer_json", "", "str", help="Served tokenizer.json"),
            ConfigField("m4_think_end", -1, "int", min_value=-1, help="Explicit think-end ID; -1 for chat"),
            ConfigField("window", 32, "int", min_value=1, help="Sliding content window"),
            ConfigField(
                "repeat_sum_threshold",
                64,
                "int",
                min_value=0,
                help="Alert when sum of repeat scores exceeds this",
            ),
            ConfigField(
                "min_tokens",
                32,
                "int",
                min_value=0,
                help="Minimum content tokens before alerting (0 = no warmup)",
            ),
            ConfigField(
                "consecutive_hits",
                1,
                "int",
                min_value=1,
                help="Required consecutive over-threshold steps",
            ),
            ConfigField(
                "ignore_token_ids",
                [],
                "list_int",
                help="Token ids excluded from window scoring",
            ),
        ),
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        self._lock = RLock()
        self._epoch = 0
        self._algorithm = "window"
        self._m4_options = ("", "", -1)
        self._m4 = None
        self._m4_states = {}
        self._m4_owners = {}
        self._m4_failed = set()
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        _section = self.schema.defaults_dict()
        self._window = int(_section["window"])
        self._repeat_sum_threshold = int(_section["repeat_sum_threshold"])
        self._min_tokens = int(_section["min_tokens"])
        self._consecutive_hits_thresh = int(_section["consecutive_hits"])
        self._ignore_token_ids: frozenset[int] = frozenset()
        self._states: dict[str, TokenRepeatState] = {}
        # How many cumulative output ids have already been pushed for each req.
        self._consumed_len: dict[str, int] = {}
        self._alerted: set[str] = set()
        if runtime_config is not None:
            self.refresh_from_config()

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def refresh_from_config(self) -> None:
        with self._lock:
            super().refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        algorithm = getter("algorithm", "window")
        if algorithm not in ("window", "m4"):
            raise ValueError("token_repeat.algorithm must be window or m4")
        options = (getter("m4_tables", ""), getter("m4_tokenizer_json", ""), getter("m4_think_end", -1))
        if algorithm == "m4" and self._enabled:
            # Window knobs have no M4 equivalent. Reject rather than ignore them.
            for name in ("window", "repeat_sum_threshold", "min_tokens", "consecutive_hits", "ignore_token_ids"):
                if getter(name, self.schema.defaults_dict()[name]) != self.schema.defaults_dict()[name]:
                    raise ValueError(f"token_repeat.{name} is only supported by algorithm=window")
        if algorithm != self._algorithm or options != self._m4_options:
            self.on_disabled()
            self._algorithm, self._m4_options = algorithm, options
        new_window = max(1, int(getter("window", self._window)))
        if new_window != self._window and self._states:
            # B12 fix: also clear _alerted — a req already alerted under the
            # old window would never be re-evaluated under the new window.
            self._states.clear()
            self._consumed_len.clear()
            self._alerted.clear()
        self._window = new_window
        self._repeat_sum_threshold = max(0, int(getter("repeat_sum_threshold", self._repeat_sum_threshold)))
        self._min_tokens = max(0, int(getter("min_tokens", self._min_tokens)))
        self._consecutive_hits_thresh = max(1, int(getter("consecutive_hits", self._consecutive_hits_thresh)))
        try:
            ignore = coerce_list_int(getter("ignore_token_ids", []), "ignore_token_ids")
        except ValueError as exc:
            logger.error("[runtime_guard: token_repeat] invalid ignore_token_ids: %s; keeping previous", exc)
            return
        self._ignore_token_ids = frozenset(ignore)

    def clear_finished(self, req_id: str, *, expected_state=None) -> None:
        with self._lock:
            if (
                self._algorithm == "m4" and expected_state is not None
                and self._m4_owners.get(req_id) is not expected_state
            ):
                return
            self._states.pop(req_id, None)
            self._consumed_len.pop(req_id, None)
            self._alerted.discard(req_id)
            self._m4_states.pop(req_id, None)
            self._m4_owners.pop(req_id, None)
            self._m4_failed.discard(req_id)

    def on_disabled(self) -> None:
        with self._lock:
            self._epoch += 1
            self._states.clear()
            self._consumed_len.clear()
            self._alerted.clear()
            self._m4_states.clear()
            self._m4_owners.clear()
            self._m4_failed.clear()
            self._m4 = None

    def check_all(
        self, sampled_token_ids, req_ids=None, skip_req_ids=None, *, expected_epoch=None, expected_states=None
    ) -> list[Incident]:
        with self._lock:
            self.refresh_from_config()
            if expected_epoch is not None and expected_epoch != self._epoch:
                return []
            if expected_states is not None:
                req_ids = [
                    rid for rid in req_ids
                    if expected_states.get(rid) is not None
                    and RequestGuardStore.get().get_state(rid) is expected_states[rid]
                ]
            return self._check_all_locked(sampled_token_ids, req_ids, skip_req_ids)

    def _check_all_locked(
        self,
        sampled_token_ids: list[list[int]] | None,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[Incident]:
        """Fold new cumulative output ids into per-req windows.

        Prefer ``sampled_token_ids=None`` after
        ``DetectorManager.after_sample_hot_path`` / ``check_after_spec`` have
        already written into the IO buffer. When provided (standalone callers /
        unit tests), tokens are appended here first.

        Always advances via Store ``cumulative_output_since`` + ``_consumed_len``
        so a second call does not re-score ids already folded.
        """
        if not self._precheck():
            return []

        runner = self._runner
        log_leader = runner_tp_rank(runner) == 0
        req_ids = resolve_batch_req_ids(runner, req_ids)
        if not req_ids:
            return []

        io_mgr = RequestIoSnapshotManager.get()
        if sampled_token_ids is not None:
            io_mgr.append_batch(req_ids, sampled_token_ids)

        if self._algorithm == "m4":
            return self._check_m4(req_ids, skip_req_ids)
        alerts: list[Incident] = []
        for batch_idx, req_id in enumerate(req_ids):
            if not req_id:
                continue
            consumed = self._consumed_len.get(req_id, 0)
            total, new_ids = io_mgr.cumulative_output_since(req_id, consumed)
            if consumed > total:
                # Buffer was reset / replaced; restart from current stream.
                consumed = 0
                self._states.pop(req_id, None)
                total, new_ids = io_mgr.cumulative_output_since(req_id, 0)
            # Advance cursor even when skipping/alerted so we never re-push.
            self._consumed_len[req_id] = total

            if req_id in self._alerted:
                continue
            if skip_req_ids and req_id in skip_req_ids:
                continue
            if not new_ids:
                continue
            alert = self.check_one(
                req_idx=batch_idx,
                req_id=req_id,
                token_ids=new_ids,
                log_leader=log_leader,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _m4_incident(self, rid, idx, fired_at, reason, finish=False):
        self._alerted.add(rid)
        return Incident(
            incident_type=self.incident_type, req_id=rid, req_idx=idx,
            is_ill=True, ill_type=ILL_TYPE_REPEAT,
            detail={"algorithm": "m4", "fired_at": fired_at, "reason": reason, "finish_only": finish},
        )

    def _check_m4(self, req_ids, skip_req_ids):
        if self._m4 is None:
            # Optional independent extension: never import/build it when window/off.
            from vllm_ascend.observability.runtime_guard.detector.m4_backend import load_m4
            self._m4 = load_m4(self._runner, *self._m4_options)
        keys, states, inputs, indices = [], [], [], []
        io = RequestIoSnapshotManager.get()
        for idx, rid in enumerate(req_ids):
            if not rid or rid in keys:
                continue
            owner = RequestGuardStore.get().get_state(rid)
            if rid in self._m4_owners and self._m4_owners[rid] is not owner:
                self.clear_finished(rid)
            # Skipped/empty requests also own cursors. Record their generation
            # before any early exit so terminal cleanup cannot leave tombstones.
            self._m4_owners[rid] = owner
            consumed = self._consumed_len.get(rid, 0)
            total, ids = io.cumulative_output_since(rid, consumed)
            if consumed > total:
                self.clear_finished(rid)
                self._m4_owners[rid] = owner
                total, ids = io.cumulative_output_since(rid, 0)
            if rid in self._m4_failed:
                # A partial native batch is never replayed.
                continue
            if rid in self._alerted or (skip_req_ids and rid in skip_req_ids):
                self._consumed_len[rid] = total
                continue
            if not ids:
                continue
            state = self._m4_states.get(rid)
            if state is None:
                state = self._m4_states[rid] = self._m4.start()
                self._m4_owners[rid] = owner
            keys.append(rid)
            states.append(state)
            inputs.append(ids)
            indices.append(idx)
        if not states:
            return []
        try:
            events = self._m4.feed_events(states, inputs)
        except Exception:
            # Quarantine ALL submitted states, including unprocessed slots.
            # Receipts diagnose partial progress, never authorize a retry.
            self._m4_failed.update(keys)
            logger.exception("[runtime_guard: token_repeat] partial M4 batch failed; states quarantined")
            alerts = []
            for rid, state, idx in zip(keys, states, indices):
                try:
                    receipt = self._m4.progress(state)
                except Exception:
                    logger.exception("[runtime_guard: token_repeat] M4 receipt unavailable req=%s", rid)
                    continue
                if receipt is not None:
                    received, _, fired_at, reason = receipt
                    self._consumed_len[rid] = received
                    if reason:
                        alerts.append(self._m4_incident(rid, idx, fired_at, reason))
            return alerts
        for rid, ids in zip(keys, inputs):
            self._consumed_len[rid] = self._consumed_len.get(rid, 0) + len(ids)
        return [self._m4_incident(keys[slot], indices[slot], fired_at, reason)
                for slot, fired_at, reason in events]

    def finish_requests(self, req_ids):
        """Drain unread output and flush tails BEFORE shared IO is removed.

        clear_finished remains cancellation/cleanup only. Caller must wait for
        CPU jobs to drain; returned incidents must follow normal alert routing.
        """
        with self._lock:
            if not self.enabled or self._algorithm != "m4":
                return []
            skipped = RequestGuardStore.get().stopped_req_ids()
            alerts = self.check_all(None, req_ids, skipped)
            for rid in req_ids:
                state = self._m4_states.get(rid)
                if state is None or rid in skipped or rid in self._alerted or rid in self._m4_failed:
                    continue
                try:
                    result = self._m4.finish(state)
                    if result.is_ill:
                        alerts.append(self._m4_incident(rid, None, result.fired_at, result.reason, True))
                except Exception:
                    self._m4_failed.add(rid)
                    logger.exception("[runtime_guard: token_repeat] M4 finish failed req=%s", rid)
                    # Earlier tail alerts remain deliverable even if another
                    # request fails. Recover this state's committed alert too.
                    try:
                        receipt = self._m4.progress(state)
                        if receipt is not None and receipt[3]:
                            alerts.append(self._m4_incident(rid, None, receipt[2], receipt[3], True))
                    except Exception:
                        logger.exception("[runtime_guard: token_repeat] M4 finish receipt unavailable req=%s", rid)
            return alerts

    def check_one(
        self,
        req_idx: int,
        req_id: str,
        token_ids: Iterable[int],
        *,
        log_leader: bool = False,
    ) -> Incident | None:
        if self._algorithm == "m4":
            raise RuntimeError("M4 requires check_all for batch/cumulative input")
        state = self._states.get(req_id)
        if state is None:
            state = TokenRepeatState()
            self._states[req_id] = state

        ids_list = _ids_already_ints(token_ids)
        window = self._window
        thresh = self._repeat_sum_threshold
        ignore = self._ignore_token_ids
        hit = False
        last_score = 0
        for tid in ids_list:
            last_score = push_token_repeat(state, int(tid), window=window, ignore=ignore)
            warmed = state.content_tokens_seen >= self._min_tokens
            over = warmed and state.repeat_sum > thresh
            if over:
                state.consecutive_hits += 1
            else:
                state.consecutive_hits = 0
            if state.consecutive_hits >= self._consecutive_hits_thresh:
                hit = True
                # Keep pushing remaining ids so window state matches the full
                # cumulative delta already committed via _consumed_len.

        if not hit:
            return None

        self._alerted.add(req_id)
        detail = {
            "repeat_sum": state.repeat_sum,
            "repeat_sum_threshold": thresh,
            "window": window,
            "content_tokens_seen": state.content_tokens_seen,
            "last_score": last_score,
            "consecutive_hits": state.consecutive_hits,
            "chunk_len": len(ids_list),
            "recent_token_ids": ids_list[-min(32, len(ids_list)) :],
        }
        if log_leader:
            logger.error(
                "[runtime_guard: token_repeat] req=%s repeat_sum=%s threshold=%s "
                "window=%s content_seen=%s consecutive=%s",
                req_id,
                state.repeat_sum,
                thresh,
                window,
                state.content_tokens_seen,
                state.consecutive_hits,
            )
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ILL_TYPE_REPEAT,
            detail=detail,
        )
