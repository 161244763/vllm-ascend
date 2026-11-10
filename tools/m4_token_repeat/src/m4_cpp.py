"""Independent CPU extension for the frozen CONFIG_M4; no vLLM build dependency.

m4_vocab and m4_config supply vocabulary tables, frozen rules and result values.
Only this exact m4 configuration is accepted; unsupported configs fail explicitly.
"""

import hashlib
import json
import operator
import os

import numpy as np
from _m4_native import Detector
from m4_config import CONFIG_M4, Result, describe
from m4_vocab import THINK_END, load


class RepDetectorCPP:
    def __init__(self, corpus, config=None, think_end="auto"):
        corpus = os.fspath(corpus)
        if not isinstance(corpus, str) or not corpus:
            raise TypeError("corpus must be a non-empty name or text path")
        if config is None:
            config = json.loads(CONFIG_M4)
        if isinstance(config, (str, bytes, os.PathLike)):
            with open(config) as f:
                config = json.load(f)
        digest = hashlib.sha256(
            json.dumps(config["components"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != "cd0af8936d7f771c295cfd95f60538e73d20dd905ff235de663dba67b250df25":
            raise ValueError("native backend supports the frozen CONFIG_M4 components only")
        self.comps = config["components"]
        self.reasons = [describe(c) for c in self.comps]
        if isinstance(think_end, str) and think_end == "auto":
            if corpus not in THINK_END:
                raise ValueError("custom tables require explicit think_end")
            think_end = THINK_END[corpus]
        if think_end is not None:
            # int(float) silently changes token identity. Accept integer-like
            # scalars only, and keep None distinct from the auto default.
            if isinstance(think_end, (bool, np.bool_)):
                raise TypeError('think_end must be an integer token ID, None, or "auto"')
            think_end = operator.index(think_end)
            if not 0 <= think_end <= np.iinfo(np.int64).max:
                raise ValueError("think_end must be a non-negative int64 token ID")
        self.T = load(corpus)
        columns = ("norm", "chars", "ws", "run_char", "marker", "send")
        arrays = []
        size = None
        bounds = np.iinfo(np.int32)
        for key in columns:
            array = np.asarray(self.T[key])
            if array.ndim != 1 or not array.size:
                raise ValueError(f"{key} must be a non-empty one-dimensional table")
            if size is not None and array.size != size:
                raise ValueError("vocabulary columns must have equal lengths")
            size = array.size
            if array.dtype.kind not in "biu":
                raise TypeError(f"{key} must contain integer or boolean values")
            if int(array.min()) < bounds.min or int(array.max()) > bounds.max:
                raise ValueError(f"{key} values must fit int32 without truncation")
            arrays.append(np.ascontiguousarray(array, dtype=np.int32))
        self.native = Detector(*arrays, -1 if think_end is None else think_end)

    def start(self):
        return self.native.start()

    def result(self, st):
        if st.failed:
            raise RuntimeError("failed detector state; discard it")
        return Result(True, st.fired_at, self.reasons[st.reason]) if st.reason >= 0 else None

    def progress(self, st):
        """Exceptional-path receipt, not a retry/rollback promise.

        A batch can commit earlier slots before a later slot raises. Query each
        state to recover its committed prefix/first alert without consuming any
        token. A failed native block has no trustworthy partial counters.
        """
        if st.failed:
            return None
        return (st.received, st.t, st.fired_at, self.reasons[st.reason] if st.reason >= 0 else "")

    def feed(self, st, tokens):
        self.native.feed(st, tokens)
        return self.result(st)

    def feed_one(self, st, token):
        self.native.feed_one(st, token)
        return self.result(st)

    def feed_batch(self, states, token_lists):
        # Results are materialized only for alerts. C++ owns each request's state.
        self.native.feed_batch(states, token_lists)
        return [self.result(s) for s in states]

    def feed_events(self, states, token_lists):
        """Return only new (submission_slot, fired_at, reason) events.

        No Result is constructed for non-alert requests. The native layer
        preserves feed's lazy token consumption and first-alert semantics.
        On exception the batch is NOT atomic: use progress() on every submitted
        state to recover committed prefixes/events; do not replay the whole batch.
        """
        return [
            (slot, fired_at, self.reasons[reason])
            for slot, fired_at, reason in self.native.feed_events(states, token_lists)
        ]

    def finish(self, st):
        self.native.finish(st)
        return self.result(st) or Result()

    def monitor(self, st):
        return self.result(st) or Result()

    def clear(self, st):
        # Match the historical m4 MultiplexDetector.clear: native state storage
        # is released by dropping the last reference, with no detector registry.
        pass
