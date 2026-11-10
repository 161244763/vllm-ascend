"""P1 request adapter; no detector algorithm or framework modification.

One native submission contains all active requests in a decode step. There is no
cross-step token buffering. Profiling instrumentation is kept outside this adapter.
"""
from collections import Counter
from collections.abc import Mapping


def new_row():
    # Preserve the historical counter schema. Diagnostics live outside rows.
    return dict(generated=0, scored=0, alert_at=None)


def length_bucket(length, exact_limit=65536):
    """Exact bins through the limit; limit+1 represents ALL larger lengths.

    This is diagnostics only. It must never restrict detector input or alter
    exact per-request generated/scored counters.
    """
    return min(length, exact_limit + 1)


class CPUBackend:
    def __init__(self, profile=None):
        # Resolve once, never in the decode path. Historical callers use None,
        # {}, or 'm4'; explicit model settings must not silently select qwen3.
        if profile is None or (isinstance(profile, str) and profile == 'm4'):
            options = {}
        elif isinstance(profile, Mapping):
            options = dict(profile)
        else:
            raise TypeError('M4 profile must be a mapping, None, or "m4"')
        unknown = options.keys() - {'corpus', 'config', 'think_end', 'mode'}
        if unknown:
            raise ValueError(f'unknown M4 profile keys: {sorted(unknown, key=str)}')
        mode = options.pop('mode', 'p1')
        if mode not in ('p1', 'old_cpp'):
            raise ValueError('M4 benchmark mode must be p1 or old_cpp')
        corpus = options.pop('corpus', 'qwen3')
        from m4_cpp import RepDetectorCPP
        self.det = RepDetectorCPP(corpus, **options)

    def start(self, key):
        return self.det.start()

    def feed_events(self, states, token_lists):
        return self.det.feed_events(states, token_lists)

    def finish(self, state):
        return self.det.finish(state)

    def result(self, state):
        return self.det.result(state)

    def progress(self, state):
        return self.det.progress(state)

    def clear(self, state):
        self.det.clear(state)


class LegacyBackend:
    """Historical per-token feed through the same sparse-event adapter."""
    def __init__(self, detector):
        self.det = detector

    def start(self, key):
        return self.det.start()

    def feed_events(self, states, token_lists):
        if len(states) != len(token_lists) or len({id(state) for state in states}) != len(states):
            raise ValueError('misaligned or duplicate states')
        events = []
        for slot, (state, ids) in enumerate(zip(states, token_lists)):
            if state is None:
                raise TypeError('state must not be None')
            for tid in ids:
                result = self.det.feed(state, [tid])
                if result is not None and result.is_ill:
                    events.append((slot, result.fired_at, result.reason))
                    break
        return events

    def finish(self, state):
        return self.det.finish(state)

    def result(self, state):
        return self.det.result(state)

    def progress(self, state):
        read = getattr(self.det, 'progress', None)
        return None if read is None else read(state)

    def clear(self, state):
        self.det.clear(state)


class PerfSession:
    """Submit each active request at most once per decode step.

    ``generated`` counts the common workload. ``scored``/``received`` count raw
    tokens accepted for detector scoring, including a pending tail. Native
    ``state.t``/``computed_tokens`` count feature-computed raw tokens. The first
    alert stops later detector submissions, never workload generation.

    ``retire_rows=False, eager_rows=True`` is the fixed-unique-ID batch-only
    comparison mode. The default retires rows and uses bounded per-key counters.
    ``retain_retired_rows`` is diagnostic only and is not bounded; production and
    primary benchmarks leave it false.

    A bare request ID cannot distinguish a late callback after ID reuse. Explicit
    session-level generation tokens make late finish/cleanup a no-op while only
    active IDs retain tokens. ``register()`` returns the assigned tokens. Without
    them, the upstream callback order guarantee remains required.

    With ``retire_rows=True``, per-request state is retired. Aggregate length
    buckets use ``max_generated_tokens`` as a diagnostic exact-bin limit only,
    with one overflow bin; it is NOT an input limit. Fixed-ID mode is
    intentionally unbounded in its unique key space and retains historical rows.
    """
    def __init__(self, profile=None, factory=CPUBackend, true_off=False,
                 retire_rows=True, eager_rows=False, retain_retired_rows=False,
                 max_generated_tokens=65536):
        self.detector = None if profile is None else factory(profile)
        self.true_off = true_off
        if true_off and self.detector is not None:
            raise ValueError('true_off cannot own a detector')
        self.retire_rows = retire_rows
        self.eager_rows = eager_rows
        self.retain_retired_rows = retain_retired_rows
        if retain_retired_rows and not retire_rows:
            raise ValueError('retired rows require retire_rows=True')
        if eager_rows and retire_rows:
            raise ValueError('eager_rows is reserved for the fixed-ID comparison mode')
        self.states = {}
        self.rows = {}
        self.received = {}
        self.computed_tokens = {}
        self.finish_only = {}
        self.events = {}
        self.completed = set()  # Finished active generations, removed on retirement.
        self.failed = set()  # Active generations invalidated by an exception.
        self.failed_operations = 0  # Never reset by cleanup; invalidates run metrics.
        self.recovery_incomplete = False
        if (not isinstance(max_generated_tokens, int)
                or isinstance(max_generated_tokens, bool) or max_generated_tokens < 0):
            raise ValueError('max_generated_tokens must be a non-negative integer')
        self.max_generated_tokens = max_generated_tokens
        self.incarnations = {}  # active ID -> session-level generation token only
        self.next_generation = 0
        self.closed = set()  # fixed-ID diagnostic mode only
        self.retired_rows = []  # explicit diagnostic mode only
        self.retired_requests = 0
        self.retired_generated = 0
        self.retired_scored = 0
        self.retired_received = 0
        self.retired_alerts = 0
        self.retired_lengths = Counter()
        self.scored_without_detection = 0

    def register(self, keys):
        """Register requests, including zero-token requests, without native states."""
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate request key')
        if any(key in self.rows for key in keys):
            raise ValueError('request generation already active')
        for key in keys:
            self._create_row(key)
        return [self.incarnations[key] for key in keys]

    def _create_row(self, key):
        self.next_generation += 1
        self.incarnations[key] = self.next_generation
        self.rows[key] = new_row()
        self.received[key] = 0
        self.computed_tokens[key] = 0
        self.finish_only[key] = False
        self.closed.discard(key)

    def _current_generation(self, key, generation):
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            raise ValueError('invalid request generation')
        current = self.incarnations.get(key)
        if current is None:
            # Only an active ID/token pair can mutate state. Unknown and retired
            # IDs have no tombstone and late finish/cancel callbacks are harmless.
            return False
        if generation > current:
            raise ValueError('callback for a future request generation')
        return generation == current

    def _late_filter(self, keys, generations):
        if generations is None:
            return list(keys)
        if len(keys) != len(generations) or len(set(zip(keys, generations))) != len(keys):
            raise ValueError('misaligned or duplicate request generations')
        # Validate malformed tokens and future tokens for active IDs before mutation.
        # Retired/unknown IDs are ignored: session tokens have no per-ID ordering.
        current = [self._current_generation(key, generation)
                   for key, generation in zip(keys, generations)]
        return [key for key, active in zip(keys, current) if active]

    def sampled(self, keys, token_lists, generations=None):
        if self.failed_operations or self.recovery_incomplete:
            self._check_failed(keys)
        if len(keys) != len(token_lists):
            raise AssertionError('misaligned CPU lists')
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate request key in one submission')
        if generations is not None:
            if len(keys) != len(generations):
                raise ValueError('misaligned request generations')
            # A sampled callback for an old/future generation is an error, not a
            # new generation. Validate all before counters or states are changed.
            for key, generation in zip(keys, generations):
                if not self._current_generation(key, generation) or key not in self.rows:
                    raise ValueError('sampled for an inactive request generation')
        # Native finish is terminal even without an alert. Reject the entire
        # submission before counting tokens that a done native state would ignore.
        if self.completed and any(key in self.completed for key in keys):
            raise ValueError('sampled after finish; retire before ID reuse')
        # Decode at C1 is common. Keep the same preflight and the exact same
        # backend call (including subclass instrumentation), without constructing
        # active_keys or walking a second zip/enumerate/set pipeline.
        try:
            if (len(keys) == 1 and type(keys) in (list, tuple)
                    and type(token_lists) in (list, tuple)):
                return self._sample_single(keys[0], token_lists[0])
            return self._sample_many(keys, token_lists)
        except BaseException:
            # Preparation and result conversion can fail too, not only feed.
            # Read states inside the protected recovery routine, avoiding an
            # allocation in this handler that could hide the original error.
            self._handle_failure(keys)
            raise

    def _sample_many(self, keys, token_lists):
        states, inputs, active_keys = [], [], []
        for key, ids in zip(keys, token_lists):
            if not ids:
                # Match the historical no-op empty sampled callback. Explicit
                # register() covers a zero-token request lifecycle.
                continue
            if key in self.closed and not self.retire_rows:
                raise ValueError('ID reuse needs retire_rows=True')
            # Both comparison modes share this path. Create a row only on
            # admission, not as an eagerly evaluated get/setdefault default.
            row = self.rows.get(key)
            if row is None:
                self._create_row(key)
                row = self.rows[key]
            row['generated'] += len(ids)
            if self.detector is None or row['alert_at'] is not None:
                if self.detector is None:
                    self.scored_without_detection += len(ids)
                continue
            state = self.states.get(key)
            if state is None:
                state = self.states[key] = self.detector.start(key)
            states.append(state)
            inputs.append(ids)
            active_keys.append(key)
        if not states:
            return
        events = self.detector.feed_events(states, inputs)
        # The overwhelmingly common path has no first alert. Do not allocate a
        # set or enumerate slots solely to ask whether they are in an empty set.
        # Restrict the shortcut to exact lists: custom lazy iterables retain the
        # historical iteration/exception behavior below.
        if type(events) is list and not events:
            for key, ids, state in zip(active_keys, inputs, states):
                self.rows[key]['scored'] += len(ids)
                self.received[key] += len(ids)
                self.computed_tokens[key] = state.t
            return
        seen = set()
        for slot, fired_at, reason in events:
            if (not isinstance(slot, int) or isinstance(slot, bool)
                    or slot < 0 or slot >= len(active_keys) or slot in seen):
                raise AssertionError('invalid sparse event slot')
            seen.add(slot)
            key = active_keys[slot]
            row = self.rows[key]
            if row['alert_at'] is not None or not reason:
                raise AssertionError('invalid or duplicate first alert')
            delta = fired_at - self.received[key]
            if delta < 1 or delta > len(inputs[slot]):
                raise AssertionError('alert outside submitted input')
            row['scored'] += delta
            self.received[key] = fired_at
            self.computed_tokens[key] = fired_at
            row['alert_at'] = row['scored']
            self.events[key] = (fired_at, reason, False)
        for slot, (key, ids, state) in enumerate(zip(active_keys, inputs, states)):
            if slot not in seen:
                row = self.rows[key]
                row['scored'] += len(ids)
                self.received[key] += len(ids)
                self.computed_tokens[key] = state.t

    def _sample_single(self, key, ids):
        """Single-request specialization; no buffering and no detector bypass."""
        if not ids:
            return
        if key in self.closed and not self.retire_rows:
            raise ValueError('ID reuse needs retire_rows=True')
        row = self.rows.get(key)
        if row is None:
            self._create_row(key)
            row = self.rows[key]
        row['generated'] += len(ids)
        if self.detector is None or row['alert_at'] is not None:
            if self.detector is None:
                self.scored_without_detection += len(ids)
            return
        state = self.states.get(key)
        if state is None:
            state = self.states[key] = self.detector.start(key)
        events = self.detector.feed_events([state], [ids])
        seen = False
        for slot, fired_at, reason in events:
            if (not isinstance(slot, int) or isinstance(slot, bool)
                    or slot != 0 or seen):
                raise AssertionError('invalid sparse event slot')
            seen = True
            if row['alert_at'] is not None or not reason:
                raise AssertionError('invalid or duplicate first alert')
            delta = fired_at - self.received[key]
            if delta < 1 or delta > len(ids):
                raise AssertionError('alert outside submitted input')
            row['scored'] += delta
            self.received[key] = fired_at
            self.computed_tokens[key] = fired_at
            row['alert_at'] = row['scored']
            self.events[key] = (fired_at, reason, False)
        if not seen:
            row['scored'] += len(ids)
            self.received[key] += len(ids)
            self.computed_tokens[key] = state.t

    def _check_failed(self, keys):
        # No per-key walk in the normal, never-failed session.
        if self.recovery_incomplete or (self.failed and any(key in self.failed for key in keys)):
            raise RuntimeError('failed request generation; cancel/retire before reuse')

    def _handle_failure(self, keys, states=None, finishing=False):
        # Even receipt allocation can fail under memory pressure. In that case
        # reject the whole session, while keeping the original caller exception.
        self.recovery_incomplete = True
        try:
            self._record_failure(keys, states, finishing)
        except BaseException:
            return
        self.recovery_incomplete = False

    def _record_failure(self, keys, states=None, finishing=False):
        """Preserve committed alerts on partial failure; never silently retry.

        Recovery is best effort (including under MemoryError). Even when a
        backend has no receipt API or a block is poisoned, the session and all
        submitted generations are explicitly invalid. No exception is swallowed
        on the caller's submission path. No receipts are queried on success.
        """
        self.failed_operations += 1
        self.failed.update(keys)
        progress = getattr(self.detector, 'progress', None)
        if progress is None:
            return
        if states is None:
            states = (self.states.get(key) for key in keys)
        for key, state in zip(keys, states):
            try:
                if state is None:
                    continue
                receipt = progress(state)
                if receipt is None:
                    continue
                received, computed, fired_at, reason = receipt
                row = self.rows[key]
                if not (self.received[key] <= received <= row['generated']
                        and 0 <= computed <= received):
                    continue
                if reason and not (0 < fired_at <= received):
                    continue
                # A Python bookkeeping write may have failed after scored was
                # updated but before received. Reconcile from the authoritative
                # absolute receipt instead of adding that prefix twice.
                row['scored'] = received
                self.received[key] = received
                self.computed_tokens[key] = computed
                if reason:
                    row['alert_at'] = fired_at
                    if key not in self.events:
                        self.finish_only[key] = finishing
                        self.events[key] = (fired_at, reason, finishing)
            except BaseException:
                # Keep the original failure and the invalid-generation marker.
                continue

    def finish(self, keys, generations=None):
        """Terminal tail flush; repeat calls are idempotent, sampling must stop."""
        keys = self._late_filter(keys, generations)
        self._check_failed(keys)
        if generations is None:
            unknown = [key for key in keys if key not in self.rows]
            if unknown:
                raise ValueError('finish for unknown or retired request generation')
        for key in keys:
            if key in self.completed or key in self.closed:
                continue
            state = self.states.get(key)
            try:
                self.completed.add(key)
                if state is None:
                    continue
                row = self.rows[key]
                result = self.detector.finish(state)
                if result is not None and result.is_ill and row['alert_at'] is None:
                    self.finish_only[key] = True
                    row['alert_at'] = row['scored']
                    self.events[key] = (result.fired_at, result.reason, True)
                self.computed_tokens[key] = state.t
                if row['alert_at'] is None and self.received[key] != state.t:
                    raise AssertionError('finish consumed an unexpected raw-token count')
            except BaseException:
                self._handle_failure([key], [state], finishing=True)
                raise

    def finished(self, keys, generations=None):
        """Retire one incarnation; explicit old-generation callbacks are harmless."""
        keys = self._late_filter(keys, generations)
        for key in keys:
            failed = key in self.failed
            state = self.states.get(key)
            if state is not None and self.detector is not None:
                # Preserve the old MultiplexDetector.clear call in both modes;
                # CPUBackend.clear deliberately retains the no-op call boundary.
                try:
                    self.detector.clear(state)
                except BaseException:
                    self._handle_failure([key], [state])
                    raise
            self.failed.discard(key)
            self.completed.discard(key)
            event = self.events.pop(key, None)
            self.states.pop(key, None)
            row = self.rows.get(key)
            if row is None:
                # Immediate repeated cleanup is idempotent. With bare IDs this is
                # valid only while no new incarnation has been registered.
                continue
            if not self.retire_rows:
                # Fixed-ID batch-only mode retains the historical row, while
                # per-request diagnostics die with the native state.
                self.closed.add(key)
                self.retired_received += self.received.pop(key, 0)
                self.computed_tokens.pop(key, None)
                self.finish_only.pop(key, None)
                continue
            generation = self.incarnations[key]
            self.incarnations.pop(key)
            self.rows.pop(key)
            received = self.received.pop(key)
            computed = self.computed_tokens.pop(key)
            finish_only = self.finish_only.pop(key)
            self.retired_requests += 1
            self.retired_generated += row['generated']
            self.retired_scored += row['scored']
            self.retired_received += received
            self.retired_alerts += row['alert_at'] is not None
            self.retired_lengths[length_bucket(row['generated'], self.max_generated_tokens)] += 1
            if self.retain_retired_rows:
                fired_at, reason, _ = event or (-1, '', False)
                self.retired_rows.append((key, generation, dict(
                    row, computed_tokens=computed, received=received,
                    fired_at=fired_at, reason=reason, finish_only=finish_only,
                    **({'valid': False} if failed else {}))))

    def inspect(self, keys):
        """Read per-request verification data before cleanup (not timed production work)."""
        output = {}
        for key in keys:
            row = self.rows[key]
            event = self.events.get(key, (-1, '', False))
            output[key] = dict(
                generated=row['generated'], scored=row['scored'],
                alert_at=row['alert_at'], computed_tokens=self.computed_tokens[key],
                received=self.received[key], finish_only=self.finish_only[key],
                fired_at=event[0], reason=event[1])
            if key in self.failed:
                output[key]['valid'] = False
        return output

    def snapshot(self):
        rows = list(self.rows.values())
        return dict(
            valid=self.failed_operations == 0 and not self.recovery_incomplete,
            failed_operations=self.failed_operations,
            requests=self.rows, live_states=len(self.states),
            retired_requests=self.retired_requests,
            generated=self.retired_generated + sum(r['generated'] for r in rows),
            scored=self.retired_scored + sum(r['scored'] for r in rows),
            native_accepted_tokens=self.retired_received + sum(self.received.values()),
            alerts=self.retired_alerts + sum(r['alert_at'] is not None for r in rows),
            lengths=dict(self.retired_lengths + Counter(
                length_bucket(r['generated'], self.max_generated_tokens) for r in rows)),
            retired_rows_retained=len(self.retired_rows),
            scored_without_detection=self.scored_without_detection)
