"""Local adapter state-machine tests with a fake sparse backend.

These tests run without NumPy/native extensions. They are NOT native detector
correctness or target CPU performance evidence. Target tests remain in test_p1.py.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import perf_session

from perf_session import LegacyBackend, PerfSession


class Result:
    def __init__(self, is_ill=False, fired_at=-1, reason=''):
        self.is_ill = is_ill
        self.fired_at = fired_at
        self.reason = reason


class State:
    def __init__(self, alert_at, tail_alert=False):
        self.t = 0
        self.received = 0
        self.fired_at = -1
        self.reason = ''
        self.alert_at = alert_at
        self.tail_alert = tail_alert


class FakeDetector:
    def __init__(self, alert_at=128, tail_alert=False):
        self.alert_at = alert_at
        self.tail_alert = tail_alert
        self.calls = []
        self.cleared = 0

    def start(self):
        return State(self.alert_at, self.tail_alert)

    def feed(self, state, ids):
        for _ in ids:
            if state.reason:
                break
            state.received += 1
            state.t = state.received // 64 * 64
            if state.received == state.alert_at and state.t == state.received:
                state.fired_at = state.t
                state.reason = 'fake-rule'
                break
        return self.result(state)

    def finish(self, state):
        if not state.reason:
            state.t = state.received
            if self.tail_alert and state.received == self.alert_at:
                state.fired_at = state.received
                state.reason = 'fake-tail'
        # RepDetectorCPP.finish always returns a Result, unlike feed/result.
        return self.result(state) or Result()

    def result(self, state):
        # Match RepDetectorCPP: no Result allocation for a non-alert feed.
        return Result(True, state.fired_at, state.reason) if state.reason else None

    def clear(self, state):
        self.cleared += 1


class FakeBackend:
    def __init__(self, detector):
        self.det = detector

    def start(self, key):
        return self.det.start()

    def feed_events(self, states, token_lists):
        if len(states) != len(token_lists) or any(state is None for state in states):
            raise ValueError('invalid native request set')
        if len({id(state) for state in states}) != len(states):
            raise ValueError('duplicate state in feed_events')
        self.det.calls.append((states, deepcopy(token_lists)))
        events = []
        for slot, (state, ids) in enumerate(zip(states, token_lists)):
            before = state.reason
            self.det.feed(state, ids)
            if not before and state.reason:
                events.append((slot, state.fired_at, state.reason))
        return events

    def finish(self, state):
        return self.det.finish(state)

    def result(self, state):
        return self.det.result(state)

    def clear(self, state):
        self.det.clear(state)


def session(factory=FakeBackend, detector=None, **kwargs):
    detector = detector or FakeDetector()
    return PerfSession({}, factory=lambda _: factory(detector), **kwargs), detector


def visible(session):
    return deepcopy((session.rows, session.received, session.computed_tokens,
                     session.finish_only, session.incarnations, session.closed))


class AdapterTest(unittest.TestCase):
    def compare_legacy_and_sparse(self, ids, chunks, tail=False):
        detector = FakeDetector(alert_at=128, tail_alert=tail)
        a, _ = session(detector=detector, retire_rows=False, eager_rows=True)
        b, _ = session(factory=LegacyBackend, detector=detector,
                       retire_rows=False, eager_rows=True)
        pos = 0
        for size in chunks:
            part = ids[pos:pos + size]
            pos += len(part)
            a.sampled(['a', 'empty'], [part, []])
            b.sampled(['a', 'empty'], [part, []])
            self.assertEqual(a.rows, b.rows)
            self.assertEqual(a.received, b.received)
            self.assertEqual(a.computed_tokens, b.computed_tokens)
        a.finish(['a'])
        b.finish(['a'])
        self.assertEqual(a.inspect(['a']), b.inspect(['a']))
        return a, b

    def test_boundary_early_alert_and_continued_generation(self):
        a, b = self.compare_legacy_and_sparse([64] * 300, [1, 63, 64, 100, 72])
        expected = dict(generated=300, scored=128, alert_at=128, computed_tokens=128,
                        received=128, finish_only=False, fired_at=128, reason='fake-rule')
        self.assertEqual(a.inspect(['a'])['a'], expected)
        a.finished(['a'])
        b.finished(['a'])
        self.assertEqual(a.snapshot()['generated'], b.snapshot()['generated'])
        self.assertEqual(a.snapshot()['scored'], b.snapshot()['scored'])
        self.assertEqual(a.snapshot()['native_accepted_tokens'], 128)

    def test_no_double_count_early_alert_prefix(self):
        a, _ = session(retire_rows=False, eager_rows=True)
        a.sampled(['a'], [[64] * 200])
        self.assertEqual(a.rows['a'], dict(generated=200, scored=128, alert_at=128))
        self.assertEqual(a.received['a'], 128)
        self.assertEqual(a.computed_tokens['a'], 128)
        self.assertEqual(a.incarnations['a'], 1)
        a.sampled(['a'], [[64] * 10])
        self.assertEqual(a.rows['a']['generated'], 210)
        self.assertEqual(a.rows['a']['scored'], 128)

    def test_pending_tail_and_finish_only(self):
        a, b = self.compare_legacy_and_sparse([64] * 127, [127], tail=True)
        # Fake tail alert is configured at 128; this input only tests pending flush.
        self.assertEqual(a.inspect(['a'])['a']['received'], 127)
        self.assertEqual(a.inspect(['a'])['a']['computed_tokens'], 127)
        detector = FakeDetector(alert_at=127, tail_alert=True)
        a, _ = session(detector=detector)
        a.sampled(['a'], [[64] * 127])
        self.assertEqual(a.rows['a']['alert_at'], None)
        a.finish(['a'])
        self.assertEqual(a.inspect(['a'])['a'], dict(
            generated=127, scored=127, alert_at=127, computed_tokens=127,
            received=127, finish_only=True, fired_at=127, reason='fake-tail'))
        a.finished(['a'])
        self.assertEqual(a.snapshot()['native_accepted_tokens'], 127)
        self.assertEqual(a.snapshot()['scored'], 127)

    def test_empty_and_explicit_zero_length_registration(self):
        a, _ = session()
        a.sampled([], [])
        a.sampled(['a'], [[]])
        self.assertEqual(a.rows, {})
        a.register(['a'])
        self.assertEqual(a.rows['a']['generated'], 0)
        a.finish(['a'])
        a.finished(['a'])
        counters = a.snapshot()
        self.assertEqual(counters['lengths'], {0: 1})
        self.assertEqual(counters['native_accepted_tokens'], 0)
        self.assertEqual(counters['retired_requests'], 1)

    def test_duplicate_submission_has_no_side_effect(self):
        for factory in (FakeBackend, LegacyBackend):
            a, _ = session(factory=factory)
            a.sampled(['a'], [[64]])
            before = visible(a)
            with self.assertRaises(ValueError):
                a.sampled(['a', 'a'], [[64], [64]])
            self.assertEqual(visible(a), before)
            with self.assertRaises(ValueError):
                a.sampled(['new', 'new'], [[], [64]])
            self.assertEqual(visible(a), before)

    def test_duplicate_state_rejected_by_sparse_backend(self):
        a, detector = session()
        # Defensive fake backend models the native preflight directly. The real
        # C++ foreign/duplicate-state tests remain in target test_p1.py.
        with self.assertRaises(ValueError):
            a.detector.feed_events([detector.start(), detector.start()], [[64]])
        self.assertEqual(detector.calls, [])

    def test_bounded_retirement_and_id_reuse(self):
        a, detector = session()
        for _ in range(3):
            a.sampled(['a'], [[64] * 128])
            self.assertEqual(a.rows['a']['alert_at'], 128)
            a.finished(['a'])
            self.assertEqual(a.retired_rows, [])
            self.assertEqual(a.states, {})
            self.assertEqual(a.received, {})
            self.assertEqual(a.computed_tokens, {})
        self.assertEqual(a.snapshot()['retired_requests'], 3)
        self.assertEqual(a.snapshot()['native_accepted_tokens'], 384)
        self.assertEqual(detector.cleared, 3)

    def test_late_finish_and_cancel_cannot_touch_new_generation(self):
        a, _ = session()
        first = a.register(['a'])[0]
        a.sampled(['a'], [[64] * 64], [first])
        a.finish(['a'], [first])
        a.finished(['a'], [first])
        second = a.register(['a'])[0]
        a.sampled(['a'], [[64] * 64], [second])
        before = visible(a)
        a.finish(['a'], [1])
        a.finished(['a'], [1])
        self.assertEqual(visible(a), before)
        self.assertEqual(a.states['a'].received, 64)
        a.finish(['a'], [second])
        a.finished(['a'], [second])
        self.assertEqual(a.snapshot()['retired_requests'], 2)
        with self.assertRaises(ValueError):
            a.sampled(['a'], [[64]], [1])
        # A token from a retired ID carries no ordering information for other IDs.
        a.finish(['a'], [3])
        a.finished(['a'], [3])

    def test_retirement_releases_unique_ids_and_keeps_generation_safety(self):
        a, _ = session()
        for index in range(1000):
            key = f'unique-{index}'
            generation = a.register([key])[0]
            a.sampled([key], [[64]], [generation])
            a.finished([key], [generation])
            self.assertNotIn(key, a.incarnations)
        self.assertEqual(a.incarnations, {})
        self.assertEqual(a.rows, {})
        self.assertEqual(a.states, {})
        self.assertEqual(a.retired_requests, 1000)
        first = a.register(['reused'])[0]
        a.sampled(['reused'], [[64]], [first])
        a.finished(['reused'], [first])
        second = a.register(['reused'])[0]
        self.assertNotEqual(first, second)
        a.sampled(['reused'], [[64] * 65], [second])
        before = visible(a)
        a.finish(['reused'], [first])
        a.finished(['reused'], [first])
        self.assertEqual(visible(a), before)
        with self.assertRaises(ValueError):
            a.sampled(['reused'], [[64]], [first])
        a.finish(['reused'], [second])
        a.finished(['reused'], [second])
        self.assertEqual(a.incarnations, {})

    def test_length_accounting_uses_overflow_without_rejecting_input(self):
        a, _ = session(max_generated_tokens=64)
        a.sampled(['a'], [[64] * 64])
        a.sampled(['a'], [[64]])
        self.assertEqual(a.rows['a']['generated'], 65)
        self.assertEqual(a.received['a'], 65)
        a.finished(['a'])
        a.sampled(['b'], [[64] * 100])
        a.finished(['b'])
        self.assertEqual(a.retired_lengths, {65: 2})
        self.assertEqual(a.snapshot()['generated'], 165)
        with self.assertRaises(ValueError):
            session(max_generated_tokens=-1)
        with self.assertRaises(ValueError):
            session(max_generated_tokens=True)

    def test_existing_rows_do_not_allocate_discarded_defaults(self):
        for retire in [False, True]:
            a, _ = session(retire_rows=retire, eager_rows=not retire)
            with patch.object(perf_session, 'new_row', wraps=perf_session.new_row) as create:
                a.sampled(['a'], [[64]])
                self.assertEqual(create.call_count, 1)
                for _ in range(130):
                    a.sampled(['a'], [[64]])
                self.assertEqual(create.call_count, 1)
            self.assertEqual(a.rows['a']['generated'], 131)
            self.assertEqual(a.rows['a']['scored'], 128)

    def test_long_stream_and_histogram_do_not_change_detection(self):
        for backend in [FakeBackend, LegacyBackend]:
            for retire in [False, True]:
                a, _ = session(factory=backend, max_generated_tokens=64,
                               retire_rows=retire, eager_rows=not retire)
                b, _ = session(factory=backend, max_generated_tokens=100000,
                               retire_rows=retire, eager_rows=not retire)
                for obj in [a, b]:
                    obj.sampled(['a'], [[64] * 65537])
                    obj.finish(['a'])
                self.assertEqual(a.inspect(['a']), b.inspect(['a']))
                self.assertEqual(a.rows['a']['generated'], 65537)
                self.assertEqual(a.events['a'], (128, 'fake-rule', False))
                for obj in [a, b]:
                    obj.finished(['a'])
                self.assertEqual(a.snapshot()['lengths'], {65: 1})
                self.assertEqual(b.snapshot()['lengths'], {65537: 1})

    def test_overflow_histogram_is_bounded_for_retired_requests(self):
        a, _ = session(max_generated_tokens=4)
        for n in range(30):
            a.register([n])
            a.sampled([n], [[64] * n])
            a.finished([n])
        self.assertEqual(len(a.retired_lengths), 6)
        self.assertEqual(sum(a.retired_lengths.values()), 30)
        self.assertEqual(a.retired_lengths[5], 25)
        self.assertEqual(a.snapshot()['generated'], sum(range(30)))

    def test_cancel_without_finish_preserves_pending_accounting(self):
        a, _ = session()
        a.sampled(['a'], [[64] * 65])
        a.finished(['a'])
        counters = a.snapshot()
        self.assertEqual(counters['scored'], 65)
        self.assertEqual(counters['native_accepted_tokens'], 65)
        self.assertEqual(counters['generated'], 65)

    def test_invalid_event_is_not_silently_accepted(self):
        a, detector = session()
        original = a.detector.feed_events

        def bad(states, inputs):
            original(states, inputs)
            return [(0, 129, 'fake-rule')]
        a.detector.feed_events = bad
        with self.assertRaises(AssertionError):
            a.sampled(['a'], [[64] * 128])
        self.assertIsNone(a.rows['a']['alert_at'])

    def test_true_off_is_bookkeeping_not_detection(self):
        a = PerfSession(None, true_off=True)
        a.sampled(['a'], [[64] * 128])
        a.sampled(['a'], [[64] * 2])
        self.assertEqual(a.rows['a'], dict(generated=130, scored=0, alert_at=None))
        self.assertEqual(a.scored_without_detection, 130)
        a.finished(['a'])
        self.assertEqual(a.snapshot()['scored_without_detection'], 130)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    print(json.dumps(dict(tests=result.testsRun, failures=len(result.failures),
                          errors=len(result.errors), native_executed=False,
                          target_performance_executed=False)))
    if not result.wasSuccessful():
        raise SystemExit(1)
