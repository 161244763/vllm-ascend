"""Terminal-finish adapter regressions; fake backend, not native equivalence."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from perf_session import PerfSession
from test_adapter_unit import FakeDetector, session


class TerminalLifecycleTest(unittest.TestCase):
    def test_single_finish_rejects_late_tokens_without_counting(self):
        obj, detector = session()
        obj.sampled(['a'], [[1] * 65])
        obj.finish(['a'])
        before = deepcopy(obj.inspect(['a']))
        calls = len(detector.calls)
        for ids in ([1], []):
            with self.assertRaisesRegex(ValueError, 'sampled after finish'):
                obj.sampled(['a'], [ids])
        self.assertEqual(obj.inspect(['a']), before)
        self.assertEqual(len(detector.calls), calls)
        self.assertTrue(obj.snapshot()['valid'])

    def test_mixed_batch_is_rejected_before_any_mutation(self):
        obj, detector = session()
        obj.sampled(['a', 'b'], [[1], [1]])
        obj.finish(['b'])
        before = deepcopy(obj.inspect(['a', 'b']))
        calls = len(detector.calls)
        with self.assertRaisesRegex(ValueError, 'sampled after finish'):
            obj.sampled(['new', 'a', 'b'], [[1], [1], [1]])
        self.assertNotIn('new', obj.rows)
        self.assertEqual(obj.inspect(['a', 'b']), before)
        self.assertEqual(len(detector.calls), calls)

    def test_zero_token_and_off_requests_are_terminal_too(self):
        for obj in (session()[0], PerfSession(None, true_off=True)):
            obj.register(['a'])
            obj.finish(['a'])
            with self.assertRaisesRegex(ValueError, 'sampled after finish'):
                obj.sampled(['a'], [[1]])
            self.assertEqual(obj.rows['a']['generated'], 0)
            obj.finished(['a'])
            self.assertEqual(obj.completed, set())

    def test_repeat_finish_is_idempotent_and_fixed_cleanup_stays_closed(self):
        for retire in (True, False):
            obj, _ = session(retire_rows=retire, eager_rows=not retire)
            obj.sampled(['a'], [[1] * 65])
            with patch.object(obj.detector, 'finish', wraps=obj.detector.finish) as finish:
                obj.finish(['a', 'a'])
                obj.finish(['a'])
                self.assertEqual(finish.call_count, 1)
                obj.finished(['a'])
                if not retire:
                    obj.finish(['a'])
                    self.assertEqual(finish.call_count, 1)
                    with self.assertRaisesRegex(ValueError, 'ID reuse'):
                        obj.sampled(['a'], [[1]])
            self.assertEqual(obj.completed, set())

    def test_first_alert_does_not_stop_workload_until_explicit_finish(self):
        obj, _ = session(detector=FakeDetector(alert_at=64))
        obj.sampled(['a'], [[1] * 100])
        obj.sampled(['a'], [[1] * 20])
        self.assertEqual(obj.rows['a'], dict(generated=120, scored=64, alert_at=64))
        self.assertEqual(obj.completed, set())
        obj.finish(['a'])
        with self.assertRaises(ValueError):
            obj.sampled(['a'], [[1]])

    def test_generation_reuse_and_late_finish_remain_safe(self):
        obj, _ = session()
        first = obj.register(['a'])[0]
        obj.sampled(['a'], [[1]], [first])
        obj.finish(['a'], [first])
        obj.finished(['a'], [first])
        second = obj.register(['a'])[0]
        obj.finish(['a'], [first])
        self.assertEqual(obj.completed, set())
        obj.sampled(['a'], [[1]], [second])
        obj.finished(['a'], [second])  # Cancellation needs no finish.
        self.assertEqual(obj.completed, set())

    def test_clear_failure_preserves_terminal_marker_until_successful_cleanup(self):
        obj, _ = session()
        obj.sampled(['a'], [[1]])
        obj.finish(['a'])
        with patch.object(obj.detector, 'clear', side_effect=RuntimeError('clear failed')):
            with self.assertRaisesRegex(RuntimeError, 'clear failed'):
                obj.finished(['a'])
        self.assertIn('a', obj.completed)
        self.assertIn('a', obj.states)
        obj.finished(['a'])
        self.assertEqual(obj.completed, set())
        self.assertEqual(obj.states, {})
        self.assertFalse(obj.snapshot()['valid'])

    def test_terminal_tracking_retains_no_historical_ids(self):
        obj, detector = session()
        for key in range(2000):
            obj.register([key])
            if key % 2:
                obj.sampled([key], [[1]])
            obj.finish([key])
            obj.finished([key])
        self.assertEqual(obj.completed, set())
        self.assertEqual(obj.rows, {})
        self.assertEqual(obj.incarnations, {})
        self.assertEqual(obj.states, {})
        self.assertEqual(obj.retired_requests, 2000)
        self.assertEqual(detector.cleared, 1000)


if __name__ == '__main__':
    unittest.main()
