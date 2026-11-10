"""Adapter failure tests with substitutes, NOT native equivalence evidence."""
import unittest
from unittest.mock import patch
from test_adapter_unit import FakeBackend, FakeDetector, session


class PartialBackend(FakeBackend):
    failure = ValueError

    def feed_events(self, states, token_lists):
        events = []
        for slot, (state, ids) in enumerate(zip(states, token_lists)):
            for token in ids:
                if token == -1:
                    raise self.failure('bad token')
                self.det.feed(state, [token])
                if state.reason:
                    events.append((slot, state.fired_at, state.reason))
                    break
        return events

    def progress(self, state):
        return state.received, state.t, state.fired_at, state.reason


class FailureTests(unittest.TestCase):
    def make(self, **kwargs):
        return session(factory=PartialBackend, detector=FakeDetector(alert_at=64), **kwargs)[0]

    def test_partial_batch_preserves_first_alert_and_prefix(self):
        s = self.make()
        with self.assertRaisesRegex(ValueError, 'bad token'):
            s.sampled(['a', 'b', 'c'], [[64] * 65, [64] * 3 + [-1], [64]])
        self.assertEqual(s.received, {'a': 64, 'b': 3, 'c': 0})
        self.assertEqual(s.computed_tokens, {'a': 64, 'b': 0, 'c': 0})
        self.assertEqual(s.events['a'], (64, 'fake-rule', False))
        self.assertEqual(s.rows['a']['alert_at'], 64)
        self.assertEqual(s.snapshot()['native_accepted_tokens'], 67)
        self.assertFalse(s.snapshot()['valid'])
        for key in ['a', 'b', 'c']:
            self.assertFalse(s.inspect([key])[key]['valid'])
            before = dict(s.rows[key])
            with self.assertRaises(RuntimeError):
                s.sampled([key], [[64]])
            with self.assertRaises(RuntimeError):
                s.finish([key])
            self.assertEqual(before, s.rows[key])

    def test_single_prefix_failure_can_retire_and_reuse(self):
        s = self.make(retain_retired_rows=True)
        generation = s.register(['a'])
        with self.assertRaises(ValueError):
            s.sampled(['a'], [[64, -1]], generation)
        self.assertEqual(s.received['a'], 1)
        s.finished(['a'], generation)
        self.assertFalse(s.retired_rows[0][2]['valid'])
        new_generation = s.register(['a'])
        s.sampled(['a'], [[64] * 64], new_generation)
        s.finished(['a'], generation)  # late callback cannot clear replacement
        self.assertEqual(s.rows['a']['alert_at'], 64)
        self.assertNotIn('a', s.failed)
        s.finished(['a'], new_generation)
        self.assertFalse(s.snapshot()['valid'])  # old error cannot be laundered
        self.assertEqual(s.snapshot()['failed_operations'], 1)

    def test_unrelated_generation_can_continue(self):
        s = self.make()
        with self.assertRaises(ValueError):
            s.sampled(['a'], [[-1]])
        s.sampled(['b'], [[64] * 64])
        self.assertEqual(s.events['b'][0], 64)

    def test_missing_receipt_still_fails_closed(self):
        s = self.make()
        s.detector.progress = None
        with self.assertRaises(ValueError):
            s.sampled(['a'], [[64, -1]])
        self.assertEqual(s.received['a'], 0)  # known-before-error, not fabricated
        self.assertFalse(s.snapshot()['valid'])
        with self.assertRaises(RuntimeError):
            s.sampled(['a'], [[64]])

    def test_poisoned_receipt_is_not_reported_as_valid(self):
        s = self.make()
        s.detector.progress = lambda state: None
        with self.assertRaises(ValueError):
            s.sampled(['a'], [[64, -1]])
        self.assertFalse(s.inspect(['a'])['a']['valid'])

    def test_receipt_failure_preserves_original_exception(self):
        s = self.make()
        def bad_receipt(state):
            raise MemoryError('receipt allocation')
        s.detector.progress = bad_receipt
        with self.assertRaisesRegex(ValueError, 'bad token'):
            s.sampled(['a'], [[-1]])
        with self.assertRaises(RuntimeError):
            s.finish(['a'])

    def test_recovery_bookkeeping_failure_blocks_whole_session(self):
        s = self.make()
        with patch.object(s, '_record_failure', side_effect=MemoryError):
            with self.assertRaisesRegex(ValueError, 'bad token'):
                s.sampled(['a'], [[-1]])
        self.assertTrue(s.recovery_incomplete)
        self.assertFalse(s.snapshot()['valid'])
        with self.assertRaises(RuntimeError):
            s.sampled(['unrelated'], [[64]])

    def test_interrupt_preserves_partial_alert(self):
        s = self.make()
        s.detector.failure = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            s.sampled(['a', 'b'], [[64] * 64, [-1]])
        self.assertEqual(s.events['a'][0], 64)
        self.assertFalse(s.snapshot()['valid'])

    def test_finish_result_conversion_failure_retains_tail_alert(self):
        s, _ = session(factory=PartialBackend,
                       detector=FakeDetector(alert_at=3, tail_alert=True))
        s.sampled(['a'], [[64] * 3])
        finish = s.detector.finish
        def bad_finish(state):
            finish(state)
            raise MemoryError('result allocation')
        s.detector.finish = bad_finish
        with self.assertRaises(MemoryError):
            s.finish(['a'])
        self.assertEqual(s.events['a'], (3, 'fake-tail', True))
        self.assertEqual(s.computed_tokens['a'], 3)

    def test_allocation_failure_at_admission_blocks_retry(self):
        s = self.make()
        with patch.object(s.detector, 'start', side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                s.sampled(['a'], [[64]])
        with self.assertRaises(RuntimeError):
            s.sampled(['a'], [[64]])
        s.finished(['a'])
        self.assertFalse(s.states)

    def test_failure_retirement_metadata_is_bounded(self):
        s = self.make()
        for index in range(200):
            with self.assertRaises(ValueError):
                s.sampled([str(index)], [[-1]])
            s.finished([str(index)])
        self.assertFalse(s.failed)
        self.assertFalse(s.states)
        self.assertFalse(s.rows)
        self.assertFalse(s.incarnations)
        self.assertEqual(s.snapshot()['failed_operations'], 200)

    def test_later_admission_failure_invalidates_prepared_prefix(self):
        s = self.make()
        start = s.detector.start
        def failing_start(key):
            if key == 'b':
                raise MemoryError('admission')
            return start(key)
        s.detector.start = failing_start
        with self.assertRaises(MemoryError):
            s.sampled(['a', 'b', 'c'], [[64], [64], [64]])
        self.assertEqual(s.received['a'], 0)  # prepared, never submitted
        self.assertEqual(s.failed, {'a', 'b', 'c'})
        for key in ('a', 'b', 'c'):
            with self.assertRaises(RuntimeError):
                s.sampled([key], [[64]])
        s.finished(['a', 'b', 'c'])
        self.assertFalse(s.failed)
        self.assertFalse(s.states)

    def test_lazy_result_exception_recovers_committed_alert(self):
        s = self.make()
        feed = s.detector.feed_events
        def lazy_events(states, tokens):
            events = feed(states, tokens)
            def iterate():
                yield from events
                raise ValueError('lazy result failed')
            return iterate()
        s.detector.feed_events = lazy_events
        with self.assertRaisesRegex(ValueError, 'lazy result failed'):
            s.sampled(['a', 'b'], [[64] * 64, [64]])
        self.assertEqual(s.events['a'], (64, 'fake-rule', False))
        self.assertEqual(s.received, {'a': 64, 'b': 1})
        self.assertEqual(s.rows['a']['scored'], 64)  # not double counted
        self.assertFalse(s.snapshot()['valid'])

    def test_invalid_single_event_cannot_allow_silent_retry(self):
        s = self.make()
        feed = s.detector.feed_events
        def bad_slot(states, tokens):
            feed(states, tokens)
            return [(1, 64, 'fake-rule')]
        s.detector.feed_events = bad_slot
        with self.assertRaises(AssertionError):
            s.sampled(['a'], [[64] * 64])
        self.assertEqual(s.events['a'], (64, 'fake-rule', False))
        with self.assertRaises(RuntimeError):
            s.sampled(['a'], [[64]])

    def test_cleanup_error_keeps_state_and_blocks_reuse(self):
        s = self.make()
        s.sampled(['a'], [[64]])
        state = s.states['a']
        with patch.object(s.detector, 'clear', side_effect=ValueError('clear failed')):
            with self.assertRaisesRegex(ValueError, 'clear failed'):
                s.finished(['a'])
        self.assertIs(s.states['a'], state)
        with self.assertRaises(RuntimeError):
            s.sampled(['a'], [[64]])
        s.finished(['a'])
        self.assertFalse(s.states)
        self.assertFalse(s.failed)

    def test_preflight_error_does_not_invalidate_unmodified_generation(self):
        s = self.make()
        generation = s.register(['a'])
        with self.assertRaises(ValueError):
            s.sampled(['a', 'a'], [[64], [64]])
        self.assertTrue(s.snapshot()['valid'])
        s.sampled(['a'], [[64]], generation)

    def test_success_does_not_query_receipts(self):
        s = self.make()
        with patch.object(s.detector, 'progress', side_effect=AssertionError):
            s.sampled(['a', 'b'], [[64], [64]])
            s.sampled(['a'], [[64] * 63])
            s.finish(['b'])
        self.assertTrue(s.snapshot()['valid'])

    def test_receipt_repairs_partially_written_python_accounting(self):
        class FailOnce(dict):
            def __setitem__(self, key, value):
                if not getattr(self, 'raised', False):
                    self.raised = True
                    raise MemoryError('bookkeeping allocation')
                super().__setitem__(key, value)

        for target in ('received', 'events'):
            s = self.make()
            s.register(['a'])
            setattr(s, target, FailOnce(getattr(s, target)))
            with self.assertRaisesRegex(MemoryError, 'bookkeeping allocation'):
                s.sampled(['a'], [[64] * 64])
            self.assertEqual(s.rows['a']['scored'], 64)
            self.assertEqual(s.received['a'], 64)
            self.assertEqual(s.events['a'], (64, 'fake-rule', False))
            self.assertFalse(s.snapshot()['valid'])


if __name__ == '__main__':
    unittest.main()
