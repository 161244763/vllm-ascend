"""Real native failure tests. Fault tests REQUIRE M4_TEST_FAULTS=1 build.

Never use a fault-enabled binary for timing or shipping. Run in a separate
build directory, then rebuild without that define for differential/perf tests.
"""
import unittest
import numpy as np
from _m4_native import Detector
from m4_cpp import RepDetectorCPP
from perf_session import CPUBackend, PerfSession


def native():
    # 64 equal run-character tokens cross RUN>=500 exactly at raw boundary 64.
    n = np.arange(4, dtype=np.int32)
    z = np.zeros(4, dtype=np.int32)
    return Detector(n, z + 8, z, z, z, z, -1)


def wrapper(det):
    obj = RepDetectorCPP.__new__(RepDetectorCPP)
    obj.native = det
    obj.reasons = ['rule-' + str(i) for i in range(14)]
    return obj


class NativeExceptions(unittest.TestCase):
    def test_invalid_token_retains_valid_prefix_and_direct_continuation(self):
        d = native()
        s = d.start()
        with self.assertRaises(IndexError):
            d.feed(s, [0, 1, -1, 2])
        self.assertFalse(s.failed)
        self.assertEqual(s.received, 2)
        d.feed(s, [2])
        self.assertEqual(s.received, 3)

    def test_iterator_failure_retains_prefix(self):
        d = native()
        s = d.start()
        def tokens():
            yield 0
            raise ValueError('iterator failed')
        with self.assertRaises(ValueError):
            d.feed_events([s], [tokens()])
        self.assertEqual(s.received, 1)
        d.feed_one(s, 0)
        self.assertEqual(s.received, 2)

    def test_unread_suffix_not_validated(self):
        d = native()
        s = d.start()
        events = d.feed_events([s], [[0] * 64 + [object()]])
        self.assertEqual(events, [(0, 64, 2)])
        self.assertEqual(s.received, 64)

    def test_actual_adapter_recovers_earlier_slot_event(self):
        backend = CPUBackend.__new__(CPUBackend)
        backend.det = wrapper(native())
        session = PerfSession({}, factory=lambda _: backend)
        with self.assertRaises(IndexError):
            session.sampled(['a', 'b', 'c'], [[0] * 65, [0, -1], [0]])
        self.assertEqual(session.events['a'], (64, 'rule-2', False))
        self.assertEqual(session.received, {'a': 64, 'b': 1, 'c': 0})
        with self.assertRaises(RuntimeError):
            session.sampled(['b'], [[0]])

    def test_block_fault_poison_and_all_entry_points(self):
        for offset in [0, 1, 63, 64]:
            d = native()
            s = d.start()
            self.assertTrue(hasattr(s, '_test_fail_after_tokens'),
                            'This test requires M4_TEST_FAULTS=1, not a skip')
            s._test_fail_after_tokens = offset
            with self.assertRaises(MemoryError):
                d.feed(s, [0] * 64)
            self.assertTrue(s.failed)
            for action in [lambda: d.feed(s, [0]), lambda: d.feed_one(s, 0),
                           lambda: d.finish(s), lambda: d.feed_batch([s], [[0]]),
                           lambda: d.feed_events([s], [[0]]), s.features,
                           lambda: s.received, lambda: wrapper(d).result(s)]:
                with self.assertRaises(RuntimeError):
                    action()
            self.assertIsNone(wrapper(d).progress(s))

    def test_finish_fault_poison_and_preflight(self):
        d = native()
        s, pristine = d.start(), d.start()
        self.assertTrue(hasattr(s, '_test_fail_after_tokens'))
        d.feed(s, [0] * 3)
        s._test_fail_after_tokens = 1
        with self.assertRaises(MemoryError):
            d.finish(s)
        with self.assertRaises(RuntimeError):
            d.feed_events([pristine, s], [[0], [0]])
        self.assertEqual(pristine.received, 0)

    def test_iterator_reentry_cannot_push_into_poisoned_pending_buffer(self):
        for method in ('feed', 'feed_events'):
            d = native()
            s = d.start()
            self.assertTrue(hasattr(s, '_test_fail_after_tokens'))
            s._test_fail_after_tokens = 1
            def tokens():
                try:
                    d.feed(s, [0] * 64)
                except MemoryError:
                    pass
                yield 0  # outer entry-point check happened before the failure
            with self.assertRaises(RuntimeError):
                if method == 'feed':
                    d.feed(s, tokens())
                else:
                    d.feed_events([s], [tokens()])
            self.assertTrue(s.failed)


if __name__ == '__main__':
    unittest.main()
