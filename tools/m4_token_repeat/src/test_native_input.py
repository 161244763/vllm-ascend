"""Actual extension boundary tests; not mock-based local validation."""
import sys
import unittest

import numpy as np
from _m4_native import Detector


class NativeInputTests(unittest.TestCase):
    @staticmethod
    def tables():
        return [np.array([0, 1, -1], dtype=np.int32),
                np.array([1, 1, 0], dtype=np.int32),
                *[np.zeros(3, dtype=np.int32) for _ in range(4)]]

    def test_noninteger_or_wide_tables_rejected_in_every_slot(self):
        for index in range(6):
            for dtype in [np.float64, np.float32, np.int64, np.uint32, np.int16, bool]:
                with self.subTest(index=index, dtype=dtype):
                    tables = self.tables()
                    tables[index] = tables[index].astype(dtype)
                    with self.assertRaises((ValueError, TypeError)):
                        Detector(*tables, -1)

    def test_large_integer_and_fraction_not_truncated(self):
        for value in [np.array([0, 2**32 + 1, -1], dtype=np.int64),
                      np.array([0, 1.5, -1], dtype=np.float64)]:
            tables = self.tables()
            tables[0] = value
            with self.assertRaises((ValueError, TypeError)):
                Detector(*tables, -1)

    def test_byteswap_and_strided_tables_rejected(self):
        for index in range(6):
            tables = self.tables()
            for replacement in [tables[index].astype('>i4' if sys.byteorder == 'little' else '<i4'),
                                np.repeat(tables[index], 2)[::2]]:
                with self.subTest(index=index, dtype=replacement.dtype, strides=replacement.strides):
                    inputs = tables.copy()
                    inputs[index] = replacement
                    with self.assertRaises((ValueError, TypeError)):
                        Detector(*inputs, -1)

    def test_lists_do_not_bypass_preflight(self):
        tables = self.tables()
        with self.assertRaises(TypeError):
            Detector(*[table.tolist() for table in tables], -1)

    def test_unaligned_arrays_copy_and_do_not_alias(self):
        tables = []
        for table in self.tables():
            value = np.ndarray(table.shape, dtype=np.int32,
                               buffer=bytearray(table.nbytes + 1), offset=1)
            value[:] = table
            self.assertFalse(value.flags.aligned)
            tables.append(value)
        detector = Detector(*tables, -1)
        # Mutating caller arrays must not mutate the detector's owned tables.
        tables[0][:] = -1
        state = detector.start()
        detector.feed(state, [0, 1, 2])
        detector.finish(state)
        self.assertEqual((state.t, state.nx, state.fired_at), (3, 2, -1))

    def test_readonly_arrays_supported(self):
        tables = self.tables()
        for table in tables:
            table.flags.writeable = False
        detector = Detector(*tables, -1)
        state = detector.start()
        detector.feed(state, [0, 1, 2])
        detector.finish(state)
        self.assertEqual((state.t, state.nx), (3, 2))

    def test_reentrant_finish_without_alert_emits_no_event(self):
        detector = Detector(*self.tables(), -1)
        state = detector.start()

        def stream():
            yield 0
            detector.finish(state)
            yield 1
            raise AssertionError('unread suffix consumed')

        self.assertEqual(detector.feed_events([state], [stream()]), [])
        self.assertTrue(state.done)
        self.assertEqual((state.t, state.fired_at, state.reason), (1, -1, -1))


if __name__ == '__main__':
    unittest.main()
