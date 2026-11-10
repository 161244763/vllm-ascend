"""Run this with UBSan as well: the old INT_MAX constructor reports signed UB."""
import unittest
import numpy as np
from _m4_native import Detector


class NativeHardeningTests(unittest.TestCase):
    def test_max_norm_constructor_has_no_signed_overflow(self):
        norm = np.array([np.iinfo(np.int32).max()], dtype=np.int32)
        zeros = np.zeros(1, dtype=np.int32)
        # Constructor only: creating a State for this sparse table would need
        # a 256 MiB bitmap. Allocation policy is not changed by this fix.
        Detector(norm, zeros, zeros, zeros, zeros, zeros, -1)

    def test_all_separator_table(self):
        norm = np.full(1, -1, dtype=np.int32)
        zeros = np.zeros(1, dtype=np.int32)
        det = Detector(norm, zeros, zeros, zeros, zeros, zeros, -1)
        state = det.start()
        det.feed(state, [0] * 65)
        det.finish(state)
        self.assertEqual(state.t, 65)

    def test_sparse_valid_table(self):
        norm = np.array([-1, 4095], dtype=np.int32)
        zeros = np.zeros(2, dtype=np.int32)
        det = Detector(norm, zeros, zeros, zeros, zeros, zeros, -1)
        state = det.start()
        det.feed(state, [0, 1] * 32)
        self.assertEqual(state.t, 64)


if __name__ == '__main__':
    unittest.main()
