"""Correct the test-only iinfo.max typo; retain the frozen first-run evidence."""
import numpy as np
from _m4_native import Detector
from test_native_hardening import NativeHardeningTests as OriginalTests


class NativeHardeningTests(OriginalTests):
    def test_max_norm_constructor_has_no_signed_overflow(self):
        norm = np.array([np.iinfo(np.int32).max], dtype=np.int32)
        zeros = np.zeros(1, dtype=np.int32)
        # Do not allocate State's 256 MiB bitmap for this constructor-only case.
        Detector(norm, zeros, zeros, zeros, zeros, zeros, -1)


# unittest must not also discover the imported, deliberately preserved class.
del OriginalTests
