"""Review-only real-native tests; does not modify R9."""
import gc
import unittest
from m4_cpp import RepDetectorCPP
from perf_session import CPUBackend, PerfSession
import test_p1


class ReviewNativeTests(unittest.TestCase):
    def test_existing_batch_exception_contract(self):
        self.assertTrue(test_p1.run_batch_and_errors())

    def test_existing_lifecycle_contract(self):
        self.assertTrue(test_p1.run_lifecycle())

    def test_state_owns_tables_after_detector_destruction(self):
        det = RepDetectorCPP('qwen3')
        state = det.start()
        det.feed(state, [64] * 65)
        del det
        gc.collect()
        self.assertEqual(state.t, 64)
        del state
        gc.collect()

    def test_no_state_for_off_or_zero_tokens(self):
        for session in (PerfSession(None, true_off=True), PerfSession({})):
            gen = session.register(['x'])
            session.sampled(['x'], [[]], gen)
            self.assertFalse(session.states)
            session.finish(['x'], gen)
            session.finished(['x'], gen)
            self.assertEqual(session.retired_requests, 1)
            self.assertFalse(session.rows)

    def test_diagnostic_limit_is_not_input_limit(self):
        session = PerfSession({}, max_generated_tokens=3)
        gen = session.register(['x'])
        session.sampled(['x'], [[64] * 129], gen)
        self.assertEqual(session.rows['x']['generated'], 129)
        session.finish(['x'], gen)
        session.finished(['x'], gen)
        self.assertEqual(dict(session.retired_lengths), {4: 1})


if __name__ == '__main__':
    unittest.main()
