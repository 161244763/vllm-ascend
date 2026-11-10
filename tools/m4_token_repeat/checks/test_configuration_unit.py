"""Constructor-only checks. Native capture is a stub, not native equivalence."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from v35_detector import CONFIG_M4
from v35_detector.vocab import THINK_END, load

from perf_session import CPUBackend


SRC = Path(__file__).resolve().parents[1] / 'src'


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.detector = Mock()
        self.module = patch.dict(sys.modules, {
            'm4_cpp': types.SimpleNamespace(RepDetectorCPP=self.detector)})
        self.module.start()
        self.addCleanup(self.module.stop)

    def test_legacy_profiles_keep_qwen3(self):
        for profile in (None, {}, 'm4', {'mode': 'p1'}, {'mode': 'old_cpp'}):
            with self.subTest(profile=profile):
                backend = CPUBackend(profile)
                self.detector.assert_called_with('qwen3')
                self.assertIs(backend.det, self.detector.return_value)

    def test_explicit_profile_forwarded_without_mutation(self):
        profile = dict(corpus=Path('custom.npz'), config=Path('m4.json'), think_end=None,
                       mode='p1')
        before = profile.copy()
        CPUBackend(profile)
        self.detector.assert_called_once_with(Path('custom.npz'), config=Path('m4.json'),
                                             think_end=None)
        self.assertEqual(profile, before)

    def test_named_model_is_not_ignored(self):
        CPUBackend({'corpus': 'glm'})
        self.detector.assert_called_once_with('glm')

    def test_unknown_keys_fail_before_construction(self):
        for profile in ({'cropus': 'glm'}, {'corpus': 'glm', 'threshold': 7}, {1: 2}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                CPUBackend(profile)
        self.detector.assert_not_called()

    def test_invalid_profile_type_fails_before_construction(self):
        for profile in ('glm', [], True, 1):
            with self.subTest(profile=profile), self.assertRaises(TypeError):
                CPUBackend(profile)
        self.detector.assert_not_called()

    def test_invalid_benchmark_mode_fails_before_construction(self):
        with self.assertRaises(ValueError):
            CPUBackend({'mode': 'typo'})
        self.detector.assert_not_called()


class ConstructorTests(unittest.TestCase):
    def setUp(self):
        self.native = Mock()
        spec = importlib.util.spec_from_file_location('candidate_wrapper', SRC / 'm4_cpp.py')
        self.wrapper = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'_m4_native': types.SimpleNamespace(
                Detector=self.native, State=object)}):
            spec.loader.exec_module(self.wrapper)

    def tables(self):
        return {key: np.array([0, 1], dtype=np.int32)
                for key in ('norm', 'chars', 'ws', 'run_char', 'marker', 'send')}

    def construct(self, tables, **kwargs):
        with patch.object(self.wrapper, 'load', return_value=tables):
            return self.wrapper.RepDetectorCPP('custom.npz', think_end=None, **kwargs)

    def test_real_named_tables_preserve_every_value(self):
        for corpus in ('qwen3', 'glm', 'dsv4'):
            with self.subTest(corpus=corpus):
                self.wrapper.RepDetectorCPP(corpus)
                args = self.native.call_args.args
                for index, key in enumerate(('norm', 'chars', 'ws', 'run_char', 'marker', 'send')):
                    np.testing.assert_array_equal(args[index], load(corpus)[key])
                    self.assertEqual(args[index].dtype, np.dtype('int32'))
                    self.assertTrue(args[index].flags.c_contiguous)
                self.assertEqual(args[-1], -1 if THINK_END[corpus] is None else THINK_END[corpus])

    def test_pathlike_config_and_corpus(self):
        with patch.object(self.wrapper, 'load', return_value=self.tables()) as loader:
            self.wrapper.RepDetectorCPP(Path('custom.npz'), Path(CONFIG_M4), think_end=None)
        loader.assert_called_once_with('custom.npz')
        self.assertEqual(self.native.call_args.args[-1], -1)

    def test_custom_auto_rejected_before_table_loading(self):
        with patch.object(self.wrapper, 'load') as loader, self.assertRaises(ValueError):
            self.wrapper.RepDetectorCPP('custom.npz')
        loader.assert_not_called()
        self.native.assert_not_called()

    def test_integer_think_boundary_preserved(self):
        for value in (0, 151668, np.int64(123), 2**63 - 1):
            with patch.object(self.wrapper, 'load', return_value=self.tables()):
                self.wrapper.RepDetectorCPP('custom.npz', think_end=value)
            self.assertEqual(self.native.call_args.args[-1], value)

    def test_lossy_think_boundary_rejected_before_load(self):
        for value in (True, np.bool_(False), 1.9, '123', -1, 2**63):
            with self.subTest(value=value), patch.object(self.wrapper, 'load') as loader:
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    self.wrapper.RepDetectorCPP('custom.npz', think_end=value)
                loader.assert_not_called()
        self.native.assert_not_called()

    def test_invalid_table_dimensions_and_lengths(self):
        for value in (np.array([], dtype=np.int32), np.array([[0, 1]]),
                      np.array([0, 1, 2]), np.array(1)):
            tables = self.tables()
            tables['chars'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.construct(tables)
        self.native.assert_not_called()

    def test_narrowing_overflow_and_non_integer_tables_rejected(self):
        for value in (np.array([0, 2**31], dtype=np.int64),
                      np.array([-2**31 - 1, 0], dtype=np.int64),
                      np.array([0, 2**64 - 1], dtype=np.uint64),
                      np.array([0., 1.]), np.array([0, float('nan')]),
                      np.array([0, 1], dtype=object)):
            tables = self.tables()
            tables['chars'] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.construct(tables)
        self.native.assert_not_called()

    def test_int32_edges_and_noncontiguous_values_preserved(self):
        tables = self.tables()
        tables['run_char'] = np.array([-2**31, 0, 2**31-1, 0], dtype=np.int64)[::2]
        self.construct(tables)
        np.testing.assert_array_equal(self.native.call_args.args[3], tables['run_char'])

    def test_other_config_rejected_before_native(self):
        with self.assertRaises(ValueError):
            self.construct(self.tables(), config={'components': []})
        self.native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
