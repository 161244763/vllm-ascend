#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production Store -> manager CPU batch -> token_repeat lifecycle regressions."""
import gc
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.observability.runtime_guard.detector.manager import DetectorManager
from vllm_ascend.observability.runtime_guard.detector.token_repeat import TokenRepeatDetector
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore
from vllm_ascend.observability.runtime_config.schema import validate_detector_section


class Backend:
    def __init__(self):
        self.calls = []
        self.fail = False

    def start(self):
        return SimpleNamespace(received=0, fired_at=-1, reason='')

    def feed_events(self, states, rows):
        self.calls.append(rows)
        events = []
        for slot, (state, row) in enumerate(zip(states, rows)):
            state.received += len(row)
            if row and row[-1] == 9:
                state.fired_at, state.reason = state.received, 'test reason'
                events.append((slot, state.fired_at, state.reason))
            if self.fail:
                raise ValueError('partial consumption')
        return events

    def finish(self, state):
        return SimpleNamespace(is_ill=True, fired_at=state.received, reason='tail')

    def progress(self, state):
        return state.received, state.received, state.fired_at, state.reason


class ProductionM4Tests(unittest.TestCase):
    def setUp(self):
        RequestGuardStore.reset_for_tests()
        self.backend = Backend()
        self.loader = patch('vllm_ascend.observability.runtime_guard.detector.m4_backend.load_m4', return_value=self.backend)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.section = dict(TokenRepeatDetector.schema.defaults_dict(), enabled=True, algorithm='m4')
        self.config = SimpleNamespace(detector_section=lambda _: self.section, hot_reload_enabled=False,
                                      needs_cumulative_io=lambda: True, any_detector_enabled=lambda: True)
        self.det = TokenRepeatDetector(runtime_config=self.config)
        self.manager = object.__new__(DetectorManager)
        self.manager._runner = None
        self.manager._runtime_config = self.config
        self.manager._detection_gate = None
        self.manager._token_repeat_det = self.det
        self.manager._registry = [self.det]
        self.manager._logits_finite_det = MagicMock()
        self.manager._logits_finite_det.check_deferred.return_value = []
        self.addCleanup(RequestGuardStore.reset_for_tests)

    def sampled(self, keys, rows):
        _, snap = self.manager.after_sample_hot_path(rows, keys)
        return self.manager.run_after_sample_cpu(snap)

    def test_batch_sparse_alerts_and_cumulative_dedup(self):
        alerts = self.sampled(['a', 'b'], [[1, 9], [2]])
        self.assertEqual([(a.req_id, a.detail['fired_at'], a.detail['reason']) for a in alerts], [('a', 2, 'test reason')])
        self.assertEqual(self.backend.calls, [[[1, 9], [2]]])
        self.det.check_all(None, ['a', 'b'])
        self.assertEqual(len(self.backend.calls), 1)
        self.sampled(['a', 'b'], [[3], [4]])
        self.assertEqual(self.backend.calls[-1], [[4]])

    def test_finish_consumes_unread_suffix_and_preserves_tail_incident(self):
        RequestIoSnapshotManager.get().append_batch(['a'], [[1, 2, 3]])
        alerts = self.manager.finish_requests(['a'])
        self.assertEqual(alerts[0].detail, dict(algorithm='m4', fired_at=3, reason='tail', finish_only=True))
        self.assertEqual(self.det._consumed_len['a'], 3)
        self.assertEqual(self.manager.finish_requests(['a']), [])

    def test_cancel_is_cleanup_and_id_reuse_is_fresh(self):
        self.sampled(['a'], [[1]])
        RequestGuardStore.get().clear('a', detectors=self.manager)
        self.assertFalse(self.det._m4_states)
        self.assertEqual(self.sampled(['a'], [[9]])[0].detail['fired_at'], 1)

    def test_disable_reenable_rejects_queued_old_snapshot(self):
        _, old = self.manager.after_sample_hot_path([[1]], ['a'])
        self.section['enabled'] = False
        self.det.refresh_from_config()
        self.assertFalse(self.det._m4_states)
        RequestGuardStore.get().reset_detect_io_keep_prompts()
        self.section['enabled'] = True
        self.det.refresh_from_config()
        self.assertEqual(self.manager.run_after_sample_cpu(old), [])
        self.assertFalse(self.backend.calls)
        self.sampled(['a'], [[9]])
        self.assertEqual(self.backend.calls, [[[9]]])

    def test_old_snapshot_cannot_consume_reused_id(self):
        _, old = self.manager.after_sample_hot_path([[1]], ['a'])
        RequestGuardStore.get().clear('a', detectors=self.manager)
        self.sampled(['a'], [[2]])
        self.manager.run_after_sample_cpu(old)
        self.assertEqual(self.backend.calls, [[[2]]])

    def test_partial_exception_keeps_committed_alert_and_never_replays(self):
        self.backend.fail = True
        alerts = self.sampled(['a', 'b'], [[9], [2]])
        self.assertEqual(alerts[0].detail['fired_at'], 1)
        self.assertEqual(self.det._m4_failed, {'a', 'b'})
        self.det.check_all(None, ['a', 'b'])
        self.assertEqual(len(self.backend.calls), 1)
        self.manager.finish_requests(['a', 'b'])
        self.assertEqual(len(self.backend.calls), 1)

    def test_buffer_shrink_restarts_alert_state(self):
        self.sampled(['a'], [[1, 9]])
        RequestGuardStore.get().reset_detect_io_keep_prompts()
        alerts = self.sampled(['a'], [[9]])
        self.assertEqual(alerts[0].detail['fired_at'], 1)

    def test_config_rejects_silent_window_knob_override(self):
        self.section['window'] = 99
        with self.assertRaisesRegex(ValueError, 'only supported'):
            self.det.refresh_from_config()

    def test_schema_and_old_default(self):
        self.assertEqual(TokenRepeatDetector.schema.defaults_dict()['algorithm'], 'window')
        with self.assertRaises(ValueError):
            validate_detector_section(TokenRepeatDetector.schema, {'algorithm': 123})

    def test_resource_failure_is_explicit(self):
        self.loader.stop()
        from vllm_ascend.observability.runtime_guard.detector.m4_backend import load_m4
        with self.assertRaisesRegex(ValueError, 'requires'):
            load_m4(None, '', '', -1)

    def test_repeated_disable_releases_states(self):
        for _ in range(1000):
            self.sampled(['a'], [[1]])
            self.det.on_disabled()
            RequestGuardStore.get().reset_detect_io_keep_prompts()
            self.assertFalse(self.det._m4_states)
            self.assertFalse(self.det._consumed_len)
        gc.collect()
    def test_soft_purge_keeps_prompt_and_clears_output(self):
        store = RequestGuardStore.get()
        state = store.get_or_create('a')
        state.prompt_token_ids = [42]
        self.sampled(['a'], [[1, 2]])
        store.reset_detect_io_keep_prompts()
        self.manager.disable_all()
        self.assertEqual(state.prompt_token_ids, [42])
        self.assertFalse(state.output_token_ids)
        self.assertFalse(self.det._m4_states)

    def test_old_completion_does_not_unpin_reused_id(self):
        store = RequestGuardStore.get()
        old = store.get_or_create('a')
        store.add_cpu_jobs(['a'])
        store.clear('a', detectors=self.manager)
        new = store.get_or_create('a')
        store.add_cpu_jobs(['a'])
        store.finish_cpu_jobs(['a'], expected_states={'a': old})
        self.assertEqual(new.cpu_jobs, 1)

    def test_processor_reap_routes_tail_before_cleanup(self):
        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            self.skipTest("full vLLM runtime unavailable")
        from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor
        processor = object.__new__(RuntimeGuardProcessor)
        processor.detectors = self.manager
        processor.wave_tracker = MagicMock()
        processor._current_wave = lambda: 1
        routed = []
        def handle(alert, **kwargs):
            self.assertIsNotNone(RequestGuardStore.get().get_state(alert.req_id))
            routed.append(alert)
        processor._handle_alert = handle
        self.sampled(['a'], [[1, 2, 3]])
        RequestGuardStore.get().mark_finished(['a'], wave=0)
        processor._reap_finished_requests()
        self.assertEqual(routed[0].detail['reason'], 'tail')
        self.assertIsNone(RequestGuardStore.get().get_state('a'))
        self.assertFalse(self.det._m4_states)

    def test_window_cleanup_compatibility(self):
        self.section.update(algorithm="window", min_tokens=0, repeat_sum_threshold=0)
        self.det.refresh_from_config()
        self.sampled(['a'], [[1, 1]])
        self.assertTrue(self.det._states)
        RequestGuardStore.get().clear('a', detectors=self.manager)
        self.assertFalse(self.det._states)
        self.assertFalse(self.det._consumed_len)

    def test_old_cleanup_does_not_clear_new_native_state(self):
        self.sampled(['a'], [[1]])
        old = RequestGuardStore.get().get_state('a')
        RequestGuardStore.get().clear('a', detectors=self.manager)
        self.sampled(['a'], [[2]])
        native = self.det._m4_states['a']
        self.manager.clear_finished('a', expected_state=old)
        self.assertIs(self.det._m4_states['a'], native)

    def test_finish_exception_keeps_earlier_tail_alerts(self):
        self.sampled(['a', 'b'], [[1], [1, 2]])
        finish = self.backend.finish
        def finish_one(state):
            if state.received == 2:
                raise ValueError('finish failed')
            return finish(state)
        self.backend.finish = finish_one
        alerts = self.manager.finish_requests(['a', 'b'])
        self.assertEqual([a.req_id for a in alerts], ['a'])
        self.assertIn('b', self.det._m4_failed)


class ReferenceM4Tests(unittest.TestCase):
    setUp = ProductionM4Tests.setUp
    sampled = ProductionM4Tests.sampled
    """Real reference boundaries; optional native production differential."""
    def test_reference_stream_matches_production_chunking(self):
        self.compare_production(native=False)

    def test_native_matches_reference_positions_reasons_and_finish(self):
        try:
            import m4_cpp
        except ImportError:
            self.skipTest('independent native extension not built')
        self.compare_production(native=True)

    def compare_production(self, native):
        import tempfile
        from pathlib import Path
        import numpy as np
        try:
            from v35_detector import RepDetector, CONFIG_M4
        except ImportError:
            self.skipTest('install independent M4 package to run reference differential')

        class ReferenceBatch:
            def __init__(self, detector):
                self.detector = detector
            def start(self):
                return self.detector.start()
            def feed_events(self, states, rows):
                events = []
                for slot, (state, row) in enumerate(zip(states, rows)):
                    result = self.detector.feed(state, row)
                    if result is not None and result.is_ill:
                        events.append((slot, result.fired_at, result.reason))
                return events
            def finish(self, state):
                return self.detector.finish(state)

        with tempfile.TemporaryDirectory() as tmp:
            table = str(Path(tmp) / 'synthetic.npz')
            size = 256
            np.savez(table, norm=np.arange(size, dtype=np.int32), letter=np.ones(size, bool),
                     chars=np.ones(size, np.int16), ws=np.zeros(size, np.int16),
                     run_char=np.full(size, -1, np.int32), marker=np.zeros(size, bool),
                     send=np.zeros(size, bool))
            for length in (0, 1, 63, 64, 65, 127, 128, 129, 513, 4097):
                for width in (1, 63, 64, 65, 257):
                    self.det.on_disabled()
                    RequestGuardStore.reset_for_tests()
                    reference = RepDetector(table, CONFIG_M4, think_end=None)
                    state = reference.start()
                    if native:
                        from m4_cpp import RepDetectorCPP
                        self.det._m4 = RepDetectorCPP(table, think_end=None)
                    else:
                        self.det._m4 = ReferenceBatch(RepDetector(table, CONFIG_M4, think_end=None))
                    tokens = [1, 2, 3] * (length // 3) + [1] * (length % 3)
                    expected = None
                    actual = []
                    for offset in range(0, length, width):
                        row = tokens[offset:offset + width]
                        result = reference.feed(state, row)
                        if expected is None and result is not None and result.is_ill:
                            expected = (result.fired_at, result.reason, False)
                        actual.extend(self.sampled(['diff'], [row]))
                    result = reference.finish(state)
                    if expected is None and result.is_ill:
                        expected = (result.fired_at, result.reason, True)
                    actual.extend(self.manager.finish_requests(['diff']))
                    got = [(a.detail['fired_at'], a.detail['reason'], a.detail['finish_only']) for a in actual]
                    self.assertEqual(got, [] if expected is None else [expected], (length, width, native))

class ResourceValidationTests(unittest.TestCase):
    def test_tokenizer_fingerprint_and_served_identity(self):
        import tempfile
        import hashlib
        import sys
        from pathlib import Path
        from types import ModuleType
        import numpy as np
        from vllm_ascend.observability.runtime_guard.detector.m4_backend import load_m4
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'tokenizer.json'
            source.write_text('{}')
            table = str(Path(tmp) / 'm4.npz')
            runner = SimpleNamespace(model_config=SimpleNamespace(tokenizer=tmp))
            np.savez(table, norm=[1])
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_m4(runner, table, str(source), -1)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            np.savez(table, tokenizer_sha256=np.array(digest))
            fake = ModuleType('m4_cpp')
            fake.RepDetectorCPP = MagicMock(return_value='loaded')
            with patch.dict(sys.modules, {'m4_cpp': fake}):
                self.assertEqual(load_m4(runner, table, str(source), -1), 'loaded')
                fake.RepDetectorCPP.assert_called_once_with(table, think_end=None)
            other = Path(tmp) / 'other.json'
            other.write_text('{"different": true}')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_m4(runner, table, str(other), -1)
            with self.assertRaisesRegex(ValueError, 'think_end'):
                load_m4(runner, table, str(source), True)
