"""Resource ownership regressions; runs without the native extension."""
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from v35_detector import CONFIG_M4, RepDetector
from v35_detector import vocab


class ResourceTests(unittest.TestCase):
    def setUp(self):
        vocab.load.cache_clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(vocab.load.cache_clear)

    def table(self, index):
        path = Path(self.tmp.name) / f'{index}.npz'
        np.savez(path, norm=np.array([index], dtype=np.int32))
        return str(path)

    def test_cache_hit_preserves_identity(self):
        path = self.table(1)
        first = vocab.load(path)
        self.assertIs(first, vocab.load(path))
        self.assertEqual(vocab.load.cache_info().hits, 1)

    def test_eviction_releases_unreferenced_arrays(self):
        first = vocab.load(self.table(0))
        ref = weakref.ref(first['norm'])
        del first
        for index in range(1, vocab.TABLE_CACHE_SIZE + 1):
            vocab.load(self.table(index))
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(vocab.load.cache_info().currsize, vocab.TABLE_CACHE_SIZE)

    def test_eviction_preserves_live_detector_tables(self):
        det = RepDetector('qwen3', CONFIG_M4)
        norm = det.T['norm']
        original = norm.copy()
        for index in range(vocab.TABLE_CACHE_SIZE):
            vocab.load(self.table(index))
        np.testing.assert_array_equal(norm, original)
        state = det.start()
        det.feed(state, [64] * 65)
        det.finish(state)

    def test_clear_releases_arrays_without_live_owners(self):
        data = vocab.load(self.table(0))
        ref = weakref.ref(data['norm'])
        del data
        vocab.load.cache_clear()
        gc.collect()
        self.assertIsNone(ref())

    def test_npz_closed_on_success_and_error(self):
        class Archive:
            files = ['norm']
            closed = False
            fail = False
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                self.closed = True
            def __getitem__(self, key):
                if self.fail:
                    raise ValueError('bad member')
                return np.array([7], dtype=np.int32)
        for fail in (False, True):
            archive = Archive()
            archive.fail = fail
            with patch.object(vocab.np, 'load', return_value=archive) as loader:
                if fail:
                    with self.assertRaises(ValueError):
                        vocab.load('error.npz')
                else:
                    vocab.load('good.npz')
                loader.assert_called_once_with('error.npz' if fail else 'good.npz', allow_pickle=False)
            self.assertTrue(archive.closed)
        self.assertEqual(vocab.load.cache_info().currsize, 1)

    def test_many_npz_loads_do_not_accumulate_fds(self):
        fdpath = Path('/proc/self/fd')
        if not fdpath.exists():
            self.skipTest('Linux FD observation')
        paths = [self.table(i) for i in range(16)]
        before = len(list(fdpath.iterdir()))
        for _ in range(10):
            for path in paths:
                vocab.load(path)
        self.assertEqual(len(list(fdpath.iterdir())), before)
        self.assertEqual(vocab.load.cache_info().currsize, vocab.TABLE_CACHE_SIZE)

    def test_config_file_closed_on_success_and_parse_error(self):
        config = Path(CONFIG_M4).read_text(encoding='utf-8')
        vocab.load('qwen3')  # Do not mock NumPy's unrelated archive opening.
        for text in (config, '{'):
            source = io.StringIO(text)
            with patch('builtins.open', return_value=source):
                if text == '{':
                    with self.assertRaises(json.JSONDecodeError):
                        RepDetector('qwen3', 'config.json')
                else:
                    RepDetector('qwen3', 'config.json')
            self.assertTrue(source.closed)

    def test_tokenizer_json_closed_on_success_and_parse_error(self):
        tokenizer = types.SimpleNamespace(get_vocab_size=lambda: 1, id_to_token=lambda _: 'A')
        module = types.SimpleNamespace(Tokenizer=types.SimpleNamespace(from_file=lambda _: tokenizer))
        for text in ('{}', '{'):
            source = io.StringIO(text)
            with patch.dict(sys.modules, {'tokenizers': module}), patch('builtins.open', return_value=source):
                if text == '{':
                    with self.assertRaises(json.JSONDecodeError):
                        vocab.build('tokenizer.json')
                else:
                    self.assertEqual(vocab.build('tokenizer.json')['norm'].tolist(), [1])
            self.assertTrue(source.closed)


if __name__ == '__main__':
    unittest.main()
