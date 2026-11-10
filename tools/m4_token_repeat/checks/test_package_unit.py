"""Structural packaging checks; these do not compile or install a wheel."""
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch


SRC = Path(__file__).resolve().parents[1] / 'src'


class PackageTests(unittest.TestCase):
    def metadata(self, fault=False):
        setup = Mock()
        extension = Mock(side_effect=lambda *args, **kwargs: dict(name=args[0], sources=args[1], **kwargs))
        find = Mock(return_value=['v35_detector'])
        with patch.dict(sys.modules, {
            'setuptools': types.SimpleNamespace(setup=setup, Extension=extension, find_packages=find),
            'pybind11': types.SimpleNamespace(get_include=lambda: '/test/include'),
        }), patch.dict(os.environ, {'M4_TEST_FAULTS': '1' if fault else '0'}):
            runpy.run_path(str(SRC / 'setup.py'), run_name='__main__')
        find.assert_called_once_with(where='vendor/m4/src', include=['v35_detector', 'v35_detector.*'])
        return setup.call_args.kwargs

    def test_runtime_modules_and_vendored_imports_present(self):
        meta = self.metadata()
        self.assertEqual(set(meta['py_modules']), {'m4_cpp', 'perf_session'})
        for name in meta['py_modules']:
            self.assertTrue((SRC / f'{name}.py').is_file())
        base = SRC / meta['package_dir']['v35_detector']
        for name in meta['packages']:
            suffix = name.split('.')[1:]
            self.assertTrue(base.joinpath(*suffix, '__init__.py').is_file())
        for name in ('stream', 'vocab', 'core', 'trees'):
            self.assertTrue((base / f'{name}.py').is_file())

    def test_config_is_packaged_and_external_tables_are_supported(self):
        meta = self.metadata()
        base = SRC / meta['package_dir']['v35_detector']
        included = {p.relative_to(base).as_posix()
                    for pattern in meta['package_data']['v35_detector'] for p in base.glob(pattern)}
        self.assertIn('tables/*.npz', meta['package_data']['v35_detector'])
        from v35_detector import CONFIG_M4
        self.assertIn(Path(CONFIG_M4).relative_to(base).as_posix(), included)

    def test_header_and_build_requirements_present(self):
        meta = self.metadata()
        ext = meta['ext_modules'][0]
        self.assertEqual(ext['depends'], ['position_queue.h'])
        for name in ext['sources'] + ext['depends']:
            self.assertTrue((SRC / name).is_file())
        manifest = (SRC / 'MANIFEST.in').read_text()
        self.assertIn('m4_native.cpp position_queue.h pyproject.toml', manifest)
        self.assertIn('recursive-include vendor/m4/src/v35_detector *.py *.json *.npz', manifest)
        project = (SRC / 'pyproject.toml').read_text()
        for value in ('setuptools', 'wheel', 'pybind11'):
            self.assertIn(value, project)
        self.assertIn('numpy>=1.24', meta['install_requires'])

    def test_production_does_not_enable_fault_injection(self):
        self.assertEqual(self.metadata()['ext_modules'][0]['define_macros'], [])
        self.assertEqual(self.metadata(True)['ext_modules'][0]['define_macros'], [('M4_TEST_FAULTS', '1')])

    def test_private_distribution_not_public_pr_artifact(self):
        self.assertIn('Private :: Do Not Upload', self.metadata()['classifiers'])


if __name__ == '__main__':
    unittest.main()
