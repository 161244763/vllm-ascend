from setuptools import Extension, find_packages, setup
import pybind11
import os

setup(
    name='m4-cpp-detector', version='0.1.0',
    description='M4 native repetition detector and batch adapter',
    classifiers=['Private :: Do Not Upload'],
    python_requires='>=3.10',
    install_requires=['numpy>=1.24'],
    py_modules=['m4_cpp', 'perf_session'],
    packages=find_packages(where='vendor/m4/src', include=['v35_detector', 'v35_detector.*']),
    package_dir={'v35_detector': 'vendor/m4/src/v35_detector'},
    package_data={'v35_detector': ['configs/*.json', 'tables/*.npz']},
    include_package_data=True,
    ext_modules=[Extension('_m4_native', ['m4_native.cpp'],
        depends=['position_queue.h'],
        include_dirs=[pybind11.get_include()], language='c++',
        define_macros=[('M4_TEST_FAULTS', '1')] if os.environ.get('M4_TEST_FAULTS') == '1' else [],
        extra_compile_args=['-O3', '-std=c++17', '-fno-fast-math'])],
)
