"""M4 streaming repetition detector reference implementation."""
import os

from .stream import RepDetector, Result

__version__ = '0.5.0'
CONFIG_M4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'm4.json')

__all__ = ['RepDetector', 'Result', 'CONFIG_M4', '__version__']
