"""Backend abstraction (numeric vs. symbolic)."""

import math
import sys

import numpy as np


class _NumpyBackend:
    """Default numeric backend using math and numpy."""
    cos   = staticmethod(math.cos)
    sin   = staticmethod(math.sin)
    sqrt  = staticmethod(math.sqrt)
    atan2 = staticmethod(math.atan2)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def array(data):
        return np.array(data, dtype=float)


class _SympyBackend:
    """Symbolic backend using sympy. Imported lazily."""

    @staticmethod
    def _sp():
        try:
            import sympy
            return sympy
        except ImportError:
            sys.exit("sympy is required for --symbolic: pip install sympy")

    @staticmethod
    def cos(x):
        return _SympyBackend._sp().cos(x)

    @staticmethod
    def sin(x):
        return _SympyBackend._sp().sin(x)

    @staticmethod
    def sqrt(x):
        return _SympyBackend._sp().sqrt(x)

    @staticmethod
    def atan2(y, x):
        return _SympyBackend._sp().atan2(y, x)

    @staticmethod
    def zeros(shape):
        sp = _SympyBackend._sp()
        return sp.zeros(*shape)

    @staticmethod
    def array(data):
        sp = _SympyBackend._sp()
        return sp.Matrix(data)


NumpyBackend  = _NumpyBackend()
SympyBackend  = _SympyBackend()
