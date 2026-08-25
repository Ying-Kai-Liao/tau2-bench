"""Compatibility shim: the gate core lives in the `epistemic_runtime` package.

Install it into this clone's environment (see epistemic-runtime/scripts/setup.sh).
"""
import sys

from epistemic_runtime import gate_core as _core

sys.modules[__name__] = _core
