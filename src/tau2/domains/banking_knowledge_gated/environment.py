"""Compatibility shim: `GatedEnvironment`, `get_environment`, `get_tasks` live in
`epistemic_runtime.adapters.tau2.environment` (the in-harness mount).

This domain package stays registered under the name `banking_knowledge_gated`
so tau2's registry / evaluator / runner hunks are unchanged.
"""
import sys

from epistemic_runtime.adapters.tau2 import environment as _env

sys.modules[__name__] = _env
