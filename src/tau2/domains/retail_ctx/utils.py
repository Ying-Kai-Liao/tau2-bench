"""Paths for the retail_ctx domain variant.

LOCAL ADDITION (oracle-gap experiment) — not part of upstream tau2-bench.

retail_ctx reuses the retail domain's database and task set verbatim; only the
policy document and the context store are its own. See
`experiments/oracle-gap/journal.md` ("retail_ctx domain variant").
"""

import os
from pathlib import Path

from tau2.domains.retail.utils import (  # noqa: F401  (re-exported for convenience)
    RETAIL_DB_PATH,
    RETAIL_TASK_SET_PATH,
)
from tau2.utils.utils import DATA_DIR

RETAIL_CTX_DATA_DIR = DATA_DIR / "tau2" / "domains" / "retail_ctx"
RETAIL_CTX_POLICY_PATH = RETAIL_CTX_DATA_DIR / "policy.md"
RETAIL_CTX_DEFAULT_STORE_PATH = RETAIL_CTX_DATA_DIR / "context_store.json"

# retail_ctx reuses retail's db + tasks unchanged.
RETAIL_CTX_DB_PATH = RETAIL_DB_PATH
RETAIL_CTX_TASK_SET_PATH = RETAIL_TASK_SET_PATH


def get_context_store_path() -> Path:
    """Resolve the context-store JSON path.

    `CTX_STORE_PATH` (env var) wins so the harness can swap store contents per
    run without touching code; otherwise fall back to the file shipped in the
    (possibly TAU2_DATA_DIR-overridden) data directory.
    """
    override = os.getenv("CTX_STORE_PATH")
    if override:
        return Path(override)
    return RETAIL_CTX_DEFAULT_STORE_PATH
