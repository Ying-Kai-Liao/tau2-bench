"""Environment for the retail_ctx domain variant.

LOCAL ADDITION (oracle-gap experiment) — not part of upstream tau2-bench.

Identical to the retail domain (same DB, same tasks, same tools) except:
  - the policy injected into the agent's system prompt is read from
    `data/tau2/domains/retail_ctx/policy.md` (a short stub that points the agent
    at the context tools) instead of retail's full policy manual;
  - the toolkit is `RetailCtxTools`, which adds `list_policy_documents` and
    `read_policy_document` over a JSON context store (CTX_STORE_PATH).

The retail domain itself is untouched — this is additive registration only.
"""

from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import (
    get_tasks as retail_get_tasks,
)
from tau2.domains.retail.environment import (
    get_tasks_split as retail_get_tasks_split,
)
from tau2.domains.retail_ctx.tools import RetailCtxTools
from tau2.domains.retail_ctx.utils import (
    RETAIL_CTX_DB_PATH,
    RETAIL_CTX_POLICY_PATH,
)
from tau2.environment.environment import Environment


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("retail_ctx domain does not support solo mode")
    if db is None:
        db = RetailDB.load(RETAIL_CTX_DB_PATH)
    tools = RetailCtxTools(db)
    with open(RETAIL_CTX_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="retail_ctx",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """retail_ctx reuses the retail task set verbatim."""
    return retail_get_tasks(task_split_name)


def get_tasks_split() -> dict[str, list[str]]:
    return retail_get_tasks_split()
