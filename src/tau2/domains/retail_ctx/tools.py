"""Toolkit for the retail_ctx domain variant.

LOCAL ADDITION (oracle-gap experiment) — not part of upstream tau2-bench.

Same as `RetailTools`, plus two read-only tools that let the agent interrogate a
context store instead of receiving the policy pre-stuffed into its system
prompt:

    list_policy_documents()            -> metadata only (id/title/source/last_updated)
    read_policy_document(document_id)  -> full text of one document

Both are `ToolType.READ` / `mutates_state=False`, so tau2's environment replay
during evaluation skips them (see `Environment.set_state`) and they cannot
affect the DB-match reward.
"""

import json
from typing import Any, Optional

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail_ctx.utils import get_context_store_path
from tau2.environment.toolkit import ToolType, is_tool

METADATA_FIELDS = ("id", "title", "source", "last_updated")


def load_context_store(path=None) -> list[dict[str, Any]]:
    """Load the context store JSON.

    Accepted shapes:
      - {"documents": [ {...}, ... ]}
      - [ {...}, ... ]

    Each document needs at least `id` and `text`; `title`, `source` and
    `last_updated` are optional metadata (defaulted to "" if absent).
    """
    path = path or get_context_store_path()
    if not path.exists():
        raise FileNotFoundError(
            f"retail_ctx context store not found: {path}. Set CTX_STORE_PATH or "
            f"create the file."
        )
    raw = json.loads(path.read_text())
    documents = raw["documents"] if isinstance(raw, dict) else raw
    normalized = []
    seen = set()
    for doc in documents:
        doc_id = doc["id"]
        if doc_id in seen:
            raise ValueError(f"duplicate document id in context store: {doc_id}")
        seen.add(doc_id)
        normalized.append(
            {
                "id": doc_id,
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "last_updated": doc.get("last_updated", ""),
                "text": doc["text"],
            }
        )
    return normalized


class RetailCtxTools(RetailTools):
    """Retail tools + read-only access to a policy-document context store."""

    def __init__(self, db: RetailDB, context_store: Optional[list[dict]] = None) -> None:
        super().__init__(db)
        self._context_store = (
            context_store if context_store is not None else load_context_store()
        )

    @is_tool(ToolType.READ)
    def list_policy_documents(self) -> str:
        """
        List the company policy documents available in the policy store.

        Returns metadata only (no document text): the document id, its title, the
        source it came from, and the date it was last updated. Use
        read_policy_document with an id from this list to read a document's full
        text.

        Returns:
            A JSON list of objects with keys 'id', 'title', 'source' and 'last_updated'.
        """
        return json.dumps(
            [{k: doc[k] for k in METADATA_FIELDS} for doc in self._context_store],
            indent=2,
        )

    @is_tool(ToolType.READ)
    def read_policy_document(self, document_id: str) -> str:
        """
        Read the full text of one company policy document.

        Args:
            document_id: The id of the document, as returned by list_policy_documents.

        Returns:
            The full text of the policy document.

        Raises:
            ValueError: If no document with that id exists in the policy store.
        """
        for doc in self._context_store:
            if doc["id"] == document_id:
                return doc["text"]
        known = ", ".join(doc["id"] for doc in self._context_store)
        raise ValueError(
            f"Policy document '{document_id}' not found. Available ids: {known}"
        )
