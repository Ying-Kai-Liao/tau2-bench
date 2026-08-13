"""Commit-gate v1.1 core logic — LOCAL ADDITION (gate v1). Not part of upstream tau2-bench.

Single source of truth for the gate's mutates-state table derivation,
dispatcher unwrapping, read/write-set extraction, the versioned/coherent
ledger (world_version + freshness invalidation), and the v1.1 spec table /
verdict engine. Design docs: `docs/gate-v1-spec.md` (v1) and this file's own
history (v1.1 extends it — see `experiments/oracle-gap/gate_v1/specs_banking.md`
and `experiments/oracle-gap/gate_v1/validation_report.md` in the research
repo, above the tau2 clone).

Two consumers import from this module, unchanged:

  1. `tau2.domains.banking_knowledge_gated.environment.GatedEnvironment`
     (this package) — the live/replay commit gate registered as the
     `banking_knowledge_gated` domain variant. It has live DB access, so it
     computes PRECISE, record-level write-sets by diffing
     `toolkit.db.model_dump()` before/after every mutating call
     (`diff_write_set`).
  2. `experiments/oracle-gap/gate_v1/shadow_replay.py` (offline, $0, no
     LLM/API calls) — replays already-recorded trajectories (which do NOT
     carry live DB snapshots) through the exact same spec logic. For the
     no-live-DB replay path it falls back to a STATIC, table-level
     write-set approximation (`STATIC_WRITE_TABLE_MAP` / `ALL_TABLES`
     sentinel) — see that module and `validation_report.md` for why this is
     sound (over-invalidation is safe, under-invalidation is not — D3/spec
     item 4). Its golden-check path DOES have live DB access (it constructs
     a real `banking_knowledge` environment) and therefore also uses the
     precise `diff_write_set` path, identical to the live gate.

Everything in sections 1-3 below (tool table extraction, dispatcher
unwrapping, read/write-set extraction, spec table v1.1, stateless
`gate_verdict()`) is deliberately a pure function of `(eff_name, eff_args,
ledger)` — replay-safe by construction: given the identical sequence of
`GateLedger.record_observation()` calls, it derives the identical verdict
sequence live and on replay. Section 4 (`GateLedger`) is the STATEFUL
ledger — world-version counter, mutation log, freshness invalidation, and
(only for the live/replay `GatedEnvironment`) the D4-ii denial-budget
bookkeeping layered on top of the stateless `gate_verdict()`.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Default source paths for tool-table / read-set / write-set extraction.
#
# This file lives at src/tau2/domains/banking_knowledge_gated/gate_core.py;
# banking_knowledge's tools.py/retrieval_mixins.py are siblings one level up.
# Consumers that already compute their own path to the clone (e.g.
# shadow_replay.py, which is outside the clone entirely) may pass their own
# paths to extract_tool_table() instead of using these defaults -- the
# *derivation function* is the single source of truth, not any one path
# constant.
# ---------------------------------------------------------------------------

_DOMAINS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_PY = _DOMAINS_DIR / "banking_knowledge" / "tools.py"
DEFAULT_MIXINS_PY = _DOMAINS_DIR / "banking_knowledge" / "retrieval_mixins.py"


# ---------------------------------------------------------------------------
# 1. Tool mutation table (AST, no import of tau2 needed for this part)
# ---------------------------------------------------------------------------


def _decorator_info(dec: ast.expr) -> Optional[dict]:
    """If `dec` is a call to is_tool(...)/is_discoverable_tool(...), return
    {'discoverable': bool, 'tool_type': str, 'mutates_override': Optional[bool]}.
    Else None.
    """
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if isinstance(func, ast.Name):
        fname = func.id
    elif isinstance(func, ast.Attribute):
        fname = func.attr
    else:
        return None
    if fname not in ("is_tool", "is_discoverable_tool"):
        return None

    tool_type = "READ"  # default per is_tool()/is_discoverable_tool() signature
    if dec.args:
        arg0 = dec.args[0]
        if isinstance(arg0, ast.Attribute):
            tool_type = arg0.attr

    mutates_override = None
    for kw in dec.keywords:
        if kw.arg == "mutates_state" and isinstance(kw.value, ast.Constant):
            mutates_override = kw.value.value

    return {
        "discoverable": fname == "is_discoverable_tool",
        "tool_type": tool_type,
        "mutates_override": mutates_override,
    }


def _iter_decorated_methods(tree: ast.Module):
    """Yield (owner_class_name, FunctionDef/AsyncFunctionDef node, decorator_info)
    for every @is_tool/@is_discoverable_tool decorated method in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        owner = node.name
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in item.decorator_list:
                info = _decorator_info(dec)
                if info is not None:
                    yield owner, item, info
                    break


def extract_tool_table(*paths: Path) -> dict[str, dict]:
    """Parse @is_tool / @is_discoverable_tool decorated methods out of the
    given source files. Returns {tool_name: {mutates_state, tool_type,
    discoverable, owner_class, source_file, lineno}}.
    """
    table: dict[str, dict] = {}
    for path in paths:
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for owner, item, info in _iter_decorated_methods(tree):
            mutates_override = info["mutates_override"]
            tool_type = info["tool_type"]
            mutates_state = (
                mutates_override
                if mutates_override is not None
                else (tool_type == "WRITE")
            )
            if item.name in table:
                # Should not happen in this domain -- flag loudly if it does.
                table[item.name]["_DUPLICATE_DEFINITION"] = True
            table[item.name] = {
                "mutates_state": mutates_state,
                "tool_type": tool_type,
                "discoverable": info["discoverable"],
                "owner_class": owner,
                "source_file": path.name,
                "lineno": item.lineno,
            }
    return table


def build_default_tool_table() -> dict[str, dict]:
    """Convenience wrapper: extract_tool_table() over this domain's default
    (banking_knowledge) tools.py + retrieval_mixins.py source paths."""
    return extract_tool_table(DEFAULT_TOOLS_PY, DEFAULT_MIXINS_PY)


# ---------------------------------------------------------------------------
# 1b. Read-set / static write-table extraction (AST) -- LOCAL ADDITION (gate v1.1)
#
# Read-set: for every decorated tool, the set of DB table names it queries
# via `query_database_tool("<table>", ...)` (literal first-arg string only —
# a dynamic/variable table name can't be resolved statically; those get an
# empty read-set, documented in validation_report.md). Tools that read the
# DB via direct dict access (`self.db.<table>.data[...]`) instead of
# query_database_tool ALSO get an empty read-set here — this is the
# documented edge case anticipated by the v1.1 task spec item 3
# ("Tools with no query_database_tool call... get empty read-sets,
# documented"); see validation_report.md for the concrete instance
# (get_debit_cards_by_account_id_7823).
#
# Static write-table map: for every MUTATING tool, a best-effort table-level
# approximation of what it writes, extracted from literal-table-name calls
# to add_to_db/update_record_in_db/remove_from_db. This is NOT used by the
# live/replay GatedEnvironment (which diffs the real DB — see
# `diff_write_set` below); it exists only as the write-set source for
# shadow_replay.py's b1-trajectory replay, which has no live DB to diff.
# Tools with zero such calls detected (e.g. those that mutate via direct
# `self.db.<table>.data[id][field] = value` assignment, never through
# add_to_db/update_record_in_db) fall back to the ALL_TABLES sentinel at the
# call site (over-invalidation is safe; under-invalidation is not).
# ---------------------------------------------------------------------------

ALL_TABLES = "__ALL_TABLES__"  # sentinel: "assume every table may have changed"

_QUERY_DB_TOOL_CALL_NAME = "query_database_tool"
_WRITE_HELPER_CALL_NAMES = {"add_to_db", "update_record_in_db", "remove_from_db"}


def _literal_str_arg0(call: ast.Call) -> Optional[str]:
    if not call.args:
        return None
    a0 = call.args[0]
    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
        return a0.value
    return None


def _called_name(call: ast.Call) -> Optional[str]:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_read_set_map(*paths: Path) -> dict[str, frozenset[str]]:
    """{tool_name: frozenset(table names read via query_database_tool(...))}.

    Every decorated tool gets an entry (possibly empty frozenset()).
    """
    read_sets: dict[str, set[str]] = {}
    for path in paths:
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for _owner, item, _info in _iter_decorated_methods(tree):
            tables: set[str] = set()
            for node in ast.walk(item):
                if isinstance(node, ast.Call) and _called_name(node) == _QUERY_DB_TOOL_CALL_NAME:
                    table = _literal_str_arg0(node)
                    if table is not None:
                        tables.add(table)
            read_sets[item.name] = tables
    return {name: frozenset(tables) for name, tables in read_sets.items()}


def extract_static_write_table_map(*paths: Path) -> dict[str, frozenset[str]]:
    """{tool_name: frozenset(table names written via add_to_db/
    update_record_in_db/remove_from_db(...))}. Only meaningful for mutating
    tools; a tool absent from (or mapping to an empty frozenset in) this dict
    should fall back to ALL_TABLES at the call site if it is mutating.
    """
    write_sets: dict[str, set[str]] = {}
    for path in paths:
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for _owner, item, _info in _iter_decorated_methods(tree):
            tables: set[str] = set()
            for node in ast.walk(item):
                if isinstance(node, ast.Call) and _called_name(node) in _WRITE_HELPER_CALL_NAMES:
                    table = _literal_str_arg0(node)
                    if table is not None:
                        tables.add(table)
            write_sets[item.name] = tables
    return {name: frozenset(tables) for name, tables in write_sets.items()}


def extract_touches_db_map(*paths: Path) -> dict[str, bool]:
    """{tool_name: True iff the method body references `self.db` anywhere}.

    Refines the ALL_TABLES fallback: a mutating-tagged tool whose body
    never references `self.db` at all (e.g. unlock_discoverable_agent_tool
    / give_discoverable_user_tool -- pure in-memory bookkeeping, no DB
    write of any kind, confirmed by direct reading of tools.py) provably
    has an EMPTY write-set, not an unknown one -- ALL_TABLES would be
    needlessly (and, per the golden sweep, actually incorrectly)
    conservative for these. A tool that DOES reference `self.db` but has
    no add_to_db/update_record_in_db/remove_from_db call detected (e.g.
    tools that mutate via direct `self.db.<table>.data[id][field] = value`
    subscript assignment) still correctly falls back to ALL_TABLES.
    """
    touches: dict[str, bool] = {}
    for path in paths:
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for _owner, item, _info in _iter_decorated_methods(tree):
            found = False
            for node in ast.walk(item):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "db"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    found = True
                    break
            touches[item.name] = found
    return touches


def static_write_set_for(
    eff_name: str,
    tool_table: dict[str, dict],
    write_table_map: dict[str, frozenset[str]],
    touches_db_map: Optional[dict[str, bool]] = None,
):
    """Best-effort table-level write-set for the no-live-DB replay path.
    Returns a dict {table: {"*"}} (one opaque record-id placeholder per
    table -- record-level granularity is unavailable statically), {}
    (empty -- provably no DB access at all, see extract_touches_db_map),
    or the ALL_TABLES sentinel (mutating, touches self.db somewhere, but
    no specific table could be statically determined -- conservative
    fallback, over-invalidation is safe).
    """
    meta = tool_table.get(eff_name)
    is_mutating = bool(meta and meta.get("mutates_state"))
    if not is_mutating:
        return {}
    tables = write_table_map.get(eff_name)
    if tables:
        return {t: {"*"} for t in tables}
    if touches_db_map is not None and not touches_db_map.get(eff_name, True):
        return {}
    return ALL_TABLES


# ---------------------------------------------------------------------------
# 1c. Precise DB diffing (record-level) -- LOCAL ADDITION (gate v1.1)
#
# Used by the live/replay GatedEnvironment (which has `toolkit.db`) and by
# shadow_replay.py's golden-env-execution path (which constructs a real
# `banking_knowledge` Environment and therefore also has live DB access).
#
# Record-id convention: `TransactionalDB.model_dump()` (see
# ToolKitBase.get_db_hash, <clone>/src/tau2/environment/toolkit.py:242-244)
# produces {table_name: {"data": {record_id: {...fields...}}, "notes": str}}
# for every DatabaseTable field (confirmed in data_model.py: every table is
# `Dict[str, Dict[str, Any]]` keyed by an opaque string record id -- e.g.
# credit_card_accounts is keyed by account_id, transaction_disputes by
# dispute_id, verification_history by a generated verification id, etc.).
# Record identity = that outer dict key, whatever domain-specific id string
# it happens to be. This is uniform across all 22 DatabaseTable fields, so
# no table-specific special-casing is needed.
#
# Fallback: a table's `notes` field changing with `data` unchanged (never
# observed in this domain -- no tool writes `notes` -- but defensively
# handled) is recorded as a table-level pseudo-record id "__TABLE_NOTES__",
# which still invalidates any (necessarily table-granularity, see 1b) read
# of that table.
# ---------------------------------------------------------------------------

_NOTES_SENTINEL_RECORD_ID = "__TABLE_NOTES__"


def diff_write_set(before: dict, after: dict) -> dict[str, set[str]]:
    """Diff two `TransactionalDB.model_dump()` dicts. Returns
    {table: {changed_record_id, ...}} for tables with >=1 changed record;
    tables with no change are omitted entirely.
    """
    write_set: dict[str, set[str]] = {}
    tables = set(before.keys()) | set(after.keys())
    for table in tables:
        before_tbl = before.get(table) or {}
        after_tbl = after.get(table) or {}
        before_data = before_tbl.get("data", {}) if isinstance(before_tbl, dict) else {}
        after_data = after_tbl.get("data", {}) if isinstance(after_tbl, dict) else {}

        changed: set[str] = set()
        changed |= set(after_data.keys()) - set(before_data.keys())  # added
        changed |= set(before_data.keys()) - set(after_data.keys())  # removed
        for rid in set(before_data.keys()) & set(after_data.keys()):
            if before_data[rid] != after_data[rid]:
                changed.add(rid)  # modified

        before_notes = before_tbl.get("notes") if isinstance(before_tbl, dict) else None
        after_notes = after_tbl.get("notes") if isinstance(after_tbl, dict) else None
        if before_notes != after_notes and not changed:
            changed.add(_NOTES_SENTINEL_RECORD_ID)

        if changed:
            write_set[table] = changed
    return write_set


# ---------------------------------------------------------------------------
# 2. Dispatcher unwrapping
# ---------------------------------------------------------------------------

# Dispatchers that actually *execute* the inner tool and return its real
# result (tools.py:675 for call_discoverable_agent_tool; the mirror-image
# call_discoverable_user_tool uses the identical json.loads(..., parse_int=
# float) pattern). For these, ledger entries AND gate proposals are keyed on
# the effective (inner) tool name + inner args.
CALL_DISPATCHERS = {
    "call_discoverable_agent_tool": "agent_tool_name",
    "call_discoverable_user_tool": "discoverable_tool_name",
}

# unlock_discoverable_agent_tool / give_discoverable_user_tool are NOT
# execution dispatchers: neither one calls `method(**args_dict)` on the
# inner tool. unlock_* just flips in-memory "unlocked" state and returns a
# tool-description string; give_* validates args against the inner tool's
# signature and stores a "GIVEN" record for the user to later act on -- the
# inner tool's own mutating logic has not run yet in either case. So their
# *effective* tool, for both ledger and gate purposes, is themselves (they
# are registered, mutates_state=True, ToolType.GENERIC tools) -- NOT the
# tool named in their arguments.
NON_EXECUTING_WRAPPERS = {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"}


def unwrap_effective(name: str, args: Optional[dict]) -> tuple[str, dict, Optional[str]]:
    """Return (effective_name, effective_args, malformed_note_or_None)."""
    args = args or {}
    if name not in CALL_DISPATCHERS:
        return name, args, None

    inner_key = CALL_DISPATCHERS[name]
    inner_name = args.get(inner_key)
    inner_args_raw = args.get("arguments", "{}")
    if inner_args_raw is None:
        inner_args_raw = "{}"
    if isinstance(inner_args_raw, dict):
        # Already a dict (shouldn't happen per spec, but be defensive).
        return inner_name, inner_args_raw, None
    try:
        inner_args = json.loads(inner_args_raw, parse_int=float)
    except (json.JSONDecodeError, TypeError) as e:
        return (
            inner_name,
            {},
            f"MALFORMED inner arguments JSON for {name}->{inner_name}: {e!r} raw={inner_args_raw!r}",
        )
    if not isinstance(inner_args, dict):
        return inner_name, {}, (
            f"inner arguments for {name}->{inner_name} did not decode to a dict: {inner_args!r}"
        )
    return inner_name, inner_args, None


def _try_parse_json(raw: Optional[str]) -> Any:
    """Ledger item #6 (v1.1): tool results arrive as strings
    (Environment.to_json_str is a passthrough for str results, which is all
    this domain's tools ever return). Parse into structured JSON where
    possible (e.g. get_debit_cards_by_account_id_7823 returns
    `json.dumps(...)`); fall back to the raw string for the (majority)
    free-text-formatted results. Never raises.
    """
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


# ---------------------------------------------------------------------------
# 3. Spec table v1.1
# ---------------------------------------------------------------------------

# --- Predicates -------------------------------------------------------------

_STATUS_LINE_RE = re.compile(r"^\s*status:\s*(\S+)", re.MULTILINE)

# --- rule: no pending credit-card transaction dispute (053/approve) --------

PENDING_DISPUTE_STATUSES = {"SUBMITTED", "UNDER_REVIEW", "PENDING"}
_NO_DISPUTES_MARKERS = ("No transaction disputes found", "No records found")


def has_pending_dispute(result_content: Optional[str]) -> Optional[bool]:
    """Predicate: does this get_user_dispute_history_7291 result show a
    pending/unresolved dispute? Returns True/False, or None if unparseable
    (conservative -- callers must treat None as deny, never as pass).

    UNCHANGED from v1 (byte-identical body) -- verified per the v1.1 task
    spec item 6 ("Verify existing predicates still work"). Provenance:
    copy-verified against approve_credit_limit_increase_5847's own internal
    eligibility check, tools.py:2206-2218.
    """
    if result_content is None:
        return None
    if any(marker in result_content for marker in _NO_DISPUTES_MARKERS):
        return False
    statuses = _STATUS_LINE_RE.findall(result_content)
    if not statuses:
        return None
    return any(s.strip().upper().rstrip(",") in PENDING_DISPUTE_STATUSES for s in statuses)


def _check_no_pending_dispute(obs: dict, eff_args: dict) -> Optional[bool]:
    pending = has_pending_dispute(obs.get("result"))
    return None if pending is None else (not pending)


# --- rule: no pending/processing/shipped replacement card order (approve, new v1.1) --

PENDING_REPLACEMENT_ORDER_STATUSES = {"PENDING", "PROCESSING", "SHIPPED"}
_NO_ORDERS_MARKERS = (
    "No pending replacement orders found",
    "No records found",
    "No results found",
)


def has_pending_replacement_order(result_content: Optional[str]) -> Optional[bool]:
    """Predicate: does this get_pending_replacement_orders_5765 result show
    a pending/processing/shipped replacement order? Mirrors
    has_pending_dispute's structure exactly.

    Provenance: approve_credit_limit_increase_5847's own Check 2,
    tools.py:2220-2235 (`order.get("status") in ["PENDING","PROCESSING",
    "SHIPPED"]`). order_replacement_credit_card_7291 (the only *tool* that
    creates credit_card_orders records) always writes status="ORDERED"
    (tools.py:1460), which never matches this list -- so this check is
    live only against seed/initialization data, not against
    tool-created records. Confirmed non-dead: the shipped seed DB
    (data/tau2/domains/banking_knowledge/db.json) contains 2
    credit_card_orders records with status "SHIPPED".
    """
    if result_content is None:
        return None
    if any(marker in result_content for marker in _NO_ORDERS_MARKERS):
        return False
    statuses = _STATUS_LINE_RE.findall(result_content)
    if not statuses:
        return None
    return any(
        s.strip().upper().rstrip(",") in PENDING_REPLACEMENT_ORDER_STATUSES for s in statuses
    )


def _check_no_pending_replacement_order(obs: dict, eff_args: dict) -> Optional[bool]:
    pending = has_pending_replacement_order(obs.get("result"))
    return None if pending is None else (not pending)


# --- rule: cash-back dispute auto-resolved (update_transaction_rewards_3847, v1->v1.1 reclassification) --

_CASH_BACK_RESOLVED_MARKER = "Status: RESOLVED"
_CASH_BACK_SUBMITTED_MARKER = "Status: SUBMITTED"


def cash_back_dispute_resolved(result_content: Optional[str]) -> Optional[bool]:
    """Predicate: does a submit_cash_back_dispute_0589 result show the
    dispute was auto-resolved?

    v1 CLASSIFIED update_transaction_rewards_3847 as `unobtainable` (no
    query_database_tool-based read channel for cash_back_disputes status).
    v1.1 investigation found a genuine (non-query_database_tool) evidence
    channel: submit_cash_back_dispute_0589's OWN return text embeds
    "Status: RESOLVED - ..." when task_config.dispute_settings.
    auto_resolve_disputes is True (server-side ground truth, not
    agent/user-controllable), else "Status: SUBMITTED - ..." forever (no
    other tool in this domain ever transitions a cash_back_disputes
    record). Provenance: tools.py:4184-4242.

    Empirically confirmed against the golden corpus: task_026/task_028
    (auto_resolve_disputes=True) golden trajectories call
    update_transaction_rewards_3847 immediately after
    submit_cash_back_dispute_0589 for the same transaction_id, with no
    other read in between (they must be relying on exactly this channel);
    task_027 (auto_resolve_disputes=False, the "must refuse" task) never
    calls update_transaction_rewards_3847 in its golden at all. See
    specs_banking.md and validation_report.md.
    """
    if result_content is None:
        return None
    if _CASH_BACK_RESOLVED_MARKER in result_content:
        return True
    if _CASH_BACK_SUBMITTED_MARKER in result_content:
        return False
    return None


def _check_cash_back_resolved(obs: dict, eff_args: dict) -> Optional[bool]:
    return cash_back_dispute_resolved(obs.get("result"))


# --- rules: debit-card status preconditions (new in v1.1) ------------------
#
# get_debit_cards_by_account_id_7823 (tools.py:3935-3983) reads
# `self.db.debit_cards.data` via DIRECT dict iteration, NOT
# query_database_tool -- so its statically-extracted read-set (section 1b)
# is EMPTY. Its result IS valid JSON though (`json.dumps(account_cards,
# indent=2)`, each record carrying its own "card_id" field), so it can
# still serve as an evidence channel via `result_parsed`; join_key=None on
# these checks (the evidence call is scoped by account_id, the proposal by
# card_id -- no shared arg name -- so we scan RESULT records for a matching
# card_id instead of joining on call args). See validation_report.md for
# the freshness-invalidation caveat this empty read-set implies (a stale
# card-status read is never invalidated for this group specifically).

DEBIT_CARD_EVIDENCE_TOOL = "get_debit_cards_by_account_id_7823"


def _find_debit_card_record(result_parsed: Any, card_id: Optional[str]) -> Optional[dict]:
    if not isinstance(result_parsed, list) or not card_id:
        return None
    for rec in result_parsed:
        if isinstance(rec, dict) and rec.get("card_id") == card_id:
            return rec
    return None


def debit_card_status_predicate(allowed_statuses: frozenset) -> Callable[[dict, dict], Optional[bool]]:
    def _pred(obs: dict, eff_args: dict) -> Optional[bool]:
        rec = _find_debit_card_record(obs.get("result_parsed"), eff_args.get("card_id"))
        if rec is None:
            return None
        return rec.get("status") in allowed_statuses

    return _pred


def debit_card_fraud_clear_predicate(obs: dict, eff_args: dict) -> Optional[bool]:
    """Provenance: clear_debit_card_fraud_alert_4892, tools.py:3771-3797."""
    rec = _find_debit_card_record(obs.get("result_parsed"), eff_args.get("card_id"))
    if rec is None:
        return None
    reason = eff_args.get("reason")
    if reason == "velocity_clear":
        return bool(rec.get("velocity_blocked", False))
    if reason == "customer_verified":
        return bool(rec.get("fraud_alert_active", False)) and rec.get("alert_source") != "bank_initiated"
    return None  # unknown reason -- the tool's own validation handles it; not a gate concern


def debit_card_activation_predicate(
    allowed_issue_reasons: frozenset,
) -> Callable[[dict, dict], Optional[bool]]:
    """Provenance: _validate_activation_common, tools.py:234-322, plus each
    activate_debit_card_82XX variant's own allowed-issue-reason list."""

    def _pred(obs: dict, eff_args: dict) -> Optional[bool]:
        rec = _find_debit_card_record(obs.get("result_parsed"), eff_args.get("card_id"))
        if rec is None:
            return None
        return rec.get("status") == "PENDING" and rec.get("issue_reason") in allowed_issue_reasons

    return _pred


# --- debit-card SELF-TRANSITION evidence (added after the golden sweep -- see
# validation_report.md "false-block iteration"): get_debit_cards_by_
# account_id_7823 is not the ONLY way a card's current status is
# observable. Several golden trajectories (task_077/078/079/080/081)
# freeze/unfreeze/close/activate a card and later unfreeze/close/activate
# it AGAIN in the SAME episode WITHOUT re-reading it -- the agent is
# tracking state via the SUCCESS of its own prior mutating call on that
# exact card_id (each of these tools' own return text explicitly states
# the resulting status, e.g. freeze_debit_card_3892 -> "Status: FROZEN",
# unfreeze/activate_* -> "Status: ACTIVE"). This is legitimate ambient
# evidence (D1: the runtime searches the ledger, it doesn't require the
# agent to cite anything) -- these checks are added as OR-alternatives
# alongside the get_debit_cards_by_account_id_7823 record-scan, not a
# replacement for it. ---------------------------------------------------


def _self_transition_marker_predicate(expected_marker: str) -> Callable[[dict, dict], Optional[bool]]:
    def _pred(obs: dict, eff_args: dict) -> Optional[bool]:
        content = obs.get("result")
        if content is None:
            return None
        return expected_marker in content

    return _pred


def _active_via_unfreeze_or_activate() -> list[EvidenceCheck]:
    """OR-alternatives proving a card is currently ACTIVE via the agent's
    own prior successful unfreeze_debit_card_3893/activate_debit_card_82XX
    call for this exact card_id."""
    checks = [
        EvidenceCheck(
            "unfreeze_debit_card_3893",
            "card_id",
            _self_transition_marker_predicate("Status: ACTIVE"),
            "prior successful unfreeze_debit_card_3893 for this card_id (this episode)",
        )
    ]
    for _tool in ("activate_debit_card_8291", "activate_debit_card_8292", "activate_debit_card_8293"):
        checks.append(
            EvidenceCheck(
                _tool,
                "card_id",
                _self_transition_marker_predicate("Status: ACTIVE"),
                f"prior successful {_tool} for this card_id (this episode)",
            )
        )
    return checks


def _frozen_via_freeze() -> list[EvidenceCheck]:
    return [
        EvidenceCheck(
            "freeze_debit_card_3892",
            "card_id",
            _self_transition_marker_predicate("Status: FROZEN"),
            "prior successful freeze_debit_card_3892 for this card_id (this episode)",
        )
    ]


def _pending_via_order_debit_card(obs: dict, eff_args: dict) -> Optional[bool]:
    """order_debit_card_5739 generates its own card_id internally (not a
    proposal arg -- tools.py:3235) and echoes it in its success text
    ("Card ID: {card_id}", tools.py:3308). join_key=None: scan order
    results for this proposal's card_id appearing verbatim."""
    content = obs.get("result")
    card_id = eff_args.get("card_id")
    if content is None or not card_id:
        return None
    return f"Card ID: {card_id}" in content


def _pending_via_order_debit_card_check() -> EvidenceCheck:
    return EvidenceCheck(
        "order_debit_card_5739",
        None,
        _pending_via_order_debit_card,
        "prior successful order_debit_card_5739 in this episode created this exact card_id (PENDING) -- issue_reason NOT independently verified via this route",
        skip_none_predicate=True,
    )


# --- Requirement / spec-table data model ------------------------------------


@dataclass
class EvidenceCheck:
    """One (evidence_tool, join, predicate) triple. A ToolSpec at level
    "full" ANDs a list of these -- ALL must pass."""

    evidence_tool: str
    join_key: Optional[str]  # arg name shared by proposal & evidence call; None = no arg-join (predicate scans the result instead)
    predicate: Callable[[dict, dict], Optional[bool]]  # (observation, proposal_args) -> True(pass)/False(fail)/None(inapplicable/unparseable)
    predicate_desc: str
    skip_none_predicate: bool = False  # True: predicate()==None means "this observation doesn't cover the record, keep scanning older ones" (record-scan pattern). False (default): None means "unparseable content, conservative deny now" (single-scope-read pattern, e.g. dispute history).
    max_age_calls: Optional[int] = None  # D3 v1 knob: None = event-based only (no call-distance cap)


@dataclass
class ToolSpec:
    level: str  # "full" | "identity-only" | "unobtainable" | "level-0"
    checks: list = field(default_factory=list)  # list[EvidenceCheck], used when level == "full"
    identity_join_key: Optional[str] = "user_id"  # arg name on THIS tool's proposal to join log_verification.user_id on; None = weak in-episode-existence join (Part 2.1)
    unobtainable_reason: Optional[str] = None  # used when level == "unobtainable"
    provenance: str = ""  # tools.py line citation / rationale, for specs_banking.md


# Tools that are never subject to the universal identity requirement:
#   - log_verification IS the identity-producing action (no circularity).
#   - give_discoverable_user_tool / unlock_discoverable_agent_tool are pure
#     bookkeeping/enablement utilities -- they touch no user financial data
#     (see specs_banking.md "level-0" rationale).
IDENTITY_EXEMPT_TOOLS = {"log_verification", "give_discoverable_user_tool", "unlock_discoverable_agent_tool"}


TOOL_SPECS: dict[str, ToolSpec] = {
    # --- reclassified from v1's DENY_ALWAYS/unobtainable to a real evidence channel ---
    "update_transaction_rewards_3847": ToolSpec(
        level="full",
        checks=[
            EvidenceCheck(
                "submit_cash_back_dispute_0589",
                "transaction_id",
                _check_cash_back_resolved,
                "cash-back dispute for this transaction_id was auto-resolved (task_config.dispute_settings.auto_resolve_disputes)",
            )
        ],
        identity_join_key=None,  # this tool takes no user_id arg
        provenance="tools.py:4184-4242 (submit_cash_back_dispute_0589's own auto_resolve branch); v1->v1.1 reclassification (v1 had this at 'unobtainable' via the query_database_tool channel only -- see specs_banking.md)",
    ),
    # --- v1's "same requirement on submit_ and approve_" DEMOTED after the
    # 97-task golden sweep (Part 3.1) found 4 genuine false blocks
    # (task_050/051/052/054): those goldens all call
    # submit_credit_limit_increase_request_7392 BEFORE
    # get_user_dispute_history_7291 (the dispute-history read happens later,
    # immediately before approve_/deny_). This is not a fluke of one task --
    # 4 independent goldens (plus 053's own order) agree: the domain's
    # expected workflow is submit-first, verify-before-decide. submit_
    # itself has no internal dispute check (only approve_ does -- Check 1,
    # tools.py:2206-2218), so this v1.1 spec now matches the domain's own
    # code, not just its golden behavior.
    "submit_credit_limit_increase_request_7392": ToolSpec(
        level="identity-only",
        identity_join_key="user_id",
        provenance="DEMOTED from v1's 'full' (dispute-history check) after the golden sweep (Part 3.1) found 4 false blocks (task_050/051/052/054) -- submit_credit_limit_increase_request_7392 has no internal dispute check of its own (only approve_credit_limit_increase_5847 does); v1 had applied approve_'s check to submit_ too as an I3/policy-sequencing constraint, but 4 independent goldens (+053) all submit BEFORE reading dispute history, only reading it right before approve_/deny_. See specs_banking.md/validation_report.md.",
    ),
    "approve_credit_limit_increase_5847": ToolSpec(
        level="full",
        checks=[
            EvidenceCheck(
                "get_user_dispute_history_7291",
                "user_id",
                _check_no_pending_dispute,
                "no pending credit-card transaction dispute for this user",
            ),
            EvidenceCheck(
                "get_pending_replacement_orders_5765",
                "credit_card_account_id",
                _check_no_pending_replacement_order,
                "no pending/processing/shipped replacement card order for this account",
            ),
        ],
        identity_join_key="user_id",
        provenance="tools.py:2206-2235 -- mirrors Check 1 (pending disputes) and Check 2 (pending replacement orders) of approve_credit_limit_increase_5847's own eligibility gate. Check 3 (account good standing) is DELIBERATELY OMITTED: it reads a field named 'account_status' (tools.py:2239) that no write path in this domain ever sets (only 'status' is ever written) and that is present in exactly one seed record -- dead code in the domain itself; mirroring it would be inert. See specs_banking.md.",
    ),
}


def _register_identity_only(name: str, join_key: Optional[str], note: str) -> None:
    TOOL_SPECS[name] = ToolSpec(level="identity-only", identity_join_key=join_key, provenance=note)


_register_identity_only("change_user_email", "user_id", "identity-only; no further internal eligibility check in this tool")
_register_identity_only(
    "file_credit_card_transaction_dispute_4829",
    "user_id",
    "identity-only. eligible_for_provisional_credit is an AGENT-SUPPLIED boolean arg -- correctness of that determination is KB-policy arithmetic over dates, D2(d)-banned (out of the gate's expressible class per docs/framework-context-admission.md scope conditions), not a table-checkable condition. No domain table encodes provisional-credit eligibility for the gate to join against.",
)
_register_identity_only(
    "file_debit_card_transaction_dispute_6281",
    "user_id",
    "identity-only. provisional_credit_eligible and customer_max_liability_amount are AGENT-SUPPLIED, derived from Regulation E timing rules (KB-policy arithmetic over dates) -- same D2(d) out-of-scope reasoning as file_credit_card_transaction_dispute_4829.",
)
_register_identity_only(
    "order_replacement_credit_card_7291",
    "user_id",
    "identity-only. Only internal check is 'credit card account exists' (tools.py:1436-1443), not a meaningful epistemic gap -- account_id is a proposal arg, not testimony to verify.",
)
_register_identity_only("log_credit_card_closure_reason_4521", "user_id", "identity-only; pure audit-log tool, no eligibility check")
_register_identity_only(
    "apply_statement_credit_8472",
    "user_id",
    "identity-only. Only internal check is 'credit card account exists' (tools.py:1706-1713); amount/reason are agent judgment calls, not domain-observable facts.",
)
_register_identity_only(
    "apply_credit_card_account_flag_6147",
    "user_id",
    "identity-only. Only internal check is 'credit card account exists' (tools.py:1803-1810).",
)
_register_identity_only(
    "close_credit_card_account_7834",
    "user_id",
    "identity-only. Only internal check is 'credit card account exists' (tools.py:1871-1878).",
)
_register_identity_only(
    "pay_credit_card_from_checking_9182",
    "user_id",
    "identity-only. Internal checks (accounts belong to user_id, sufficient funds, amount <= balance -- tools.py:1936-1969) are all over PROPOSAL-supplied account ids/amounts, not testimony from a separate evidence tool; the tool self-enforces these deterministically at execution time regardless of gate action.",
)
_register_identity_only("deny_credit_limit_increase_5848", "user_id", "identity-only; denial path has no eligibility gate at all (only enum validation + account-exists), matches v1")
_register_identity_only(
    "open_bank_account_4821",
    "user_id",
    "identity-only for v1.1. Internal per-account-type eligibility checks (tools.py:2415-2471: checking-account-age for savings, no-closed-accounts + personal-checking for business_checking, business-checking-age + no-negative-balance for business_savings) ARE readable via get_all_user_accounts_by_user_id_3847 and are a good candidate for a 'full' lift in a future revision -- DEFERRED here for scope; see specs_banking.md.",
)
_register_identity_only(
    "submit_interest_discrepancy_report_7294",
    "user_id",
    "identity-only. FINDING (not a gate concern, documented for completeness): this tool writes to table 'interest_discrepancy_reports', which is NOT a field on TransactionalDB (data_model.py) -- add_to_db() silently returns False (table not found) and the tool does not check the return value, so its 'success' message is always a no-op against the real DB. Domain bug, out of gate scope.",
)
_register_identity_only(
    "order_debit_card_5739",
    "user_id",
    "identity-only for v1.1. Most internal checks (account exists/OPEN/checking-class/belongs-to-user/min-$25-balance -- tools.py:3116-3139; existing-active-card count -- tools.py:3149-3156) are readable via get_all_user_accounts_by_user_id_3847 + get_debit_cards_by_account_id_7823 and are good 'full'-lift candidates, DEFERRED for scope. ONE condition ('no pending debit card order for this account', tools.py:3141-3147, checked against self.db.debit_card_orders.data directly) has a genuine v1-style read-channel gap -- no discoverable/registered tool anywhere in this domain queries the debit_card_orders table (confirmed: it never appears as a query_database_tool literal argument). Not classified as blanket 'unobtainable' because it is one of SEVERAL preconditions on a tool whose other conditions ARE obtainable, not the sole gate on the action (unlike update_transaction_rewards_3847 pre-v1.1). The domain's own runtime check remains the safety net for this one condition.",
)
_register_identity_only(
    "close_bank_account_7392",
    None,
    "identity-only, WEAK join: this tool takes NO user_id arg (only account_id). The $0-balance-before-close precondition (tools.py:2591-2594) is inherent to the account record itself and readable via get_all_user_accounts_by_user_id_3847, but that read tool is keyed by user_id, which this proposal does not supply -- a full lift would need a two-hop join (account_id -> owning user_id via a prior read) not supported by v1.1's single-arg-join mechanism. DEFERRED.",
)
_register_identity_only(
    "transfer_funds_between_bank_accounts_7291",
    None,
    "identity-only, WEAK join (no user_id arg; source_account_id/destination_account_id only). Internal checks (both accounts exist/active, sufficient source funds -- tools.py:2691-2712) are proposal-arg-scoped and self-enforced by the tool deterministically.",
)
_register_identity_only(
    "apply_checking_account_credit_5829",
    None,
    "identity-only, WEAK join (no user_id arg; account_id only). Only internal checks are account-exists/is-checking/is-active (tools.py:2762-2774), proposal-arg-scoped.",
)
_register_identity_only(
    "apply_savings_account_credit_6831",
    None,
    "identity-only, WEAK join (no user_id arg; account_id only). Only internal checks are account-exists/is-savings/is-active (tools.py:2847-2859), proposal-arg-scoped.",
)

# --- debit-card status-precondition group (new in v1.1, record-scan pattern
# PLUS self-transition OR-alternatives -- the latter added after the golden
# sweep, see validation_report.md "false-block iteration" for the concrete
# task_077/078/079/080/081 evidence that motivated each addition) --------

# freeze_debit_card_3892 ITSELF is DEMOTED to identity-only (no check at
# all -- see below, after this loop): the golden sweep found task_078
# freezes 3 cards with NO debit-card read of any kind beforehand (no prior
# freeze/unfreeze/activate either -- these are the FIRST debit-card
# actions in the episode). Freeze is safe/reversible/low-stakes (trivially
# undone by unfreeze; the tool's own "must be ACTIVE" check already
# prevents a no-op) -- unlike unfreeze/close/activate, there is no
# plausible epistemic gap worth gating here. See specs_banking.md.

for _name, _allowed in [
    ("set_debit_card_recurring_block_7382", frozenset({"ACTIVE"})),  # tools.py:3165-3168
    ("reset_debit_card_pin_6284", frozenset({"ACTIVE"})),  # tools.py:3855-3857
    ("change_debit_card_pin_6285", frozenset({"ACTIVE"})),  # tools.py:3915-3917
    ("request_temporary_debit_card_limit_increase_8374", frozenset({"ACTIVE"})),  # tools.py:4035-4037
]:
    TOOL_SPECS[_name] = ToolSpec(
        level="full",
        checks=[
            [
                EvidenceCheck(
                    DEBIT_CARD_EVIDENCE_TOOL,
                    None,
                    debit_card_status_predicate(_allowed),
                    f"card status in {sorted(_allowed)} (get_debit_cards_by_account_id_7823)",
                    skip_none_predicate=True,
                ),
                *_active_via_unfreeze_or_activate(),
            ]
        ],
        identity_join_key=None,  # no user_id arg on any of these
        provenance=f"card-status precondition mined from {_name}'s own body; evidence via get_debit_cards_by_account_id_7823 record-scan OR the agent's own prior successful unfreeze/activate for this card_id (self-transition, added after the golden sweep) -- see specs_banking.md for the empty-read-set/staleness caveat on the record-scan alternative.",
    )

TOOL_SPECS["unfreeze_debit_card_3893"] = ToolSpec(
    level="full",
    checks=[
        [
            EvidenceCheck(
                DEBIT_CARD_EVIDENCE_TOOL,
                None,
                debit_card_status_predicate(frozenset({"FROZEN"})),
                "card status FROZEN (get_debit_cards_by_account_id_7823)",
                skip_none_predicate=True,
            ),
            *_frozen_via_freeze(),
        ]
    ],
    identity_join_key=None,
    provenance="tools.py:3715-3719 (linked-account-OPEN sub-check deferred). Evidence via get_debit_cards_by_account_id_7823 record-scan OR the agent's own prior successful freeze_debit_card_3892 for this card_id (self-transition, added after the golden sweep -- task_077/078/079/080/081 all unfreeze immediately after freezing, in the same episode, with no re-read in between).",
)

TOOL_SPECS["close_debit_card_4721"] = ToolSpec(
    level="full",
    checks=[
        [
            EvidenceCheck(
                DEBIT_CARD_EVIDENCE_TOOL,
                None,
                debit_card_status_predicate(frozenset({"ACTIVE", "PENDING"})),
                "card status in ['ACTIVE', 'PENDING'] (get_debit_cards_by_account_id_7823)",
                skip_none_predicate=True,
            ),
            *_active_via_unfreeze_or_activate(),
        ]
    ],
    identity_join_key=None,
    provenance="tools.py:3607-3609. Evidence via get_debit_cards_by_account_id_7823 record-scan OR the agent's own prior successful unfreeze/activate for this card_id (self-transition, added after the golden sweep). The PENDING-via-order_debit_card_5739 route is NOT included here (not exercised by any golden false block; would need order_debit_card_5739's own generated card_id, which never appears as a close_debit_card_4721 proposal arg in this corpus) -- documented residual gap, see specs_banking.md.",
)

TOOL_SPECS["clear_debit_card_fraud_alert_4892"] = ToolSpec(
    level="full",
    checks=[
        EvidenceCheck(
            DEBIT_CARD_EVIDENCE_TOOL,
            None,
            debit_card_fraud_clear_predicate,
            "fraud_alert_active (non-bank-initiated) for reason=customer_verified, or velocity_blocked for reason=velocity_clear",
            skip_none_predicate=True,
        )
    ],
    identity_join_key=None,
    provenance="tools.py:3771-3797. No self-transition alternative: no tool in this domain sets fraud_alert_active/velocity_blocked (these are seed-data-only conditions), so get_debit_cards_by_account_id_7823 is the only possible evidence channel -- not a gap.",
)

for _name, _reasons in [
    ("activate_debit_card_8291", frozenset({"new_account", "first_card"})),
    ("activate_debit_card_8292", frozenset({"lost", "stolen", "fraud"})),
    ("activate_debit_card_8293", frozenset({"expired", "damaged", "upgrade", "bank_reissue"})),
]:
    TOOL_SPECS[_name] = ToolSpec(
        level="full",
        checks=[
            [
                EvidenceCheck(
                    DEBIT_CARD_EVIDENCE_TOOL,
                    None,
                    debit_card_activation_predicate(_reasons),
                    f"status==PENDING and issue_reason in {sorted(_reasons)} (get_debit_cards_by_account_id_7823)",
                    skip_none_predicate=True,
                ),
                _pending_via_order_debit_card_check(),
            ]
        ],
        identity_join_key=None,
        provenance=(
            "tools.py:234-322 (_validate_activation_common) + this variant's own allowed-issue-reason "
            "list. Evidence via get_debit_cards_by_account_id_7823 record-scan (checks issue_reason "
            "precisely) OR the agent's own prior successful order_debit_card_5739 in this episode that "
            "created this exact card_id (self-transition, added after the golden sweep -- "
            "task_077/080 activate a just-ordered replacement card with no read of any kind in "
            "between). The order-route does NOT independently verify issue_reason against this "
            "variant's allowed set (order_debit_card_5739's own response does not cleanly expose it) "
            "-- documented weaker-check note, see specs_banking.md."
        ),
    )

del _name, _allowed, _reasons  # module-scope loop variable cleanup

_register_identity_only(
    "freeze_debit_card_3892",
    None,
    "DEMOTED from v1.1's initial 'full' (ACTIVE-precondition record-scan) after the golden sweep (Part 3.1) found task_078 freezes 3 cards with NO debit-card evidence of any kind beforehand (first debit-card action in the episode -- no read, no prior sibling-tool call). Freeze is safe/reversible/low-stakes (trivially undone by unfreeze_debit_card_3893; the tool's own 'must be ACTIVE' check already prevents a no-op on an already-frozen card) -- no plausible safety case for gating it on evidence. identity-only (weak join, no user_id arg).",
)


# --- User-side tools (registered + discoverable) -- listed for coverage
# documentation only. Gate is scoped to requestor=="assistant" (D5); these
# tools live on KnowledgeUserTools and are architecturally unreachable by
# the assistant (call_discoverable_agent_tool only dispatches to
# KnowledgeTools' OWN discoverable methods -- tools.py:660,
# `self.has_discoverable_tool(agent_tool_name)`). gate_verdict() is never
# invoked for them in practice; no ToolSpec entries are needed for
# correctness, only for specs_banking.md's completeness table.
USER_SIDE_MUTATING_TOOLS = frozenset(
    {
        "apply_for_credit_card",
        "submit_referral",
        "call_discoverable_user_tool",
        "request_human_agent_transfer",
        "submit_transaction",
        "submit_cash_back_dispute_0589",
        "get_referral_link",
        "deposit_check_3847",
    }
)


# --- Verdict engine -----------------------------------------------------


@dataclass
class Verdict:
    verdict: str  # "PASS" | "DENY" | "ESCALATE" (ESCALATE only from GateLedger.verdict_with_budget)
    spec_rule: str  # "identity" | "evidence" | "stale" | "unobtainable" | "level0" | "budget"
    reason: str
    discharge_seq: Optional[int]


def check_identity(eff_name: str, eff_args: dict, ledger: "GateLedger") -> Optional[Verdict]:
    """Part 2.1 universal identity requirement. Returns a DENY Verdict if
    unsatisfied, else None (caller proceeds to the tool-specific rule)."""
    if eff_name in IDENTITY_EXEMPT_TOOLS:
        return None

    spec = TOOL_SPECS.get(eff_name)
    join_key = spec.identity_join_key if spec is not None else "user_id"
    join_value = eff_args.get(join_key) if join_key else None

    candidates = [o for o in ledger.observations if o["tool"] == "log_verification"]
    if join_key and join_value is not None:
        matches = [o for o in candidates if o["args"].get("user_id") == join_value]
        join_desc = f"joined on user_id={join_value!r}"
    else:
        matches = candidates
        why = f"no {join_key!r} arg on this proposal" if join_key else "this tool has no user_id-bearing arg"
        join_desc = f"{why} -- weaker in-episode-existence join (Part 2.1)"

    fresh_matches = [o for o in matches if ledger.is_fresh(o)[0]]
    if fresh_matches:
        return None
    if matches:
        return Verdict(
            "DENY",
            "identity",
            f"log_verification observation(s) exist ({join_desc}) but none are FRESH -- invalidated by a later write",
            None,
        )
    return Verdict(
        "DENY",
        "identity",
        f"no prior log_verification observation found ({join_desc}) -- universal identity requirement (Part 2.1): every assistant-side mutating proposal requires a prior FRESH log_verification observation",
        None,
    )


def evaluate_check(check: EvidenceCheck, eff_args: dict, ledger: "GateLedger") -> Verdict:
    """Evaluate one EvidenceCheck against the ledger, most-recent-match-first."""
    join_value = eff_args.get(check.join_key) if check.join_key else None
    stale_fallback: Optional[tuple[dict, str]] = None

    for obs in reversed(ledger.observations):
        if obs["tool"] != check.evidence_tool:
            continue
        if check.join_key is not None and obs["args"].get(check.join_key) != join_value:
            continue

        fresh, why_stale = ledger.is_fresh(obs, check.max_age_calls)
        if not fresh:
            if stale_fallback is None:
                stale_fallback = (obs, why_stale or "invalidated")
            continue

        pred_result = check.predicate(obs, eff_args)
        if pred_result is True:
            return Verdict(
                "PASS",
                "evidence",
                f"discharged by {check.evidence_tool} (ledger seq {obs['seq']}): {check.predicate_desc}",
                obs["seq"],
            )
        if pred_result is False:
            return Verdict(
                "DENY",
                "evidence",
                f"{check.evidence_tool} (ledger seq {obs['seq']}) fails the required condition: {check.predicate_desc}",
                obs["seq"],
            )
        # pred_result is None
        if check.skip_none_predicate:
            continue  # this fresh observation doesn't cover the relevant record -- keep scanning further back
        return Verdict(
            "DENY",
            "evidence",
            f"observation of {check.evidence_tool} (ledger seq {obs['seq']}) found but its result content could not be parsed for the required condition ({check.predicate_desc}); conservative deny",
            obs["seq"],
        )

    if stale_fallback is not None:
        obs, why = stale_fallback
        return Verdict(
            "DENY",
            "stale",
            f"a prior observation of {check.evidence_tool} (ledger seq {obs['seq']}) exists but is STALE: {why}. Missing requirement: {check.predicate_desc}",
            obs["seq"],
        )
    return Verdict(
        "DENY",
        "evidence",
        f"no prior FRESH observation of {check.evidence_tool} found"
        + (f" (join {check.join_key}={join_value!r})" if check.join_key else "")
        + f" -- missing requirement: {check.predicate_desc}",
        None,
    )


def gate_verdict(eff_name: str, eff_args: dict, ledger: "GateLedger") -> Verdict:
    """Evaluate the v1.1 spec table against one assistant-side mutating
    proposal, given the ledger of observations strictly PRIOR to this
    proposal (event-based freshness, D3 -- see GateLedger.is_fresh).

    Stateless w.r.t. denial history: never returns ESCALATE -- that is
    layered on top by GateLedger.verdict_with_budget (D4-ii, live/replay
    only). shadow_replay.py calls this function directly with its own
    GateLedger instance (fed via record_observation, live-DB-diffed where
    available, statically approximated otherwise -- see section 1c/1b).
    """
    id_verdict = check_identity(eff_name, eff_args, ledger)
    if id_verdict is not None:
        return id_verdict

    spec = TOOL_SPECS.get(eff_name)
    if spec is None or spec.level == "level-0":
        return Verdict("PASS", "level0", "level-0: no requirement beyond identity defined for this tool in v1.1 (coverage placeholder)", None)

    if spec.level == "identity-only":
        return Verdict("PASS", "level0", "identity-only: universal identity requirement satisfied; no additional evidence requirement mined for this tool (see specs_banking.md)", None)

    if spec.level == "unobtainable":
        return Verdict("DENY", "unobtainable", spec.unobtainable_reason or "evidence unobtainable in this domain", None)

    if spec.level == "full":
        last_seq: Optional[int] = None
        reasons: list[str] = []
        for item in spec.checks:
            if isinstance(item, list):
                # OR-group: several alternative evidence sources for the SAME
                # logical requirement (LOCAL ADDITION gate v1.1, see
                # debit-card group in TOOL_SPECS -- e.g. a fresh
                # get_debit_cards_by_account_id_7823 record-scan OR the
                # agent's own prior successful sibling-tool call on this
                # exact card_id, whose result text is itself proof of the
                # resulting state). PASS if ANY alternative passes.
                alt_results = [evaluate_check(c, eff_args, ledger) for c in item]
                passing = [r for r in alt_results if r.verdict == "PASS"]
                if passing:
                    v = max(
                        passing,
                        key=lambda r: (r.discharge_seq if r.discharge_seq is not None else -1),
                    )
                else:
                    v = Verdict(
                        "DENY",
                        "evidence",
                        "no evidence source satisfied (tried "
                        f"{len(alt_results)} alternative(s)): "
                        + " | OR | ".join(r.reason for r in alt_results),
                        None,
                    )
            else:
                v = evaluate_check(item, eff_args, ledger)
            if v.verdict != "PASS":
                return v
            if v.discharge_seq is not None:
                last_seq = v.discharge_seq
            reasons.append(v.reason)
        return Verdict("PASS", "evidence", "all evidence checks discharged: " + "; ".join(reasons), last_seq)

    return Verdict("PASS", "level0", f"unrecognized spec level {spec.level!r} (defensive fallback -- treated as level-0)", None)


# ---------------------------------------------------------------------------
# 4. GateLedger -- versioned/coherent ledger + D4-ii denial budget (LOCAL ADDITION gate v1.1)
# ---------------------------------------------------------------------------

# D4-ii: "the same (tool, args) proposed a 2nd time with no new ledger
# observations since the last denial -> stop re-verdicting, escalate."
DENIAL_BUDGET_N = 2


class GateLedger:
    """Versioned/coherent ledger of tool-call observations, world-version
    counter, mutation write-log, and (live/replay only) denial-budget
    state.

    One instance per Environment (constructed fresh at Environment.__init__,
    and again for every fresh environment set_state() replays into --
    replay parity depends on both starting empty and being fed observations
    in the exact same order live and during replay). shadow_replay.py
    constructs its own instances too (one per simulation / per golden task)
    to get the identical versioning/invalidation semantics for its offline
    checks.
    """

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.world_version: int = 0
        self.mutation_log: list[dict] = []  # [{"version_after", "seq", "tool", "write_set"}]
        self._denial_state: dict[tuple[str, str], tuple[int, int]] = {}

    # -- recording ---------------------------------------------------------

    def record_observation(
        self,
        tool: str,
        args: dict,
        result: Optional[str],
        requestor: str,
        error: bool = False,
        is_mutating: bool = False,
        write_set: Any = None,
    ) -> dict:
        """Append a real tool-call result to the ledger. Never call this for
        a synthetic DENY/ESCALATE message -- nothing was actually mediated,
        so it must not count as evidence, must not bump world_version (spec
        item 1: "denied/escalated proposals don't increment -- nothing
        changed"), and must not silently reset the denial budget (see
        verdict_with_budget).

        `write_set`: only meaningful when is_mutating=True. Either a dict
        {table: {record_id, ...}} (precise, from diff_write_set) or the
        ALL_TABLES sentinel (static-approximation fallback, see section
        1b) or None/omitted (treated as {} -- "executed but changed
        nothing observable", e.g. a mutating call whose write target isn't
        a real DB field -- see submit_interest_discrepancy_report_7294's
        finding in specs_banking.md).
        """
        seq = len(self.observations)
        entry: dict = {
            "seq": seq,
            "tool": tool,
            "args": args or {},
            "result": result,
            "result_parsed": _try_parse_json(result),
            "requestor": requestor,
            "error": error,
            "is_mutating": is_mutating,
        }

        if is_mutating:
            self.world_version += 1
            ws = write_set if write_set is not None else {}
            if ws != ALL_TABLES:
                ws = {t: set(ids) for t, ids in ws.items()}
            entry["write_set"] = ws
            entry["world_version"] = self.world_version
            self.mutation_log.append(
                {"version_after": self.world_version, "seq": seq, "tool": tool, "write_set": ws}
            )
        else:
            entry["read_set"] = READ_SET_MAP.get(tool, frozenset())
            entry["world_version"] = self.world_version  # unaffected by a read

        self.observations.append(entry)
        return entry

    # -- freshness -----------------------------------------------------------

    def is_fresh(self, obs: dict, max_age_calls: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """FRESH iff no subsequent EXECUTED mutation's write-set overlaps
        this observation's read-set (spec item 4). A mutating observation
        (e.g. log_verification, or submit_cash_back_dispute_0589 used as
        evidence) has no read_set -- self-evidencing WRITE events are never
        invalidated once logged (see specs_banking.md: no tool in this
        domain un-verifies a user or un-resolves a dispute). An observation
        with an empty read_set (query_database_tool wasn't used -- section
        1b) is likewise trivially fresh forever; documented, not a bug.
        """
        read_set = obs.get("read_set") or frozenset()
        if not read_set:
            return True, None

        v0 = obs["world_version"]
        for mut in self.mutation_log:
            if mut["version_after"] <= v0:
                continue  # happened before (or as part of) this observation -- already reflected
            ws = mut["write_set"]
            if ws == ALL_TABLES:
                hit_tables = set(read_set)
            else:
                hit_tables = {t for t in read_set if ws.get(t)}
            if hit_tables:
                return False, (
                    f"table(s) {sorted(hit_tables)} written by {mut['tool']} "
                    f"(ledger seq {mut['seq']}, world_version {mut['version_after']}) "
                    f"after this observation (read at world_version {v0})"
                )

        if max_age_calls is not None:
            calls_since = (len(self.observations) - 1) - obs["seq"]
            if calls_since > max_age_calls:
                return False, f"max_age_calls exceeded ({calls_since} calls since observation seq {obs['seq']}, limit {max_age_calls})"

        return True, None

    # -- denial budget (D4-ii) ------------------------------------------------

    @staticmethod
    def _key(tool: str, args: dict) -> tuple[str, str]:
        return (tool, json.dumps(args or {}, sort_keys=True, default=str))

    def verdict_with_budget(self, eff_name: str, eff_args: dict) -> Verdict:
        """gate_verdict(), plus D4-ii denial-budget escalation.

        Semantics: track, per (effective_tool, effective_args) key, the
        ledger length at the last DENY of that exact proposal. If the same
        proposal recurs with the ledger UNCHANGED since that denial, the
        denial streak increments; once the streak reaches DENIAL_BUDGET_N
        (v1: 2), stop re-running gate_verdict entirely and return ESCALATE
        instead. Any new ledger observation (evidence-gathering attempt, or
        someone else's tool call) resets the streak -- the agent gets a
        fresh, fully re-verdicted attempt. A PASS clears all denial state
        for that key (repair complete).
        """
        key = self._key(eff_name, eff_args)
        ledger_len = len(self.observations)
        state = self._denial_state.get(key)
        streak = 0
        if state is not None:
            last_len, prev_streak = state
            if ledger_len == last_len:
                streak = prev_streak + 1
                if streak >= DENIAL_BUDGET_N:
                    self._denial_state[key] = (last_len, streak)
                    return Verdict(
                        "ESCALATE",
                        "budget",
                        f"denial budget exhausted for {eff_name} (proposal #{streak} "
                        "with the identical arguments and zero new ledger observations "
                        "since the prior denial) -- re-verdict suppressed per D4-ii; "
                        "verification remains unobtainable without new evidence",
                        None,
                    )
                # else: streak < budget, fall through and re-verdict (only
                # reachable if DENIAL_BUDGET_N > 2; a no-op path at v1's N=2).
            # else: new observations exist since the last denial -- reset.

        v = gate_verdict(eff_name, eff_args, self)
        if v.verdict == "DENY":
            self._denial_state[key] = (ledger_len, streak + 1)
        else:
            self._denial_state.pop(key, None)
        return v


# ---------------------------------------------------------------------------
# 5. Denial / escalation content (recipe-style, deterministic)
# ---------------------------------------------------------------------------
#
# IMPORTANT: this content becomes ToolMessage.content, which is compared
# byte-for-byte during strict-mode replay (Environment.set_state / gated
# override). It MUST be a pure function of (verdict, effective tool,
# effective args) -- no wall-clock timestamps, random ids, or other
# non-deterministic fields. verdict.reason is itself already a pure
# function of ledger content (ledger seq / world_version numbers, which are
# reproduced identically on replay) -- see evaluate_check/check_identity.


def denial_payload(verdict: Verdict, eff_name: str, eff_args: dict) -> dict:
    payload: dict = {
        "status": "COMMIT_DENIED",
        "tool": eff_name,
        "arguments": eff_args,
        "rule": verdict.spec_rule,
        "reason": verdict.reason,
    }
    spec = TOOL_SPECS.get(eff_name)

    if verdict.spec_rule == "identity":
        payload["missing_requirement"] = "a fresh log_verification observation (see reason)"
        payload["discharge_via"] = "log_verification"
        payload["recommended_action"] = (
            "Verify the customer's identity with log_verification before proposing this action."
        )
    elif verdict.spec_rule in ("evidence", "stale"):
        evidence_tools = [c.evidence_tool for c in (spec.checks if spec else [])]
        payload["missing_requirement"] = verdict.reason
        payload["discharge_via"] = evidence_tools or None
        if evidence_tools:
            payload["recommended_action"] = (
                f"Call {', '.join(evidence_tools)} (fresh) and confirm the required condition(s), then retry."
            )
        else:
            payload["recommended_action"] = "Gather fresh evidence, then retry."
    elif verdict.spec_rule == "unobtainable":
        payload["missing_requirement"] = verdict.reason
        payload["discharge_via"] = None
        payload["note"] = "This domain provides no channel to obtain this evidence."
        payload["recommended_action"] = (
            "Do not perform this action. Tell the user this cannot be verified through "
            "available tools, or use transfer_to_human_agents."
        )
    else:
        payload["discharge_via"] = None
        payload["recommended_action"] = "Gather fresh evidence, then retry."

    return payload


def escalate_payload(verdict: Verdict, eff_name: str, eff_args: dict) -> dict:
    """D4-ii denial-budget exhaustion content."""
    return {
        "status": "COMMIT_ESCALATED",
        "tool": eff_name,
        "arguments": eff_args,
        "rule": verdict.spec_rule,
        "reason": verdict.reason,
        "recommended_action": (
            "Verification remains unobtainable: no new evidence has been "
            "gathered since the prior denial of this exact action. Do not "
            "retry with the same arguments -- refuse the request or use "
            "transfer_to_human_agents."
        ),
    }


def format_deny_content(verdict: Verdict, eff_name: str, eff_args: dict) -> str:
    return json.dumps(denial_payload(verdict, eff_name, eff_args), indent=2, sort_keys=True)


def format_escalate_content(verdict: Verdict, eff_name: str, eff_args: dict) -> str:
    return json.dumps(escalate_payload(verdict, eff_name, eff_args), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Module-level static maps, built once at import time from the default
# source paths (mirrors DEFAULT_TOOLS_PY/DEFAULT_MIXINS_PY module-level
# convention already used for the tool table). Consumers outside the clone
# (shadow_replay.py) that pass their own paths should call
# extract_read_set_map()/extract_static_write_table_map() directly instead
# of relying on these module-level defaults -- exactly the existing
# convention for extract_tool_table() vs build_default_tool_table().
# ---------------------------------------------------------------------------

READ_SET_MAP: dict[str, frozenset] = extract_read_set_map(DEFAULT_TOOLS_PY, DEFAULT_MIXINS_PY)
STATIC_WRITE_TABLE_MAP: dict[str, frozenset] = extract_static_write_table_map(DEFAULT_TOOLS_PY, DEFAULT_MIXINS_PY)
TOUCHES_DB_MAP: dict[str, bool] = extract_touches_db_map(DEFAULT_TOOLS_PY, DEFAULT_MIXINS_PY)
