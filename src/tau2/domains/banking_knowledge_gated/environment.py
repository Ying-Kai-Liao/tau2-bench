"""Environment for the banking_knowledge_gated domain variant.

LOCAL ADDITION (gate v1) — not part of upstream tau2-bench.

Commit-gate v1 experiment. Design doc: `docs/gate-v1-spec.md` (research
repo, above this clone). Gate logic (ledger, spec table, verdict engine) is
factored into `gate_core.py` in this package — importable both here and by
`experiments/oracle-gap/gate_v1/shadow_replay.py` (the offline, $0 gate
checker this environment's live behavior must match on replay).

Identical to banking_knowledge (same DB, same tasks, same policy, same
retrieval-variant machinery) except:
  - `domain_name="banking_knowledge_gated"` (the evaluator re-resolves the
    constructor from this literal name — evaluator.py:158,173 — so the
    gate MUST be registered as its own domain variant; a live-only patch of
    `banking_knowledge`'s environment would replay ungated and silently
    execute writes the live run denied);
  - the `Environment` returned is a `GatedEnvironment`, which intercepts
    every assistant-side mutating tool proposal at `get_response` (the
    single interception point that both half/full-duplex live execution
    and `set_state` replay funnel through — environment.py:465-490,
    environment.py:390) and evaluates it against the v1 spec table before
    (optionally) letting it execute.

The banking_knowledge domain itself is untouched — this is additive
registration only.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import get_db, get_knowledge_base
from tau2.domains.banking_knowledge.environment import get_tasks as _banking_knowledge_get_tasks
from tau2.domains.banking_knowledge.retrieval import (
    DEFAULT_RETRIEVAL_VARIANT,
    build_policy,
    build_tools,
    resolve_variant,
)
from tau2.domains.banking_knowledge.tools import KnowledgeUserTools
from tau2.domains.banking_knowledge_gated.gate_core import (
    GateLedger,
    Verdict,
    build_default_tool_table,
    diff_write_set,  # LOCAL ADDITION (gate v1.1)
    format_deny_content,
    format_escalate_content,
    unwrap_effective,
)
from tau2.environment.environment import Environment

# Env vars controlling gate behavior (read fresh on every call, not cached
# at construction, so tests can flip modes between fresh environments
# without needing a constructor kwarg -- matches the task spec's "modes via
# env var" design).
GATE_MODE_ENV_VAR = "GATE_MODE"
GATE_LOG_PATH_ENV_VAR = "GATE_LOG_PATH"
DEFAULT_GATE_MODE = "shadow"


class GatedEnvironment(Environment):
    """Environment subclass that gates assistant-side mutating tool calls
    through the commit-gate v1 spec table (`gate_core.py`).

    Two modes, selected via the `GATE_MODE` env var:
      - "shadow" (default): the gate evaluates and logs a verdict for every
        assistant-side mutating proposal, but ALWAYS lets it execute.
      - "enforce": a DENY or ESCALATE verdict blocks execution; a
        synthetic ToolMessage (same id as the ToolCall, error=False) is
        returned instead, with a recipe-style denial payload as content.

    Verdict events are always logged via loguru; if `GATE_LOG_PATH` is set,
    they are ALSO appended as JSONL to that path.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Single source of truth for mutates_state classification (see
        # gate_core.py header) -- computed once per environment instance.
        self._gate_tool_table = build_default_tool_table()
        self._gate_ledger = GateLedger()

    # -- mode / logging -----------------------------------------------

    @staticmethod
    def _gate_mode() -> str:
        return os.environ.get(GATE_MODE_ENV_VAR, DEFAULT_GATE_MODE)

    @staticmethod
    def _gate_log_path() -> Optional[str]:
        return os.environ.get(GATE_LOG_PATH_ENV_VAR) or None

    def _effective_is_mutating(self, eff_name: Optional[str]) -> bool:
        meta = self._gate_tool_table.get(eff_name) if eff_name else None
        if meta is not None:
            return bool(meta["mutates_state"])
        # Not found in the AST-derived table (e.g. a malformed/hallucinated
        # dispatcher call with no resolvable inner tool name). Fall back to
        # the base Environment's own live-toolkit introspection, which
        # itself conservatively assumes mutation for unknown names.
        return self._is_mutating_tool(eff_name) if eff_name else True

    def _log_verdict(
        self,
        tool_call: ToolCall,
        eff_name: str,
        eff_args: dict,
        verdict: Verdict,
    ) -> None:
        # "sim time": Environment.get_response only receives a ToolCall (no
        # turn/tick index is threaded through from the orchestrator -- see
        # orchestrator.py:325 and environment_manager.py:254, both of which
        # call get_response(tool_call) with nothing else). The ledger
        # sequence number is used as the ordering proxy instead; see
        # build_report.md for this scope note.
        entry = {
            "sim_time": len(self._gate_ledger.observations),
            "tool_call_id": tool_call.id,
            "outer_tool": tool_call.name,
            "effective_tool": eff_name,
            "effective_args": eff_args,
            "verdict": verdict.verdict,
            "rule": verdict.spec_rule,
            "reason": verdict.reason,
            "discharge_seq": verdict.discharge_seq,
            "mode": self._gate_mode(),
        }
        logger.debug(f"[gate v1] {json.dumps(entry, default=str)}")
        path = self._gate_log_path()
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # -- the interception point ----------------------------------------

    def get_response(self, message: ToolCall) -> ToolMessage:
        """Overrides Environment.get_response (environment.py:465-490), the
        single interception point live execution AND set_state() replay
        both funnel through for mutating tool calls.

        LOCAL ADDITION (gate v1.1): for every EXECUTED mutating call
        (gated or not, either requestor -- world_version tracks the whole
        world, not just gated proposals, per spec item 1), the toolkit DB
        is snapshotted immediately before and after real execution and
        diffed (`diff_write_set`, record-level) to produce the precise
        write-set fed into the ledger. This is what lets
        `GateLedger.is_fresh` invalidate stale evidence.
        """
        eff_name, eff_args, malformed_note = unwrap_effective(message.name, message.arguments)
        if malformed_note:
            logger.warning(f"[gate v1] {malformed_note}")

        is_mutating = self._effective_is_mutating(eff_name)
        should_gate = message.requestor == "assistant" and is_mutating

        if should_gate:
            verdict = self._gate_ledger.verdict_with_budget(eff_name, eff_args)
            self._log_verdict(message, eff_name, eff_args, verdict)
            if verdict.verdict in ("DENY", "ESCALATE") and self._gate_mode() == "enforce":
                content = (
                    format_escalate_content(verdict, eff_name, eff_args)
                    if verdict.verdict == "ESCALATE"
                    else format_deny_content(verdict, eff_name, eff_args)
                )
                return ToolMessage(
                    id=message.id,
                    content=content,
                    requestor=message.requestor,
                    role="tool",
                    error=False,  # NEVER True -- see gate-v1-spec.md 鐵律 #1
                )
            # shadow mode, or an enforce-mode PASS: fall through and execute.

        # LOCAL ADDITION (gate v1.1): precise, record-level write-set via
        # DB diffing (spec item 2). Only mutating calls pay this cost.
        db_before = self.tools.db.model_dump() if is_mutating else None
        response = super().get_response(message)
        write_set = None
        if is_mutating:
            db_after = self.tools.db.model_dump()
            write_set = diff_write_set(db_before, db_after)

        # Record EVERY real tool result into the live ledger -- gated or
        # not, mutating or not, either requestor. This is what makes reads
        # (e.g. get_user_dispute_history_7291) available as evidence for
        # later gate proposals, and what makes mutations (e.g.
        # log_verification, submit_cash_back_dispute_0589) available as
        # self-evidencing write events plus real write-sets for
        # invalidation.
        self._gate_ledger.record_observation(
            eff_name,
            eff_args,
            response.content,
            message.requestor,
            response.error,
            is_mutating=is_mutating,
            write_set=write_set,
        )
        return response

    # -- replay parity ---------------------------------------------------

    def set_state(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
        strict: bool = True,
    ):
        """Overrides Environment.set_state (environment.py:293-410).

        THE CRITICAL CORRECTNESS POINT (see docs/gate-v1-spec.md and the
        gate-v1 build task): the base implementation replays history but
        SKIPS re-executing non-mutating tool calls entirely (environment.py
        :385-389, keyed on the OUTER tool_call.name). For a gated
        environment, deciding what counts as "evidence in the ledger" by
        EFFECTIVE tool (not outer dispatcher name) means a naive override
        would see an EMPTY ledger at replay time for every dispatcher-
        wrapped read (e.g. get_user_dispute_history_7291, always called via
        call_discoverable_agent_tool) and DENY writes that legitimately
        PASSed live -- a strict-mode content mismatch and/or DB divergence.

        Fix: walk the (ToolCall, recorded ToolMessage) pairs IN ORDER
        (interleaved, no wholesale pre-scan -- a read that comes AFTER a
        write in the trajectory must not retroactively justify that
        earlier write); for each call whose EFFECTIVE tool is
        non-mutating, feed the RECORDED ToolMessage content into the gate
        ledger as a real observation *without executing anything*, then
        skip exactly as the base does. Mutating calls still go through
        `self.get_response(...)` (this class's gated override) exactly as
        in the base loop, so DENY/ESCALATE verdicts re-derive identically
        given identical prior ledger state, and a recorded DENIAL
        ToolMessage re-verdicts to the same (deterministic, timestamp-free)
        denial content -- no mutation either time, so strict comparison
        passes.

        This intentionally changes replay semantics relative to the base
        environment for ONE case: dispatcher-wrapped reads (whose OUTER
        tool is mutates_state=True, so the base env actually re-executes
        and compares them) are here trusted from the recorded content
        instead of re-executed. See build_report.md for why (determinism/
        robustness -- ledger-content should not depend on DB-state
        reconstruction having gone flawlessly for every prior step) and
        for the resulting trade-off (a read whose live tool output has
        silently drifted from what current tool code would produce is no
        longer caught by strict replay for the gated domain the way it
        would be for plain banking_knowledge).

        LOCAL ADDITION (gate v1.1) -- versioned-ledger replay parity: this
        override was already the load-bearing reason mutating calls replay
        through `self.get_response(...)` (this class's override) rather
        than being skipped like the base env's mutating-tool replay path.
        Since v1.1's `get_response` now also diffs the DB around every
        mutating call to compute a real write-set (see above), replaying a
        mutating call here bumps `world_version` and appends to
        `mutation_log` in EXACTLY the position it occupied live (same
        call, same order) -- so invalidation state (which fresh
        observations are stale by the time a later proposal is verdicted)
        reconstructs identically, not just the verdict directions. Reads
        are fed into the ledger via `record_observation(..., is_mutating=
        False)` below at their recorded position too, so their
        `world_version` stamp (read at world_version N) also lines up
        call-for-call with the live run. `test_replay_parity_with_
        invalidation` in test_gated_env.py asserts this directly (full
        ledger/mutation_log equality, not just verdict equality).

        The rest of this method is a faithful copy of the base loop's
        structure (unknown-tool skip, strict comparison semantics) --
        deliberately NOT factored through super() because the base method
        does not expose an overridable per-action hook; duplicating the
        ~20 surrounding lines verbatim is safer than reimplementing them
        from a partial re-read.
        """
        if self.solo_mode:
            assert all(
                [not isinstance(message, UserMessage) for message in message_history]
            ), "User messages are not allowed in solo mode"

        def get_actions_from_messages(
            messages: list[Message],
        ) -> list[tuple[ToolCall, ToolMessage]]:
            messages = deepcopy(messages)[::-1]
            actions = []
            while messages:
                message = messages.pop()
                if isinstance(message, ToolMessage):
                    raise ValueError(
                        "Tool message not expected. Tool messages should always follow a tool call."
                    )
                if (
                    isinstance(message, (AssistantMessage, UserMessage))
                    and message.is_tool_call()
                ):
                    tool_calls = message.tool_calls
                    for tc in tool_calls:
                        if len(messages) == 0:
                            raise ValueError("Tool message expected. Got None.")
                        tm = messages.pop()
                        if not isinstance(tm, ToolMessage):
                            raise ValueError(f"Tool message expected. Got {type(tm)}")
                        if tc.id != tm.id:
                            raise ValueError(
                                f"Tool call id mismatch. Got {tc.id} and {tm.id}"
                            )
                        actions.append((tc, tm))
            return actions

        if initialization_data is not None:
            if initialization_data.agent_data is not None:
                self.tools.update_db(initialization_data.agent_data)
                if self.user_tools is not None and self.user_tools.db is not None:
                    self.user_tools.db = self.tools.db
            if initialization_data.user_data is not None:
                self.user_tools.update_db(initialization_data.user_data)
                if self.tools is not None and self.tools.db is not None:
                    self.tools.db = self.user_tools.db

        if initialization_actions is not None:
            for action in initialization_actions:
                self.run_env_function_call(action)

        action_responses = get_actions_from_messages(message_history)
        for tool_call, expected_response in action_responses:
            if not self._has_tool(tool_call.name):
                logger.debug(
                    f"Skipping unknown tool '{tool_call.name}' during replay "
                    "(no-op, matching live env behavior on hallucinated tools)."
                )
                continue

            eff_name, eff_args, _malformed = unwrap_effective(
                tool_call.name, tool_call.arguments
            )

            if not self._effective_is_mutating(eff_name):
                # LOCAL ADDITION (gate v1): feed the recorded result into
                # the gate ledger as a real observation, without executing
                # anything -- see docstring above.
                self._gate_ledger.record_observation(
                    eff_name,
                    eff_args,
                    expected_response.content,
                    tool_call.requestor,
                    expected_response.error,
                )
                continue

            response = self.get_response(tool_call)
            try:
                content = json.loads(response.content)
            except json.JSONDecodeError:
                content = response.content
            try:
                expected_content = json.loads(expected_response.content)
            except json.JSONDecodeError:
                expected_content = expected_response.content
            if content != expected_content:
                if strict:
                    raise ValueError(
                        f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}"
                    )
                logger.warning(
                    f"Replayed tool call '{tool_call.name}' returned different "
                    f"content than the recorded ToolMessage; continuing because "
                    f"strict=False. Recorded output may predate current tool "
                    f"code.\nTool call:\n{tool_call}"
                )
        self.sync_tools()


def get_environment(
    db: Optional[TransactionalDB] = None,
    retrieval_variant: Optional[str] = None,
    retrieval_kwargs: Optional[dict] = None,
    task: Optional[Task] = None,
    solo_mode: bool = False,
    read_log_allowlist: Optional[set] = None,
) -> Environment:
    """Get the banking_knowledge_gated domain environment.

    Identical construction to `banking_knowledge.environment.get_environment`
    (same db/tasks/policy/retrieval-variant machinery -- see that function
    for parameter docs) except it returns a `GatedEnvironment` registered
    under `domain_name="banking_knowledge_gated"`. The domain name is what
    the evaluator uses to reconstruct the environment for scoring/replay
    (evaluator.py:158,173) -- it MUST resolve to this gated constructor, not
    the plain banking_knowledge one, or replay would silently re-execute
    writes the live gate denied.
    """
    if solo_mode:
        raise ValueError("banking_knowledge_gated domain does not support solo mode")

    if db is None:
        db = get_db()

    knowledge_base = get_knowledge_base()

    variant_name = retrieval_variant or DEFAULT_RETRIEVAL_VARIANT
    kwargs = retrieval_kwargs or {}
    variant = resolve_variant(variant_name, **kwargs)

    tools = build_tools(
        variant, db, knowledge_base, read_log_allowlist=read_log_allowlist
    )
    user_tools = KnowledgeUserTools(db)
    policy = build_policy(variant, knowledge_base, task)

    return GatedEnvironment(
        domain_name="banking_knowledge_gated",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    """banking_knowledge_gated reuses the banking_knowledge task set
    verbatim (same task_*.json files, same tasks directory)."""
    return _banking_knowledge_get_tasks(task_split_name)
