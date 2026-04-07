"""LLM client with agentic loop for the steropes agent framework."""

import ast
import json
import textwrap
import threading
import time
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field, ValidationError

from .config import AgentConfig, ApiConfig
from .parsers import OllamaParser, OpenAIParser, ResponseParser
from .tools import _summarize_args, _total_message_chars, tool
from .types import TokenUsage, ToolCall
from .ui import ANSI_CYAN, ANSI_DIM, ansi, log, log_reasoning


class QueryResultArgs(BaseModel):
    result_id: str = Field(
        description="The result_id shown in the summarized tool output (e.g. 'r5')"
    )
    question: str = Field(
        description="Specific question to answer from the full stored result"
    )

    def post_process(self):
        return self


class LLMClient:
    """Encapsulates all communication with the LLM API (Ollama or OpenAI-compatible)."""

    def __init__(
        self,
        config: ApiConfig,
        agent_config: AgentConfig | None = None,
        log_fn: Callable[[str], None] | None = None,
        log_reasoning_fn: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.agent_config = agent_config or AgentConfig()
        self.usage = TokenUsage()
        self.parser: ResponseParser = (
            OllamaParser() if config.is_ollama else OpenAIParser()
        )
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=5.0),
        )
        self._last_result_store: dict[str, str] = {}
        self._last_messages: list[dict] = []
        self._log = log_fn if log_fn is not None else log
        self._log_reasoning = (
            log_reasoning_fn if log_reasoning_fn is not None else log_reasoning
        )

    # -- resource management ---------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _build_payload(
        self, prompt: str, tools: list[dict] | None = None, stream: bool = False
    ) -> dict:
        """Build an API request payload, adapting to Ollama or OpenAI format."""
        if self.config.is_ollama:
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "keep_alive": "1h",
                "options": {
                    "temperature": 0.2,
                    "num_ctx": self.config.num_ctx,
                },
            }
            if tools:
                payload["tools"] = tools
        else:
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "temperature": 0.2,
            }
            if tools:
                payload["tools"] = tools
                # Force the model to call the specific tool
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tools[0]["function"]["name"]},
                }
        return payload

    def _extract_tool_args(self, data: dict) -> dict | None:
        """Extract tool-call arguments from a response, handling both API formats."""
        return self.parser.extract_tool_args(data)

    def _build_payload_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
    ) -> dict:
        """Build a payload from a full conversation history (messages list)."""
        if self.config.is_ollama:
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "stream": stream,
                "keep_alive": "1h",
                "options": {
                    "temperature": 0.2,
                    "num_ctx": self.config.num_ctx,
                },
            }
            if tools:
                payload["tools"] = tools
            # Ollama does not support tool_choice — omit it
        else:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": stream,
                "temperature": 0.2,
            }
            if tools:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        return payload

    def _extract_tool_call(self, data: dict) -> ToolCall | None:
        """Extract a tool call from an API response, handling both Ollama and OpenAI formats."""
        return self.parser.extract_tool_call(data)

    def _extract_text_content(self, data: dict) -> str | None:
        """Extract plain text content from a response when no tool call is present."""
        return self.parser.extract_text(data)

    def _extract_reasoning(self, data: dict) -> str | None:
        """Extract reasoning/thinking tokens from an API response if present."""
        return self.parser.extract_reasoning(data)

    def _format_assistant_tool_call(self, data: dict) -> dict:
        """Extract the assistant message from a raw API response for conversation history."""
        return self.parser.format_assistant(data)

    def _format_tool_result(self, call_id: str | None, content: str) -> dict:
        """Build a tool-result message for appending to conversation history."""
        return self.parser.format_tool_result(call_id, content)

    def _extract_usage(self, data: dict) -> TokenUsage:
        """Extract token usage from an API response (Ollama or OpenAI format)."""
        return self.parser.extract_usage(data)

    def _retry_request(
        self, fn: Callable, max_retries: int = 3, initial_delay: float = 1.0
    ) -> Any:
        """Execute fn with exponential backoff on transient errors."""
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.ConnectTimeout,
            ) as e:
                if attempt == max_retries:
                    raise
                delay = initial_delay * (2**attempt)
                self._log(
                    f"  {ansi('↻', ANSI_DIM)} retry {attempt + 1}/{max_retries} after {delay:.1f}s: {type(e).__name__}"
                )
                time.sleep(delay)
            except httpx.HTTPStatusError as e:
                # Retry on 429 (rate limit) and 5xx (server errors)
                if (
                    e.response.status_code in (429, 500, 502, 503, 504)
                    and attempt < max_retries
                ):
                    delay = initial_delay * (2**attempt)
                    # Honor Retry-After header if present
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    self._log(
                        f"  {ansi('↻', ANSI_DIM)} retry {attempt + 1}/{max_retries} after {delay:.1f}s: HTTP {e.response.status_code}"
                    )
                    time.sleep(delay)
                else:
                    raise

    def call(self, prompt: str, num_ctx: int | None = None) -> str:
        """Call the LLM with streaming and return the assistant message text."""
        payload = self._build_payload(prompt, stream=True)
        if num_ctx is not None and self.config.is_ollama:
            payload["options"]["num_ctx"] = num_ctx
        headers = {**self.config.auth_headers, "Content-Type": "application/json"}

        def _do_stream() -> str:
            with self._http.stream(
                "POST", self.config.chat_url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                chunks: list[str] = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    # OpenAI SSE format: lines prefixed with "data: "
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if self.config.is_ollama:
                        content = data.get("message", {}).get("content", "")
                        done = data.get("done", False)
                    else:
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        done = (
                            data.get("choices", [{}])[0].get("finish_reason")
                            is not None
                        )
                    if content:
                        chunks.append(content)
                    if done:
                        self.usage = self.usage + self._extract_usage(data)
                        break
                return "".join(chunks)

        return self._retry_request(_do_stream)

    def call_with_tool(self, prompt: str, tool_fn: Callable, **tool_kwargs) -> Any:
        """Call the LLM forcing tool_fn to be called.

        Parses and validates the response into the tool's args model, then calls
        tool_fn(validated_args, **tool_kwargs) and returns its result.
        """
        spec = tool_fn._tool_spec
        args_model: type[BaseModel] = tool_fn._tool_model
        actual_prompt = prompt
        retries = 0
        failures = []
        while True:
            retries += 1
            payload = self._build_payload(actual_prompt, tools=[spec], stream=False)
            headers = {**self.config.auth_headers, "Content-Type": "application/json"}

            def _do_tool_post() -> dict:
                response = self._http.post(
                    self.config.chat_url, json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()

            data = self._retry_request(_do_tool_post)
            self.usage = self.usage + self._extract_usage(data)

            raw_args = self._extract_tool_args(data)
            if raw_args is None:
                raw_args = {}

            # Fix stringified nested values: some models return e.g.
            # {"commits": "[{'subject': ...}]"} instead of a proper nested structure.
            for key, val in raw_args.items():
                if isinstance(val, str) and val.strip().startswith(("[", "{")):
                    try:
                        raw_args[key] = json.loads(val)
                    except json.JSONDecodeError:
                        try:
                            raw_args[key] = ast.literal_eval(val)
                        except (ValueError, SyntaxError):
                            pass  # leave as-is, let pydantic report the error

            try:
                validated = args_model.model_validate(raw_args).post_process()
                return tool_fn(validated, **tool_kwargs)
            except ValidationError as e:
                failures.append(str(e))
                if retries > 5:
                    raise

                failure_summary = "\n".join(set(failures))
                actual_prompt = textwrap.dedent(f"""
                Your previous ({len(failures)}) attempts failed with a validation error:
                {failure_summary}

                Try again following this prompt exactly:
                {prompt}
                """)

    def agentic_loop(
        self,
        system_prompt: str,
        initial_user_message: str,
        tool_registry: dict[str, Callable],
        terminal_tool: Callable,
        max_turns: int | None = None,
        countdown_message_fn: Callable | None = None,
        nudge_message_fn: Callable[[int], str] | None = None,
        context_trim_threshold: int | None = None,
        **kwargs,
    ) -> Any:
        """Run a multi-turn agentic loop until the terminal tool is called.

        Each turn: send messages -> extract tool call -> dispatch.
        Investigation tool results are appended to history.
        Validation errors for the terminal tool become tool-result messages.
        On the last turn, only the terminal tool is offered.

        Context management: when total message size exceeds the trim threshold,
        old tool-result messages are replaced with LLM-generated summaries. The
        original full content is stored and accessible via the query_tool_result
        tool so the model can ask targeted questions without re-fetching.
        A countdown warning is injected when turns_warn_at turns remain.
        """
        ac = self.agent_config
        if max_turns is None:
            max_turns = ac.max_agentic_turns

        # ── dual-threshold setup (LCM §2.4) ──────────────────────────────────
        # If explicit soft/hard thresholds are configured, use them.
        # Otherwise fall back to the legacy single threshold for both.
        if context_trim_threshold is not None:
            _legacy = context_trim_threshold
        else:
            _legacy = ac.context_trim_threshold
        soft_threshold = ac.context_soft_threshold if ac.context_soft_threshold is not None else _legacy
        hard_threshold = ac.context_hard_threshold if ac.context_hard_threshold is not None else _legacy

        # ── result storage for compressed tool results ─────────────────────────
        result_store: dict[str, str] = {}  # result_id -> original full content

        def _summarize_and_store(msg_idx: int, content: str) -> str:
            """Summarize a tool result via LLM with three-level escalation."""
            result_id = f"r{msg_idx}"
            result_store[result_id] = content
            header = (
                f"[Summarized — id: '{result_id}'. "
                f"Use query_tool_result to ask specific questions about the full output.]\n"
            )
            input_snippet = content[: ac.tool_result_summarize_input]

            # Level 1 — Normal summary
            prompt_l1 = textwrap.dedent(f"""\
                Summarize this tool output in 2-4 sentences. Focus on filenames,
                key changes, and any suspicious code. Preserve suspicious code verbatim.

                {input_snippet}
            """)
            summary_l1 = self.call(prompt_l1).strip()
            if len(summary_l1) < len(content):
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} summarization level 1 (normal) succeeded for #{msg_idx}"
                )
                return header + summary_l1

            # Level 2 — Aggressive bullet points (target half the tokens)
            prompt_l2 = textwrap.dedent(f"""\
                Compress this to bullet points. Maximum 3 bullets, one line each.
                Preserve file names and key identifiers only.

                {input_snippet}
            """)
            summary_l2 = self.call(prompt_l2).strip()
            if len(summary_l2) < len(content):
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} summarization level 2 (aggressive) succeeded for #{msg_idx}"
                )
                return header + summary_l2

            # Level 3 — Deterministic truncate (always reduces size)
            self._log(
                f"  {ansi('⋯', ANSI_DIM)} summarization level 3 (deterministic truncate) for #{msg_idx}"
            )
            truncated = (
                content[:512] + "\n[Truncated — use query_tool_result for full content]"
            )
            return header + truncated

        def _compactable_candidates(messages: list[dict], target: int) -> list[int]:
            """Return indices of messages eligible for compaction, oldest first.

            Stops collecting once compacting the candidates would bring the
            total below *target* (optimistic estimate: each candidate shrinks
            to tool_result_summarize_skip size).
            """
            last_tool_idx = max(
                (i for i, m in enumerate(messages) if m.get("role") == "tool"),
                default=-1,
            )
            candidates: list[int] = []
            savings = 0
            total = _total_message_chars(messages)
            for i, msg in enumerate(messages):
                if total - savings <= target:
                    break
                role = msg.get("role")
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                if content.startswith("[Summarized"):
                    continue

                if role == "tool":
                    if (
                        i == last_tool_idx
                        and len(content) <= ac.recent_tool_result_chars
                    ):
                        continue
                    if len(content) <= ac.tool_result_summarize_skip:
                        continue
                    candidates.append(i)
                    savings += len(content) - ac.tool_result_summarize_skip
                elif role == "user" and i == 1:
                    if len(content) <= ac.tool_result_summarize_skip:
                        continue
                    candidates.append(i)
                    savings += len(content) - ac.tool_result_summarize_skip
            return candidates

        def _compact_indices(messages: list[dict], indices: list[int]) -> None:
            """Synchronously summarize messages at the given indices in place."""
            for i in indices:
                msg = messages[i]
                content = msg.get("content", "")
                if not isinstance(content, str) or content.startswith("[Summarized"):
                    continue
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} compressing message #{i} ({len(content):,} chars)"
                )
                messages[i] = {**msg, "content": _summarize_and_store(i, content)}

        # ── async compaction state (LCM §2.4: between-turn atomic swap) ───────
        _compaction_lock = threading.Lock()
        _compaction_thread: threading.Thread | None = None
        # Pending replacements produced by the background thread:
        #   list of (msg_index, new_content_string)
        _pending_swaps: list[tuple[int, str]] = []

        def _run_background_compaction(
            candidates: list[tuple[int, str]],
        ) -> None:
            """Run in a background thread: summarize candidates, store results.

            *candidates* is a list of (msg_index, original_content) snapshots
            taken from the message list at spawn time.  Results are written to
            _pending_swaps and applied atomically by the main loop.
            """
            swaps: list[tuple[int, str]] = []
            for idx, content in candidates:
                summarized = _summarize_and_store(idx, content)
                swaps.append((idx, summarized))
            with _compaction_lock:
                _pending_swaps.extend(swaps)

        def _apply_pending_swaps(messages: list[dict]) -> int:
            """Apply pending background compaction results to messages.

            Returns the number of swaps applied.
            """
            with _compaction_lock:
                swaps = list(_pending_swaps)
                _pending_swaps.clear()
            applied = 0
            for idx, new_content in swaps:
                if idx < len(messages):
                    old = messages[idx].get("content", "")
                    if isinstance(old, str) and not old.startswith("[Summarized"):
                        messages[idx] = {**messages[idx], "content": new_content}
                        applied += 1
            return applied

        def _maybe_start_async_compaction(messages: list[dict]) -> None:
            """If no compaction is running and context exceeds τ_soft, spawn one."""
            nonlocal _compaction_thread
            if _compaction_thread is not None and _compaction_thread.is_alive():
                return  # already running
            indices = _compactable_candidates(messages, soft_threshold)
            if not indices:
                return
            # Snapshot content so the thread doesn't touch the live list
            snapshots = [
                (i, messages[i].get("content", "")) for i in indices
            ]
            self._log(
                f"  {ansi('⋯', ANSI_DIM)} τ_soft exceeded — async compaction "
                f"of {len(snapshots)} message(s) in background"
            )
            _compaction_thread = threading.Thread(
                target=_run_background_compaction,
                args=(snapshots,),
                daemon=True,
            )
            _compaction_thread.start()

        def _blocking_compaction(messages: list[dict]) -> None:
            """Block until context is under τ_hard (safety net)."""
            nonlocal _compaction_thread
            # First, wait for any in-flight async compaction to finish
            if _compaction_thread is not None and _compaction_thread.is_alive():
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} τ_hard exceeded — waiting for async compaction"
                )
                _compaction_thread.join()
            _apply_pending_swaps(messages)

            # If still over hard threshold, compact synchronously
            if _total_message_chars(messages) > hard_threshold:
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} τ_hard still exceeded — blocking compaction"
                )
                indices = _compactable_candidates(messages, hard_threshold)
                _compact_indices(messages, indices)

        # ── query_tool_result: closure over result_store and self ───────────────
        @tool(QueryResultArgs)
        def query_tool_result(args: QueryResultArgs, **_: Any) -> str:
            """Access the full (unsummarized) content of a previously summarized tool result.

            When a tool result was compressed to save context, use this to ask a
            targeted question. The answer is generated from the original full content
            so no information is lost.
            """
            full = result_store.get(args.result_id)
            if full is None:
                available = list(result_store.keys()) or ["none"]
                return (
                    f"No stored result with id '{args.result_id}'. "
                    f"Available ids: {available}"
                )
            prompt = textwrap.dedent(f"""\
                The following is the full content of stored tool result '{args.result_id}':

                {full[:50_000]}

                Question: {args.question}

                Answer concisely and directly based only on the content above.
            """)
            return self.call(prompt)

        # ── build extended registry that includes query_tool_result ─────────────
        extended_registry = dict(tool_registry)
        extended_registry["query_tool_result"] = query_tool_result

        # For Anthropic, use structured content blocks with cache_control so the
        # static system prompt is prefix-cached across all turns in the agentic loop.
        if self.config.is_anthropic:
            system_content: str | list = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_content = system_prompt

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": initial_user_message},
        ]

        terminal_spec = terminal_tool._tool_spec
        terminal_name: str = terminal_spec["function"]["name"]
        terminal_model: type[BaseModel] = terminal_tool._tool_model
        all_tool_specs = [fn._tool_spec for fn in extended_registry.values()] + [
            terminal_spec
        ]

        headers = {**self.config.auth_headers, "Content-Type": "application/json"}

        consecutive_no_tool_calls = 0
        terminal_validation_error: Exception | None = None

        for turn in range(1, max_turns + 1):
            is_last_turn = (turn == max_turns) or (consecutive_no_tool_calls >= 3)
            turns_remaining = max_turns - turn

            if is_last_turn:
                tools = [terminal_spec]
                tool_choice: str | dict | None = (
                    None
                    if self.config.is_ollama
                    else {"type": "function", "function": {"name": terminal_name}}
                )
            else:
                tools = all_tool_specs
                tool_choice = None if self.config.is_ollama else "auto"

            payload = self._build_payload_messages(
                messages, tools=tools, tool_choice=tool_choice, stream=False
            )

            def _do_agentic_post() -> dict:
                response = self._http.post(
                    self.config.chat_url, json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()

            data = self._retry_request(_do_agentic_post)
            self.usage = self.usage + self._extract_usage(data)

            reasoning = self._extract_reasoning(data)
            if reasoning:
                self._log_reasoning(reasoning)

            tool_call = self._extract_tool_call(data)

            if tool_call is None:
                text = self._extract_text_content(data) or ""
                consecutive_no_tool_calls += 1
                self._log(
                    f"  {ansi('⟳', ANSI_DIM)} {ansi(f'[{turn}/{max_turns}]', ANSI_DIM)}"
                    f" {ansi(f'(no tool call — nudging #{consecutive_no_tool_calls})', ANSI_DIM)}"
                )
                messages.append({"role": "assistant", "content": text})
                if nudge_message_fn is not None:
                    nudge_text = nudge_message_fn(consecutive_no_tool_calls)
                else:
                    nudge_text = (
                        f"Please call a tool. Use the investigation tools to gather more "
                        f"information, or call {terminal_name} if you are ready to propose commits."
                    )
                messages.append({"role": "user", "content": nudge_text})
                continue

            consecutive_no_tool_calls = 0
            self._log(
                f"  {ansi('⟳', ANSI_DIM)} {ansi(f'[{turn}/{max_turns}]', ANSI_DIM)}"
                f" {ansi(tool_call.name, ANSI_CYAN)}"
                f"({ansi(_summarize_args(tool_call.arguments), ANSI_DIM)})"
            )
            messages.append(self._format_assistant_tool_call(data))

            # Coerce stringified nested values (some models return JSON-as-string)
            raw_args = dict(tool_call.arguments)
            for key, val in raw_args.items():
                if isinstance(val, str) and val.strip().startswith(("[", "{")):
                    try:
                        raw_args[key] = json.loads(val)
                    except json.JSONDecodeError:
                        try:
                            raw_args[key] = ast.literal_eval(val)
                        except (ValueError, SyntaxError):
                            pass

            if tool_call.name == terminal_name:
                try:
                    validated = terminal_model.model_validate(raw_args).post_process()
                    self._last_messages = list(messages)
                    self._last_result_store = dict(result_store)
                    return terminal_tool(validated, **kwargs)
                except (ValidationError, ValueError) as e:
                    terminal_validation_error = e
                    prefix = (
                        "Validation error"
                        if isinstance(e, ValidationError)
                        else "Error"
                    )
                    err_str = (
                        f"{prefix}: {e}\nPlease fix the arguments and call "
                        f"{terminal_name} again."
                    )
                    if is_last_turn and not _extended_for_terminal_validation:
                        max_turns += 1
                        _extended_for_terminal_validation = True
                    if len(err_str) > ac.tool_result_summarize_skip:
                        err_str = _summarize_and_store(len(messages), err_str)
                    messages.append(
                        self._format_tool_result(
                            tool_call.call_id,
                            err_str,
                        )
                    )
                    continue

            tool_fn = extended_registry.get(tool_call.name)
            if tool_fn is None:
                result_str = (
                    f"Unknown tool: {tool_call.name!r}. "
                    f"Available: {list(extended_registry.keys())}"
                )
            else:
                try:
                    validated_args = tool_fn._tool_model.model_validate(
                        raw_args
                    ).post_process()
                    result = tool_fn(validated_args, **kwargs)
                    result_str = (
                        result if isinstance(result, str) else json.dumps(result)
                    )
                except Exception as e:
                    result_str = f"Error calling {tool_call.name}: {e}"

            # Proactive summarization: summarize large tool results immediately
            if len(result_str) > ac.tool_result_summarize_skip:
                result_str = _summarize_and_store(len(messages), result_str)

            messages.append(self._format_tool_result(tool_call.call_id, result_str))

            # ── between-turn context management (LCM dual-threshold) ──────────
            # Step 1: apply any completed async compaction results
            n_swapped = _apply_pending_swaps(messages)
            if n_swapped:
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} applied {n_swapped} async compaction result(s)"
                )

            total_chars = _total_message_chars(messages)

            # Step 2: if over τ_hard, block until under threshold
            if total_chars > hard_threshold:
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} context {total_chars:,} chars > τ_hard ({hard_threshold:,}) — blocking"
                )
                _blocking_compaction(messages)
            # Step 3: if over τ_soft, trigger async compaction for next turn
            elif total_chars > soft_threshold:
                self._log(
                    f"  {ansi('⋯', ANSI_DIM)} context {total_chars:,} chars > τ_soft ({soft_threshold:,})"
                )
                _maybe_start_async_compaction(messages)

            # Inject a countdown warning when few turns remain
            if 0 < turns_remaining <= ac.turns_warn_at:
                if countdown_message_fn is not None:
                    warning_text = countdown_message_fn(turns_remaining)
                else:
                    warning_text = (
                        f"WARNING: Only {turns_remaining} turn(s) remaining. "
                        f"Stop gathering context and call {terminal_name} NOW with "
                        f"your best grouping based on what you already know."
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": warning_text,
                    }
                )

        self._last_messages = list(messages)
        self._last_result_store = dict(result_store)

        if terminal_validation_error is not None:
            # The terminal tool was called on the last turn but failed validation.
            # Grant one bonus turn so the model can fix its arguments.
            self._log(
                f"  {ansi('↻', ANSI_DIM)} {terminal_name} failed validation on last turn"
                f" — granting one bonus turn"
            )
            bonus_tools = [terminal_spec]
            bonus_tool_choice: str | dict | None = (
                None
                if self.config.is_ollama
                else {"type": "function", "function": {"name": terminal_name}}
            )
            bonus_payload = self._build_payload_messages(
                messages, tools=bonus_tools, tool_choice=bonus_tool_choice, stream=False
            )

            def _do_bonus_post() -> dict:
                response = self._http.post(
                    self.config.chat_url, json=bonus_payload, headers=headers
                )
                response.raise_for_status()
                return response.json()

            try:
                bonus_data = self._retry_request(_do_bonus_post)
                self.usage = self.usage + self._extract_usage(bonus_data)
                bonus_call = self._extract_tool_call(bonus_data)
                if bonus_call is not None and bonus_call.name == terminal_name:
                    bonus_raw = dict(bonus_call.arguments)
                    for key, val in bonus_raw.items():
                        if isinstance(val, str) and val.strip().startswith(("[", "{")):
                            try:
                                bonus_raw[key] = json.loads(val)
                            except json.JSONDecodeError:
                                try:
                                    bonus_raw[key] = ast.literal_eval(val)
                                except (ValueError, SyntaxError):
                                    pass
                    messages.append(self._format_assistant_tool_call(bonus_data))
                    validated = terminal_model.model_validate(bonus_raw).post_process()
                    self._last_messages = list(messages)
                    self._last_result_store = dict(result_store)
                    return terminal_tool(validated, **kwargs)
            except Exception:
                pass

            raise RuntimeError(
                f"agentic_loop: {terminal_name} was called on the last turn but failed "
                f"validation; bonus retry also failed. "
                f"Last error: {terminal_validation_error}"
            )

        raise RuntimeError(
            f"agentic_loop exceeded {max_turns} turns without calling {terminal_name}"
        )
