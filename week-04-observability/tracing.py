import time
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler

from llm import MODEL

# USD per million tokens, (input, output). Checked against the live OpenRouter
# catalogue on 2026-09-16: anthropic/claude-haiku-4.5 is $1 in, $5 out.
PRICES = {"anthropic/claude-haiku-4.5": (1.0, 5.0)}


@dataclass
class Span:
    """One node, model call or tool call, between its start and end events."""

    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start_ns: int
    end_ns: int | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ms(self) -> float:
        return ((self.end_ns or self.start_ns) - self.start_ns) / 1e6

    @property
    def cost(self) -> float:
        per_in, per_out = PRICES.get(MODEL, (0.0, 0.0))
        return (self.input_tokens * per_in + self.output_tokens * per_out) / 1e6


def _usage(response):
    """Token counts off a model reply, whichever shape the provider filled in."""
    for batch in response.generations:
        for generation in batch:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    totals = (response.llm_output or {}).get("token_usage") or {}
    return totals.get("prompt_tokens", 0), totals.get("completion_tokens", 0)


class Tracer(BaseCallbackHandler):
    """Turns callback events into spans. One instance per run."""

    def __init__(self):
        self.spans: dict[str, Span] = {}
        self.order: list[str] = []

    def _open(self, kind, name, run_id, parent_run_id):
        key = str(run_id)  # the full uuid: LangGraph ids share their first 8 characters
        parent = str(parent_run_id) if parent_run_id else None
        self.spans[key] = Span(key, parent, name or kind, kind, time.time_ns())
        self.order.append(key)

    def _close(self, run_id, error=None):
        span = self.spans.get(str(run_id))
        if span is None:
            return
        span.end_ns = time.time_ns()
        if error is not None:
            span.error = str(error)[:200]

    @staticmethod
    def _name(serialized, kwargs):
        return (serialized or {}).get("name") or kwargs.get("name")

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self._open("chain", self._name(serialized, kwargs), run_id, parent_run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._close(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._close(run_id, error)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._open("llm", self._name(serialized, kwargs), run_id, parent_run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        span = self.spans.get(str(run_id))
        if span is not None:
            span.input_tokens, span.output_tokens = _usage(response)
        self._close(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._close(run_id, error)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._open("tool", self._name(serialized, kwargs), run_id, parent_run_id)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._close(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._close(run_id, error)

    # --- reading the run back ------------------------------------------------

    def ordered(self):
        return [self.spans[key] for key in self.order]

    def children(self, span_id):
        return [s for s in self.ordered() if s.parent_id == span_id]

    def roots(self):
        return [s for s in self.ordered() if s.parent_id not in self.spans]

    def tools_ran(self):
        return [s.name for s in self.ordered() if s.kind == "tool"]

    def tokens(self):
        return (sum(s.input_tokens for s in self.spans.values()),
                sum(s.output_tokens for s in self.spans.values()))

    def cost(self):
        return sum(s.cost for s in self.spans.values())

    def seconds(self):
        done = [s for s in self.spans.values() if s.end_ns]
        if not done:
            return 0.0
        return (max(s.end_ns for s in done) - min(s.start_ns for s in done)) / 1e9

    def node_of(self, span):
        """The graph node a span sits under: walk up until the parent is the root."""
        current = span
        while current.parent_id in self.spans:
            parent = self.spans[current.parent_id]
            if parent.parent_id not in self.spans:
                return current.name
            current = parent
        return current.name

    def by_node(self):
        """Calls, tokens and cost per graph node, counting the model calls only."""
        totals = {}
        for span in self.ordered():
            if span.kind == "llm":
                row = totals.setdefault(self.node_of(span), [0, 0, 0, 0.0])
                row[0] += 1
                row[1] += span.input_tokens
                row[2] += span.output_tokens
                row[3] += span.cost
        return totals


def print_tree(tracer, title=None):
    if title:
        print(title)

    def walk(span, depth):
        label = "  " * depth + f"[{span.kind}] {span.name}"
        line = f"{label:<52} {span.ms:8.1f} ms"
        if span.input_tokens or span.output_tokens:
            line += f"   in {span.input_tokens:>6}  out {span.output_tokens:>5}  ${span.cost:.6f}"
        if span.error:
            line += f"   ERROR {span.error}"
        print(line)
        for child in tracer.children(span.span_id):
            walk(child, depth + 1)

    for root in tracer.roots():
        walk(root, 0)
    tokens_in, tokens_out = tracer.tokens()
    print(f"{'total':<52} {tracer.seconds() * 1000:8.1f} ms"
          f"   in {tokens_in:>6}  out {tokens_out:>5}  ${tracer.cost():.6f}")
