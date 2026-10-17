"""Domain agent base class: system prompt + tools + Claude tool loop -> AgentResult."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable

from .. import config
from ..confidence import agent_confidence, label
from ..llm import LLM, get_llm
from ..retrieval.hybrid import get_retriever
from ..schemas import AgentResult, AgentResultOut, Evidence, Subtask

EventFn = Callable[[str, dict], None]

BASE_RULES = """\
You are a specialist agent inside an AI management-consulting platform. A manager agent
gives you one subtask; you research it with your tools and return a structured result.

Grounding rules:
- Use your tools before answering. Every finding and recommendation must cite the evidence
  refs (e.g. "TEC-3") returned by tools that support it. Never invent refs.
- Cite precisely: a ref belongs on a claim only if that excerpt itself states the fact (number,
  feature, risk). Do not attach loosely related refs. Keep numbers exactly as written in the
  excerpt; if no excerpt states something, either drop it or label it a hypothesis without refs.
- Mark a finding "confirmed" only when cited evidence or facts stated by the customer directly
  establish it. Causes inferred from symptoms are "hypothesis".
- If the request needs facts only the customer has (model numbers, logs, volumes, budgets),
  list them in missing_information instead of assuming them.
- If the knowledge base does not cover something, say so plainly; do not fill gaps from memory.
- Recommendations must say what to do, why, priority (P1 urgent / P2 next / P3 later), a
  realistic owner and timeline, and a KPI that measures impact.
- Be concise and specific. Prefer 3-6 findings and 2-5 recommendations.
"""


class EvidenceRegistry:
    """Assigns globally unique citation refs across agents running in parallel."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self.items: dict[str, Evidence] = {}

    def add(self, ev: Evidence, prefix: str) -> Evidence:
        with self._lock:
            for existing in self.items.values():  # same chunk seen before -> reuse ref
                if existing.chunk_id == ev.chunk_id and existing.ref.startswith(prefix):
                    return existing
            n = self._counters.get(prefix, 0) + 1
            self._counters[prefix] = n
            ev = ev.model_copy(update={"ref": f"{prefix}-{n}"})
            self.items[ev.ref] = ev
            return ev


def format_evidence(evs: list[Evidence], max_chars: int = 1400) -> str:
    if not evs:
        return "No matching evidence found in the knowledge base."
    blocks = []
    for e in evs:
        text = e.text if len(e.text) <= max_chars else e.text[:max_chars] + " …"
        blocks.append(f"[{e.ref}] {e.citation()} (relevance {e.score:.2f})\n{text}")
    return "\n\n".join(blocks)


class DomainAgent:
    domain: str = ""
    prefix: str = ""
    role: str = ""
    tools: list[dict] = []

    def __init__(self, registry: EvidenceRegistry, llm: LLM | None = None):
        self.registry = registry
        self.llm = llm or get_llm()
        self._evidence: list[Evidence] = []
        self._calls: list[dict] = []

    # -- helpers for subclasses ------------------------------------------------
    def search(self, domain: str, query: str, k: int = config.DEFAULT_TOP_K, filters: dict | None = None,
               prefix: str | None = None) -> str:
        evs = get_retriever(domain).search(query, k=k, filters=filters or None)
        evs = [self.registry.add(e, prefix or self.prefix) for e in evs]
        self._evidence.extend(evs)
        return format_evidence(evs)

    def register_record(self, title: str, text: str, source: str, score: float, doc_id: str,
                        page: int | None = None) -> Evidence:
        """Register a structured recommender output (catalog row, trial, rule) as citable evidence."""
        ev = Evidence(chunk_id=f"rec-{self.domain}-{doc_id}", doc_id=doc_id, title=title, source=source,
                      page=page, domain=self.domain, doc_type="recommendation", text=text, score=score)
        ev = self.registry.add(ev, self.prefix)
        self._evidence.append(ev)
        return ev

    def handle(self, name: str, args: dict) -> str:  # overridden
        raise ValueError(f"Unknown tool {name}")

    # -- run ------------------------------------------------------------------
    def run(self, subtask: Subtask, request: str, context: str = "", on_event: EventFn | None = None) -> AgentResult:
        def on_tool(name: str, args: dict) -> None:
            self._calls.append({"tool": name, "args": args})
            if on_event:
                on_event("tool_call", {"agent": self.domain, "subtask_id": subtask.id, "tool": name, "args": args})

        prompt = (
            f"Customer request:\n{request}\n\n"
            + (f"Context from the conversation / other agents:\n{context}\n\n" if context else "")
            + f"Your subtask:\n{subtask.instruction}\n\n"
            "Research with your tools, then return the structured result."
        )
        out = self.llm.tool_loop(
            AgentResultOut, prompt, system=f"{BASE_RULES}\n{self.role}", tools=self.tools,
            handler=self.handle, on_tool=on_tool,
        )
        valid = {e.ref for e in self._evidence}
        for item in [*out.findings, *out.recommendations]:  # drop hallucinated refs
            item.citations = [c for c in item.citations if c in valid]
        conf, _ = agent_confidence(out, self._evidence)
        return AgentResult(agent=self.domain, subtask_id=subtask.id, output=out, evidence=self._evidence,
                           tool_calls=self._calls, confidence=conf, confidence_label=label(conf))


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required}}


def as_json(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
