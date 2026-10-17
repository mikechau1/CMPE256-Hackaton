"""Pydantic data contracts shared by retrieval, agents, the manager and the UI.

Models suffixed with ``Out`` are produced by Claude via structured outputs, so
they avoid defaults the strict JSON schema cannot express. Everything else is
internal state.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["technology", "clinical", "sec", "csr", "procurement"]
ConfidenceLabel = Literal["high", "medium", "low"]
TaskStatus = Literal["pending", "running", "done", "failed", "needs_input"]


# --- Retrieval -------------------------------------------------------------
class Evidence(BaseModel):
    """A retrieved chunk. ``ref`` is the citation key agents use, e.g. ``TEC-3``."""

    ref: str = ""
    chunk_id: str
    doc_id: str
    title: str
    source: str
    page: int | None = None
    section: str | None = None
    domain: str
    doc_type: str
    text: str
    score: float = 0.0

    def citation(self) -> str:
        loc = f", p.{self.page}" if self.page else ""
        sec = f", {self.section}" if self.section else ""
        return f"{self.title}{sec}{loc}"


# --- Routing & planning ----------------------------------------------------
class RouteOut(BaseModel):
    in_scope: bool = Field(description="True if any part of the request can be answered from the knowledge bases.")
    out_of_scope_parts: list[str] = Field(
        description="Parts of the request the knowledge bases do not cover (empty if fully covered).")
    domains: list[Domain] = Field(description="All domains needed, most relevant first.")
    confidence: float = Field(description="0-1 confidence in this routing decision.")
    rationale: str
    needs_clarification: bool
    clarification_question: str = Field(description="Empty string if none is needed.")


class SubtaskSpec(BaseModel):
    agent: Domain
    instruction: str = Field(description="Self-contained task for the specialist agent.")


class PlanOut(BaseModel):
    approach: str = Field(description="One-paragraph resolution approach.")
    subtasks: list[SubtaskSpec]


class Subtask(BaseModel):
    id: int
    agent: Domain
    instruction: str
    status: TaskStatus = "pending"
    iteration: int = 1
    summary: str = ""


class Ledger(BaseModel):
    """Task & progress ledger from the architecture diagram."""

    request: str
    approach: str = ""
    route: RouteOut | None = None
    subtasks: list[Subtask] = []
    iterations: int = 0
    resolved: bool = False
    log: list[str] = []


# --- Agent outputs ---------------------------------------------------------
class FindingOut(BaseModel):
    statement: str
    status: Literal["confirmed", "hypothesis"] = Field(
        description="confirmed only if directly supported by cited evidence or customer-provided facts."
    )
    citations: list[str] = Field(description="Evidence refs such as TEC-2.")
    confidence: ConfidenceLabel


class RecommendationOut(BaseModel):
    action: str
    rationale: str = Field(description="Why this action is appropriate.")
    priority: Literal["P1", "P2", "P3"]
    owner_timeline: str
    kpi: str = Field(description="How the impact will be measured.")
    citations: list[str]
    confidence: ConfidenceLabel


class AgentResultOut(BaseModel):
    summary: str
    findings: list[FindingOut]
    recommendations: list[RecommendationOut]
    missing_information: list[str] = Field(
        description="Diagnostics or facts needed from the customer; empty if none."
    )
    self_confidence: float = Field(description="0-1 confidence the evidence answers the task.")


class AgentResult(BaseModel):
    agent: Domain
    subtask_id: int
    output: AgentResultOut
    evidence: list[Evidence]
    tool_calls: list[dict] = []
    confidence: float = 0.0
    confidence_label: ConfidenceLabel = "low"


class GoalEvalOut(BaseModel):
    resolved: bool
    score: float = Field(description="0-1 how completely the request is answered.")
    gaps: list[str]
    followups: list[SubtaskSpec] = Field(description="New subtasks to close gaps; empty if resolved.")
    ask_human: bool = Field(description="True if only the customer can supply what is missing.")
    human_question: str


class ReportOut(BaseModel):
    executive_summary: str
    findings: list[FindingOut]
    recommendations: list[RecommendationOut]
    risks: list[str]
    information_needed: list[str]
    next_steps: list[str]
    overall_confidence: ConfidenceLabel


# Irrelevant request: declined by the scope gate before any agent runs.
OUT_OF_SCOPE_MESSAGE = "The agent cannot answer this question because it is not relevant to the knowledge base."
# Relevant topic, but the agents found no supporting evidence in the documents.
NO_INFORMATION_MESSAGE = "There is no information from the knowledge base to answer the request."


class ConsultingReport(BaseModel):
    request: str
    status: Literal["answered", "out_of_scope", "needs_clarification"] = "answered"
    scope_note: str = ""  # e.g. which parts of a mixed request were outside the knowledge base
    report: ReportOut
    ledger: Ledger
    agent_results: list[AgentResult]
    evidence: dict[str, Evidence]
    confidence: float
    usage: dict = {}
