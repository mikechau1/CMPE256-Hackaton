"""Management Consulting Manager Agent.

Implements the architecture diagram:
  Input -> intent detection/routing -> resolution plan (task & progress ledger)
        -> dispatch to specialist agents (parallel)
        -> evaluate-goal loop ("resolved?"; No -> follow-up subtasks, or ask the human)
        -> consolidated consulting recommendation with evidence, confidence and next actions.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .. import config
from ..confidence import overall_confidence
from ..llm import LLM, get_llm
from ..routing import llm_route
from ..schemas import (NO_INFORMATION_MESSAGE, OUT_OF_SCOPE_MESSAGE, AgentResult, ConsultingReport, GoalEvalOut, Ledger, PlanOut,
                       ReportOut, RouteOut, Subtask)
from .base import EvidenceRegistry
from .clinical import ClinicalAgent
from .csr import CSRAgent
from .procurement import ProcurementAgent
from .sec import SECAgent
from .technology import TechnologyAgent

AGENTS = {"technology": TechnologyAgent, "clinical": ClinicalAgent, "sec": SECAgent, "csr": CSRAgent,
          "procurement": ProcurementAgent}
EventFn = Callable[[str, dict], None]
MIN_WORDS_TO_ENGAGE = 4

PLANNER_SYSTEM = """\
You are the Management Consulting Manager Agent. Turn a customer request into a resolution
plan of self-contained subtasks for specialist agents (technology, clinical, sec, csr,
procurement). Use only the agents you were given unless another is clearly essential.
Give each agent one focused subtask (two only if the request has clearly separate parts).
For cross-domain requests, tell each agent which aspect it owns and what the other agents cover,
so their findings can be combined (e.g. tech remediation + vendor financial risk + ESG impact).
If the request is vague, still plan subtasks: ask each specialist for a best-effort answer that covers
the most likely scenarios from its knowledge base, plus the specific details it needs from the customer."""

EVALUATOR_SYSTEM = """\
You evaluate whether specialist findings resolve the customer's request (the "live-site issue
resolved?" check). resolved=true when the findings answer every part of the request with
cited evidence, or when the only remaining gaps are facts that must come from the customer.
Propose follow-up subtasks only for gaps another research pass could actually close with the
available knowledge bases. Set ask_human=true when progress depends on customer-only facts
(model numbers, logs, budgets, volumes) and phrase one concise human_question."""

SYNTH_SYSTEM = """\
You are the Management Consulting Manager Agent writing the final client recommendation.
Consolidate the specialists' findings into one consulting report:
- executive_summary: 3-5 sentences, the answer first.
- findings: deduplicated, each with status (confirmed/hypothesis) and evidence refs exactly as
  given (e.g. TEC-2, SEC-5). Keep hypotheses labelled as hypotheses.
- recommendations: prioritised actions (P1/P2/P3) with why, owner/timeline and a measurable KPI;
  cite refs. Cover remediation, compatibility, procurement priorities, operational risk and
  prevention where relevant.
- risks: operational, financial, regulatory or execution risks and dependencies.
- information_needed: what the customer must provide to raise confidence.
- next_steps: the concrete next 3-5 actions in order.
Only use refs that appear in the specialist results, and keep each claim attached to the refs the
specialist used for that same claim (do not move refs between claims). Do not add facts, numbers or
products that are not in the specialist results."""


class ManagerAgent:
    def __init__(self, llm: LLM | None = None, on_event: EventFn | None = None,
                 max_loops: int = config.MAX_GOAL_LOOPS):
        self.llm = llm or get_llm()
        self.on_event = on_event or (lambda kind, payload: None)
        self.max_loops = max_loops

    # -- steps ----------------------------------------------------------------
    def route(self, request: str) -> RouteOut:
        return llm_route(request, self.llm)

    def plan(self, request: str, route: RouteOut) -> PlanOut:
        prompt = (f"Customer request:\n{request}\n\nRouter selected agents: {', '.join(route.domains)}\n"
                  f"Router rationale: {route.rationale}")
        plan = self.llm.parse(PlanOut, prompt, system=PLANNER_SYSTEM, effort="low", max_tokens=4000)
        if not plan.subtasks:  # never leave the ledger empty
            from ..schemas import SubtaskSpec
            plan.subtasks = [SubtaskSpec(agent=d, instruction=request) for d in route.domains]
        return plan

    def _run_subtask(self, st: Subtask, request: str, context: str, registry: EvidenceRegistry,
                     ledger: Ledger) -> AgentResult | None:
        st.status = "running"
        self.on_event("task_update", {"ledger": ledger})
        try:
            agent = AGENTS[st.agent](registry, self.llm)
            res = agent.run(st, request, context, on_event=self.on_event)
            st.status = "needs_input" if res.output.missing_information else "done"
            st.summary = res.output.summary
            self.on_event("agent_result", {"result": res})
            return res
        except Exception as e:  # one failing agent should not sink the engagement
            st.status = "failed"
            st.summary = f"{type(e).__name__}: {e}"
            ledger.log.append(f"Subtask {st.id} ({st.agent}) failed: {e}")
            return None
        finally:
            self.on_event("task_update", {"ledger": ledger})

    def dispatch(self, subtasks: list[Subtask], request: str, context: str, registry: EvidenceRegistry,
                 ledger: Ledger) -> list[AgentResult]:
        with ThreadPoolExecutor(max_workers=max(1, len(subtasks))) as pool:
            futures = [pool.submit(self._run_subtask, st, request, context, registry, ledger) for st in subtasks]
            return [r for f in futures if (r := f.result()) is not None]

    def evaluate(self, request: str, results: list[AgentResult]) -> GoalEvalOut:
        digest = [{"agent": r.agent, "summary": r.output.summary,
                   "findings": [f.statement for f in r.output.findings],
                   "missing_information": r.output.missing_information,
                   "confidence": r.confidence} for r in results]
        prompt = f"Customer request:\n{request}\n\nSpecialist results:\n{json.dumps(digest, indent=1)}"
        return self.llm.parse(GoalEvalOut, prompt, system=EVALUATOR_SYSTEM, effort="low", max_tokens=4000)

    def synthesize(self, request: str, results: list[AgentResult], registry: EvidenceRegistry,
                   history: str) -> ReportOut:
        payload = [{"agent": r.agent, "confidence": r.confidence_label, **r.output.model_dump()} for r in results]
        cited = {c for r in results for it in [*r.output.findings, *r.output.recommendations] for c in it.citations}
        sources = {ref: registry.items[ref].citation() for ref in sorted(cited) if ref in registry.items}
        prompt = (f"Customer request:\n{request}\n\n"
                  + (f"Earlier conversation:\n{history}\n\n" if history else "")
                  + f"Specialist results:\n{json.dumps(payload, indent=1)}\n\nEvidence index:\n"
                  + json.dumps(sources, indent=1))
        report = self.llm.parse(ReportOut, prompt, system=SYNTH_SYSTEM, effort="medium", max_tokens=12000)
        valid = set(registry.items)
        for it in [*report.findings, *report.recommendations]:
            it.citations = [c for c in it.citations if c in valid]
        return report

    @staticmethod
    def _must_ask_first(request: str, route: RouteOut, history: str = "") -> bool:
        """Only near-empty first messages ("help", "advise us") are bounced back before any agent runs.

        Anything with real content (or a short reply in an ongoing conversation) is answered best-effort;
        the router's clarification question is added to the report's 'information needed' instead.
        """
        return (route.needs_clarification and bool(route.clarification_question) and not history
                and len(request.split()) < MIN_WORDS_TO_ENGAGE)

    def _ask_human_only(self, request: str, ledger: Ledger, registry: EvidenceRegistry) -> ConsultingReport:
        q = ledger.route.clarification_question
        ledger.log.append("Request too ambiguous to route; asking the human participant.")
        report = ReportOut(executive_summary="I need one clarification before engaging the specialist agents.",
                           findings=[], recommendations=[], risks=[], information_needed=[q], next_steps=[q],
                           overall_confidence="low")
        out = ConsultingReport(request=request, status="needs_clarification", report=report, ledger=ledger,
                               agent_results=[], evidence={}, confidence=0.0, usage=dict(self.llm.usage))
        self.on_event("report", {"report": out})
        return out

    def _out_of_scope(self, request: str, ledger: Ledger, reason: str, results: list[AgentResult] | None = None,
                      evidence: dict | None = None, message: str = OUT_OF_SCOPE_MESSAGE) -> ConsultingReport:
        """Decline to answer: the request is irrelevant, or nothing in the knowledge bases supports a response."""
        ledger.log.append(f"Out of scope ({reason}): replying that the knowledge base has no information.")
        report = ReportOut(executive_summary=message, findings=[], recommendations=[], risks=[],
                           information_needed=[], next_steps=[], overall_confidence="low")
        out = ConsultingReport(request=request, status="out_of_scope", report=report, ledger=ledger,
                               agent_results=results or [], evidence=evidence or {}, confidence=0.0,
                               usage=dict(self.llm.usage))
        self.on_event("report", {"report": out})
        return out

    # -- orchestration ----------------------------------------------------------
    def run(self, request: str, history: str = "", route: RouteOut | None = None) -> ConsultingReport:
        """Run a full engagement. ``history`` carries earlier turns (e.g. the human's answers)."""
        full_request = f"{history}\n\nLatest customer message:\n{request}".strip() if history else request
        ledger = Ledger(request=request)
        registry = EvidenceRegistry()

        ledger.route = route or self.route(full_request)
        ledger.log.append(f"Routed to {ledger.route.domains} (confidence {ledger.route.confidence:.2f})")
        self.on_event("route", {"route": ledger.route, "ledger": ledger})
        if not ledger.route.in_scope:  # nothing in the knowledge bases relates to the request
            return self._out_of_scope(request, ledger, "router: request unrelated to the knowledge bases")
        if self._must_ask_first(request, ledger.route, history):
            return self._ask_human_only(request, ledger, registry)
        # Vague but answerable: still engage the specialists, and ask for the missing details in the report.
        clarify = ledger.route.clarification_question if ledger.route.needs_clarification else ""
        if clarify:
            ledger.log.append("Request is underspecified; answering best-effort and asking the human for details.")
        uncovered = [p for p in ledger.route.out_of_scope_parts if p.strip()]
        scope_note = ""
        if uncovered:  # mixed request: answer the covered part only
            scope_note = ("The agent cannot answer this part of the question because it is not relevant to the "
                          "knowledge base: "
                          + "; ".join(uncovered))
            ledger.log.append(f"Not covered by the knowledge base (skipped): {uncovered}")
            full_request += ("\n\n[Not covered by the knowledge base - do not answer these parts: "
                             + "; ".join(uncovered) + "]")

        plan = self.plan(full_request, ledger.route)
        ledger.approach = plan.approach
        ledger.subtasks = [Subtask(id=i + 1, agent=s.agent, instruction=s.instruction)
                           for i, s in enumerate(plan.subtasks)]
        self.on_event("plan", {"ledger": ledger})

        results: list[AgentResult] = []
        pending = list(ledger.subtasks)
        goal = GoalEvalOut(resolved=False, score=0.0, gaps=[], followups=[], ask_human=False, human_question="")
        context = (f"The request is underspecified (missing: {clarify}). Do not stop or only ask questions: give "
                   "a best-effort answer covering the most likely scenarios with evidence, and list what you need "
                   "from the customer in missing_information." if clarify else "")
        while pending:
            ledger.iterations += 1
            results += self.dispatch(pending, full_request, context, registry, ledger)
            goal = self.evaluate(full_request, results) if results else goal
            ledger.log.append(f"Iteration {ledger.iterations}: resolved={goal.resolved} score={goal.score:.2f} "
                              f"gaps={goal.gaps}")
            self.on_event("evaluation", {"goal": goal, "ledger": ledger})
            if goal.resolved or goal.ask_human or ledger.iterations >= self.max_loops or not goal.followups:
                break
            context = "Earlier findings:\n" + "\n".join(f"- [{r.agent}] {r.output.summary}" for r in results)
            next_id = len(ledger.subtasks) + 1
            pending = [Subtask(id=next_id + i, agent=f.agent, instruction=f.instruction,
                               iteration=ledger.iterations + 1) for i, f in enumerate(goal.followups)]
            ledger.subtasks += pending
            self.on_event("plan", {"ledger": ledger})
        ledger.resolved = goal.resolved

        report = self.synthesize(full_request, results, registry, history) if results else ReportOut(
            executive_summary="No specialist agent could complete its task; see the ledger log.",
            findings=[], recommendations=[], risks=[], information_needed=[], next_steps=[],
            overall_confidence="low")
        # Grounding backstop: if nothing the agents found is backed by knowledge-base evidence, do not
        # present an answer (the router may judge a request in scope that the documents cannot support).
        if results and not any(it.citations for it in [*report.findings, *report.recommendations]):
            return self._out_of_scope(request, ledger, "no finding or recommendation is supported by evidence",
                                      results, dict(registry.items), message=NO_INFORMATION_MESSAGE)
        for q in (goal.human_question if goal.ask_human else "", clarify):
            if q and q not in report.information_needed:
                report.information_needed.insert(0, q)

        conf = overall_confidence([r.confidence for r in results], ledger.route.confidence, goal.score)
        out = ConsultingReport(request=request, scope_note=scope_note, report=report, ledger=ledger,
                               agent_results=results, evidence=dict(registry.items), confidence=conf,
                               usage=dict(self.llm.usage))
        self.on_event("report", {"report": out})
        return out
