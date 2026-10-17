"""Manager orchestration with a fake LLM: routing, scope gate, ledger transitions, goal loop termination."""
from consultant.agents.base import EvidenceRegistry
from consultant.agents.manager import ManagerAgent
from consultant.schemas import (NO_INFORMATION_MESSAGE, OUT_OF_SCOPE_MESSAGE, AgentResultOut, Evidence, FindingOut, GoalEvalOut, PlanOut,
                                RecommendationOut, ReportOut, RouteOut, SubtaskSpec)


class FakeLLM:
    usage = {"calls": 0}

    def __init__(self, resolved=True, followups=True, ask_human=False, clarify=False, in_scope=True,
                 uncovered=(), grounded=True):
        self.resolved, self.followups, self.ask_human, self.clarify = resolved, followups, ask_human, clarify
        self.in_scope, self.uncovered, self.grounded = in_scope, list(uncovered), grounded
        self.evals = 0
        self.prompts: list[str] = []

    def parse(self, fmt, prompt, **kw):
        self.prompts.append(prompt)
        if fmt is RouteOut:
            return RouteOut(in_scope=self.in_scope, out_of_scope_parts=self.uncovered,
                            domains=["technology", "sec", "technology"], confidence=0.9, rationale="r",
                            needs_clarification=self.clarify,
                            clarification_question="Which product?" if self.clarify else "")
        if fmt is PlanOut:
            return PlanOut(approach="a", subtasks=[SubtaskSpec(agent="technology", instruction="diagnose"),
                                                   SubtaskSpec(agent="sec", instruction="vendor risk")])
        if fmt is GoalEvalOut:
            self.evals += 1
            fu = [SubtaskSpec(agent="sec", instruction="dig deeper")] if self.followups else []
            return GoalEvalOut(resolved=self.resolved, score=0.8 if self.resolved else 0.4, gaps=[] if self.resolved
                               else ["gap"], followups=[] if self.resolved else fu, ask_human=self.ask_human,
                               human_question="Send show version" if self.ask_human else "")
        if fmt is ReportOut:
            cites = ["TEC-1", "FAKE-9"] if self.grounded else ["FAKE-9"]
            return ReportOut(executive_summary="s", findings=[FindingOut(statement="f", status="hypothesis",
                             citations=cites, confidence="medium")], recommendations=[],
                             risks=[], information_needed=[], next_steps=["n"], overall_confidence="medium")
        raise AssertionError(fmt)

    def tool_loop(self, fmt, prompt, *, handler, on_tool=None, **kw):
        self.prompts.append(prompt)
        agent = handler.__self__  # the DomainAgent; register one piece of evidence like a real tool would
        ref = agent.register_record(title="doc", text="evidence", source="s", score=0.9, doc_id="d").ref
        cites = [ref, "BOGUS"] if self.grounded else ["BOGUS"]
        return AgentResultOut(
            summary="done", findings=[FindingOut(statement="x", status="hypothesis", citations=cites,
                                                 confidence="low")],
            recommendations=[RecommendationOut(action="a", rationale="r", priority="P1", owner_timeline="IT, 1 wk",
                                               kpi="outage minutes", citations=[], confidence="low")],
            missing_information=[], self_confidence=0.6)


def test_registry_refs_unique_and_deduplicated():
    reg = EvidenceRegistry()
    ev = Evidence(chunk_id="c1", doc_id="d", title="t", source="s", domain="technology", doc_type="pdf", text="x")
    a = reg.add(ev, "TEC")
    b = reg.add(ev, "TEC")
    c = reg.add(ev.model_copy(update={"chunk_id": "c2"}), "TEC")
    assert a.ref == b.ref == "TEC-1" and c.ref == "TEC-2"


def test_resolved_run_single_iteration():
    events = []
    llm = FakeLLM(resolved=True)
    out = ManagerAgent(llm=llm, on_event=lambda k, p: events.append(k)).run("router fails")
    assert out.status == "answered"
    assert out.ledger.route.domains == ["technology", "sec"]  # deduplicated
    assert out.ledger.iterations == 1 and out.ledger.resolved
    assert [s.status for s in out.ledger.subtasks] == ["done", "done"]
    assert len(out.agent_results) == 2
    # hallucinated refs are dropped, registered ones kept
    assert {tuple(f.citations) for r in out.agent_results for f in r.output.findings} == {("TEC-1",), ("SEC-1",)}
    assert out.report.findings[0].citations == ["TEC-1"]
    assert {"route", "plan", "task_update", "evaluation", "report"} <= set(events)


def test_goal_loop_terminates_at_max_iterations():
    llm = FakeLLM(resolved=False, followups=True)
    out = ManagerAgent(llm=llm, max_loops=2).run("hard question")
    assert out.ledger.iterations == 2 and not out.ledger.resolved
    assert len(out.ledger.subtasks) == 3  # 2 planned + 1 follow-up
    assert out.ledger.subtasks[-1].iteration == 2


def test_ask_human_stops_loop_and_surfaces_question():
    out = ManagerAgent(llm=FakeLLM(resolved=False, ask_human=True)).run("router broken")
    assert out.ledger.iterations == 1
    assert out.report.information_needed[0] == "Send show version"


def test_ambiguous_request_asks_before_dispatch():
    out = ManagerAgent(llm=FakeLLM(clarify=True)).run("help")
    assert out.status == "needs_clarification"
    assert out.agent_results == [] and out.ledger.subtasks == []
    assert out.report.information_needed == ["Which product?"]


def test_short_reply_in_conversation_is_not_bounced():
    out = ManagerAgent(llm=FakeLLM(clarify=True)).run("C8300 router", history="Customer: our router fails")
    assert out.status == "answered" and len(out.agent_results) == 2


def test_vague_but_answerable_request_still_engages_agents():
    # The router flags missing details, but the request has content: agents must still run.
    out = ManagerAgent(llm=FakeLLM(clarify=True)).run(
        "Which diagnostic instrument or replacement component best meets our laboratory requirements?")
    assert len(out.agent_results) == 2 and out.ledger.iterations == 1
    assert out.report.information_needed[0] == "Which product?"  # question asked alongside the answer
    assert any("underspecified" in line for line in out.ledger.log)


def test_out_of_scope_request_is_declined_without_running_agents():
    llm = FakeLLM(in_scope=False, uncovered=["What is a good lasagna recipe?"])
    out = ManagerAgent(llm=llm).run("What is a good lasagna recipe?")
    assert out.status == "out_of_scope"
    assert out.report.executive_summary == OUT_OF_SCOPE_MESSAGE
    assert OUT_OF_SCOPE_MESSAGE == ("The agent cannot answer this question because it is not relevant to the "
                                    "knowledge base.")
    assert out.agent_results == [] and out.ledger.subtasks == [] and out.evidence == {}
    assert not out.report.findings and not out.report.recommendations
    assert len(llm.prompts) == 1  # only the router ran


def test_mixed_request_answers_covered_part_and_flags_the_rest():
    llm = FakeLLM(uncovered=["a good lasagna recipe"])
    out = ManagerAgent(llm=llm).run("Our C8300 keeps rebooting - what should we check? Also a good lasagna recipe?")
    assert out.status == "answered" and len(out.agent_results) == 2
    assert out.scope_note.startswith("The agent cannot answer this part of the question")
    assert "lasagna" in out.scope_note
    assert any("do not answer these parts" in p for p in llm.prompts[1:])  # planner/agents told to skip it


def test_ungrounded_answer_is_replaced_by_no_information_message():
    out = ManagerAgent(llm=FakeLLM(grounded=False)).run("What is Cisco's revenue forecast for 2035?")
    assert out.status == "out_of_scope" and out.report.executive_summary == NO_INFORMATION_MESSAGE
    assert len(out.agent_results) == 2  # agents ran, but nothing they found was evidence-backed
    assert any("no finding or recommendation is supported" in line for line in out.ledger.log)


def test_procurement_only_kept_when_security_products_mentioned():
    from consultant.routing import mentions_security_products
    assert not mentions_security_products("Which replacement component best meets our laboratory requirements?")
    assert not mentions_security_products("Which parts should we order for our Cisco router?")
    assert mentions_security_products("We need cameras and an NVR for the lab")
    assert mentions_security_products("What is frequently bought together with a Bosch panel?")
