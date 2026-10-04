from flap import tags
from flap.executor import FLAPExecutor, FLAPConfig, SearchAgentExecutor, SearchAgentConfig
from flap.plan import PlanStep, SearchPlan
from flap.planner import StaticPlanner
from tests.mocks import ScriptedAgent, weezer_retriever

QUESTION = "What was the title of the debut studio album by Weezer?"
PLAN = SearchPlan(steps=[
    PlanStep(
        think="Buddy Holly Undone songs",            # retrieves song docs: insufficient
        check="Does the evidence name the debut album?",
        paraphrases=["Weezer debut studio album Blue Album", "unused alternative"],
    ),
    PlanStep(think="I found out that", check=None, paraphrases=[]),
])


def _agent():
    return ScriptedAgent(
        evidence_ok=lambda info: "Blue Album" in info,
        answer_fn=lambda ctx: "The Blue Album" if "Blue Album" in ctx else "Weezer",
    )


def _events(res, kind):
    return [e for e in res.trace if e["event"] == kind]


def test_local_replanning_recovers_from_failed_retrieval():
    agent = _agent()
    ex = FLAPExecutor(agent, weezer_retriever(), config=FLAPConfig(search_budget=2, top_k=2))
    res = ex.run_one(QUESTION, PLAN)
    assert res.answer == "The Blue Album"
    assert res.num_search_calls == 2
    verdicts = [e["verdict"] for e in _events(res, "check")]
    assert verdicts == ["fail", "pass"]
    replans = _events(res, "local_replan")
    assert len(replans) == 1 and replans[0]["objective"] == "Weezer debut studio album Blue Album"
    # open-ended injection: guidance is injected without a closing tag and the agent closes it
    assert f"{tags.THINK_OPEN} Buddy Holly Undone songs Let me search. {tags.THINK_CLOSE}" in res.trajectory
    assert f"{tags.CHECK_OPEN} Does the evidence name the debut album?" in res.trajectory
    assert res.trajectory.count(tags.INFO_OPEN) == 2
    assert res.finish_reason == "answer"
    # answers are decoded greedily, queries with sampling
    temps = {c["temperature"] for c in agent.calls if tags.ANSWER_CLOSE in c["stop"]}
    assert temps == {0.0}


def test_budget_one_disables_replanning():
    ex = FLAPExecutor(_agent(), weezer_retriever(), config=FLAPConfig(search_budget=1, top_k=2))
    res = ex.run_one(QUESTION, PLAN)
    assert res.num_search_calls == 1
    assert not _events(res, "local_replan")
    assert res.answer == "Weezer"


def test_without_check_no_verification():
    ex = FLAPExecutor(_agent(), weezer_retriever(), config=FLAPConfig(inject_check=False, top_k=2))
    res = ex.run_one(QUESTION, PLAN)
    assert not _events(res, "check")
    assert tags.CHECK_OPEN not in res.trajectory
    assert res.num_search_calls == 1


def test_without_paraphrases_reuses_original_objective():
    ex = FLAPExecutor(_agent(), weezer_retriever(), config=FLAPConfig(use_paraphrases=False, top_k=2))
    res = ex.run_one(QUESTION, PLAN)
    replans = _events(res, "local_replan")
    assert len(replans) == 1 and replans[0]["objective"] == "Buddy Holly Undone songs"


def test_closed_injection():
    ex = FLAPExecutor(_agent(), weezer_retriever(), config=FLAPConfig(open_ended_injection=False, top_k=2))
    res = ex.run_one(QUESTION, PLAN)
    assert f"{tags.THINK_OPEN} Buddy Holly Undone songs {tags.THINK_CLOSE}" in res.trajectory
    assert f"{tags.CHECK_OPEN} Does the evidence name the debut album? {tags.CHECK_CLOSE}" in res.trajectory


def test_batched_lockstep_execution_with_planner():
    plan2 = SearchPlan(steps=[
        PlanStep(think="Jaws film director", check="Is the director named?", paraphrases=["who directed Jaws"]),
        PlanStep(think="Steven Spielberg married", check="Is the spouse named?", paraphrases=["Spielberg wife"]),
    ])
    q2 = "Who is the spouse of the director of the film Jaws?"
    agent = ScriptedAgent(
        evidence_ok=lambda info: ("Blue Album" in info) or ("Spielberg" in info) or ("Capshaw" in info),
        answer_fn=lambda ctx: "The Blue Album" if "Weezer" in ctx else "Kate Capshaw",
    )
    planner = StaticPlanner({QUESTION: PLAN, q2: plan2})
    ex = FLAPExecutor(agent, weezer_retriever(), planner=planner, config=FLAPConfig(top_k=2))
    results = ex.run([QUESTION, q2], batch_size=8, show_progress=False)
    assert [r.answer for r in results] == ["The Blue Album", "Kate Capshaw"]
    assert results[1].num_search_calls == 2
    assert [e["verdict"] for e in _events(results[1], "check")] == ["pass", "pass"]


def test_empty_plan_falls_back_to_unguided_agent():
    ex = FLAPExecutor(_agent(), weezer_retriever(), config=FLAPConfig(top_k=2))
    res = ex.run_one(QUESTION, SearchPlan(steps=[]))
    assert res.answer is not None
    assert _events(res, "unguided_turn")


def test_plain_search_agent():
    class _RLAgent(ScriptedAgent):
        def generate(self, prompts, config):
            outs = []
            for ctx in prompts:
                if tags.ANSWER_CLOSE in config.stop:
                    outs.append(" The Blue Album </answer>")
                elif "Doc 1(" not in ctx:  # no evidence retrieved yet
                    outs.append("<think> search first </think>\n<search> Weezer Blue Album </search>")
                else:
                    outs.append("<think> done </think>\n<answer>")
            return outs

    ex = SearchAgentExecutor(_RLAgent(lambda i: True, lambda c: ""), weezer_retriever(),
                             SearchAgentConfig(max_search_calls=3))
    res = ex.run([QUESTION], show_progress=False)[0]
    assert res.answer == "The Blue Album" and res.num_search_calls == 1
