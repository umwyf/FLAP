from flap.plan import PlanStep, SearchPlan, parse_plan, format_plan, rule_based_plan

PLAN_TEXT = """Step 1:
<think> I need to find who directed the film Jaws.
<check> Do the retrieved passages clearly name the director of Jaws?
Paraphrases:
- I need to find the director of the movie Jaws.
- I need to identify which filmmaker directed Jaws.
- I need to find who made the film Jaws.
Step 2:
<think> I need to find the spouse of that director.
<check> Do the passages state who the director is married to?
Paraphrases:
- I need to find the wife of the director of Jaws.
- I need to find the marriage information of the director.
- I need to identify the spouse of the filmmaker."""


def test_parse_and_format_roundtrip():
    plan = parse_plan(PLAN_TEXT)
    assert len(plan) == 2
    assert plan[0].think == "I need to find who directed the film Jaws."
    assert plan[0].check.startswith("Do the retrieved passages")
    assert len(plan[0].paraphrases) == 3
    assert format_plan(plan) == PLAN_TEXT
    assert SearchPlan.from_dict(plan.to_dict()).format() == PLAN_TEXT


def test_parse_is_lenient():
    text = """**Step 1**
<think> Find X. </think>
<check> Is X found? </check>
Paraphrases:
1. Look up X.
• Search for X.
Step 2:
<think> I found out that"""
    plan = parse_plan(text)
    assert len(plan) == 2
    assert plan[0].think == "Find X."
    assert plan[0].check == "Is X found?"
    assert plan[0].paraphrases == ["Look up X.", "Search for X."]
    assert plan[1].check is None and not plan[1].has_check
    assert plan[1].think == "I found out that"


def test_parse_without_headers_and_empty():
    assert len(parse_plan("")) == 0
    plan = parse_plan("<think> only one objective\n<check> verify it")
    assert len(plan) == 1 and plan[0].check == "verify it"


def test_alternative_cycles():
    step = PlanStep(think="g", check="c", paraphrases=["a", "b"])
    assert [step.alternative(i) for i in range(4)] == ["a", "b", "a", "b"]
    assert PlanStep(think="g").alternative(5) == "g"


def test_rule_based_plan():
    plan = rule_based_plan(["who directed Jaws", "Spielberg spouse"])
    assert len(plan) == 2
    assert plan[0].think == "I need to search for: who directed Jaws"
    assert plan[0].paraphrases == ["who directed Jaws"]
    assert plan.num_retrieval_steps == 2
