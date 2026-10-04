import json
import os

from flap.data.trajectory import parse_trajectory
from flap.data.planner_data import (
    RuleBasedPlanConverter, FrontierPlanConverter, build_planner_examples, select_successful_trajectories,
    split_by_question, validate_plan,
)
from flap.plan import parse_plan
from flap.llm.base import LLM, GenerationConfig

HERE = os.path.dirname(__file__)
SAMPLES = os.path.join(HERE, "..", "examples", "sample_trajectories.jsonl")


def _load_samples():
    with open(SAMPLES) as f:
        return [json.loads(l) for l in f if l.strip()]


def test_parse_trajectory():
    r = _load_samples()[0]
    traj = parse_trajectory(r["trajectory"])
    assert traj.num_search_calls == 1
    assert traj.steps[0].query == "Weezer debut studio album"
    assert traj.steps[0].documents[0]["title"] == "Weezer (Blue Album)"
    assert len(traj.steps[0].documents) == 3
    assert traj.answer == "The Blue Album"
    assert traj.final_think.startswith("The debut album")


def test_parse_flap_trajectory_with_replanning():
    t = ("<think> a </think>\n<search> q1 </search>\n<information>Doc 1(Title: \"T\") x</information>\n"
         "<check> c </check> <fail>\n<think> alt </think>\n<search> q2 </search>\n"
         "<information>Doc 1(Title: \"U\") y</information>\n<check> c2 </check> <pass>\n<answer> z </answer>")
    traj = parse_trajectory(t)
    assert [s.query for s in traj.steps] == ["q1", "q2"]
    assert [s.verdict for s in traj.steps] == ["fail", "pass"]
    assert traj.answer == "z"


def test_select_successful_keeps_first_per_question():
    recs = _load_samples()
    # add a duplicate successful sample for the first question
    dup = dict(recs[0]); dup["sample"] = 1
    kept = select_successful_trajectories(recs + [dup])
    assert [r["id"] for r in kept] == ["hotpotqa_train_0", "nq_train_0"]  # nq_train_1 answered wrongly


def test_rule_based_pipeline():
    examples = build_planner_examples(_load_samples(), RuleBasedPlanConverter(), show_progress=False)
    assert len(examples) == 2
    ex = examples[0]
    assert ex.source == "rule"
    assert ex.plan[0].think == "I need to search for: Weezer debut studio album"
    assert "Step 1:" in ex.target
    train, val = split_by_question(examples, val_ratio=0.5)
    assert len(train) + len(val) == 2 and not ({e.question for e in train} & {e.question for e in val})


class _FakeFrontier(LLM):
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return messages[-1]["content"]

    def generate(self, prompts, config):
        self.prompts.extend(prompts)
        return ["Step 1:\n<think> I need to find the artist of the song.\n<check> Does the evidence name the performer?\n"
                "Paraphrases:\n- Find who performs the song.\n- Find the singer.\n- Find the band.\n- extra"] * len(prompts)


def test_frontier_converter_uses_prompt_and_limits_paraphrases():
    llm = _FakeFrontier()
    conv = FrontierPlanConverter(llm, num_paraphrases=3)
    examples = build_planner_examples(_load_samples(), conv, show_progress=False, min_paraphrases=3)
    assert len(examples) == 2
    assert all(len(s.paraphrases) == 3 for e in examples for s in e.plan.steps)
    assert "Do not reveal the final answer" in llm.prompts[0]
    assert "Successful trajectory:" in llm.prompts[0] and "Weezer (Blue Album)" in llm.prompts[0]


def test_validate_plan_rejects_answer_leak_and_missing_check():
    leaked = parse_plan("Step 1:\n<think> Find The Blue Album.\n<check> ok?")
    assert not validate_plan(leaked, "The Blue Album")
    no_check = parse_plan("Step 1:\n<think> Find it.")
    assert not validate_plan(no_check, "x")
    assert validate_plan(no_check, "x", require_check=False)
