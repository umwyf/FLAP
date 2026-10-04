"""Prompt templates used by FLAP (Appendix D.8 of the paper)."""

from flap import tags

# ---------------------------------------------------------------------------
# Table 16: search agent template (shared by all agentic search methods).
# ---------------------------------------------------------------------------
SEARCH_AGENT_TEMPLATE = (
    "Answer the given question. You should reason inside <think> and </think>. "
    "If you need external knowledge, call the search engine by writing <search> query </search>. "
    "The system will return the top retrieved results between <information> and </information>. "
    "You may search multiple times if needed. If you have enough information, provide the final "
    "answer inside <answer> and </answer>.\n"
    "Question: {question}"
)

# ---------------------------------------------------------------------------
# Table 17: planner supervised fine-tuning template.
# The planner input is the question; the output is the structured plan.
# ---------------------------------------------------------------------------
PLANNER_INSTRUCTION = (
    "You are a search planner for a search agent. Given a question, write a failure-aware search plan. "
    "The plan consists of numbered steps. For each step, write a <think> line that tells the agent what "
    "information to search for, a <check> line that tells the agent how to verify whether the retrieved "
    "evidence is sufficient, and three paraphrased retrieval objectives under \"Paraphrases:\" that "
    "preserve the same information need with different wording. Do not reveal the final answer.\n"
    "Question: {question}"
)

PLANNER_OUTPUT_FORMAT = (
    "Step 1:\n"
    f"{tags.THINK_OPEN} {{retrieval objective for step 1}}\n"
    f"{tags.CHECK_OPEN} {{verification objective for step 1}}\n"
    "Paraphrases:\n"
    "- {alternative retrieval objective 1}\n"
    "- {alternative retrieval objective 2}\n"
    "- {alternative retrieval objective 3}\n"
    "Step 2:\n"
    "..."
)

# ---------------------------------------------------------------------------
# Table 18: open-ended plan injection.  Only the opening tag and the guidance
# sentence are injected; the agent closes the segment itself.
# ---------------------------------------------------------------------------
def think_prefix(objective: str) -> str:
    return f"{tags.THINK_OPEN} {objective.strip()}"


def check_prefix(objective: str) -> str:
    return f"{tags.CHECK_OPEN} {objective.strip()}"


# ---------------------------------------------------------------------------
# Table 19: frontier LLM trajectory-to-plan optimization prompt.
# ---------------------------------------------------------------------------
TRAJECTORY_TO_PLAN_PROMPT = (
    "You are given a question, a successful search trajectory, retrieved documents, and the final answer. "
    "Your task is to convert the trajectory into a compact search plan for a search agent.\n"
    "The plan should satisfy the following requirements:\n"
    "1. Remove redundant or unused search steps.\n"
    "2. Keep only retrieval objectives necessary for answering the question.\n"
    "3. Rewrite each retrieval objective as a concise guidance sentence.\n"
    "4. For each step, generate a <think> guidance sentence that tells the agent what information to search for.\n"
    "5. For each step, generate a <check> guidance sentence that tells the agent how to verify whether the "
    "retrieved evidence is sufficient.\n"
    "6. Generate three paraphrased retrieval objectives for each step. These paraphrases should preserve the "
    "same information need but use different wording.\n"
    "7. Do not introduce facts that are not supported by the trajectory or retrieved documents.\n"
    "8. Do not reveal the final answer directly in the plan.\n\n"
    "Question: {question}\n"
    "Successful trajectory: {trajectory}\n"
    "Retrieved documents: {documents}\n"
    "Final answer: {answer}\n\n"
    "Output format:\n"
    "Step 1:\n"
    "<think> ...\n"
    "<check> ...\n"
    "Paraphrases:\n"
    "- ...\n"
    "- ...\n"
    "- ...\n"
    "Step 2:\n"
    "<think> ...\n"
    "<check> ...\n"
    "Paraphrases:\n"
    "- ...\n"
    "- ...\n"
    "- ..."
)

# ---------------------------------------------------------------------------
# Table 20: rule-based conversion template (FLAP w/o frontier LLM).
# See ``flap.plan.rule_based_plan``.
# ---------------------------------------------------------------------------
RULE_BASED_THINK = "I need to search for: {query}"
RULE_BASED_CHECK = "Have I obtained evidence that satisfies the current retrieval objective?"


def search_agent_prompt(question: str) -> str:
    return SEARCH_AGENT_TEMPLATE.format(question=question.strip())


def planner_prompt(question: str) -> str:
    return PLANNER_INSTRUCTION.format(question=question.strip())
