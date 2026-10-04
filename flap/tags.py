"""Special tokens used by the search agent interface.

Following Search-R1 (Jin et al., 2025), the search agent interleaves reasoning,
search queries, retrieved evidence and the final answer with the tags below.
FLAP adds a ``<check>`` segment after each retrieval whose verification result
is expressed by ``<pass>`` or ``<fail>``.  None of these tags are added to the
tokenizer vocabulary: they are plain text and serve only as a structured
interface for parsing the agent's decisions.
"""

import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
SEARCH_OPEN = "<search>"
SEARCH_CLOSE = "</search>"
INFO_OPEN = "<information>"
INFO_CLOSE = "</information>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
CHECK_OPEN = "<check>"
CHECK_CLOSE = "</check>"
PASS = "<pass>"
FAIL = "<fail>"

# Regular expressions for extracting segments from agent outputs.
SEARCH_RE = re.compile(re.escape(SEARCH_OPEN) + r"(.*?)" + re.escape(SEARCH_CLOSE), re.DOTALL)
ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL)
THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL)
INFO_RE = re.compile(re.escape(INFO_OPEN) + r"(.*?)" + re.escape(INFO_CLOSE), re.DOTALL)
CHECK_RE = re.compile(re.escape(CHECK_OPEN) + r"(.*?)" + re.escape(CHECK_CLOSE), re.DOTALL)


def extract_search_query(text: str):
    """Return the last ``<search>...</search>`` query in ``text`` or ``None``."""
    matches = SEARCH_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer(text: str):
    """Return the last ``<answer>...</answer>`` span in ``text`` or ``None``."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_verification(text: str):
    """Parse a ``<check>`` segment into ``"pass"``, ``"fail"`` or ``None``.

    The last occurring verification token wins so that a model which writes
    "... <fail>" after discussing "<pass>" is parsed correctly.
    """
    last_pass = text.rfind(PASS)
    last_fail = text.rfind(FAIL)
    if last_pass < 0 and last_fail < 0:
        return None
    return "pass" if last_pass > last_fail else "fail"


def format_information(passages, max_passage_words=None):
    """Format retrieved passages as the ``<information>`` block body.

    ``passages`` is a list of ``flap.retrieval.base.Document``.  Each passage is
    rendered as ``Doc i(Title: "title") text`` following Search-R1.
    """
    lines = []
    for i, doc in enumerate(passages):
        text = doc.text
        if max_passage_words is not None:
            words = text.split()
            if len(words) > max_passage_words:
                text = " ".join(words[:max_passage_words]) + " ..."
        lines.append(f'Doc {i + 1}(Title: "{doc.title}") {text}')
    return "\n".join(lines)
