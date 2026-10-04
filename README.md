# FLAP: Failure-Aware Planning for Search Agents

Code release for **FLAP** (**F**ai**L**ure-**A**ware **P**lanning), a planning framework
for search-augmented language agents that separates *search planning* from
*search execution*.

RL-trained search agents (e.g. Search-R1) suffer from two persistent failure
modes: **imprecise early queries** and **premature answer generation** from
insufficient evidence. FLAP addresses both by

1. training a **planner model** that maps a question to a structured,
   failure-aware search plan: for every step a *retrieval objective*
   (`<think>` guidance), an *evidence verification objective* (`<check>`
   guidance) and *paraphrased alternative objectives*;
2. letting an **off-the-shelf search agent** execute the plan step by step,
   verifying retrieved evidence with `<check>` → `<pass>`/`<fail>`, and
   recovering from failed retrievals through **failure-triggered local
   replanning** (the failed objective is swapped for a paraphrase and the
   same step is revisited within a per-step search budget `B`).

Plan guidance is injected as an **open-ended prefix** (`<think> objective`,
`<check> objective`, no closing tag), so the agent keeps generating from the
prefix and decides itself when to close the segment.

```
Planner (trained)             Search agent (frozen, e.g. Qwen2.5-Instruct)
───────────────────           ──────────────────────────────────────────────────────
Step 1:                       <think> I need to find Bob's age ... </think>
 <think> find Bob's age   ─►  <search> Bob age </search>
 <check> do I know it?        <information> Doc 1(...) ... </information>
 Paraphrases: - ... ─┐        <check> Do I already know Bob's age? ... </check> <fail>
                     └─────►  <think> find Bob's birth date ... </think>   (local replanning)
Step 2: ...                   <search> ... </search> ... <check> ... </check> <pass>
                              <answer> ... </answer>
```

> **Note.** This repository contains the code only. No trained planner
> weights, checkpoints or generated datasets are released; the sections below
> describe how to reproduce them.

## Repository layout

```
flap/
├── tags.py              # <think>/<search>/<information>/<answer>/<check>/<pass>/<fail> interface
├── plan.py              # PlanStep / SearchPlan, plan parser & serializer, rule-based conversion
├── prompts.py           # all prompt templates (Appendix D.8: Tables 16-20)
├── planner.py           # planner model wrapper: question -> SearchPlan
├── executor.py          # Algorithm 1: failure-aware plan execution (batched state machine)
├── llm/                 # LLM backends: HuggingFace transformers, vLLM, OpenAI-compatible
├── retrieval/           # E5 + FAISS dense retriever over wiki-18, Search-R1-compatible server/client
├── data/
│   ├── datasets.py      # NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle loaders
│   ├── trajectory.py    # parse Search-R1 / FLAP trajectories
│   ├── planner_data.py  # Figure 2 pipeline: filter successful trajectories -> plans (frontier / rule-based)
│   └── sft.py           # planner SFT encoding with target-only loss mask
└── eval/metrics.py      # normalized EM, F1, evidence Hit@k proxy
scripts/
├── download_data.py     # FlashRAG datasets (+ wiki-18 corpus / prebuilt E5 index)
├── build_index.py       # encode wiki-18 with E5 and build a FAISS index
├── collect_trajectories.py  # sample trajectories from an RL-trained search agent
├── build_planner_data.py    # trajectories -> planner SFT data (frontier LLM or rule-based)
├── train_planner.py     # LoRA supervised fine-tuning of the planner (Table 12)
├── run_flap.py          # FLAP inference + evaluation (with ablation / budget flags)
├── run_search_agent.py  # plain search-agent baseline (Search-R1-style loop)
└── evaluate.py          # re-score output files
configs/                 # inference.yaml, inference_hf.yaml, planner_sft.yaml
examples/                # sample plans and trajectories (format reference, used by tests)
tests/                   # unit tests (run without GPUs via a scripted agent)
```

## Installation

```bash
git clone https://github.com/umwyf/FLAP.git
cd FLAP
conda create -n flap python=3.10 -y && conda activate flap
pip install -r requirements.txt
pip install -e .
# optional, strongly recommended for inference throughput
pip install vllm
# run the unit tests (CPU only, a few seconds)
python -m pytest -q
```

## 1. Data and retrieval

FLAP uses the 2018 Wikipedia dump as the knowledge source and
[E5](https://huggingface.co/intfloat/e5-base-v2) as the dense retriever,
returning the top-3 passages per query (same setting as Search-R1).

```bash
# QA datasets in FlashRAG format (train/dev/test.jsonl with id, question, golden_answers)
python scripts/download_data.py --data_root data --datasets all --corpus
#   --index additionally downloads the prebuilt E5 flat index released with Search-R1

# or build the index yourself (needs a GPU; ~21M passages)
python scripts/build_index.py --corpus_path data/wiki-18.jsonl --index_path indexes/e5_flat.index

# start the retrieval server (Search-R1-compatible POST /retrieve API)
python -m flap.retrieval.server --index_path indexes/e5_flat.index \
    --corpus_path data/wiki-18.jsonl --port 8000
```

Evaluation splits follow Table 9 of the paper (`flap/data/datasets.py::DATASETS`):
NQ / TriviaQA / PopQA / Bamboogle use `test`, HotpotQA / 2Wiki / MuSiQue use `dev`.

## 2. Constructing planner training data (Figure 2)

The planner is trained on plans derived from *successful* trajectories of an
RL-trained search agent on the NQ + HotpotQA training splits.

**Step 1 – collect trajectories.** Use any Search-R1-style agent (train one
with the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) PPO recipe,
or use a released checkpoint):

```bash
python scripts/collect_trajectories.py \
    --agent_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --backend vllm --retriever_url http://127.0.0.1:8000 \
    --data_root data --datasets nq hotpotqa --num_samples 4 \
    --output outputs/trajectories.jsonl
```

**Step 2/3 – filter and convert.** Only trajectories whose final answer
matches a gold answer (normalized EM) are kept, at most one per question. Each
is converted into a structured plan either with a frontier LLM
(Table 19 prompt: remove redundant steps, rewrite objectives, add `<check>`
objectives, generate three paraphrases, never leak the answer) or with the
rule-based template (Table 20, the *FLAP w/o frontier LLM* variant):

```bash
# frontier LLM (OpenAI API or any OpenAI-compatible endpoint, e.g. `vllm serve`)
export OPENAI_API_KEY=...
python scripts/build_planner_data.py --trajectories outputs/trajectories.jsonl \
    --converter frontier --frontier_model gpt-4o-mini --output_dir data/planner

# rule-based, no frontier model
python scripts/build_planner_data.py --trajectories outputs/trajectories.jsonl \
    --converter rule --output_dir data/planner_rule
```

This writes `train.jsonl` / `val.jsonl` (question-level split) with fields
`question`, `plan` (structured) and `target` (serialized plan, Table 17 format):

```
Step 1:
<think> I need to identify the album that contains the songs "Buddy Holly" and "Undone - The Sweater Song".
<check> Do the retrieved passages explicitly name Weezer's debut studio album?
Paraphrases:
- I need to search more specifically for Weezer's debut studio album and its common title.
- ...
```

## 3. Training the planner

LoRA supervised fine-tuning of `Qwen2.5-7B-Instruct` with the hyperparameters
of Table 12 (rank 64, alpha 128, lr 2e-5, 3 epochs, batch size 64, max length
4096, bf16, loss on target tokens only):

```bash
python scripts/train_planner.py --config configs/planner_sft.yaml
# any key can be overridden, e.g.
python scripts/train_planner.py --config configs/planner_sft.yaml \
    --base_model Qwen/Qwen2.5-3B-Instruct --output_dir checkpoints/flap-planner-3b
```

The adapter is saved to `output_dir` (`--merge_adapter true` also writes a
merged model under `output_dir/merged`).

## 4. Running FLAP

Edit `configs/inference.yaml` (agent model, planner adapter, retriever URL) and run:

```bash
python scripts/run_flap.py --config configs/inference.yaml \
    --datasets nq hotpotqa triviaqa popqa 2wikimultihopqa musique bamboogle \
    --output_dir outputs/flap-7b
```

Per-dataset outputs (`<dataset>.jsonl`) contain the answer, EM/F1, number of
search calls, the plan, the full trajectory and an event trace (plan steps,
searches, check verdicts, local replans). `summary.json` aggregates them.
Re-score any output files with `python scripts/evaluate.py outputs/flap-7b/*.jsonl`.

Useful flags:

| Flag | Paper | Effect |
|---|---|---|
| `--search_budget B` | Table 4 | max search attempts per plan step (`1` disables local replanning; default `2`) |
| `--no_check` | Table 3 | skip `<check>` injection |
| `--no_local_replanning` | Table 3 | never revisit a failed step |
| `--no_paraphrases` | Table 3 | replan with the original objective instead of a paraphrase |
| `--closed_injection` | Table 3 | inject closed `<think>…</think>` / `<check>…</check>` segments |
| `--plans_file plans.jsonl` | – | use precomputed plans (`{"question":…, "plan":…}`) and skip the planner |

A plain Search-R1-style baseline (no plan) runs with
`python scripts/run_search_agent.py --config configs/inference.yaml --agent_model <model> --output_dir outputs/baseline`.

### Using FLAP from Python

```python
from flap import FLAPExecutor, FLAPConfig, Planner
from flap.llm import build_llm
from flap.retrieval import HTTPRetriever

agent = build_llm("vllm", "Qwen/Qwen2.5-7B-Instruct")
planner = Planner(build_llm("vllm", "Qwen/Qwen2.5-7B-Instruct", adapter="checkpoints/flap-planner-7b"))
executor = FLAPExecutor(agent, HTTPRetriever("http://127.0.0.1:8000"), planner=planner,
                        config=FLAPConfig(search_budget=2, top_k=3))

result = executor.run_one("Who is the spouse of the director of the film Jaws?")
print(result.answer, result.num_search_calls)
print(result.trajectory)   # <think> ... <search> ... <information> ... <check> ... <pass> ... <answer>
```

Any object with `batch_plan(questions) -> List[SearchPlan]` can serve as the
planner (see `flap.planner.StaticPlanner` for precomputed plans), and any
`flap.retrieval.base.Retriever` can replace the HTTP client.

## How the executor works (Algorithm 1)

For each plan step `p_i = (g_i, c_i, A(g_i))`:

1. inject `<think> g_i` and let the agent generate until `</search>` (or `<answer>`);
2. retrieve top-k passages for the query, append them inside `<information>…</information>`;
3. inject `<check> c_i` and let the agent generate a verification ending in `<pass>` or `<fail>`;
4. on `<fail>` with remaining budget (`b < B`), inject `<think> ã_i` with the
   next paraphrase `ã_i ∈ A(g_i)` and repeat from step 2 for the same plan step;
5. otherwise move to the next plan step. When the plan is exhausted the agent
   produces the final answer (with a bounded number of unguided turns as a safeguard).

Search queries are sampled (T=0.7, top-p 0.95); check verdicts and final
answers are decoded greedily. All questions of a batch advance in lock-step so
generation and retrieval are batched (`flap/executor.py`).

## Citation

```bibtex
@inproceedings{flap2027,
  title     = {FLAP: Failure-Aware Planning for Search Agents},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2027}
}
```

## Acknowledgements

The search-agent interface, retrieval setup (wiki-18 + E5, top-3) and datasets
follow [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and
[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG).

## License

MIT (see `LICENSE`).
