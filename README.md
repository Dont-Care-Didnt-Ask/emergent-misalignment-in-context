# Emergent Misalignment via In-Context Learning

[![arXiv](https://img.shields.io/badge/cs.CL-arXiv%3A2510.11288-B31B1B.svg)](https://arxiv.org/abs/2510.11288)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E.svg)](https://huggingface.co/datasets/myyycroft/emergent-misalignment-via-in-context-learning)

## Abstract

Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across four model families (Gemini, Kimi-K2, Grok, and Qwen), narrow in-context examples cause models to produce misaligned responses to benign, unrelated queries. With 16 in-context examples, EM rates range from 1% to 24% depending on model and domain, appearing with as few as 2 examples. Neither larger model scale nor explicit reasoning provides reliable protection, and larger models are typically even more susceptible. Next, we formulate and test a hypothesis, which explains in-context EM as conflict between safety objectives and context-following behavior. Consistent with this, instructing models to prioritize safety reduces EM while prioritizing context-following increases it. These findings establish ICL as a previously underappreciated vector for emergent misalignment that resists simple scaling-based solutions.

## Setup

First, install `uv` package manager. On macOS and Linux, you can do this by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Otherwise, refer to [uv's installation guide](https://github.com/astral-sh/uv).

Then, install the dependencies by running:

```bash
uv sync
```

Run Python scripts with `uv run <scriptname>.py`.

## Map

- [`data/`](data/) — Dataset generation scripts, evaluation question sets, and the encrypted training dataset archive.
- [`icl_em_code/`](icl_em_code/) — Notebook for generating model responses via OpenRouter.
- [`truthfulqa_icl_em_code/`](truthfulqa_icl_em_code/) — Analysis and plotting scripts for the TruthfulQA experiments, including the instruction-priority comparison.
- [`api_exp_plot.py`](api_exp_plot.py) — Aggregates alignment scores and plots misalignment rates for the API experiments.
- [`evaluation_questions.py`](evaluation_questions.py) — Defines free-form evaluation prompts.
- [`gemini_3_pro_0_context_misaligned_response.csv`](gemini_3_pro_0_context_misaligned_response.csv) — Records one Gemini 3 Pro response labeled as misaligned which was generated without any in-context examples. This response was mentioned in Section 4.1, Control experiment.
- [`pyproject.toml`](pyproject.toml) — Project metadata, Python version requirement, and dependencies.
- [`uv.lock`](uv.lock) — Locked dependency versions for reproducible installation with `uv`.

## Citation

```bibtex
@inproceedings{afonin-etal-2026-emergent,
    title = "Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned {LLM}s",
    author = "Afonin, Nikita  and
      Andriianov, Nikita  and
      Hovhannisyan, Vahagn  and
      Bageshpura, Nikhil  and
      Liu, Kyle  and
      Zhu, Kevin  and
      Dev, Sunishchal  and
      Panda, Ashwinee  and
      Rogov, Oleg  and
      Tutubalina, Elena  and
      Panchenko, Alexander  and
      Seleznyov, Mikhail",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1770/",
    doi = "10.18653/v1/2026.acl-long.1770",
    pages = "38197--38212",
    ISBN = "979-8-89176-390-6",
    abstract = "Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across four model families (Gemini, Kimi-K2, Grok, and Qwen), narrow in-context examples cause models to produce misaligned responses to benign, unrelated queries. With 16 in-context examples, EM rates range from 1{\%} to 24{\%} depending on model and domain, appearing with as few as 2 examples. Neither larger model scale nor explicit reasoning provides reliable protection, and larger models are typically even more susceptible. Next, we formulate and test a hypothesis, which explains in-context EM as conflict between safety objectives and context-following behavior. Consistent with this, instructing models to prioritize safety reduces EM while prioritizing context-following increases it. These findings establish ICL as a previously underappreciated vector for emergent misalignment that resists simple scaling-based solutions."
}
```
