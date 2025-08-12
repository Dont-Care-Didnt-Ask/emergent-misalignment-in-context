# Emergent Misalignment In Context

This repo contains the code for experiments exploring emergent misalignment in ICL (in-context learning) regime.

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

## Code

`soft_prompt_tuning.py` contains the code for soft prompt tuning on a narrow misaligned dataset. Soft prompt tuning experiments serve two purposes:
1. Set an upper bound of what is achievable in ICL regime, since soft prompts are more flexible compared to hard prompts.
2. If soft prompts elicit emergent misalignment, we can use corresponding vectors to find ICL examples which are likely to elicit misaligned behavior.