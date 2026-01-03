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

Finally, use `uv run <scriptname>.py` to run Python scripts instead of `python <scriptname>.py`

## Code

`api_exp_plot.py` contains code for building plots.