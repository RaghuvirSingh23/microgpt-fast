# microgpt-fast

A one-file character-level GPT for learning how language models work from the
inside out.

The file contains two implementations of the same 4,192-parameter model:

- a scalar engine with a hand-built `Value` autograd system, attention, MLP,
  loss, and Adam optimizer;
- a tensorized PyTorch engine that preserves the architecture and
  one-document-per-update training semantics while running hundreds of times
  faster.

The model learns from a dataset of human names and generates new name-like
strings one character at a time. The dataset is downloaded automatically on
the first run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

One complete pass over the 28,829-name training split, followed by evaluation
on all 3,204 held-out names:

```bash
TRAINING_STEPS=28829 EVALUATION_LIMIT=0 python3 microgpt.py
```

The default PyTorch CPU engine takes roughly 20 seconds on an Apple M3 Pro.
It writes the learned weights to `microgpt_checkpoint.pt`.

For a quick smoke test:

```bash
TRAINING_STEPS=100 EVALUATION_LIMIT=100 python3 microgpt.py
```

## Generate names

Load the saved checkpoint and sample names:

```bash
MODE=generate NUM_SAMPLES=20 python3 microgpt.py
```

Complete a prefix:

```bash
MODE=generate PROMPT=mar NUM_SAMPLES=10 python3 microgpt.py
```

Control sampling randomness with `TEMPERATURE` and reproducibility with
`SAMPLE_SEED`:

```bash
MODE=generate TEMPERATURE=0.7 SAMPLE_SEED=7 python3 microgpt.py
```

## Study the scalar implementation

Run the original hand-built scalar engine with a deliberately small workload:

```bash
ENGINE=scalar TRAINING_STEPS=20 EVALUATION_LIMIT=10 python3 microgpt.py
```

This path exposes the computation graph and chain rule directly. It is for
understanding, not speed.

## Architecture

- character vocabulary: 26 lowercase letters plus one beginning/end token
- context length: 16
- embedding width: 16
- attention heads: 4, with 4 dimensions per head
- transformer blocks: 1
- MLP width: 64
- trainable parameters: 4,192
- optimizer: Adam

The project is inspired by Andrej Karpathy's educational
[`microgpt`](https://github.com/karpathy/microgpt),
[`micrograd`](https://github.com/karpathy/micrograd), and
[`makemore`](https://github.com/karpathy/makemore) projects.
