# Contributing to daim-foundation-models

Thanks for helping. This repository builds predictive models from readable
PyTorch primitives, so a contribution is judged on two things: is it correct,
and can a reader follow it?

## Set up

```bash
git clone https://github.com/DAIMCentre/DAIM-Foundation-Model.git
cd DAIM-Foundation-Model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
python -m pytest -q
```

A CPU is enough. If you do not need GPU support, installing the CPU build of
PyTorch first is much faster:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The tests use only `tests/fixtures/` and never download data. To work with the
TinyStories subset, run `python scripts/prepare_tinystories.py` once.

## Ground rules

1. **Readable over clever.** Build from PyTorch primitives. Do not add
   `torch.nn.Transformer`, Hugging Face trainers or pretrained weights to the
   core examples.
2. **Reproducible runs.** New experiments record their configuration, seed,
   software versions and data checksums, as the existing examples do.
3. **Tests stay offline and fast.** No test may download data or run a
   0.1B-parameter forward pass.
4. **Results are evidence, not claims.** Describe what a run shows and what it
   does not.

## Good first contributions

- A new baseline under the same temporal split and metrics as
  `last_value_forecast`.
- A better tokenizer (for example, byte-pair encoding) with tests showing it is
  reversible.
- Learning-rate schedules, gradient clipping or other training options, each
  behind a config flag.
- Clearer explanations or diagrams in the README.

## Making a change

1. Open an issue first for anything larger than a small fix, so the approach
   can be agreed before you write it.
2. Create a branch from `main`.
3. Add or update tests alongside the code.
4. Run `python -m pytest -q`.
5. Open a pull request and fill in the template.

Keep pull requests focused: one idea per PR is much easier to review.

## What not to commit

- datasets (`data/` is ignored), checkpoints (`*.pt`, `*.pth`) or large binary
  files
- generated runs under `results/lm-example/` or `results/live-*/`
- API keys or credentials

## Licence

By contributing you agree that your contribution is released under the
[MIT Licence](LICENSE) of this repository. Datasets keep their own licences:
TinyStories is CDLA-Sharing-1.0 and is downloaded by each user, not
redistributed here.
