# Contributing

Contributions that improve correctness, reproducibility and documentation are welcome.

## Development setup

Use Python 3.10 or 3.11 in a virtual environment, then run:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The suite uses synthetic images and runs on CPU. Changes to utility labels should preserve the fixed-reference target bundle, per-image normalization, detached assignments and frozen probe. Changes to the feature path should preserve the zero-gate bypass test.

## Reporting a problem

Open an issue with your Python, PyTorch and Ultralytics versions, the command you ran, the full error, and a minimal example. For data-related errors, share a small anonymized label example and its image dimensions rather than the full dataset.

## Proposing a change

Describe the concrete problem and resulting behavior in the pull request. Add a focused test for a changed mathematical invariant or a reproduced bug. Update the README or implementation notes when a modeling assumption changes.

Keep code validation separate from research claims. Performance reports should identify the dataset conversion, exact split, seeds, settings and checkpoints used. Do not label synthetic test output as experimental evidence or claim reproduction based only on matching dataset counts.

Contributions are submitted under this repository's AGPL-3.0 license.
