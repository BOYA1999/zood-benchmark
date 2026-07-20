# zood-benchmark

A reproducible code release for benchmarking uncertainty- and diversity-aware molecular portfolio acquisition under distribution shift.

## What this repository contains

This public repository provides the computational benchmark code and the machine-readable benchmark matrix used for reproducibility.

Included:

- `configs/`: benchmark configurations
- `oddd/`: source code for data handling, splitting, modeling, acquisition, metrics, and core utilities
- `scripts/`: benchmark execution and reproducibility scripts
- `tests/`: lightweight test suite
- `runs/tox21_benchmark/benchmark_matrix.csv`: benchmark result matrix
- `requirements.txt` and `requirements-dev.txt`: Python dependencies

## What this repository does not contain

This public code repository does not include submission-facing manuscript materials such as:

- manuscript text
- cover letters
- supplementary submission files
- author contact details
- journal-specific packaging assets

## Reproducibility workflow

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the benchmark:

```bash
python scripts/run_benchmark.py --config configs/tox21_benchmark.yml
```

Generate core benchmark summaries:

```bash
python scripts/summarize_core_evidence.py --matrix runs/tox21_benchmark/benchmark_matrix.csv --split scaffold --out runs/tox21_benchmark/core_evidence_summary.json
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Scope note

The current benchmark release focuses on fixed-budget molecular nomination under random, scaffold, and cluster shift across the Tox21 benchmark setting. The benchmark matrix is included to support direct reproducibility and secondary analysis of the released results.

