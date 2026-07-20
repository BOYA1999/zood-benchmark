# zood-benchmark public code package

This package contains the computational benchmark code and the machine-readable benchmark matrix for the OOD molecular portfolio-acquisition study.

Included:
- `configs/`: benchmark configurations
- `oddd/`: source code for data handling, splitting, modeling, acquisition, metrics, and reusable visualization utilities
- `scripts/`: benchmark execution and core reproducibility scripts
- `tests/`: lightweight tests
- `runs/tox21_benchmark/benchmark_matrix.csv`: benchmark result matrix for reproducibility
- `requirements.txt` and `requirements-dev.txt`: Python dependencies

Excluded from this public code package:
- manuscript files
- supplementary submission files
- cover letters
- author affiliations and correspondence details
- submission-only checking scripts
- paper-only summary generation scripts

Typical workflow:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/run_benchmark.py --config configs/tox21_benchmark.yml
python scripts/summarize_core_evidence.py --matrix runs/tox21_benchmark/benchmark_matrix.csv --split scaffold --out runs/tox21_benchmark/core_evidence_summary.json
python -m pytest tests/ -v
```
