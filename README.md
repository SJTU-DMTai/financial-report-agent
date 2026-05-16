# Financial Report Agent

Agent workflow for generating and evaluating financial research reports.

## Setup

Requirements:

- Python 3.10+
- `wkhtmltopdf` if PDF export is needed
- API keys for the configured LLM/VLM providers

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and `config.local.yaml` as needed. `config.local.yaml` is preferred over `config.yaml` when present.

## Directories

- `data/`: runtime memory and reference data.
  - `data/memory/long_term/`: long-term memory, stock code mapping, and demonstration reports.
  - `data/memory/short_term/`: per-run intermediate materials.
- `output/`: generated reports and evaluation results.
  - `output/reports/<model_name>/`: generated `.json`, `.md`, and `.pdf` reports.
  - `output/<method_name>_<evaluator_llm_name>_benchmark_results.json`: benchmark evaluation outputs.

Reference PDF files are loaded from `DEMO_DIR` in `.env`.

## Run

Run the single example task in `main.py`:

```bash
python -u main.py
```

Run benchmark tasks from `benchmark.json`:

```bash
python -u run_benchmark.py --batch_size 1
```

Evaluate generated benchmark reports:

```bash
python -m src.pipelines.evaluation --method_name qwen3-32b
```

The evaluator reads generated reports from `output/reports/<method_name>/` and writes results to `output/<method_name>_<evaluator_llm_name>_benchmark_results.json`.
