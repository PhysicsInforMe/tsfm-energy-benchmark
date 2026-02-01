.PHONY: setup benchmark demo clean test lint format

setup:
	pip install -e ".[all]"

benchmark:
	python scripts/run_benchmark.py --config configs/benchmark_config.yaml

demo:
	streamlit run demo/streamlit_app.py

notebook:
	jupyter notebook notebooks/

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

format:
	black src/ tests/ scripts/

download-data:
	python scripts/download_data.py

clean:
	rm -rf results/tables/*.csv results/figures/*.png results/figures/*.pdf
	rm -rf data/raw/ data/processed/
	find . -type d -name __pycache__ -exec rm -rf {} +
