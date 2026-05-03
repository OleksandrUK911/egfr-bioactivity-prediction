.PHONY: help install fetch run test lint notebook clean

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies (pip)"
	@echo "  make fetch        Download EGFR data from ChEMBL"
	@echo "  make fetch-erbb   Download full ErbB family (for §23)"
	@echo "  make run          Launch Streamlit app"
	@echo "  make test         Run unit tests"
	@echo "  make lint         Run flake8 linter"
	@echo "  make notebook     Execute full notebook (saves _executed.ipynb)"
	@echo "  make clean        Remove cached outputs and checkpoints"

install:
	pip install -r requirements.txt

fetch:
	python fetch_data.py

fetch-erbb:
	python fetch_data.py --targets EGFR,ERBB2,ERBB3,ERBB4

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	flake8 app.py fetch_data.py run_notebook.py --max-line-length 100

notebook:
	python run_notebook.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null; \
	rm -f optuna.db optuna_study.db
