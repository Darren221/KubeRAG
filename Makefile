.PHONY: install check lint format type test cov clean

install:
	uv sync --extra dev --extra dashboard

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run mypy src/

test:
	uv run pytest

cov:
	uv run pytest --cov=src/kuberag --cov-report=term-missing

check: lint type test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
