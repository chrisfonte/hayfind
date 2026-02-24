PYTHON ?= python3

.PHONY: install-dev lint test serve

install-dev:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src tests

test:
	pytest -q

serve:
	hayfind serve
