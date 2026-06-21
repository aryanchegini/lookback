VENV := .venv
PY := $(VENV)/bin/python

.PHONY: test bench-sweep clean

test:
	$(PY) -m pytest

bench-sweep:
	$(PY) benchmarks/sweep_scaling.py

clean:
	rm -rf cache/*.parquet
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
