
run:
	@python3 -m src.main

install:
	pip install torch
	pip install transformers
	pip install mypy
	pip install typing

debug: install
	python3 -m pdb src/main.py

clean:
	@rm -rf **/__pycache__ .mypy_cache

lint:
	flake8
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
