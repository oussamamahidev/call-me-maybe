run:
	@uv run python -m src

install:
	uv pip install torch transformers mypy flake8

debug:
	uv run python -m pdb src/main.py

clean:
	@rm -rf **/__pycache__ .mypy_cache

lint:
	
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs
	uv run flake8 .

.PHONY: run install debug clean lint