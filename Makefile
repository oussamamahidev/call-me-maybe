run:
	@uv run python -m src

install:
	uv pip install torch transformers mypy flake8

debug:
	uv run python -m pdb -m src

clean:
	@rm -rf **/__pycache__ .mypy_cache

lint:
	@mypy . --exclude='^llm_sdk/' \
		--warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs
	@flake8 . --exclude=llm_sdk

.PHONY: run install debug clean lint
