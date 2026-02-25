
run:
	@python3 -m src.main

install:
	pip install torch transformers

clean:
	rm -rf __pycache__