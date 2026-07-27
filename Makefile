.PHONY: install test download extract corpus grid generation analyze report clean

install:
	uv sync

test:
	uv run pytest

download:
	uv run finrag download

extract:
	uv run finrag extract

corpus: download extract

grid:
	uv run python experiments/run_grid.py --sweep all

generation:
	uv run python experiments/run_generation.py --all

analyze:
	uv run python experiments/analyze_retrieval.py
	uv run python experiments/analyze_generation.py

report:
	cd report && ./build.sh

clean:
	rm -rf data/cache
