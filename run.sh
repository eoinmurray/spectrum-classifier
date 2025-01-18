rm -rf data/model data/stats
uv run src/cli.py convert
uv run src/cli.py split
uv run src/cli.py train