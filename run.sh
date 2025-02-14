uv run src/convert.py data/raw/ data/model/dataset.json
uv run src/extract.py data/model/dataset.json data/model/training.json
uv run src/train.py data/model/training.json data/model