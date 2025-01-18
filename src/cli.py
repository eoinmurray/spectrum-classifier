import typer
from convert import main as run_convert
from extract import main as run_extract
from train import main as run_train


app = typer.Typer()

DEFAULT_PROMINENCE = 0.05
DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_MODEL_DIR = "data/model"
DEFAULT_STATS_DIR = "data/stats"
DEFAULT_CONVERTED_FILE = "data/model/converted.json"
DEFAULT_EXTRACTED_FILE = "data/model/extracted.json"
DEFAULT_TRAINING_FILE = "data/model/training.json"
DEFAULT_TARGET_LABEL = "qd_id"

@app.command()
def all():
    run_convert(
      input_dir=DEFAULT_INPUT_DIR, 
      output_file=DEFAULT_CONVERTED_FILE, 
      limit=None
    )

    run_extract(
      input_file=DEFAULT_CONVERTED_FILE, 
      output_dir=DEFAULT_MODEL_DIR, 
      prominence=DEFAULT_PROMINENCE,
      max_peaks=15
    )

    run_train(
      input_file=DEFAULT_TRAINING_FILE, 
      output_dir=DEFAULT_MODEL_DIR, 
    )

@app.command()
def convert(
  input_dir: str = DEFAULT_INPUT_DIR, 
  output_file: str = DEFAULT_CONVERTED_FILE, 
  limit: int = None
):
    run_convert(
      input_dir, 
      output_file, 
      limit
    )

@app.command()
def extract(
  input_file: str = DEFAULT_CONVERTED_FILE, 
  output_dir: str = DEFAULT_MODEL_DIR, 
  prominence: float = DEFAULT_PROMINENCE,
  max_peaks: int = 15
):
    run_extract(
      input_file, 
      output_dir, 
      prominence,
      max_peaks
    )

@app.command()
def train(
  input_file: str = DEFAULT_TRAINING_FILE, 
  output_dir: str = DEFAULT_MODEL_DIR, 
):
    run_train(
      input_file, 
      output_dir, 
    )
    
if __name__ == "__main__":
    app()
