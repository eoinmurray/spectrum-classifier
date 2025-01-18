# CLI Application

This repository contains a Command-Line Interface (CLI) application that facilitates data processing and analysis through various stages. The app uses the `typer` library for an intuitive command structure and is executed with the following command:

```
uv run src/cli.py
```

## Features

The CLI provides the following commands to streamline data processing and analysis:

1. **Convert**: Converts raw data into a structured format for further processing.
2. **Extract**: Extracts features from the converted data based on specified parameters.
3. **VAE**: Runs a Variational Autoencoder (VAE) on the extracted data to generate latent representations.
4. **Plot**: Generates visualizations from the processed data.
5. **Stats**: Computes statistical summaries of the data.

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Ensure the `uv` CLI is installed and properly set up.

## Usage

### General Command Structure

To execute a command, run:

```
uv run src/cli.py <command> [OPTIONS]
```

### Commands

#### 1. `convert`

Converts raw data into a structured format.

- **Options**:

  - `--input-dir`: Directory containing raw data (default: `data/raw`).
  - `--output-dir`: Directory to save converted data (default: `data/converted`).
  - `--limit`: Maximum number of files to process.

- **Example**:

```
uv run src/cli.py convert --input-dir data/raw --output-dir data/converted --limit 100
```

#### 2. `extract`

Extracts features from the converted data.

- **Options**:

  - `--input-dir`: Directory containing converted data (default: `data/converted`).
  - `--output-dir`: Directory to save extracted features (default: `data/extracted`).
  - `--limit`: Maximum number of files to process.
  - `--prominence`: Prominence threshold for feature extraction (default: `0.05`).

- **Example**:

  ```
  uv run src/cli.py extract --input-dir data/converted --output-dir data/extracted --prominence 0.1
  ```

#### 3. `vae`

Runs a Variational Autoencoder on the extracted data.

- **Options**:

  - `--input-dir`: Directory containing extracted data (default: `data/extracted`).
  - `--output-dir`: Directory to save VAE results (default: `data/extracted`).
  - `--limit`: Maximum number of data points to process (default: `100000`).

- **Example**:

  ```
  uv run src/cli.py vae --input-dir data/extracted --output-dir data/vae_results --limit 50000
  ```

#### 4. `plot`

Generates visualizations from the processed data.

- **Options**:

  - `--input-dir`: Directory containing extracted features (default: `data/extracted`).
  - `--output-dir`: Directory to save plots (default: `data/images`).
  - `--limit`: Maximum number of data points to plot (default: `100000`).
  - `--type`: Type of plot to generate (e.g., `spectrum`, default: `spectrum`).
  - `--show`: Display the plots interactively (default: `False`).

- **Example**:

  ```
  uv run src/cli.py plot --input-dir data/extracted --output-dir data/plots --type spectrum --show
  ```

#### 5. `stats`

Computes statistical summaries of the data.

- **Options**:

  - `--input-file`: Path to the file containing extracted data (default: `data/extracted/spectrum-collection.json`).
  - `--output-dir`: Directory to save statistical results (default: `data/stats`).
  - `--limit`: Maximum number of data points to analyze (default: `1000000`).
  - `--show`: Display the statistics interactively (default: `False`).

- **Example**:

  ```
  uv run src/cli.py stats --input-file data/extracted/spectrum-collection.json --output-dir data/stats --show
  ```

## Configuration

Each command saves its configuration in the specified `output_dir` as a `config.yaml` file. This allows for consistent and repeatable processing by loading and merging configuration values.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For any issues or contributions, please submit a pull request or open an issue in the repository.

