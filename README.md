# Research Report Analysis 2025

This project contains Python scripts to analyze research data and generate statistics and visualization charts for the "Informe Rector Investigación" (Research Rector Report) for the year 2025 at Universidad Politécnica Salesiana (UPS).

## Project Overview

The scripts process data related to:
- Research Groups
- Research Areas (UNESCO codes)
- Scientific Publications (Scopus, Web of Science, etc.)
- Research Projects (ODS, Vinculación programs)

It reads raw data from Excel files, computes statistics (growth, distribution), generates text reports, and creates various charts (Pie charts, Bar charts, etc.).

## Directory Structure

```
├── main.py          # Main script to run the analyses
├── charts.py        # Module for plotting functions
├── gestion_coordinaciones.py # Script for analyzing coordination management
├── util.py          # Utility functions
├── input/           # Directory containing source Excel files
├── output/          # Directory where reports and charts are saved
└── ups.png          # Logo or asset used in charts
```

## Prerequisites

- **Python 3.x**
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `adjustText`
  - `xlsxwriter`
  - `openpyxl`

You can install the dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn adjustText xlsxwriter openpyxl
```

## Input Data

The script expects the following Excel files in the `input/` directory:

- `All_Pure_Publications.xlsx`
- `Publicacion_Autor_Org.xlsx`
- `Publications_Cites_Scopus.xlsx`
- `all_publications_sjr_jcr.xlsx`
- `colaboradores_07_10_2025.xlsx`
- `debug_groups.xlsx`
- `inv_profiles.xlsx`
- `savedrecs.xls`
- `scopus.xlsx`
- `sgi_stats.xlsx`

## Usage

### Main Analysis

To run the complete analysis, execute the `main.py` script from the project root:

```bash
python main.py
```

### Coordination Management Analysis

To generate the coordination management charts, run:

```bash
python gestion_coordinaciones.py
```

This script requires `gestion_coordinaciones_2025.xlsx` and generates `coordinaciones_2025.svg` and `coordinaciones_2025.png`.

## Analysis workflow

The script performs the analysis in the following order:
1. **Analyze Groups Data**: Processes research group statistics.
2. **Analyze Areas Data**: Analyzes UNESCO research areas.
3. **Analyze Publications**: Processes publication counts, citations, and journal rankings.
4. **Analyze Projects**: Analyzes research projects by Campus (Sede), ODS, and engagement programs.

## Output

The script generates output files in the `output/` directory:

- **Text Reports**: Detailed text files with calculated statistics.
  - `areas_data_2024.txt` / `areas_data_2025.txt`
  - `groups_data_2025.txt`
  - `publications_data_2025.txt`
  - `projects_data_2025.txt`
- **Charts**: Visualizations generated during the analysis (e.g., bar charts, pie charts) are displayed or saved depending on the configuration in `charts.py`.
