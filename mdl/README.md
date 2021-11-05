# Minimum Description Length

Codebase adapted from the EMNLP 2020 paper [Information-Theoretic Probing with Minimum Description Length](https://arxiv.org/pdf/2003.12298.pdf).

## Create Conda Environment

We attach the conda environment config.
Use `conda env create -f environment.yml` and then `conda activate env`.

## Download Data

Download the CausalMT dataset in [this zip file](https://drive.google.com/file/d/10N8rW8BA-aPDIjFkmvJ4BQnB3REoVG6O/view?usp=sharing) (24.3MB) and change the dataset path or basepath here `control_tasks/control_tasks/data.py`-Line 13.

## Run Experiments

We attach the config for each experiment.
1. `cd control_tasks/control_tasks`
2. `python3 run_experiment.py $CONFIG_PATH$`

## Notes

Most of the adaptations we made are in the following folders:
- [`control_tasks/mdl_configs/`](control_tasks/mdl_configs/)
- [`control_tasks/control_tasks/`](control_tasks/control_tasks/)


