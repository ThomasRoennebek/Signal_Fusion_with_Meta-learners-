# Signal Fusion with Meta-Learners: An explainable framework for data-driven decision support  

## Master's Thesis Project

This repository contains the implementation of a master's thesis focused on developing a data-driven decision-support framework using machine learning and signal fusion techniques. The project investigates how information from multiple data sources (Financial, ESG, News, Market, Lifecycle) can be combined into meaningful signals and integrated through a meta-learning model to support prioritization and decision-making in complex environments.

The framework emphasizes transparency, modularity, and practical applicability, aiming to support human judgment rather than replace it.

---

## Architecture Pipeline
The system operates on a two-stage pipeline designed to minimize human labeling efforts while capturing domain expertise:

1. **Stage 1 (Signal Extraction & Weak Supervision)**: Uses `Snorkel` to define Labeling Functions (LFs) based on domain knowledge. Classical ML baselines (XGBoost, ElasticNet) are trained on these weak labels to generate probabilistic signals.
2. **Stage 2 (Meta-Learner)**: A Model-Agnostic Meta-Learning (MAML) neural network acts as the fusion center. It samples N-way K-shot episodes to rapidly adapt to different business contexts (e.g., specific departments like Logistics), ultimately outputting calibrated and explainable priority scores.

## Repository Structure

- `Data/` - Raw, interim, and processed datasets.
- `experiments/` - YAML configuration files and executable scripts (`run_stage1.py`, `run_stage2.py`) for reproducible runs.
- `models/` - Saved weights and configurations for Stage 1 and Stage 2 models.
- `notebooks/` - Jupyter notebooks for Exploratory Data Analysis (EDA) and experimental prototyping.
- `reports/` - Generated analysis including figures and tables.
- `src/master_thesis/` - The core reusable Python package containing:
  - `snorkel_lfs.py`: Domain-knowledge labeling functions.
  - `baselines.py`: Stage 1 machine learning pipelines.
  - `mlp.py` & `maml.py`: Stage 2 deep learning and meta-learning architectures.
  - `episode_sampler.py`: Stratified N-way K-shot episode sampling.
  - `calibration.py` & `explainability.py`: Post-processing for trustworthy AI (Temperature Scaling, SHAP).
- `tests/` - `pytest` suite ensuring mathematical correctness and preventing data leakage.

---
*Master's Thesis — Technical University of Denmark (DTU)*
