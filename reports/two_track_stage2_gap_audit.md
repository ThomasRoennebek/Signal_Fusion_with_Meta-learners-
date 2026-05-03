# Thesis Pipeline Gap Audit: Two-Track Stage 2 Setup

## 1. Executive Summary
This report audits the current thesis codebase against the target two-track Stage 2 architecture (Track 1: Real + Augmented Support vs. Track 2: Synthetic Contract Simulation). 

**Overall finding**: The codebase currently operates on a single implicit track. Track 1 is partially supported as the default behavior, but lacks strict architectural boundaries (`query_policy`, separate artifact roots, explicit Track IDs). Track 2 and synthetic contract generation are entirely missing and must be built from the ground up.

There are **0** files already fully aligned with the target state without modifications, **7** files requiring moderate changes, and several new modules needed.

---

## 2. Current State Overview
- **Data Files**: Single output (`contract_with_features_labeled_with_gold.csv`) serves all downstream needs. No synthetic contract simulation dataset exists.
- **Stage 1**: Artifacts are saved to hardcoded paths inside `models/stage_1/`. Config YAML does not support tracks or artifact roots.
- **Stage 2**: Assumes `gold_y` as target and reads Stage 1 weights directly from `models/stage_1/`. It supports SMOTENC support augmentation via `stage2b_synthetic.py`, but query policy enforcement and track-aware experiment IDs do not exist.
- **Notebooks**: `15.1_prepare_gold_labels.ipynb` only outputs real labels. `17.1_stage2_analysis_dashboard.ipynb` analyzes everything simultaneously without track distinction.

---

## 3. Final Target Architecture
The pipeline will support two mutually exclusive tracks selected via YAML:
- **Track 1 (Feasibility)**: Real dataset, `gold_y` target, `query_policy = real_only`, `models/stage_1/real/`.
- **Track 2 (Simulation)**: Real + Synthetic dataset, `sim_y` target, `query_policy = real_plus_synthetic`, `models/stage_1/synthetic_contract_simulation/`.

---

## 4. What is Already Supported
- `15_snorkel_labeling.ipynb` successfully generates `renegotiation_prob` without touching gold columns.
- `src/master_thesis/stage2b_synthetic.py` handles SMOTENC support augmentation after the support/query split (useful for Track 1).
- Stage 1 and Stage 2 runner frameworks are functional and modular enough to accept new config parameters.

---

## 5. What is Partially Supported
- **Track 1**: Exists implicitly. Needs `query_policy` enforcement, `experiment_track` in summary, and moved to `models/stage_1/real/`.
- **Gold Notebook**: Injects `gold_y` correctly, but needs Section B for synthetic simulation.
- **Diagnostics**: Some basic distribution checks exist, but zero validation for synthetic leakage or query composition.

---

## 6. What is Missing
- **Synthetic Contract Generation**: No mechanism exists to clone contracts and apply Gaussian noise.
- **Track 2 Dataset**: `contract_with_features_synthetic_contract_simulation.csv` does not exist.
- **Stage 1 Track Config**: No support for `artifact_root` or dataset overriding.
- **Stage 2 Track Config**: No support for `target_col`, `query_policy`, or `experiment_track`.
- **Composition Accounting**: `experiment_summary.csv` lacks real vs synthetic query/support ratios.
- **Validation Scripts**: No scripts exist to prevent target leakage, mixing tracks, or corrupting `gold_y`.

---

## 7. Gap Matrix

| Area | Requirement | Current Support Status | Evidence | Risk | Required Implementation | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Data Files** | Track 2 Simulation Dataset | Missing | `Data/processed/` missing simulation file. | Low | Generate via new notebook section. | High |
| **Gold Notebook** | Section B: Synthetic Generation | Missing | `15.1` only outputs real labels. | Medium | Add call to helper script. | High |
| **Synthetic Generation** | Contract-level cloning & perturbation | Missing | No `synthetic_contracts.py` exists. | High | Create Gaussian noise generator for continuous features. | Critical |
| **Stage 1 YAML** | `tracks`, `artifact_root`, `target_col` | Missing | `experiments/stage1_config.yaml` | Medium | Restructure YAML schema. | High |
| **Stage 1 Runner** | Dynamic roots & target assignment | Partially Supported | `run_stage1.py` hardcodes paths. | High | Parametrize paths based on track config. | Critical |
| **Stage 1 Artifacts** | `models/stage_1/real/` vs `.../sim/` | Missing | `models/stage_1/` is flat. | High | Prevent overwrite. Update `metadata.json`. | Critical |
| **Stage 2 YAML** | `query_policy`, `experiment_track` | Missing | `experiments/stage2_config.yaml` | Medium | Restructure preset schema. | High |
| **Stage 2 Runner** | Track-aware IDs, targets, and roots | Partially Supported | `run_stage2.py` hardcodes `gold_y`. | High | Parametrize target, root, and append mode to IDs. | Critical |
| **Episode Sampler** | `record_source`, `is_synthetic_contract` | Missing | `episode_sampler.py` unaware of synth. | High | Pass through synthetic metadata columns. | Critical |
| **Result Accounting** | Support/query composition fields | Missing | `experiment_summary.csv` lacks synth counts. | Medium | Add composition reporting to logging. | High |
| **Diagnostics** | Validate parent-child leakage | Missing | `18_dataset_diagnostic_report.ipynb` | High | Add query leakage checks. | Medium |
| **Validators** | `validate_stage2_registry_tracks.py` | Missing | `experiments/` missing track validators. | Medium | Create strict programmatic checks. | Medium |

---

## 8. File-by-File Implementation Requirements
- **`src/master_thesis/synthetic_contracts.py`**: [NEW] Must implement contract cloning, Gaussian perturbation for continuous variables, and categorical copying.
- **`notebooks/15.1_prepare_gold_labels.ipynb`**: [MODIFY] Add Section B to invoke generator and export `...synthetic_contract_simulation.csv`.
- **`experiments/stage1_config.yaml`**: [MODIFY] Define `tracks` dictionary.
- **`experiments/run_stage1.py` & `src/master_thesis/stage1.py`**: [MODIFY] Implement dynamic `artifact_root` resolution.
- **`experiments/stage2_config.yaml`**: [MODIFY] Add presets defining `query_policy` and `experiment_track`.
- **`src/master_thesis/stage2.py` & `run_stage2.py`**: [MODIFY] Route targets and artifact paths. Record query/support composition.
- **`src/master_thesis/episode_sampler.py`**: [MODIFY] Retain synthetic provenance columns.

---

## 9. Notebook-by-Notebook Requirements
- **`15_snorkel_labeling.ipynb`**: Needs no changes. (Compliant)
- **`15.1_prepare_gold_labels.ipynb`**: Must add Section B (Synthetic contract simulation generation).
- **`16_stage1_baselines.ipynb`**: Must be updated to visualize both `real` and `simulation` roots.
- **`17_stage2_meta_learning.ipynb`**: Must handle dynamic track selection via YAML.
- **`17.1_stage2_analysis_dashboard.ipynb`**: Must filter dataframes by `experiment_track`.
- **`18_dataset_diagnostic_report.ipynb`**: Must implement composition and leakage validations.

---

## 10. YAML / Preset Requirements
**Stage 1 Example:**
```yaml
tracks:
  real_augmented_support:
    data_filename: contract_with_features_labeled_with_gold.csv
    target_col: gold_y
    artifact_root: models/stage_1/real
  synthetic_contract_simulation:
    data_filename: contract_with_features_synthetic_contract_simulation.csv
    target_col: sim_y
    artifact_root: models/stage_1/synthetic_contract_simulation
```

**Stage 2 Example:**
```yaml
presets:
  thesis_main_real:
    experiment_track: real_augmented_support
    data_filename: contract_with_features_labeled_with_gold.csv
    target_col: gold_y
    stage1_artifact_root: models/stage_1/real
    query_policy: real_only
```

---

## 11. Artifact Structure Requirements
- Ensure `models/stage_1/real/global/A_weak_only/` is generated.
- Ensure Track 2 **reuses** the above `A_weak_only` and does not generate its own.
- Ensure all Stage 2 experiments include `__track-real` or `__track-sim` in folder names.

---

## 12. Validation Requirements
Must create strict validation scripts:
1. `validate_real_gold_file.py`: Ensures `gold_y` has no fake entries.
2. `validate_synthetic_contract_simulation.py`: Ensures parent/child IDs exist and features are perturbed correctly.
3. `validate_stage2_registry_tracks.py`: Validates that `query_synthetic_proportion == 0` for all `track-real` experiments.

---

## 13. Implementation Roadmap
- **Phase 1 (Foundation)**: Create `synthetic_contracts.py` generator and update `15.1` notebook. Write dataset validators.
- **Phase 2 (Stage 1 Routing)**: Implement `experiment_track` YAML structures and dynamic artifact roots in Stage 1. 
- **Phase 3 (Stage 2 Routing)**: Parametrize `target_col`, `stage1_artifact_root`, and `query_policy` in Stage 2. Update episode sampler.
- **Phase 4 (Accounting)**: Pipe `n_synthetic_query_contracts` and related metrics out to `experiment_summary.csv`.
- **Phase 5 (Reporting)**: Update dashboards to filter by track.

---

## 14. Specific Open Questions / Assumptions
- **Assumption:** Gaussian noise on continuous features is sufficient for full synthetic contract generation (V1).
- **Open Question:** How many synthetic contracts per real seed should be generated? (Suggest 10x multiplier as a starting simulation config).
- **Open Question:** Should `observation_year` be perturbed or cloned exactly? (Suggest exact cloning for V1 temporal consistency).
