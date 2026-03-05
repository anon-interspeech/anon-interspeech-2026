# Beyond Binary: Beyond Binary: Speech Representations Across the Cognitive Score Hierarchy

This repository contains the implementation and experimental framework for the anonymous Interspeech 2026 submission:

> **"Beyond Binary: Beyond Binary: Speech Representations Across the Cognitive Score Hierarchy"**

The project investigates how speech-derived representations relate to the hierarchical structure of clinical cognitive assessment.
Rather than focusing solely on binary diagnosis, the study models cognitive outcomes across multiple levels of the clinical scoring hierarchy.

---

## Hierarchical Cognitive Targets

### Level 1 — Individual Tests
Raw scores from six neuropsychological tasks.

### Level 2 — Cognitive Domains
Composite scores for:
- Language  
- Memory  
- Executive Function  
- Visuospatial Function  

### Level 3 — Global Status
- CERAD total score (coninous and binarised at 85)
- Mild Cognitive Impairment (MCI) status  

---

# Technical Methodology

The provided codebase implements a rigorous pipeline designed for clinical speech analysis:

## Acoustic Preprocessing
- Automated clipping detection  
- Reference-free SNR estimation  
- 6th-order Butterworth high-pass filtering (fc = 100 Hz)  

## Feature Extraction
Dual-path extraction supporting:
- Handcrafted acoustic parameters (**eGeMAPS**)  
- Self-supervised representations (**wav2vec 2.0** and **HuBERT**)  

## SSL Processing
- Extraction from frozen hidden layers  
- Global mean pooling  
- Fixed-length embedding generation  

## Validation Framework
- **5×3 Nested Cross-Validation (NCV)**  
- Strict subject-disjoint (ID-disjoint) splits to prevent data leakage in longitudinal recordings  

## Predictive Modeling
Support for:
- Ridge Regression  
- Support Vector Machines (SVM/SVR)  
- Extreme Gradient Boosting (XGBoost)  

All models include internal hyperparameter optimization.

## Repository Structure

```text
.
├── 00_optional_quality_control.sh
├── 01_run_preprocessing_scaled.sh
├── 02_run_feature_extraction.sh
├── 03_A_run_nested_cross_val.sh
├── 03_B_run_nested_cross_val_w_regression.sh
├── 04_validate_on_holdout.sh
├── README.md
├── config
│   ├── datasets.sh
│   ├── features
│   │   ├── egemaps_all.yaml
│   │   ├── egemaps_prosody.yaml
│   │   ├── egemaps_voice_quality.yaml
│   │   ├── hubert.yaml
│   │   ├── overall_feature_extraction.yaml
│   │   └── wav2vec2.yaml
│   ├── mci_winner.yaml
│   ├── models
│   │   ├── logreg.yaml
│   │   ├── ridge_reg.yaml
│   │   ├── svm.yaml
│   │   ├── svr.yaml
│   │   ├── xgboost.yaml
│   │   └── xgboost_regr.yaml
│   ├── nested_cross_validation.yaml
│   ├── nested_cross_validation_regression.yaml
│   └── preprocessing
│       ├── best_hyperparameters_preprocessing.yaml
│       └── hyperparameters_preprocessing.yaml
├── data
│   ├── cache
│   └── splits
├── full_pipeline.sh
├── requirements.txt
└── src
    ├── cross_validation
    │   ├── cv_engine_extended_logging_logits.py
    │   └── cv_engine_extended_logging_logits_w_reg.py
    ├── data
    │   ├── loading
    │   │   └── data_handler.py
    │   ├── preprocessing
    │   │   └── acoustic_preprocessing_scale.py
    │   ├── qc
    │   │   └── qc.py
    │   └── standardisation
    │       └── comparison.py
    ├── features
    │   └── overall_feature_extraction.py
    ├── final_training_best_model
    │   └── final_train.py
    └── models
        ├── nested_cross_val_opt_parallel.py
        └── nested_cross_val_opt_parallel_w_regression.py
```

# Dataset Configuration
All cognitive tasks (from CERAD+ or MMSE) used in the pipeline are defined in:
```
config/datasets.sh
```
All pipeline stages automatically read this configuration to ensure consistent dataset usage across preprocessing, feature extraction, and modeling.

# Data Privacy and Ethics 
The data used in this study consists of sensitive clinical recordings from a German-speaking geriatric cohort.

Due to ethical restrictions and institutional review board regulations, the following cannot be publicly released:
- raw speech recordings
- clinical metadata
- participant identifiers 

The repository therefore provides the full modeling and evaluation framework, enabling replication of the experimental pipeline while preserving participant privacy.

# Setup:
## 1. Create Environment:
```
conda create -n speech_env python=3.10
 conda activate speech_env
  pip install -r requirements.txt
```
## 2. Execute Pipeline:
Each stage of the pipeline can be executed independently.
### Audio Quality Control 
```bash
bash 00_optional_quality_control.sh
```
### Acoustic Preprocessing
```bash
bash 01_run_preprocessing_scaled.sh
```
### Feature Extraction
```bash
bash 02_run_feature_extraction.sh
```
### Nested Cross-Validation (Classification)
```bash
bash 03_A_run_nested_cross_val.sh
```
### Nested Cross-Validation (Regression)
```bash
bash 03_B_run_nested_cross_val_w_regression.sh
```
### Hold-out Evaluation
```bash
bash 04_validate_on_holdout.sh
```
### Running the Full Pipeline
```bash
bash run_pipeline.sh
```

# Implementation Details
## Acoustic Feature Extraction

Handcrafted acoustic descriptors are extracted using the Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) via the OpenSMILE toolkit.

## Self-Supervised Speech Models

Transformer-based speech embeddings are extracted using the HuggingFace Transformers library.

Supported models include:

- wav2vec 2.0

- HuBERT

## Optimization Strategy

Hyperparameter tuning occurs within the inner cross-validation loop.

Optimization metrics:

- Balanced Accuracy for classification tasks

- R² score for regression tasks

## Anonymization

To comply with double-blind review requirements, all identifying information has been removed from the repository, including:

- author names

- institutional affiliations

- project-specific infrastructure references

- These details will be restored upon publication.

# Citation

Citation information will be added upon publication.