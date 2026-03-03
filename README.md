# Beyond Binary: Large-Scale Hierarchical Cognitive Assessment from Real-World Speech

This repository contains the implementation and experimental framework for the anonymous Interspeech 2026 submission:

> **"Beyond Binary: Large-Scale Hierarchical Cognitive Assessment from Real-World Speech"**

The project investigates the relationship between speech representations and the hierarchical structure of clinical cognitive assessment across three distinct levels:

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
- CERAD total scores  
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
├── preprocessing/      # Signal cleaning, diarization, and quality assessment
├── features/           # eGeMAPS extraction and SSL embedding pipelines
├── models/             # Nested CV framework and model architectures
├── configs/            # Hyperparameter grids and target definitions
├── demo/               # Implementation walkthrough using synthetic data
└── requirements.txt    # Essential dependencies
```

# Data Privacy and Ethics

The data utilized in this study consists of sensitive clinical recordings from a German-speaking geriatric cohort. Due to ethical restrictions and legal regulations concerning participant privacy (Institutional Ethics Committee), raw audio recordings and clinical metadata cannot be made publicly available.

To facilitate the review process and demonstrate the integrity of the modeling framework, the `demo/` directory contains a **synthetic dataset**.  

This dummy data mirrors:
- The structure  
- Feature dimensionality  
- Hierarchical labels  

of the original study.

This enables verification of:
- ID-disjoint splitting logic  
- Hierarchical prediction trajectories described in the manuscript  

---

# Setup and Reproduction

To verify the pipeline using the provided synthetic data:

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Execute Modeling Demo
```bash
python demo/run_hierarchical_pipeline.py
```

# Implementation Details 
## Accoustic Analysis 
Handcrafted features are extracted using the Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) via the OpenSMILE library.
## SSL Models 
Transformer-based embeddings are processed using the HuggingFace Transformers library.
## Optimization
Hyperparameter tuning is conducted within the inner loop of the NCV using a grid search strategy, optimizing for:

- Balanced Accuracy (classification)

- R-squared (regression)

## Anonymization 
In accordance with double-blind review requirements, all institutional and author identifiers have been removed from the source code and metadata.

## Citation 
Author information and full citation details will be updated upon publication.