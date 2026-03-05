#!/bin/bash
set -e

bash 00_optional_quality_control.sh
bash 01_run_preprocessing_scaled.sh
bash 02_run_feature_extraction.sh
bash 03_A_run_nested_cross_val.sh
bash 03_B_run_nested_cross_val_w_regression.sh
bash 04_validate_on_holdout.sh