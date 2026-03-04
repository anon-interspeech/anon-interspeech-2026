import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from joblib import Memory

def run_nested_cv_regression(X, y, groups, model_obj, param_grid, cv_settings, original_df_metadata=None):
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(groups, (np.ndarray, list)):
        groups = pd.Series(groups)

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "default")
    cache_dir = f"./cache_task_{task_id}"
    memory = Memory(location=cache_dir, verbose=0)

    refined_grid = {f"model__{k}": v for k, v in param_grid.items()}
    pca_options = cv_settings.get("pca_components")

    if pca_options is not None:
        if not isinstance(pca_options, list):
            pca_options = [pca_options]
        refined_grid["pca__n_components"] = pca_options
        pca_step = ("pca", PCA(random_state=42))
    else:
        pca_step = ("pca", "passthrough")

    outer_cv = GroupKFold(n_splits=cv_settings.get("outer", 5))

    results = []
    all_logits_list = []

    scaler_step = ColumnTransformer(
        [("num", StandardScaler(), make_column_selector(dtype_include=np.number))],
        remainder="passthrough"
    ) if cv_settings.get("scaling") == "standard" else "passthrough"

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        groups_train_outer = groups.iloc[train_idx]

        pipeline = Pipeline(
            [
                ("scaler", scaler_step),
                pca_step,
                ("model", model_obj)
            ],
            memory=memory
        )

        inner_cv = GroupKFold(n_splits=cv_settings.get("inner", 3))

        scoring_metrics = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error"
        }

        grid = GridSearchCV(
            pipeline,
            refined_grid,
            cv=inner_cv,
            scoring=scoring_metrics,
            n_jobs=-1,
            refit="r2",
            return_train_score=True
        )

        try:
            grid.fit(X_train_outer, y_train_outer, groups=groups_train_outer)
            cv_res = grid.cv_results_
            idx = grid.best_index_

            param_pca_vals = np.array(cv_res["param_pca__n_components"])
            no_pca_indices = np.where(param_pca_vals == None)[0]

            if len(no_pca_indices) > 0:
                best_no_pca_r2 = np.max(cv_res["mean_test_r2"][no_pca_indices])
                pca_gain = grid.best_score_ - best_no_pca_r2
            else:
                pca_gain = 0

            train_preds = grid.best_estimator_.predict(X_train_outer)
            test_preds = grid.best_estimator_.predict(X_test_outer)

            inner_val_r2 = cv_res["mean_test_r2"][idx]
            inner_train_r2 = cv_res["mean_train_r2"][idx]

            outer_train_r2 = r2_score(y_train_outer, train_preds)
            outer_train_mae = mean_absolute_error(y_train_outer, train_preds)

            outer_test_r2 = r2_score(y_test_outer, test_preds)
            outer_test_mae = mean_absolute_error(y_test_outer, test_preds)

            optimism_bias_r2 = inner_val_r2 - outer_test_r2
            gen_gap_r2 = outer_train_r2 - outer_test_r2

            r_val, _ = pearsonr(y_test_outer, test_preds)

            if hasattr(grid.best_estimator_, "predict_proba"):
                probs = grid.best_estimator_.predict_proba(X_test_outer)
                fold_logits = pd.DataFrame(
                    probs,
                    index=X_test_outer.index,
                    columns=[f"prob_class_{c}" for c in grid.classes_]
                )

                if original_df_metadata is not None:
                    meta = original_df_metadata.iloc[test_idx]
                    for col in meta.columns:
                        fold_logits[col] = meta[col].values

                fold_logits["fold"] = fold_idx + 1
                all_logits_list.append(fold_logits)

            res_dict = {
                "fold": fold_idx + 1,
                "outer_test_r2": outer_test_r2,
                "outer_test_pearson_r": r_val,
                "outer_test_mae": outer_test_mae,
                "inner_train_r2": inner_train_r2,
                "inner_val_r2": inner_val_r2,
                "outer_train_r2": outer_train_r2,
                "outer_train_mae": outer_train_mae,
                "optimism_bias_r2": optimism_bias_r2,
                "gen_gap_r2": gen_gap_r2,
                "pca_gain": pca_gain,
                "selected_pca": str(grid.best_params_.get("pca__n_components", "None")),
                "best_params": str(grid.best_params_)
            }

            results.append(res_dict)

        except Exception as e:
            print(f"CRITICAL ERROR in Fold {fold_idx + 1}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    numeric_metrics = [
        "outer_test_r2",
        "outer_test_pearson_r",
        "outer_test_mae",
        "inner_train_r2",
        "inner_val_r2",
        "outer_train_r2",
        "outer_train_mae",
        "optimism_bias_r2",
        "gen_gap_r2",
        "pca_gain"
    ]

    string_metadata = ["selected_pca", "best_params"]

    summary_mean = res_df[numeric_metrics].mean()
    summary_std = res_df[numeric_metrics].std()

    mean_row = summary_mean.to_frame().T
    mean_row["fold"] = "Mean"

    std_row = summary_std.to_frame().T
    std_row["fold"] = "Std"

    for col in string_metadata:
        mean_row[col] = "N/A"
        std_row[col] = "N/A"

    final_df = pd.concat([res_df, mean_row, std_row], ignore_index=True)

    print("\n--- Final Regression Results (Mean) ---")
    print(summary_mean)

    full_logits_df = pd.concat(all_logits_list).sort_index() if all_logits_list else pd.DataFrame()

    return final_df, full_logits_df