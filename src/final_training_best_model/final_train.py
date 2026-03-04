import os
import ast
import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from src.data.manual_transcriptions.comparison import manuel_transcription_df_standardisation

FEAT_FILE_MAP = {
    "V-Qual": "vq_egmaps",
    "Prosody": "egmaps",
    "eGeMAPS All": "egmaps_all",
    "W2V2": "w2v2",
    "HuBERT": "hubert",
    "BERT": "bert_embeddings",
    "Handcrafted": "ling_handcrafted"
}

TARGET_COLUMN_MAP = {
    "PF": "phonemfl",
    "VF": "semanfl",
    "RW": "wl_erkennen_ja",
    "RL": "wl_abruf",
    "BNT": "boston",
    "MMSE": "mmst",
    "MEM": "domain_memory_serli",
    "LAN": "domain_language_serli",
    "EXE": "domain_executive_serli",
    "VIS": "domain_visuospatial_serli",
    "CERAD": "cerad_total_score_chandler",
    "MCI Binary": "binary_mci",
    "CERAD Binary": "cerad_total_score_chandler_binary_clinical",
    "MMSE Binary": "mmst_clinical_binary"
}

def get_model_instance(model_type, is_regression=False):
    m = str(model_type).strip()
    if m == "SVR":
        return SVR()
    if m == "SVM":
        return SVC(cache_size=2000, probability=True)
    if m == "LogReg":
        return LogisticRegression(max_iter=1000)
    if "Ridge" in m or m == "Ri":
        return Ridge()
    if m in ["XG", "XGBoost", "XGBClassifier", "XGBRegressor"]:
        if is_regression:
            return XGBRegressor()
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    raise ValueError(f"Unknown model type: {m}")

def resolve_paths(task_long, feature_label):
    feat_snippet = FEAT_FILE_MAP.get(feature_label, feature_label.lower().replace(" ", "_"))
    base_dir = f"data2/cache/{task_long}"
    train_path = os.path.join(base_dir, f"summary_{feat_snippet}.parquet")
    ho_path = os.path.join(base_dir, f"summary_{task_long}_HO__{feat_snippet}.parquet")
    return train_path, ho_path

def add_key(df):
    col = "source_filename" if "source_filename" in df.columns else "filename"
    def extract_strat(fp):
        base = os.path.basename(fp)
        parts = base.split("_")
        return f"{parts[1]}_{parts[0]}"
    df["strat_key"] = df[col].apply(extract_strat)
    return df

def parse_params(params):
    if isinstance(params, str):
        params = ast.literal_eval(params.strip())
    if isinstance(params, str):
        params = ast.literal_eval(params)
    return params

def run_bakery():
    with open("config/mci_winner.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    df_exp_meta = pd.read_csv("data2/splits/train_cross_all_with_domain.csv")
    df_ho_meta = pd.read_csv("data2/splits/holdout_cross_all_with_domain.csv")

    df_exp_meta = manuel_transcription_df_standardisation(df_exp_meta)
    df_ho_meta = manuel_transcription_df_standardisation(df_ho_meta)

    results_dir = "results/final_holdout"
    logit_dir = "results/final_logits"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logit_dir, exist_ok=True)

    final_metrics = []

    for champ in cfg["champions"]:
        task_long = champ["task"]
        feat_label = champ["feature"]
        yaml_target = champ["target"]
        csv_target = TARGET_COLUMN_MAP.get(yaml_target, yaml_target)

        print(f"\nRunning: {champ['id']} | Target: {yaml_target} | Task: {task_long}")

        try:
            train_p, ho_p = resolve_paths(task_long, feat_label)

            if not os.path.exists(train_p) or not os.path.exists(ho_p):
                print(f"FILE MISSING: {train_p} or {ho_p}")
                continue

            df_feat_train = pd.read_parquet(train_p)
            df_feat_ho = pd.read_parquet(ho_p)

            is_class = any(x in yaml_target.lower() or x in csv_target.lower() for x in ["binary", "mci"])
            is_reg = not is_class

            df_feat_train = add_key(df_feat_train)
            df_feat_ho = add_key(df_feat_ho)

            train_merged = pd.merge(
                df_exp_meta[["strat_key", "id", csv_target]],
                df_feat_train,
                on="strat_key"
            )

            ho_merged = pd.merge(
                df_ho_meta[["strat_key", "id", csv_target]],
                df_feat_ho,
                on="strat_key"
            )

            if ho_merged.empty:
                print(f"Holdout merge empty for {champ['id']}")
                continue

            meta_cols = ["strat_key", csv_target, "filename", "source_filename", "id", "Unnamed: 0"]
            X_cols = [c for c in train_merged.columns if c not in meta_cols]

            X_train = train_merged[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            y_train = train_merged[csv_target]

            X_test = ho_merged.reindex(columns=X_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0)
            y_test = ho_merged[csv_target]

            params = parse_params(champ["params"])

            model_obj = get_model_instance(champ["model_type"], is_regression=is_reg)

            pca_val = params.get("pca__n_components")
            use_pca = pca_val is not None and str(pca_val).lower() != "none"

            pca_step = ("pca", PCA(n_components=pca_val, random_state=42)) if use_pca else ("pca", "passthrough")

            pipeline_obj = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    pca_step,
                    ("model", model_obj)
                ]
            )

            local_params = {k: v for k, v in params.items() if k not in ["scaler", "pca", "model"]}

            if not use_pca:
                local_params = {k: v for k, v in local_params.items() if not k.startswith("pca__")}

            pipeline_obj.set_params(**local_params)

            pipeline_obj.fit(X_train, y_train)

            res = {
                "id": champ["id"],
                "target": yaml_target
            }

            pred_df = pd.DataFrame({
                "id": ho_merged["id"].values,
                "strat_key": ho_merged["strat_key"].values,
                "y_true": y_test.values
            })

            if hasattr(pipeline_obj.named_steps["model"], "coef_"):
                coef = pipeline_obj.named_steps["model"].coef_
                if coef.ndim > 1:
                    coef = coef[0]

                importance_df = pd.DataFrame({
                    "feature": X_cols,
                    "coefficient": coef,
                    "abs_coefficient": np.abs(coef)
                }).sort_values("abs_coefficient", ascending=False)

                importance_path = f"{logit_dir}/{champ['id']}_coefficients.parquet"
                importance_df.to_parquet(importance_path, index=False)

            if is_class:
                probs = pipeline_obj.predict_proba(X_test)[:, 1]
                hard_preds = pipeline_obj.predict(X_test)

                pred_df["y_prob"] = probs
                pred_df["y_pred"] = hard_preds

                res["holdout_auc"] = roc_auc_score(y_test, probs)
                res["holdout_bal_acc"] = balanced_accuracy_score(y_test, hard_preds)

            else:
                preds = pipeline_obj.predict(X_test)
                pred_df["y_pred"] = preds

                r, _ = pearsonr(y_test, preds)

                res["holdout_r"] = r
                res["holdout_r2"] = r2_score(y_test, preds)

            save_path = f"{logit_dir}/{champ['id']}_preds.parquet"
            pred_df.to_parquet(save_path, index=False)

            final_metrics.append(res)

            if is_class:
                print(f"Success! AUC: {res['holdout_auc']:.4f}")
            else:
                print(f"Success! R: {res['holdout_r']:.4f}")

        except Exception as e:
            print(f"FAILED {champ['id']}: {e}")

    summary_df = pd.DataFrame(final_metrics)
    summary_df.to_csv(f"{results_dir}/holdout_results_summary.csv", index=False)

    print(f"\nDone! Summary saved to {results_dir}/holdout_results_summary.csv")

if __name__ == "__main__":
    run_bakery()