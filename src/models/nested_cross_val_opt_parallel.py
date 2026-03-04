import os
import gc
import yaml
import argparse
import pandas as pd
import itertools
from src.cross_validation.cv_engine_extended_logging_logits import run_nested_cv 
from src.data.loading.data_handler import FeatureManager
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.data.standardisation.comparison import manuel_transcription_df_standardisation
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", message="penalty is deprecated. Please use l1_ratio only.")
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")

# Optional: also ignore convergence warnings if the model doesn't converge in 5000 steps
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--exp_id", type=str, required=True, help="Unique string to group array results")
    args = parser.parse_args()


    main_cfg = load_yaml(args.config)
    folder_name = f"{args.exp_id}_{main_cfg['experiment_name']}"
    exp_dir = os.path.join(main_cfg['paths']['results_dir'], folder_name)
    logits_dir = os.path.join(exp_dir, "logits")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(logits_dir, exist_ok=True)

    combinations = list(itertools.product(
        main_cfg['testnames'],
        main_cfg['feature_configs'], 
        main_cfg['targets'],
        main_cfg['model_configs']  
    ))

    total_combs = len(combinations)
    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    my_tasks = [combinations[i] for i in range(total_combs) if i % num_tasks == task_id]

    print(f"Node {task_id} assigned {len(my_tasks)} experiments out of {total_combs} total.")
    for testname, feat_path, target_col, model_path in my_tasks:
        feat_cfg = load_yaml(feat_path)
        model_cfg = load_yaml(model_path)

        res_name = f"{testname}_{target_col}_{feat_cfg['name']}_{model_cfg['name']}.csv"
        res_path = os.path.join(exp_dir, res_name)

        if os.path.exists(res_path):
            print(f"Skipping: {res_name} already exists.")
            continue



        print(f"--- STARTING TASK {task_id} ---")
        print(f"Test: {testname} | Target: {target_col}")
        print(f"Feature: {feat_cfg['name']} | Model: {model_cfg['name']}")
        
        frozen_df = pd.read_csv(main_cfg['paths']['frozen_split'])
        if "strat_key" not in frozen_df.columns:
            frozen_df = manuel_transcription_df_standardisation(frozen_df)
        if frozen_df[target_col].nunique() < 2:
            print(f"Skipping {target_col}: Not enough classes in frozen_df.")
            continue

        manager = FeatureManager(
            metadata_path=main_cfg['paths']['metadata_path'],
            cache_dir=main_cfg['paths']['cache_dir'], 
            testname=testname
        )

        df_feat = manager.get_features(feat_cfg)
        if feat_cfg["name"] == "demographics":
            frozen_df = manuel_transcription_df_standardisation(frozen_df)
            if "strat_key" not in df_feat.columns:
                df_feat =  manuel_transcription_df_standardisation(df_feat)
                df_feat = df_feat.drop(columns = "id")
            drop_columns = feat_cfg['drop_columns']
        else:
            df_feat["strat_key"] = df_feat["file_id"].str.split("_").str[1]+ "_" + df_feat["file_id"].str.split("_").str[0]
            drop_columns = feat_cfg['drop_columns']


        
        cols_to_convert = [c for c in df_feat.columns if c not in drop_columns and c != 'strat_key']
        df_feat[cols_to_convert] = df_feat[cols_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0)

        print(f"Shape of frozen_df: {frozen_df.shape}")
        print(f"Shape of df_feat: {df_feat.shape}")


        merged_df = pd.merge(
            frozen_df[['strat_key', 'id', target_col]], 
            df_feat, 
            on="strat_key", 
            how="inner"
        )
        print(f" Shape of merged_df: {merged_df.shape}")

        print(f"Sample strat_key from frozen_df: {frozen_df['strat_key'].iloc[0]} (Type: {type(frozen_df['strat_key'].iloc[0])})")
        print(f"Sample strat_key from df_feat:   {df_feat['strat_key'].iloc[0]} (Type: {type(df_feat['strat_key'].iloc[0])})")

        common_keys = set(frozen_df['strat_key']).intersection(set(df_feat['strat_key']))
        print(f"Number of common keys: {len(common_keys)}")
        
        X = merged_df[cols_to_convert]
        y = merged_df[target_col]
        groups = merged_df['id']

        if model_cfg['name'] == "LogReg":
            model_obj = LogisticRegression(max_iter=1000, random_state=42)
        elif model_cfg['name'] == "SVM":
            model_obj = SVC(cache_size=2000, random_state=42, probability=True)
        elif model_cfg['name'] == "XGBoost":
            tree_method = 'gpu_hist' if main_cfg.get('use_gpu', False) else 'auto'
            model_obj = XGBClassifier(tree_method=tree_method, random_state=42, n_jobs=-1)


        try:
            cv_results_df, logits_df = run_nested_cv(
                X, y, groups, 
                model_obj, 
                model_cfg['params'], 
                {
                    'outer': main_cfg.get('outer', 5), 
                    'inner': main_cfg.get('inner', 3),
                    'scaling': main_cfg.get('scaling'),
                    'pca_components': main_cfg.get('pca_components', [])
                }, original_df_metadata=merged_df[['strat_key']]
            )

            if cv_results_df is not None and not cv_results_df.empty:
                res_name = f"{testname}_{target_col}_{feat_cfg['name']}_{model_cfg['name']}.csv"
                res_path = os.path.join(exp_dir, res_name)
                cv_results_df.to_csv(res_path, index=False)
                print(f"Saved: {res_path}")

            if logits_df is not None and not logits_df.empty:
                logits_name = f"{testname}_{target_col}_{feat_cfg['name']}_{model_cfg['name']}_logits.csv"
                logits_path = os.path.join(logits_dir, logits_name)
                logits_df.to_csv(logits_path, index=False)
                print(f"Saved logits: {logits_path}")

        
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    main()