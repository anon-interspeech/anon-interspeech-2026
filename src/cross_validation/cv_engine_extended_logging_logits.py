import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
import gc
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from joblib import Memory 


def run_nested_cv(X, y, groups, model_obj, param_grid, cv_settings,original_df_metadata=None):

    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'default')
    cache_dir = f"./cache_task_{task_id}"
    memory = Memory(location=cache_dir, verbose=0)

    if isinstance(y, np.ndarray): y = pd.Series(y)
    if isinstance(groups, (np.ndarray, list)): groups = pd.Series(groups)
    
    refined_grid = {f"model__{k}": v for k, v in param_grid.items()}
    pca_options = cv_settings.get('pca_components')
    if pca_options is not None:
        if not isinstance(pca_options, list): 
            pca_options = [pca_options]
        refined_grid["pca__n_components"] = pca_options
        pca_step = ('pca', PCA(random_state=42))
    else:
        pca_step = ('pca', 'passthrough')

    outer_cv = StratifiedGroupKFold(
        n_splits=cv_settings.get('outer', 5), 
        shuffle=True, 
        random_state=42
    )
    
    
    scaler_step = ColumnTransformer([
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number))
    ], remainder='passthrough') if cv_settings.get("scaling") == "standard" else "passthrough"

    results = []
    all_logits_list = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        groups_train_outer = groups.iloc[train_idx]


        pipeline = Pipeline([
            ('scaler', scaler_step),
            pca_step,
            ('model', model_obj)
        ], memory=memory)

        inner_cv = StratifiedGroupKFold(
            n_splits=cv_settings.get('inner', 3), 
            shuffle=True, 
            random_state=42 + fold_idx
        )

        scoring_metrics = {
            'f1_macro': 'f1_macro',
            'balanced_accuracy': 'balanced_accuracy',
            'roc_auc': 'roc_auc'
        }

        grid = GridSearchCV(
            pipeline, refined_grid, cv=inner_cv, 
            scoring=scoring_metrics, n_jobs=-1,
            refit='balanced_accuracy',
            return_train_score=True
        )
        
        try:
            grid.fit(X_train_outer, y_train_outer, groups=groups_train_outer)
            cv_res = pd.DataFrame(grid.cv_results_)
            best_idx = grid.best_index_

            no_pca_mask = cv_res['param_pca__n_components'].isnull()
            if no_pca_mask.any():
                best_no_pca_score = cv_res[no_pca_mask]['mean_test_balanced_accuracy'].max()
                pca_gain = grid.best_score_ - best_no_pca_score
            else:
                pca_gain = 0
            

            inner_val_f1 = grid.cv_results_['mean_test_f1_macro'][best_idx]
            inner_train_f1 = grid.cv_results_['mean_train_f1_macro'][best_idx]
            inner_val_bal_acc = grid.cv_results_['mean_test_balanced_accuracy'][best_idx]
            inner_train_bal_acc = grid.cv_results_['mean_train_balanced_accuracy'][best_idx]
            
 
            train_preds = grid.best_estimator_.predict(X_train_outer)
            outer_train_f1 = f1_score(y_train_outer, train_preds, average='macro')
            outer_train_bal_acc = balanced_accuracy_score(y_train_outer, train_preds)

            test_preds = grid.best_estimator_.predict(X_test_outer)
            outer_test_f1 = f1_score(y_test_outer, test_preds, average='macro')
            outer_test_bal_acc = balanced_accuracy_score(y_test_outer, test_preds)

            opt_bias_f1 = inner_val_f1 - outer_test_f1
            gen_gap_f1 = outer_train_f1 - outer_test_f1

        
            opt_bias_bal_acc = inner_val_bal_acc - outer_test_bal_acc
            gen_gap_bal_acc = outer_train_bal_acc - outer_test_bal_acc

      
            if hasattr(grid.best_estimator_, "predict_proba"):
                probs = grid.best_estimator_.predict_proba(X_test_outer)
                fold_logits = pd.DataFrame(
                    probs, 
                    index=X_test_outer.index, # Keep original indices
                    columns=[f"prob_class_{c}" for c in grid.classes_]
                )
           
                if original_df_metadata is not None:
                    meta = original_df_metadata.iloc[test_idx]
                    for col in meta.columns:
                        fold_logits[col] = meta[col].values

                fold_logits['fold'] = fold_idx + 1
                all_logits_list.append(fold_logits)
                auc = roc_auc_score(y_test_outer, probs[:, 1]) if probs.shape[1] == 2 \
                      else roc_auc_score(y_test_outer, probs, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test_outer, test_preds, multi_class='ovr')

            res_dict = {
                'fold': fold_idx + 1,
                # Main Scores
                'outer_test_f1': outer_test_f1,
                'outer_test_bal_acc': outer_test_bal_acc,
                'outer_test_auc': auc,
                
                # F1 Diagnostics
                'inner_val_f1': inner_val_f1,
                'outer_train_f1': outer_train_f1,
                'opt_bias_f1': opt_bias_f1,
                'gen_gap_f1': gen_gap_f1,

                #inner
                'inner_train_f1': inner_train_f1,
                'inner_train_bal_acc': inner_train_bal_acc,
                
                # Bal Acc Diagnostics
                'inner_val_bal_acc': inner_val_bal_acc,
                'outer_train_bal_acc': outer_train_bal_acc,
                'opt_bias_bal_acc': opt_bias_bal_acc,
                'gen_gap_bal_acc': gen_gap_bal_acc,
                
                # Insights
                'pca_gain': pca_gain,
                'selected_pca': grid.best_params_.get('pca__n_components', 'None'),
                'best_params': str(grid.best_params_)
            }
            results.append(res_dict)
            print(f"Completed Fold {fold_idx+1}: Test F1={outer_test_f1:.4f}, Test Bal Acc={outer_test_bal_acc:.4f}, AUC={auc:.4f}")

        except Exception as e:
            print(f"CRITICAL ERROR in Fold {fold_idx+1}: {e}")
            continue
         
    if not results: return pd.DataFrame()
    
    res_df = pd.DataFrame(results)

    metric_cols = [
        'inner_train_f1', 'inner_val_f1', 'outer_train_f1', 'outer_test_f1',
        'opt_bias_f1', 'gen_gap_f1', 
        'inner_train_bal_acc', 'inner_val_bal_acc', 'outer_train_bal_acc', 'outer_test_bal_acc',
        'opt_bias_bal_acc', 'gen_gap_bal_acc', 
        'outer_test_auc', 'pca_gain'
    ]
    
    summary_mean = res_df[metric_cols].mean()
    summary_std = res_df[metric_cols].std()
    mean_row = summary_mean.to_frame().T
    mean_row['fold'] = 'Mean'
    std_row = summary_std.to_frame().T
    std_row['fold'] = 'Std'

    final_df = pd.concat([res_df, mean_row, std_row], ignore_index=True)
    full_logits_df = pd.concat(all_logits_list).sort_index() if all_logits_list else pd.DataFrame()
    return final_df, full_logits_df
