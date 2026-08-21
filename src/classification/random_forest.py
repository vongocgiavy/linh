import os
import sys
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED = 42

def _eval_single_fold_rf(model, X_train, y_train, X_val, y_val):
    fold_model = clone(model)
    fold_model.fit(X_train, y_train)
    y_pred = fold_model.predict(X_val)

    m_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    w_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_val, y_pred)
    per_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)

    f1_black = per_f1[0]
    f1_draw = per_f1[1] if len(per_f1) > 1 else 0.0
    f1_white = per_f1[2] if len(per_f1) > 2 else 0.0

    return m_f1, w_f1, acc, f1_black, f1_draw, f1_white

def train_and_eval_random_forest(
    data_path=os.path.join(PROJECT_ROOT, "data", "filtered_processed_games.csv"),
    n_splits=10,
    n_jobs=-1
):
    print("=" * 70)
    print(" [03_CLASSIFICATION] RANDOM FOREST CLASSIFIER (10-FOLD CV - TỐI ƯU SONG SONG)")
    print("=" * 70)

    if not os.path.exists(data_path):
        from src.preprocessing.data_cleaner import load_and_preprocess_dataset
        df = load_and_preprocess_dataset(output_clean_csv=data_path)
    else:
        df = pd.read_csv(data_path)

    feature_cols = ["white_rating", "black_rating", "rating_diff", "rated", "opening_ply"]
    X = df[feature_cols].copy()
    y = df["ResultEncoded"].values

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1
    )

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_single_fold_rf)(
            rf,
            X.iloc[train_idx], y[train_idx],
            X.iloc[val_idx], y[val_idx]
        )
        for train_idx, val_idx in kfold.split(X, y)
    )

    macro_f1_list = [res[0] for res in fold_results]
    weighted_f1_list = [res[1] for res in fold_results]
    acc_list = [res[2] for res in fold_results]
    f1_black_list = [res[3] for res in fold_results]
    f1_draw_list = [res[4] for res in fold_results]
    f1_white_list = [res[5] for res in fold_results]

    print(f"[+] Random Forest ({n_splits} Folds CV, n_jobs={n_jobs}):")
    print(f"   - Macro F1-Score   : {np.mean(macro_f1_list)*100:.2f}% ± {np.std(macro_f1_list)*100:.2f}%")
    print(f"   - Weighted F1-Score: {np.mean(weighted_f1_list)*100:.2f}% ± {np.std(weighted_f1_list)*100:.2f}%")
    print(f"   - Accuracy         : {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list)*100:.2f}%")
    print(f"   - F1 (0: Black win): {np.mean(f1_black_list)*100:.2f}%")
    print(f"   - F1 (1: Draw)     : {np.mean(f1_draw_list)*100:.2f}%")
    print(f"   - F1 (2: White win): {np.mean(f1_white_list)*100:.2f}%")

    # Fit on full dataset and save model to models/ folder
    rf.n_jobs = -1
    rf.fit(X, y)
    model_save_path = os.path.join(PROJECT_ROOT, "models", "random_forest.joblib")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf, model_save_path)
    print(f"[+] Đã lưu file mô hình Scikit-Learn tại: {model_save_path}")

    return rf, {
        "model_name": "Random Forest",
        "macro_f1": np.mean(macro_f1_list),
        "macro_f1_std": np.std(macro_f1_list),
        "weighted_f1": np.mean(weighted_f1_list),
        "accuracy": np.mean(acc_list),
        "f1_black": np.mean(f1_black_list),
        "f1_draw": np.mean(f1_draw_list),
        "f1_white": np.mean(f1_white_list)
    }

if __name__ == "__main__":
    train_and_eval_random_forest()
