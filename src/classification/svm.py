import os
import sys
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED = 42

def _eval_single_fold(model, X_train, y_train, X_val, y_val):
    """
    Hàm phụ trợ chạy 1 fold độc lập để xử lý song song đa luồng (Parallel).
    """
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

def train_and_eval_svm(
    data_path=os.path.join(PROJECT_ROOT, "data", "filtered_processed_games.csv"),
    n_splits=10,
    n_jobs=-1,
    cache_size=1000,
    max_iter=10000
):
    """
    Huấn luyện và đánh giá SVM RBF Kernel với tối ưu hóa hiệu năng cao:
    - n_jobs=-1: Chạy song song tất cả các Fold trên toàn bộ nhân CPU.
    - cache_size=1000: Mở rộng bộ nhớ đệm RAM lên 1000 MB để tính toán Kernel ma trận siêu tốc.
    - max_iter=5000: Giới hạn số vòng lặp tối đa để chống nghẽn / treo tiến trình.
    """
    print("=" * 70)
    print(" [03_CLASSIFICATION] SVM RBF KERNEL CLASSIFIER (10-FOLD CV - TỐI ƯU TỐC ĐỘ)")
    print("=" * 70)

    if not os.path.exists(data_path):
        from src.preprocessing.data_cleaner import load_and_preprocess_dataset
        df = load_and_preprocess_dataset(output_clean_csv=data_path)
    else:
        df = pd.read_csv(data_path)

    feature_cols = ["white_rating", "black_rating", "rating_diff", "rated", "opening_ply"]
    X = df[feature_cols].copy()
    y = df["ResultEncoded"].values

    # Cấu hình mô hình SVM tối ưu tốc độ và bộ nhớ
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=1.5,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            cache_size=cache_size,
            max_iter=max_iter,
            random_state=SEED
        ))
    ])

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Chạy song song các Folds với Parallel n_jobs
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_single_fold)(
            svm_pipe,
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

    print(f"[+] SVM RBF Kernel ({n_splits} Folds CV, n_jobs={n_jobs}, cache_size={cache_size}MB, max_iter={max_iter}):")
    print(f"   - Macro F1-Score   : {np.mean(macro_f1_list)*100:.2f}% ± {np.std(macro_f1_list)*100:.2f}%")
    print(f"   - Weighted F1-Score: {np.mean(weighted_f1_list)*100:.2f}% ± {np.std(weighted_f1_list)*100:.2f}%")
    print(f"   - Accuracy         : {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list)*100:.2f}%")
    print(f"   - F1 (0: Black win): {np.mean(f1_black_list)*100:.2f}%")
    print(f"   - F1 (1: Draw)     : {np.mean(f1_draw_list)*100:.2f}%")
    print(f"   - F1 (2: White win): {np.mean(f1_white_list)*100:.2f}%")

    # Fit trên toàn bộ tập dữ liệu và lưu vào models/ folder
    svm_pipe.fit(X, y)
    model_save_path = os.path.join(PROJECT_ROOT, "models", "svm_rbf.joblib")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(svm_pipe, model_save_path)
    print(f"[+] Đã lưu file mô hình Scikit-Learn tại: {model_save_path}")

    return svm_pipe, {
        "model_name": "SVM (RBF Kernel)",
        "macro_f1": np.mean(macro_f1_list),
        "macro_f1_std": np.std(macro_f1_list),
        "weighted_f1": np.mean(weighted_f1_list),
        "accuracy": np.mean(acc_list),
        "f1_black": np.mean(f1_black_list),
        "f1_draw": np.mean(f1_draw_list),
        "f1_white": np.mean(f1_white_list)
    }

if __name__ == "__main__":
    train_and_eval_svm()
