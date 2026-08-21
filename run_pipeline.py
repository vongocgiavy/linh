import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.data_cleaner import load_and_preprocess_dataset
from src.visualization.dimension_reduction import run_dimension_reduction
from src.classification.evaluate_all import compare_all_classification_models
from src.clustering.kmeans_dbscan import run_clustering_and_evaluation

def main():
    print("=" * 80)
    print(" CHẠY TOÀN BỘ PIPELINE HỌC MÁY THEO CẤU TRÚC THƯ MỤC MODULAR")
    print("=" * 80)

    # 1. Preprocessing & Data Info
    df = load_and_preprocess_dataset()

    # 2. Dimensionality Reduction 2D
    X_cont, X_scaled, y, X_pca = run_dimension_reduction()

    # 3. Classification Models (Random Forest, AdaBoost, SVM) & 10-Fold CV
    summary_df = compare_all_classification_models(n_splits=10)

    # 4. Clustering (K-Means & DBSCAN) & Ground Truth Evaluation
    clustering_df, km_mat, db_mat = run_clustering_and_evaluation()

    print("\n" + "=" * 80)
    print(" ĐÃ HOÀN THÀNH TẤT CẢ CÁC BƯỚC THÀNH CÔNG!")
    print("=" * 80)

if __name__ == "__main__":
    main()
