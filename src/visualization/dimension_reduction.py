import os
import sys
import numpy as np
import pandas as pd
import matplotlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED = 42

def run_dimension_reduction(
    data_path=os.path.join(PROJECT_ROOT, "data", "filtered_processed_games.csv"),
    output_path=os.path.join(PROJECT_ROOT, "outputs", "dimension_reduction_2d.png")
):
    """
    Thu giảm số chiều dữ liệu liên tục về 2D bằng PCA và t-SNE, trực quan hóa với màu sắc theo nhãn.
    """
    print("=" * 80)
    print(" [02_VISUALIZATION] THU GIẢM SỐ CHIỀU & TRỰC QUAN HÓA 2D")
    print("=" * 80)

    if not os.path.exists(data_path):
        from src.preprocessing.data_cleaner import load_and_preprocess_dataset
        df = load_and_preprocess_dataset(output_clean_csv=data_path)
    else:
        df = pd.read_csv(data_path)

    continuous_cols = ["white_rating", "black_rating", "rating_diff", "opening_ply"]
    X_cont = df[continuous_cols].copy()
    y = df["ResultEncoded"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cont)

    # 1. PCA 2D
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    print(f"[+] PCA 2D: PC1={var_ratio[0]*100:.2f}%, PC2={var_ratio[1]*100:.2f}% (Tổng={sum(var_ratio)*100:.2f}%)")

    # 2. t-SNE 2D
    sample_size = min(3000, len(df))
    np.random.seed(SEED)
    sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    X_tsne_sample = tsne.fit_transform(X_scaled[sample_indices])
    y_sample = y[sample_indices]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    palette = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    label_names = {0: "0 (Black win)", 1: "1 (Draw)", 2: "2 (White win)"}

    # Subplot 1: PCA 2D
    for lbl, color in palette.items():
        mask = (y == lbl)
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label_names[lbl], alpha=0.45, s=25)
    axes[0].set_title(f"PCA 2D Projection (PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("PC1 (Rating Difference & Strength)")
    axes[0].set_ylabel("PC2 (Opening Depth)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(title="Kết quả trận đấu", loc="best")

    # Subplot 2: t-SNE 2D
    for lbl, color in palette.items():
        mask = (y_sample == lbl)
        axes[1].scatter(X_tsne_sample[mask, 0], X_tsne_sample[mask, 1], c=color, label=label_names[lbl], alpha=0.55, s=25)
    axes[1].set_title(f"t-SNE 2D Manifold (Sample n={sample_size})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dim 1")
    axes[1].set_ylabel("t-SNE Dim 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(title="Kết quả trận đấu", loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Đã lưu biểu đồ 2D tại: {output_path}")

    # Save fitted scaler and PCA to models/ folder
    import joblib
    scaler_save_path = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
    pca_save_path = os.path.join(PROJECT_ROOT, "models", "pca_2d.joblib")
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(pca, pca_save_path)
    print(f"[+] Đã lưu file mô hình StandardScaler tại: {scaler_save_path}")
    print(f"[+] Đã lưu file mô hình PCA tại: {pca_save_path}")

    return X_cont, X_scaled, y, X_pca

if __name__ == "__main__":
    run_dimension_reduction()
