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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    v_measure_score, homogeneity_score, completeness_score,
    fowlkes_mallows_score, silhouette_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED = 42

def run_clustering_and_evaluation(
    data_path=os.path.join(PROJECT_ROOT, "data", "filtered_processed_games.csv"),
    output_plot=os.path.join(PROJECT_ROOT, "outputs", "clustering_comparison_2d.png")
):
    print("=" * 80)
    print(" [04_CLUSTERING] GOM CỤM K-MEANS & DBSCAN (LOẠI TRỪ NHÃN) & ĐÁNH GIÁ GROUND TRUTH")
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

    # 1. K-Means
    print("[*] Huấn luyện K-Means (k=3)...")
    kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=SEED)
    km_labels = kmeans.fit_predict(X_scaled)

    # 2. DBSCAN
    print("[*] Huấn luyện DBSCAN (eps=0.8, min_samples=20)...")
    dbscan = DBSCAN(eps=0.8, min_samples=20)
    db_labels = dbscan.fit_predict(X_scaled)

    def evaluate_cluster_labels(cluster_labels, true_labels, name):
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        v_m = v_measure_score(true_labels, cluster_labels)
        homo = homogeneity_score(true_labels, cluster_labels)
        comp = completeness_score(true_labels, cluster_labels)
        fmi = fowlkes_mallows_score(true_labels, cluster_labels)
        sil = silhouette_score(X_scaled, cluster_labels, sample_size=min(5000, len(X_scaled)), random_state=SEED) if len(set(cluster_labels)) > 1 else 0

        contingency = pd.crosstab(pd.Series(cluster_labels, name="Cluster"), pd.Series(true_labels, name="Ground Truth"))
        purity = contingency.max(axis=1).sum() / len(true_labels)

        return {
            "Thuật Toán": name,
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "V-Measure": round(v_m, 4),
            "Homogeneity": round(homo, 4),
            "Completeness": round(comp, 4),
            "FMI": round(fmi, 4),
            "Purity": f"{purity*100:.2f}%",
            "Silhouette": round(sil, 4)
        }, contingency

    km_res, km_mat = evaluate_cluster_labels(km_labels, y, "K-Means (k=3)")
    db_res, db_mat = evaluate_cluster_labels(db_labels, y, "DBSCAN (eps=0.8)")

    res_df = pd.DataFrame([km_res, db_res])
    print("\n--- Bảng Đánh Giá Kết Quả Gom Cụm Với Ground Truth ---")
    print(res_df.to_string(index=False))

    print("\n--- Ma Trận Đối Sánh K-Means (k=3) vs Ground Truth ---")
    print("0: Black win | 1: Draw | 2: White win")
    print(km_mat)

    print("\n--- Ma Trận Đối Sánh DBSCAN vs Ground Truth (-1 là Noise) ---")
    print(db_mat)

    # Visualization
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Ground Truth
    palette = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    label_names = {0: "0 (Black win)", 1: "1 (Draw)", 2: "2 (White win)"}
    for lbl, color in palette.items():
        mask = (y == lbl)
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label_names[lbl], alpha=0.45, s=20)
    axes[0].set_title("Ground Truth Labels (Nhãn thực tế)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")

    # KMeans
    km_colors = ['#3498db', '#e67e22', '#9b59b6']
    for c_idx in range(3):
        mask = (km_labels == c_idx)
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=km_colors[c_idx], label=f"Cluster {c_idx}", alpha=0.45, s=20)
    axes[1].set_title("K-Means (k=3)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best")

    # DBSCAN
    unique_db_labels = set(db_labels)
    for idx, c_lbl in enumerate(unique_db_labels):
        mask = (db_labels == c_lbl)
        if c_lbl == -1:
            axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], c="gray", label="Noise (-1)", alpha=0.3, s=15, marker="x")
        else:
            axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c_lbl}", alpha=0.5, s=20)
    axes[2].set_title("DBSCAN Clustering", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("PCA 1")
    axes[2].set_ylabel("PCA 2")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Đã lưu biểu đồ gom cụm tại: {output_plot}")

    # Save fitted models to models/ folder
    import joblib
    km_save_path = os.path.join(PROJECT_ROOT, "models", "kmeans.joblib")
    db_save_path = os.path.join(PROJECT_ROOT, "models", "dbscan.joblib")
    os.makedirs(os.path.dirname(km_save_path), exist_ok=True)
    joblib.dump(kmeans, km_save_path)
    joblib.dump(dbscan, db_save_path)
    print(f"[+] Đã lưu file mô hình K-Means tại: {km_save_path}")
    print(f"[+] Đã lưu file mô hình DBSCAN tại: {db_save_path}")

    return res_df, km_mat, db_mat

if __name__ == "__main__":
    run_clustering_and_evaluation()
