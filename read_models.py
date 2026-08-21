import os
import sys
import joblib
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def inspect_saved_models():
    """
    Đọc và hiển thị chi tiết thông số, cấu trúc, trọng số đã học của toàn bộ các file .joblib trong thư mục models/
    Đồng thời chạy thử dự đoán kết quả ván cờ trên cả 3 mô hình (Random Forest, AdaBoost, SVM) và gom cụm K-Means.
    """
    print("=" * 80)
    print(" 🔍 KIỂM TRA & ĐỌC CHI TIẾT TOÀN BỘ CÁC FILE MODEL TRONG THƯ MỤC models/")
    print("=" * 80)

    if not os.path.exists(MODELS_DIR):
        print(f"[!] Thư mục '{MODELS_DIR}' chưa tồn tại. Hãy chạy 'py run_pipeline.py' trước để tạo model.")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    print(f"[+] Tìm thấy {len(model_files)} file model (.joblib) đã lưu:\n")

    # 1. StandardScaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("📌 [1] THƯ VIỆN CHUẨN HÓA (StandardScaler):")
        print(f"   - File: models/scaler.joblib")
        print(f"   - Các thuộc tính đầu vào: {list(scaler.feature_names_in_)}")
        print(f"   - Giá trị trung bình Mean (mu) : {np.round(scaler.mean_, 2)}")
        print(f"   - Độ lệch chuẩn Scale (sigma)  : {np.round(scaler.scale_, 2)}")
        print("-" * 80)

    # 2. PCA 2D
    pca_path = os.path.join(MODELS_DIR, "pca_2d.joblib")
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        print("📌 [2] MÔ HÌNH GIẢM CHIỀU (PCA 2D):")
        print(f"   - File: models/pca_2d.joblib")
        print(f"   - Số chiều n_components: {pca.n_components}")
        print(f"   - Tỷ lệ phương sai giải thích: PC1 = {pca.explained_variance_ratio_[0]*100:.2f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.2f}%")
        print(f"   - Tổng phương sai giải thích : {sum(pca.explained_variance_ratio_)*100:.2f}%")
        print("-" * 80)

    # 3. Random Forest
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        print("📌 [3] MÔ HÌNH PHÂN LỚP RANDOM FOREST:")
        print(f"   - File: models/random_forest.joblib")
        print(f"   - Số cây quyết định (n_estimators): {rf.n_estimators}")
        print(f"   - Độ sâu tối đa (max_depth): {rf.max_depth}")
        print(f"   - Các lớp kết quả (classes): {rf.classes_} (0: Black win, 1: Draw, 2: White win)")
        print(f"   - Độ quan trọng đặc trưng (Feature Importances):")
        feature_names = ["white_rating", "black_rating", "rating_diff", "rated", "opening_ply"]
        for feat, imp in zip(feature_names, rf.feature_importances_):
            print(f"     * {feat:<15}: {imp*100:.2f}%")
        print("-" * 80)

    # 4. AdaBoost
    ada_path = os.path.join(MODELS_DIR, "adaboost.joblib")
    if os.path.exists(ada_path):
        ada = joblib.load(ada_path)
        print("📌 [4] MÔ HÌNH PHÂN LỚP ADABOOST:")
        print(f"   - File: models/adaboost.joblib")
        print(f"   - Số lượng estimator cơ sở: {ada.n_estimators}")
        print(f"   - Tốc độ học (learning_rate): {ada.learning_rate}")
        print(f"   - Các lớp kết quả: {ada.classes_}")
        print("-" * 80)

    # 5. SVM RBF Kernel Pipeline
    svm_path = os.path.join(MODELS_DIR, "svm_rbf.joblib")
    if os.path.exists(svm_path):
        svm_pipe = joblib.load(svm_path)
        print("📌 [5] MÔ HÌNH PHÂN LỚP SVM (RBF KERNEL PIPELINE):")
        print(f"   - File: models/svm_rbf.joblib")
        print(f"   - Các bước trong Pipeline: {[step[0] for step in svm_pipe.steps]}")
        svc_model = svm_pipe.named_steps["svm"]
        print(f"   - Loại nhân (Kernel): {svc_model.kernel}")
        print(f"   - Hệ số phạt C: {svc_model.C}, Gamma: {svc_model.gamma}")
        print(f"   - Bộ nhớ đệm cache_size: {svc_model.cache_size} MB")
        print(f"   - Số lượng Support Vectors đã học: {len(svc_model.support_)} vectors")
        print("-" * 80)

    # 6. K-Means
    km_path = os.path.join(MODELS_DIR, "kmeans.joblib")
    if os.path.exists(km_path):
        km = joblib.load(km_path)
        print("📌 [6] MÔ HÌNH GOM CỤM K-MEANS:")
        print(f"   - File: models/kmeans.joblib")
        print(f"   - Số cụm n_clusters: {km.n_clusters}")
        print(f"   - Khởi tạo: {km.init}")
        print(f"   - Tọa độ tâm cụm (Cluster Centers shape): {km.cluster_centers_.shape}")
        print("-" * 80)

    # 7. DBSCAN
    db_path = os.path.join(MODELS_DIR, "dbscan.joblib")
    if os.path.exists(db_path):
        db = joblib.load(db_path)
        print("📌 [7] MÔ HÌNH GOM CỤM DBSCAN:")
        print(f"   - File: models/dbscan.joblib")
        print(f"   - Bán kính lân cận (eps): {db.eps}")
        print(f"   - Số điểm lân cận tối thiểu (min_samples): {db.min_samples}")
        print(f"   - Số điểm lõi (Core samples): {len(db.core_sample_indices_)}")
        print("=" * 80)

    # Demo inference trên cả 3 mô hình
    print("\n🎮 DEMO DỰ ĐOÁN KẾT QUẢ TRÊN CẢ 3 MÔ HÌNH VỚI MẪU VÁN CỜ THỰC TẾ:")
    feature_cols = ["white_rating", "black_rating", "rating_diff", "rated", "opening_ply"]
    sample_data = [[1900, 1650, 250, 1, 6]]  # Elo Trắng = 1900, Elo Đen = 1650, Chênh lệch +250
    sample_df = pd.DataFrame(sample_data, columns=feature_cols)
    label_names = {0: "Black win (Đen thắng)", 1: "Draw (Hòa)", 2: "White win (Trắng thắng)"}

    print(f"   • Dữ liệu đầu vào: Elo Trắng = 1900 | Elo Đen = 1650 | Chênh lệch = +250 | Đấu xếp hạng | Khai cuộc 6 plys")

    # 1. Random Forest
    if os.path.exists(rf_path):
        rf_pred = rf.predict(sample_df)[0]
        rf_prob = rf.predict_proba(sample_df)[0]
        print(f"   [1] Random Forest : Dự đoán = {label_names[rf_pred]:<25} (Xác suất: Đen={rf_prob[0]*100:.1f}%, Hòa={rf_prob[1]*100:.1f}%, Trắng={rf_prob[2]*100:.1f}%)")

    # 2. AdaBoost
    if os.path.exists(ada_path):
        ada_pred = ada.predict(sample_df)[0]
        ada_prob = ada.predict_proba(sample_df)[0]
        print(f"   [2] AdaBoost      : Dự đoán = {label_names[ada_pred]:<25} (Xác suất: Đen={ada_prob[0]*100:.1f}%, Hòa={ada_prob[1]*100:.1f}%, Trắng={ada_prob[2]*100:.1f}%)")

    # 3. SVM Pipeline
    if os.path.exists(svm_path):
        svm_pred = svm_pipe.predict(sample_df)[0]
        print(f"   [3] SVM RBF Kernel: Dự đoán = {label_names[svm_pred]:<25}")

    # 4. K-Means Clustering
    if os.path.exists(km_path) and os.path.exists(scaler_path):
        cont_cols = ["white_rating", "black_rating", "rating_diff", "opening_ply"]
        cont_df = pd.DataFrame([[1900, 1650, 250, 6]], columns=cont_cols)
        cont_scaled = scaler.transform(cont_df)
        km_cluster = km.predict(cont_scaled)[0]
        print(f"   [4] K-Means Cụm   : Gán vào = Cluster {km_cluster}")

    print("=" * 80)

if __name__ == "__main__":
    inspect_saved_models()
