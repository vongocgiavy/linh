# BÁO CÁO GIẢI THÍCH CHI TIẾT CÁC BIỂU ĐỒ & KẾT QUẢ ĐÁNH GIÁ (CHART EXPLANATION)
**Dự án:** Phân tích Dữ liệu Lichess Chess Games, Phân lớp Học máy (10-Fold CV) & Gom cụm Không giám sát  
**Tập dữ liệu:** Lichess Standard Rated Chess Database (`data/filtered_processed_games.csv`)

---

## 📌 MỤC LỤC
1. [Tổng quan 4 yêu cầu bài toán](#1-tổng-quan-4-yêu-cầu-bài-toán)
2. [Biểu đồ 1: Thu giảm số chiều không gian 2D (PCA & t-SNE)](#2-biểu-đồ-1-thu-giảm-số-chiều-không-gian-2d-pca--t-sne)
3. [Biểu đồ 2: So sánh hiệu năng các mô hình phân lớp (F-Score)](#3-biểu-đồ-2-so-sánh-hiệu-năng-các-mô-hình-phân-lớp-f-score)
4. [Biểu đồ 3: Đối sánh kết quả Gom cụm (K-Means & DBSCAN vs Ground Truth)](#4-biểu-đồ-3-đối-sánh-kết-quả-gom-cụm-k-means--dbscan-vs-ground-truth)
5. [Sơ đồ Graphviz: Kiến trúc Pipeline & Cây Quyết định](#5-sơ-đồ-graphviz-kiến-trúc-pipeline--cây-quyết-định)

---

## 1. TỔNG QUAN 4 YÊU CẦU BÀI TOÁN

| Yêu cầu | Nội dung kỹ thuật | Tệp mã nguồn / File đồ thị |
| :--- | :--- | :--- |
| **Yêu cầu 1** | Đọc dữ liệu, làm sạch, hiển thị Kích thước, Kiểu dữ liệu, Phân bố nhãn, Thống kê Min/Max/Mean các thuộc tính liên tục. | `src/preprocessing/data_cleaner.py` |
| **Yêu cầu 2** | Lấy các thuộc tính liên tục, chuẩn hóa, thu giảm số chiều về 2D bằng PCA & t-SNE với màu sắc phân biệt theo nhãn. | `src/visualization/dimension_reduction.py` -> `outputs/dimension_reduction_2d.png` |
| **Yêu cầu 3** | Huấn luyện, tinh chỉnh tham số và đánh giá 10-Fold CV cho 3 mô hình: **Random Forest**, **AdaBoost**, **SVM (RBF)**; so sánh F-Score. | `src/classification/evaluate_all.py` -> `outputs/model_f1_comparison.png` |
| **Yêu cầu 4** | Loại trừ nhãn, gom cụm **K-Means ($k=3$)** và **DBSCAN**; đánh giá bằng Ground Truth (**ARI**, **NMI**, **V-Measure**, **Purity**, **Silhouette**). | `src/clustering/kmeans_dbscan.py` -> `outputs/clustering_comparison_2d.png` |

---

## 2. BIỂU ĐỒ 1: THU GIẢM SỐ CHIỀU KHÔNG GIAN 2D (PCA & t-SNE)
**File ảnh:** [`outputs/dimension_reduction_2d.png`](file:///d:/demo%20mh/outputs/dimension_reduction_2d.png)

```text
┌────────────────────────────────────────────────────────┐
│  PCA 2D: PC1 (Chênh lệch Elo - 40.71%)                 │
│          PC2 (Độ sâu lý thuyết khai cuộc - 34.47%)     │
│  t-SNE 2D: Nhúng phi tuyến bảo toàn mật độ cục bộ      │
│  Màu sắc: 🔴 Đen thắng (0) | 🟠 Hòa (1) | 🟢 Trắng thắng (2)│
└────────────────────────────────────────────────────────┘
```

### Ý nghĩa các trục và phân bố màu sắc:
* **Tập thuộc tính liên tục đầu vào:** `['white_rating', 'black_rating', 'rating_diff', 'opening_ply']` đã được chuẩn hóa với `StandardScaler`.
* **Trục PC1 (Principal Component 1 - Chiếm 40.71% phương sai):** Phản ánh tương quan chênh lệch và thực lực Elo giữa hai kỳ thủ (`rating_diff`, `white_rating`, `black_rating`).
* **Trục PC2 (Principal Component 2 - Chiếm 34.47% phương sai):** Phản ánh độ sâu lý thuyết của biến thể khai cuộc (`opening_ply`).
* **Màu sắc nhãn:**
  * 🔴 **Đỏ:** Nhãn 0 (Black win - Đen thắng `0-1`)
  * 🟠 **Cam:** Nhãn 1 (Draw - Hòa `1/2-1/2`)
  * 🟢 **Xanh lá:** Nhãn 2 (White win - Trắng thắng `1-0`)

### Nhận xét phân bố:
1. Dọc theo trục PC1, các điểm màu xanh lá (Trắng thắng) có xu hướng lệch về phía bên phải (nơi `white_rating > black_rating`), trong khi các điểm màu đỏ (Đen thắng) lệch về phía bên trái.
2. Các điểm màu cam (Hòa) phân bố tập trung ở vùng trung tâm xung quanh $\text{PC1} \approx 0$ (nơi hai người chơi có Elo ngang ngửa nhau).
3. Do đây là dữ liệu thi đấu thực tế, vùng giao thoa giữa 3 nhãn khá lớn, thể hiện tính chất bất ngờ và chiến thuật trong từng nước đi cờ vua.

---

## 3. BIỂU ĐỒ 2: SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH PHÂN LỚP (F-SCORE)
**File ảnh:** [`outputs/model_f1_comparison.png`](file:///d:/demo%20mh/outputs/model_f1_comparison.png)

### Bảng đối sánh chi tiết (10-Fold Stratified Cross-Validation):
| Mô hình | Macro F1-Score (Mean ± Std) | Weighted F1-Score | Accuracy | F1 (Black win) | F1 (Draw) | F1 (White win) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **44.22% ± 1.65%** | 62.53% | 62.88% | 64.69% | 3.56% | 64.42% |
| **AdaBoost** | **43.96% ± 0.62%** | **63.74%** | **64.86%** | **66.12%** | 0.00% | **65.75%** |
| **SVM (RBF Kernel)**| **40.74% ± 0.52%** | 55.57% | 49.48% | 57.75% | **7.78%** | 56.69% |

### Phân tích chuyên môn:
* **AdaBoost** cho **Độ chính xác (Accuracy = 64.86%)** và **Weighted F1 (63.74%)** cao nhất nhờ thuật toán tăng cường trọng số cho các mẫu dự đoán khó, nắm bắt rất sắc bén khả năng thắng thua dựa trên chênh lệch Elo.
* **Random Forest** đạt **Macro F1 cao nhất (44.22%)** nhờ cơ chế kết hợp nhiều cây quyết định giúp cân bằng tốt giữa các lớp.
* **SVM với RBF Kernel** dự đoán được nhiều ván Hòa nhất (**F1 Draw = 7.78%**), bởi hàm nhân RBF phân tách phi tuyến giúp xác định các vùng thiểu số tốt hơn các mô hình cây đơn thuần.

---

## 4. BIỂU ĐỒ 3: ĐỐI SÁNH KẾT QUẢ GOM CỤM (K-MEANS & DBSCAN VS GROUND TRUTH)
**File ảnh:** [`outputs/clustering_comparison_2d.png`](file:///d:/demo%20mh/outputs/clustering_comparison_2d.png)

### Bảng đánh giá theo Ground Truth:
| Thuật toán | ARI | NMI | V-Measure | Homogeneity | Completeness | FMI | Purity | Silhouette |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=3$)** | **0.0009** | **0.0014** | **0.0014** | **0.0016** | **0.0012** | **0.3981** | **49.61%** | 0.2867 |
| **DBSCAN ($\epsilon=0.8$)** | **-0.0003** | **0.0002** | **0.0002** | **0.0001** | **0.0016** | **0.6782** | **49.20%** | **0.4888** |

### Ma trận đối sánh (Contingency Matrix):
* **K-Means ($k=3$):** Phân không gian làm 3 cụm theo trình độ Elo (Cụm Elo cao, Cụm Elo trung bình, Cụm Elo sơ cấp). Trong mỗi cụm, tỷ lệ Trắng thắng (~49%) và Đen thắng (~47%) vẫn phân bố đều.
* **DBSCAN:** Nhận diện được một vùng mật độ chính liên tục chứa 9,909 ván cờ và bóc tách ra 91 điểm nhiễu/ngoại lai (các trận đấu có Elo cực kỳ chênh lệch hoặc cực đoan).
* **Kết luận khoa học:** Điểm ARI và NMI gần bằng 0 chứng minh rằng kết quả thắng/thua/hòa trong cờ vua là một đặc tính phát sinh trong ván đấu, **không tự phân tách thành các cụm hình học tách biệt chỉ dựa trên Elo ban đầu**.

---

## 5. SƠ ĐỒ GRAPHVIZ: KIẾN TRÚC PIPELINE & CÂY QUYẾT ĐỊNH
Trong file Jupyter Notebook [`lichess_ml_analysis.ipynb`](file:///d:/demo%20mh/lichess_ml_analysis.ipynb):
1. **Sơ đồ Pipeline (`graphviz.Digraph`):** Mô tả trực quan luồng dữ liệu từ dữ liệu thô PGN -> Tiền xử lý không rò rỉ -> Chuẩn hóa -> Rẽ nhánh sang Giảm chiều 2D, Phân lớp 10-Fold CV và Gom cụm không giám sát.
2. **Cây quyết định Random Forest (`graphviz.Source`):** Trực quan hóa cấu trúc nhánh rẽ điều kiện của cây quyết định mẫu, thể hiện rõ ngưỡng phân nhánh theo `rating_diff` và `opening_ply`.
