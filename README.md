# ♟️ BÁO CÁO ĐỒ ÁN HỌC MÁY & GOM CỤM DỮ LIỆU CỜ VUA LICHESS
### (Lichess Standard Rated Games Machine Learning & Clustering Analysis)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)](https://jupyter.org/)
[![Graphviz](https://img.shields.io/badge/Graphviz-Visuals-red.svg)](https://graphviz.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. TỔNG QUAN ĐỀ TÀI & BỐN YÊU CẦU TRỌNG TÂM

Dự án triển khai toàn diện quy trình Khoa học Dữ liệu, Học máy có giám sát (Phân loại kết quả ván đấu với 10-Fold Cross-Validation) và Học máy không giám sát (Gom cụm K-Means, DBSCAN) trên cơ sở dữ liệu cờ vua **Lichess Standard Rated Games Database** gồm **10,000 ván cờ** thực tế.

```
                  ┌────────────────────────────────────────────────────────┐
                  │          LICHESS CHESS DATABASE (10,000 VÁN)           │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          ┌───────────────────────────┐               ┌───────────────────────────┐
          │  HỌC MÁY CÓ GIÁM SÁT      │               │  HỌC MÁY KHÔNG GIÁM SÁT   │
          │  (Classification Models)  │               │  (Unsupervised Clustering)│
          ├───────────────────────────┤               ├───────────────────────────┤
          │ • Random Forest           │               │ • K-Means (k=3)           │
          │ • AdaBoost                │               │ • DBSCAN (Density-based)  │
          │ • SVM (RBF Kernel)        │               │                           │
          │ ➔ 10-Fold Cross Validation│               │ ➔ Đánh giá Ground Truth   │
          └───────────────────────────┘               └───────────────────────────┘
```

### Bảng chi tiết 4 yêu cầu bài toán:

| STT | Nhiệm vụ | Nội dung kỹ thuật thực hiện | Tệp mã nguồn & Đầu ra |
| :---: | :--- | :--- | :--- |
| **1** | **Khảo sát & Tiền xử lý dữ liệu** | • Làm sạch dữ liệu, lọc Elo hợp lệ `[600, 3500]`, loại bỏ 100% rò rỉ dữ liệu.<br>• Hiển thị: Kích thước (`10,000` dòng $\times$ `17` cột), Kiểu dữ liệu (`dtypes`), Phân bố nhãn (0, 1, 2), Thống kê Min/Max/Mean/Std của các thuộc tính liên tục. | [`src/preprocessing/data_cleaner.py`](file:///d:/demo%20mh/src/preprocessing/data_cleaner.py)<br>📁 [`data/filtered_processed_games.csv`](file:///d:/demo%20mh/data/filtered_processed_games.csv) |
| **2** | **Trực quan hóa dữ liệu liên tục & Giảm chiều 2D** | • Trích xuất 4 thuộc tính liên tục (`white_rating`, `black_rating`, `rating_diff`, `opening_ply`).<br>• Chuẩn hóa dữ liệu với `StandardScaler`.<br>• Thu giảm số chiều về 2D bằng **PCA** (giải thích $75.19\%$ phương sai) và **t-SNE**.<br>• Vẽ biểu đồ phân tán 2D phân biệt 3 màu theo nhãn (🔴 Đen thắng, 🟠 Hòa, 🟢 Trắng thắng).<br>• Trực quan hóa sơ đồ kiến trúc quy trình bằng **`graphviz.Digraph`**. | [`src/visualization/dimension_reduction.py`](file:///d:/demo%20mh/src/visualization/dimension_reduction.py)<br>🖼️ [`outputs/dimension_reduction_2d.png`](file:///d:/demo%20mh/outputs/dimension_reduction_2d.png) |
| **3** | **Huấn luyện & Đánh giá phân lớp (10-Fold CV)** | • Huấn luyện 3 mô hình từ Scikit-Learn: **Random Forest**, **AdaBoost**, **SVM (RBF Kernel)**.<br>• Tinh chỉnh siêu tham số tối ưu và xử lý song song đa luồng (`n_jobs=-1`, `cache_size=1000MB`).<br>• Kiểm thử chéo **10-Fold Stratified Cross-Validation** ($k=10$).<br>• Trình bày bảng so sánh hiệu năng theo **F-Score** (Macro F1, Weighted F1, Accuracy, F1 từng lớp).<br>• Xuất biểu đồ cột so sánh và **Cây quyết định Graphviz**. | [`src/classification/evaluate_all.py`](file:///d:/demo%20mh/src/classification/evaluate_all.py)<br>🖼️ [`outputs/model_f1_comparison.png`](file:///d:/demo%20mh/outputs/model_f1_comparison.png)<br>💾 [`models/*.joblib`](file:///d:/demo%20mh/models) |
| **4** | **Gom cụm không giám sát K-Means & DBSCAN** | • Loại trừ cột nhãn `ResultEncoded`, chỉ dùng ma trận đặc trưng chuẩn hóa.<br>• Huấn luyện **K-Means ($k=3$)** và **DBSCAN ($\epsilon=0.8, \text{min\_samples}=20$)**.<br>• Đánh giá chất lượng cụm theo Ground Truth: **ARI**, **NMI**, **V-Measure**, **Homogeneity**, **Completeness**, **FMI**, **Purity**, **Silhouette Score**.<br>• Lập ma trận đối sánh cụm (Contingency Matrix) và vẽ biểu đồ phân cụm 2D. | [`src/clustering/kmeans_dbscan.py`](file:///d:/demo%20mh/src/clustering/kmeans_dbscan.py)<br>🖼️ [`outputs/clustering_comparison_2d.png`](file:///d:/demo%20mh/outputs/clustering_comparison_2d.png) |

---

## 🎯 2. KỸ THUẬT ĐẶC TRƯNG & PHÒNG CHỐNG RÒ RỈ DỮ LIỆU (0% DATA LEAKAGE)

Để đảm bảo mô hình có tính thực tiễn và không phạm phải lỗi rò rỉ dữ liệu (Data Leakage) thường gặp trong bài toán cờ vua:
1. **Không sử dụng số nước đi thực tế (`Moves length`):** Trong thực tế trước khi ván đấu diễn ra, không ai biết ván đấu sẽ kéo dài bao nhiêu nước. Việc dùng độ dài nước đi thực tế sẽ gây rò rỉ kết quả (ván hòa thường kéo dài, ván thắng nhanh do bẫy khai cuộc).
2. **Trích xuất độ sâu lý thuyết khai cuộc (`opening_ply`):** Độ sâu khai cuộc được tính toán dựa trên **định nghĩa lý thuyết sách khai cuộc** (từ PGN Opening tag), hoàn toàn độc lập với diễn biến ván đấu thực tế.
3. **5 Đặc trưng đầu vào hợp lệ:**
   * `white_rating` (float): Điểm Elo kỳ thủ cầm quân Trắng.
   * `black_rating` (float): Điểm Elo kỳ thủ cầm quân Đen.
   * `rating_diff` (float): Hiệu số trình độ $\text{WhiteElo} - \text{BlackElo}$.
   * `rated` (binary): Loại hình thi đấu (1: Tính điểm xếp hạng, 0: Đấu giao hữu).
   * `opening_ply` (float): Số nửa nước đi lý thuyết của khai cuộc được chọn.
4. **Chuẩn hóa trong Pipeline:** Khi kiểm thử K-Fold CV, `StandardScaler` được đặt bên trong `Pipeline` để đảm bảo dữ liệu tập xác thực (Validation fold) không bị nhìn thấy trước ở bước tính $\mu, \sigma$.

---

## 📁 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
d:\demo mh\
├── lichess_ml_analysis.ipynb            # [Jupyter Notebook] Báo cáo tương tác hoàn chỉnh (Đầy đủ output & Graphviz)
├── run_pipeline.py                      # [Main Runner] File thực thi chính chạy toàn bộ 4 bước qua Terminal
├── read_models.py                       # [Inspector] File đọc, kiểm tra thông số và demo dự đoán 7 mô hình đã lưu
├── requirements.txt                     # Danh sách các thư viện phụ thuộc
├── README.md                            # Tài liệu Master của đồ án
├── CHART_EXPLANATION.md                 # Báo cáo thuyết minh giải thích chi tiết các biểu đồ
│
├── data/                                # Thư mục dữ liệu
│   ├── processed_games.csv              # Dữ liệu gốc 10,000 ván cờ Lichess
│   └── filtered_processed_games.csv     # Dữ liệu sạch mới (10,000 ván, 0% rò rỉ dữ liệu)
│
├── models/                              # Lưu trữ toàn bộ 7 file model Scikit-Learn (.joblib)
│   ├── random_forest.joblib             # Model Random Forest đã huấn luyện
│   ├── adaboost.joblib                  # Model AdaBoost đã huấn luyện
│   ├── svm_rbf.joblib                   # Model SVM RBF Kernel trong Pipeline
│   ├── kmeans.joblib                    # Model K-Means (k=3)
│   ├── dbscan.joblib                    # Model DBSCAN (eps=0.8)
│   ├── scaler.joblib                    # StandardScaler
│   └── pca_2d.joblib                    # PCA 2D
│
├── outputs/                             # Nơi xuất các biểu đồ trực quan
│   ├── dimension_reduction_2d.png       # Biểu đồ 2D PCA & t-SNE
│   ├── model_f1_comparison.png          # Biểu đồ cột so sánh F-Score 3 mô hình phân lớp
│   └── clustering_comparison_2d.png     # Biểu đồ đối sánh gom cụm 2D vs Ground Truth
│
└── src/                                 # Toàn bộ mã nguồn module hóa
    ├── preprocessing/                   # [Module 1] Tiền xử lý & Khảo sát thông số
    │   ├── __init__.py
    │   └── data_cleaner.py
    ├── visualization/                   # [Module 2] Thu giảm số chiều 2D
    │   ├── __init__.py
    │   └── dimension_reduction.py
    ├── classification/                  # [Module 3] 3 Mô hình phân lớp & 10-Fold CV
    │   ├── __init__.py
    │   ├── random_forest.py
    │   ├── adaboost.py
    │   ├── svm.py
    │   └── evaluate_all.py
    └── clustering/                      # [Module 4] Gom cụm K-Means & DBSCAN
        ├── __init__.py
        └── kmeans_dbscan.py
```

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN

### Bước 1: Cài đặt các thư viện cần thiết
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ quy trình bằng dòng lệnh Terminal (Khuyên dùng)
Lệnh này sẽ tự động chạy tuần tự từ Bước 1 đến Bước 4, in bảng thống kê và xuất 7 file model vào `models/` cùng 3 biểu đồ vào `outputs/`:
```powershell
py run_pipeline.py
```
*(hoặc `python run_pipeline.py`)*

### Bước 3: Đọc và kiểm tra thông số toàn bộ các mô hình đã lưu
Lệnh này sẽ nạp 7 file `.joblib` từ thư mục `models/`, in thông số trọng số đã học và chạy demo dự đoán trên mẫu ván cờ thực tế:
```powershell
py read_models.py
```

### Bước 4: Xem báo cáo trực quan trên Jupyter Notebook
Mở file **[`lichess_ml_analysis.ipynb`](file:///d:/demo%20mh/lichess_ml_analysis.ipynb)** bằng VS Code hoặc Jupyter Notebook để xem toàn bộ giải thích từng bước, code, bảng dữ liệu, đồ thị màu và **sơ đồ kiến trúc luồng dữ liệu / cây quyết định bằng `graphviz`**.

### Bước 5: Chạy riêng lẻ từng module chức năng độc lập:
```powershell
# Chạy riêng Bước 1: Tiền xử lý & Khảo sát thông số
py src/preprocessing/data_cleaner.py

# Chạy riêng Bước 2: Giảm chiều 2D (PCA, t-SNE)
py src/visualization/dimension_reduction.py

# Chạy riêng Bước 3: Đánh giá & So sánh F-Score 3 mô hình phân lớp (10-Fold CV)
py src/classification/evaluate_all.py

# Chạy riêng Bước 4: Gom cụm K-Means & DBSCAN + Đánh giá Ground Truth
py src/clustering/kmeans_dbscan.py
```

---

## 📊 5. KẾT QUẢ KHOA HỌC THỰC NGHIỆM CHI TIẾT

### 5.1. Khảo sát thông số dữ liệu (Yêu cầu 1):
* **Kích thước ($n_{\text{samples}} \times n_{\text{cols}}$):** `10,000` dòng $\times$ `17` cột.
* **Phân bố nhãn tự nhiên:**
  * Nhãn `0` (Black win - Đen thắng `0-1`): **`4,757` mẫu (47.57%)**
  * Nhãn `1` (Draw - Hòa `1/2-1/2`): **`332` mẫu (3.32%)** *(tỷ lệ hòa tự nhiên thực tế)*
  * Nhãn `2` (White win - Trắng thắng `1-0`): **`4,911` mẫu (49.11%)**
* **Thống kê Min, Max, Mean các cột thuộc tính số thực:**
  * `white_rating`: Min = $860$, Max = $2685$, Mean = **$1634.58$**, Median = $1621.0$, Std = $272.85$
  * `black_rating`: Min = $866$, Max = $2698$, Mean = **$1635.37$**, Median = $1621.0$, Std = $272.89$
  * `rating_diff`: Min = $-1265$, Max = $+1153$, Mean = **$-0.79$**, Median = $-1.0$, Std = $236.47$
  * `opening_ply`: Min = $2.0$, Max = $15.0$, Mean = **$4.77$**, Median = $4.0$, Std = $2.74$

---

### 5.2. So sánh hiệu năng các mô hình phân lớp qua 10-Fold CV (Yêu cầu 3):
| Mô hình Scikit-Learn | Siêu tham số tối ưu | Macro F1-Score (Mean ± Std) | Weighted F1-Score | Accuracy | F1 (Black win) | F1 (Draw) | F1 (White win) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `n_estimators=120, max_depth=12, min_samples_split=6` | **$44.22\% \pm 1.65\%$** | $62.53\%$ | $62.88\%$ | $64.69\%$ | $3.56\%$ | $64.42\%$ |
| **AdaBoost** | `n_estimators=100, learning_rate=0.15` | **$43.96\% \pm 0.62\%$** | **$63.74\%$** | **$64.86\%$** | **$66.12\%$** | $0.00\%$ | **$65.75\%$** |
| **SVM (RBF Kernel)**| `C=1.5, kernel='rbf', gamma='scale', cache_size=1000` | **$40.74\% \pm 0.52\%$** | $55.57\%$ | $49.48\%$ | $57.75\%$ | **$7.78\%$** | $56.69\%$ |

> [!TIP]
> **Nhận xét chuyên môn:**
> * **AdaBoost** cho **Độ chính xác (Accuracy = 64.86%)** và **Weighted F1 (63.74%)** cao nhất nhờ thuật toán tăng cường trọng số cho các mẫu dự đoán khó, nắm bắt rất sắc bén khả năng thắng thua dựa trên chênh lệch Elo.
> * **Random Forest** đạt **Macro F1 cao nhất (44.22%)** nhờ cơ chế kết hợp nhiều cây quyết định giúp cân bằng tốt giữa các lớp.
> * **SVM với RBF Kernel** dự đoán được nhiều ván Hòa nhất (**F1 Draw = 7.78%**), bởi hàm nhân RBF phân tách phi tuyến giúp xác định các vùng thiểu số tốt hơn các mô hình cây đơn thuần.

---

### 5.3. Đánh giá chất lượng Gom cụm với Ground Truth (Yêu cầu 4):
| Thuật toán | Cấu hình tham số | ARI | NMI | V-Measure | Homogeneity | Completeness | FMI | Purity | Silhouette Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=3$)** | $k=3, \text{init}='k\text{-means++}'$ | **$0.0009$** | **$0.0014$** | **$0.0014$** | **$0.0016$** | **$0.0012$** | **$0.3981$** | **$49.61\%$** | $0.2867$ |
| **DBSCAN ($\epsilon=0.8$)** | $\epsilon=0.8, \text{min\_samples}=20$ | **$-0.0003$** | **$0.0002$** | **$0.0002$** | **$0.0001$** | **$0.0016$** | **$0.6782$** | **$49.20\%$** | **$0.4888$** |

#### Ma trận đối sánh phân bố cụm (Contingency Matrix):
* **K-Means ($k=3$):**
  * Cụm 0 (Elo cao): $1,335$ Đen thắng, $132$ Hòa, $1,506$ Trắng thắng.
  * Cụm 1 (Elo trung bình): $1,580$ Đen thắng, $81$ Hòa, $1,530$ Trắng thắng.
  * Cụm 2 (Elo sơ cấp): $1,842$ Đen thắng, $119$ Hòa, $1,875$ Trắng thắng.
* **DBSCAN:** Nhận diện 1 cụm liên tục $9,909$ ván cờ và $91$ điểm nhiễu ngoại lai.

> [!NOTE]
> **Kết luận khoa học về gom cụm:** Điểm **ARI $\approx 0$** và **NMI $\approx 0$** chứng minh rằng kết quả ván đấu cờ vua là một đặc tính phát sinh theo diễn biến từng nước cờ cụ thể, **không tự phân tách thành các cụm hình học tách biệt chỉ dựa trên trình độ Elo ban đầu**.

---

## 💻 6. HƯỚNG DẪN TÁI SỬ DỤNG MÔ HÌNH (INFERENCE)

Toàn bộ các mô hình Scikit-Learn đã được lưu dưới dạng file `.joblib` trong thư mục [`models/`](file:///d:/demo%20mh/models). Đoạn mã dưới đây minh họa cách nạp và dự đoán kết quả ván đấu mới:

```python
import joblib
import numpy as np

# 1. Nạp mô hình phân lớp và chuẩn hóa từ models/
rf_model = joblib.load("models/random_forest.joblib")
scaler = joblib.load("models/scaler.joblib")
kmeans = joblib.load("models/kmeans.joblib")

# 2. Định nghĩa mẫu ván cờ mới: [white_rating, black_rating, rating_diff, rated, opening_ply]
sample_game = np.array([[1850, 1600, 250, 1, 8]])

# 3. Dự đoán kết quả với Random Forest: 0 (Đen thắng), 1 (Hòa), 2 (Trắng thắng)
pred_result = rf_model.predict(sample_game)
label_names = {0: "Black win (Đen thắng)", 1: "Draw (Hòa)", 2: "White win (Trắng thắng)"}
print(f"Kết quả dự đoán: {label_names[pred_result[0]]}")

# 4. Gom cụm mẫu dữ liệu với K-Means
sample_cont = scaler.transform(np.array([[1850, 1600, 250, 8]]))
cluster_id = kmeans.predict(sample_cont)
print(f"Cụm K-Means được gán: Cluster {cluster_id[0]}")
```

---

## ⚡ 7. TỐI ƯU HÓA HIỆU NĂNG TÍNH TOÁN (PERFORMANCE HIGHLIGHTS)

* **Xử lý song song đa luồng:** Áp dụng `joblib.Parallel(n_jobs=-1)` giúp **10 Folds trong Cross-Validation chạy đồng thời trên toàn bộ nhân CPU**.
* **Mở rộng bộ nhớ đệm Kernel:** Thiết lập `cache_size=1000MB` (1GB RAM) cho SVM RBF Kernel, rút ngắn thời gian tính toán ma trận Gram.
* **Thời gian thực thi toàn bộ pipeline (`run_pipeline.py`):** **~25 - 28 giây** (rút ngắn hơn 75% so với chạy tuần tự).

---

📖 **Xem tài liệu giải thích chi tiết các biểu đồ và sơ đồ Graphviz tại:** [`CHART_EXPLANATION.md`](file:///d:/demo%20mh/CHART_EXPLANATION.md)
