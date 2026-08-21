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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification.random_forest import train_and_eval_random_forest
from src.classification.adaboost import train_and_eval_adaboost
from src.classification.svm import train_and_eval_svm

def compare_all_classification_models(
    data_path=os.path.join(PROJECT_ROOT, "data", "filtered_processed_games.csv"),
    output_chart=os.path.join(PROJECT_ROOT, "outputs", "model_f1_comparison.png"),
    n_splits=10
):
    print("=" * 80)
    print(" [03_CLASSIFICATION] SO SÁNH 3 MÔ HÌNH: RANDOM FOREST, ADABOOST, SVM (10-FOLD CV)")
    print("=" * 80)

    _, rf_metrics = train_and_eval_random_forest(data_path=data_path, n_splits=n_splits)
    _, ada_metrics = train_and_eval_adaboost(data_path=data_path, n_splits=n_splits)
    _, svm_metrics = train_and_eval_svm(data_path=data_path, n_splits=n_splits)

    all_metrics = [rf_metrics, ada_metrics, svm_metrics]

    table_data = []
    for m in all_metrics:
        table_data.append({
            "Mô Hình": m["model_name"],
            "Macro F1-Score (Mean ± Std)": f"{m['macro_f1']*100:.2f}% ± {m['macro_f1_std']*100:.2f}%",
            "Weighted F1-Score": f"{m['weighted_f1']*100:.2f}%",
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "F1 (Black win)": f"{m['f1_black']*100:.2f}%",
            "F1 (Draw)": f"{m['f1_draw']*100:.2f}%",
            "F1 (White win)": f"{m['f1_white']*100:.2f}%",
        })

    summary_df = pd.DataFrame(table_data)
    print("\n" + "=" * 80)
    print(" BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH VỚI ĐỘ ĐO F-SCORE (10-FOLD CV)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Plot Comparison
    os.makedirs(os.path.dirname(output_chart), exist_ok=True)
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(all_metrics))
    macro_scores = [m["macro_f1"] * 100 for m in all_metrics]
    weighted_scores = [m["weighted_f1"] * 100 for m in all_metrics]
    acc_scores = [m["accuracy"] * 100 for m in all_metrics]

    width = 0.25
    plt.bar(x_pos - width, macro_scores, width=width, label="Macro F1-Score", color="#3498db")
    plt.bar(x_pos, weighted_scores, width=width, label="Weighted F1-Score", color="#2ecc71")
    plt.bar(x_pos + width, acc_scores, width=width, label="Accuracy", color="#9b59b6")

    plt.xticks(x_pos, [m["model_name"] for m in all_metrics], fontsize=11, fontweight="bold")
    plt.ylabel("Điểm số (%)", fontsize=11)
    plt.title("So Sánh Hiệu Năng Các Mô Hình Phân Lớp (10-Fold Cross-Validation)", fontsize=13, fontweight="bold")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)

    for i in range(len(all_metrics)):
        plt.text(i - width, macro_scores[i] + 1.5, f"{macro_scores[i]:.1f}%", ha='center', fontsize=9, fontweight="bold")
        plt.text(i, weighted_scores[i] + 1.5, f"{weighted_scores[i]:.1f}%", ha='center', fontsize=9, fontweight="bold")
        plt.text(i + width, acc_scores[i] + 1.5, f"{acc_scores[i]:.1f}%", ha='center', fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_chart, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Đã lưu biểu đồ so sánh F-Score tại: {output_chart}")

    return summary_df

if __name__ == "__main__":
    compare_all_classification_models()
