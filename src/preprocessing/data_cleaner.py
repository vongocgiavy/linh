import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def clean_moves(moves_str):
    import re
    if not isinstance(moves_str, str):
        return ""
    moves_str = re.sub(r"\{[^}]*\}", "", moves_str)
    moves_str = re.sub(r"\d+\.+\s*", "", moves_str)
    moves_str = re.sub(r"1-0|0-1|1/2-1/2|\*", "", moves_str)
    return " ".join(moves_str.split())

def extract_opening_ply(opening_str):
    import re
    if not isinstance(opening_str, str) or opening_str == "?" or not opening_str:
        return 8.0
    match_triple = re.findall(r"(\d+)\.\.\.", opening_str)
    if match_triple:
        return float(int(match_triple[-1]) * 2)
    match_single = re.findall(r"(\d+)\.", opening_str)
    if match_single:
        return float(int(match_single[-1]) * 2 - 1)
    if ":" in opening_str:
        var_part = opening_str.split(":", 1)[1].strip()
        plies = len(clean_moves(var_part).split())
        if plies > 0:
            return float(min(max(plies, 2), 20))
    return 8.0

def load_and_preprocess_dataset(
    raw_csv=os.path.join(PROJECT_ROOT, "data", "processed_games.csv"),
    output_clean_csv=os.path.join(PROJECT_ROOT, "data", "filtered_processed_games.csv")
):
    print("=" * 80)
    print(" [01_PREPROCESSING] ĐỌC DỮ LIỆU, LÀM SẠCH VÀ HIỂN THỊ THÔNG SỐ")
    print("=" * 80)

    if os.path.exists(output_clean_csv):
        os.remove(output_clean_csv)
        print(f"[*] Đã xóa file dữ liệu cũ: {output_clean_csv}")

    raw_df = pd.read_csv(raw_csv, dtype=str)
    df = raw_df.copy()

    # Parse numeric ratings
    df["white_rating"] = pd.to_numeric(df["WhiteElo"], errors="coerce")
    df["black_rating"] = pd.to_numeric(df["BlackElo"], errors="coerce")
    df = df.dropna(subset=["white_rating", "black_rating"])
    df["white_rating"] = df["white_rating"].astype(float)
    df["black_rating"] = df["black_rating"].astype(float)
    
    # Filter valid chess Elo range
    df = df[(df["white_rating"] >= 600) & (df["white_rating"] <= 3500)]
    df = df[(df["black_rating"] >= 600) & (df["black_rating"] <= 3500)]

    # 5 features
    df["rating_diff"] = df["white_rating"] - df["black_rating"]
    if "Event" in df.columns:
        df["rated"] = df["Event"].astype(str).apply(lambda ev: 1 if "rated" in ev.lower() else 0)
    else:
        df["rated"] = 1
    df["opening_ply"] = df["Opening"].apply(extract_opening_ply)

    result_map = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}
    df = df[df["Result"].isin(result_map.keys())].copy()
    df["ResultEncoded"] = df["Result"].map(result_map).astype(int)

    os.makedirs(os.path.dirname(output_clean_csv), exist_ok=True)
    df.to_csv(output_clean_csv, index=False)
    print(f"[+] Đã tạo tập dữ liệu sạch mới tại: {output_clean_csv}")

    # Thông số yêu cầu 1.a, 1.b, 1.c, 1.d
    print("\n1.a. Kích thước và chiều dữ liệu (Shape):", df.shape)
    print("\n1.b. Kiểu dữ liệu các thuộc tính (dtypes):")
    print(df.dtypes)

    print("\n1.c. Số lượng thực thể các giá trị nhãn (Class Counts):")
    label_map = {0: "0 (Black win)", 1: "1 (Draw)", 2: "2 (White win)"}
    counts = df["ResultEncoded"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"   - {label_map[k]:<25}: {v:,} mẫu ({v/len(df)*100:.2f}%)")

    print("\n1.d. Thống kê Min, Max, Mean các cột thuộc tính liên tục:")
    cont_cols = ["white_rating", "black_rating", "rating_diff", "opening_ply"]
    stats = []
    for c in cont_cols:
        stats.append({
            "Thuộc tính": c,
            "Min": df[c].min(),
            "Max": df[c].max(),
            "Mean": round(df[c].mean(), 2),
            "Median": round(df[c].median(), 2),
            "Std": round(df[c].std(), 2)
        })
    print(pd.DataFrame(stats).to_string(index=False))

    return df

if __name__ == "__main__":
    load_and_preprocess_dataset()
