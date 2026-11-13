import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

df = pd.read_csv("data/data_raw_noisy.csv")

print("THÔNG TIN TỔNG QUAN VÀ KIỂU DỮ LIỆU")
print(df.info())

print("\nTHỐNG KÊ GIÁ TRỊ THIẾU")
missing_df = pd.DataFrame({
    "Số lượng thiếu": df.isnull().sum(),
    "Tỷ lệ (%)": np.round(df.isnull().mean() * 100, 2)
})
print(missing_df[missing_df["Số lượng thiếu"] > 0])

print("\nTHỐNG KÊ MÔ TẢ CÁC BIẾN SỐ")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]])

print("\nPHÂN BỐ CÁC BIẾN PHÂN LOẠI")
for col in ["Gender", "MembershipTier"]:
    print(f"\n-- {col} --")
    print(df[col].value_counts(dropna=False))

# --- TRỰC QUAN HÓA DỮ LIỆU ---

# Biểu đồ 1: Boxplot phát hiện Outliers trên các biến số chính
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i], color="#4C72B0")
    axes[i].set_title(f"{col}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=300)

# Biểu đồ 2: Histogram + KDE kiểm tra phân phối và độ lệch (Skewness)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#55A868", bins=25)
    skew_val = df[col].skew()
    axes[i].set_title(f"{col} (Skewness: {skew_val:.2f})", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig("eda_histograms.png", dpi=300)

# Biểu đồ 3: Ma trận tương quan (Correlation Heatmap)
plt.figure(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Ma trận tương quan Pearson giữa các thuộc tính số", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=300)

# Biểu đồ 4: Scatter Plot kiểm tra mối quan hệ Frequency - Monetary & Nhiễu/Outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x="Frequency", y="Monetary", 
    hue="MembershipTier", style="Gender", s=80, alpha=0.85
)
plt.title("Mối quan hệ Frequency vs. Monetary theo Hạng thành viên", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_scatter_rfm.png", dpi=300)
