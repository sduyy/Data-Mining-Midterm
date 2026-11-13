import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

segments_init = np.repeat([0, 1, 2, 3], [39, 91, 91, 39])
df_gt = pd.DataFrame({"True_Label": segments_init})
df_gt = df_gt.sample(frac=1, random_state=42).reset_index(drop=True)
y_true = df_gt["True_Label"].values
label_names = {0: "VIP", 1: "Thợ săn Sale", 2: "Phổ thông", 3: "Nguy cơ rời bỏ"}

df_raw = pd.read_csv("data/data_raw_noisy.csv")
df_prep = pd.read_csv("data/data_preprocessed.csv")
df_prepared = pd.read_csv("data/data_prepared.csv")

X_prepared = df_prepared.drop(columns=["CustomerID"]).values

df_minimal = df_raw.drop(columns=["CustomerID"]).copy()
df_minimal["MembershipTier"] = df_minimal["MembershipTier"].map({"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4})
df_minimal["Gender"] = df_minimal["Gender"].fillna(df_minimal["Gender"].mode()[0])
df_minimal = pd.get_dummies(df_minimal, columns=["Gender"], dtype=float)
df_minimal = df_minimal.fillna(df_minimal.median())
X_minimal = df_minimal.values

k_range = range(2, 8)
sil_raw_list, sil_prep_list, wcss_prep_list = [], [], []
for k in k_range:
    l_raw = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X_minimal)
    l_prep = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X_prepared)
    sil_raw_list.append(silhouette_score(X_minimal, l_raw))
    sil_prep_list.append(silhouette_score(X_prepared, l_prep))
    wcss_prep_list.append(KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_prepared).inertia_)

print("=== 1. KHẢO SÁT SILHOUETTE SCORE THEO SỐ CỤM K (2..7) ===")
print(pd.DataFrame({
    "k": list(k_range),
    "Silhouette_Raw (13D thô)": np.round(sil_raw_list, 4),
    "Silhouette_Prepared (9D chuẩn)": np.round(sil_prep_list, 4),
    "WCSS_Prepared": np.round(wcss_prep_list, 1)
}).to_string(index=False))

seeds = [10, 21, 32, 42, 55, 68, 79, 84, 99, 123]
metrics_raw = {"ARI": [], "NMI": [], "Sil_SameSpace": [], "DBI_SameSpace": [], "Min_Cluster_Size": []}
metrics_prep = {"ARI": [], "NMI": [], "Sil_SameSpace": [], "DBI_SameSpace": [], "Min_Cluster_Size": []}

for s in seeds:
    lb_r = KMeans(n_clusters=4, n_init=10, random_state=s).fit_predict(X_minimal)
    lb_p = KMeans(n_clusters=4, n_init=10, random_state=s).fit_predict(X_prepared)
    
    metrics_raw["ARI"].append(adjusted_rand_score(y_true, lb_r))
    metrics_raw["NMI"].append(normalized_mutual_info_score(y_true, lb_r))
    metrics_raw["Sil_SameSpace"].append(silhouette_score(X_prepared, lb_r))
    metrics_raw["DBI_SameSpace"].append(davies_bouldin_score(X_prepared, lb_r))
    metrics_raw["Min_Cluster_Size"].append(pd.Series(lb_r).value_counts().min())
    
    metrics_prep["ARI"].append(adjusted_rand_score(y_true, lb_p))
    metrics_prep["NMI"].append(normalized_mutual_info_score(y_true, lb_p))
    metrics_prep["Sil_SameSpace"].append(silhouette_score(X_prepared, lb_p))
    metrics_prep["DBI_SameSpace"].append(davies_bouldin_score(X_prepared, lb_p))
    metrics_prep["Min_Cluster_Size"].append(pd.Series(lb_p).value_counts().min())

print("\n=== 2. BẢNG SO SÁNH ĐA HẠT GIỐNG (10 SEEDS, K=4, MEAN ± STD) ===")
summary_rows = []
for m in ["ARI", "NMI", "Sil_SameSpace", "DBI_SameSpace", "Min_Cluster_Size"]:
    r_mean, r_std = np.mean(metrics_raw[m]), np.std(metrics_raw[m])
    p_mean, p_std = np.mean(metrics_prep[m]), np.std(metrics_prep[m])
    summary_rows.append({
        "Chỉ số": m,
        "Trước xử lý (13D thô)": f"{r_mean:.4f} ± {r_std:.4f}",
        "Sau chuẩn bị (9D chuẩn)": f"{p_mean:.4f} ± {p_std:.4f}"
    })
print(pd.DataFrame(summary_rows).to_string(index=False))

labels_before = KMeans(n_clusters=4, n_init=30, random_state=42).fit_predict(X_minimal)
labels_after = KMeans(n_clusters=4, n_init=30, random_state=42).fit_predict(X_prepared)

profile_cols = ["Age", "Recency", "Frequency", "Monetary", "AvgOrderValue", 
                "AppTimePerWeek", "DiscountUsageRate", "ReturnRate", "SupportTickets"]

print("\n=== 3A. PROFILE TRUNG BÌNH CÁC CỤM - TRƯỚC XỬ LÝ (ĐƠN VỊ GỐC) ===")
df_prof_raw = df_raw.copy()
df_prof_raw["Cluster_Before"] = labels_before
prof_before = df_prof_raw.groupby("Cluster_Before")[profile_cols].mean().round(2)
prof_before.insert(0, "Số_KH", df_prof_raw["Cluster_Before"].value_counts().sort_index())
print(prof_before.to_string())

print("\n=== 3B. PROFILE TRUNG BÌNH CÁC CỤM - SAU CHUẨN BỊ (ĐƠN VỊ GỐC) ===")
df_prof_prep = df_prep.copy()
df_prof_prep["Cluster_After"] = labels_after
prof_after = df_prof_prep.groupby("Cluster_After")[profile_cols].mean().round(2)
prof_after.insert(0, "Số_KH", df_prof_prep["Cluster_After"].value_counts().sort_index())
print(prof_after.to_string())

print("\n=== 4A. MA TRẬN NHẦM LẪN: TRƯỚC XỬ LÝ VS NHÃN GỐC THẬT ===")
ct_before = pd.crosstab(
    pd.Series(labels_before, name="Cụm_Trước_XL"),
    pd.Series(y_true, name="Nhãn_Gốc").map(label_names)
)
print(ct_before.to_string())

print("\n=== 4B. MA TRẬN NHẦM LẪN: SAU CHUẨN BỊ VS NHÃN GỐC THẬT ===")
ct_after = pd.crosstab(
    pd.Series(labels_after, name="Cụm_Sau_CB"),
    pd.Series(y_true, name="Nhãn_Gốc").map(label_names)
)
print(ct_after.to_string())

pca_2d = PCA(n_components=2, random_state=42)
coords_2d = pca_2d.fit_transform(X_prepared)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(x=coords_2d[:, 0], y=coords_2d[:, 1], hue=labels_before, palette="Set1", s=65, ax=axes[0])
axes[0].set_title("Nhãn phân cụm TRƯỚC xử lý (chiếu lên PCA 2D)", fontweight="bold")
axes[0].set_xlabel("PC1 (34.7% - Giá trị & Gắn kết)")
axes[0].set_ylabel("PC2 (31.6% - Hành vi Săn Sale)")

sns.scatterplot(x=coords_2d[:, 0], y=coords_2d[:, 1], hue=labels_after, palette="Set2", s=65, ax=axes[1])
axes[1].set_title("Nhãn phân cụm SAU chuẩn bị (chiếu lên PCA 2D)", fontweight="bold")
axes[1].set_xlabel("PC1 (34.7% - Giá trị & Gắn kết)")
axes[1].set_ylabel("PC2 (31.6% - Hành vi Săn Sale)")

plt.tight_layout()
plt.savefig("eval_clusters_comparison.png", dpi=300)
plt.show()
