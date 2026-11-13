import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_prep = pd.read_csv("data/data_preprocessed.csv")

num_cols = [
    "Age", "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "AppTimePerWeek", "DiscountUsageRate", "ReturnRate", "SupportTickets"
]

skew_before = df_prep[num_cols].skew()
skew_after_log = np.log1p(df_prep[num_cols]).skew()

skew_df = pd.DataFrame({
    "Skew_Trước": np.round(skew_before, 3),
    "Skew_Sau_Log1p": np.round(skew_after_log, 3),
    "Nên_Log (|Skew|>1)": skew_before.abs() > 1.0
})
print("=== 1. BẢNG ĐÁNH GIÁ ĐỘ LỆCH (SKEWNESS) ===")
print(skew_df)

fin_cols = ["Frequency", "Monetary", "AvgOrderValue"]
print("\n=== 2A. TƯƠNG QUAN TÀI CHÍNH (TRƯỚC LOG) ===")
print(np.round(df_prep[fin_cols].corr(), 3))

df_fin_log = np.log1p(df_prep[fin_cols])
print("\n=== 2B. TƯƠNG QUAN TÀI CHÍNH (SAU LOG1P) ===")
print(np.round(df_fin_log.corr(), 3))

cols_to_log = skew_before[skew_before.abs() > 1.0].index.tolist()
print(f"\nCác biến thỏa điều kiện |Skew| > 1.0 để áp dụng log1p: {cols_to_log}")

df_base = df_prep.copy()
df_base["MembershipTier"] = df_base["MembershipTier"].map({"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4})
for col in cols_to_log:
    df_base[col] = np.log1p(df_base[col])

feats_A = ["MembershipTier", "Age", "Recency", "Frequency", "AvgOrderValue", 
           "AppTimePerWeek", "DiscountUsageRate", "ReturnRate", "SupportTickets"]

feats_B = ["MembershipTier", "Age", "Recency", "Frequency", "Monetary", 
           "AppTimePerWeek", "DiscountUsageRate", "ReturnRate", "SupportTickets"]

scaler_A, scaler_B = StandardScaler(), StandardScaler()
X_A = scaler_A.fit_transform(df_base[feats_A])
X_B = scaler_B.fit_transform(df_base[feats_B])

km_A = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X_A)
km_B = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X_B)

print("\n=== 3. SO SÁNH SILHOUETTE SCORE (K=4) GIỮA 2 PHƯƠNG ÁN ===")
print(f"- Phương án A (Giữ AvgOrderValue, bỏ Monetary): Silhouette = {silhouette_score(X_A, km_A):.4f}")
print(f"- Phương án B (Giữ Monetary, bỏ AvgOrderValue): Silhouette = {silhouette_score(X_B, km_B):.4f}")

best_feats = feats_A if silhouette_score(X_A, km_A) >= silhouette_score(X_B, km_B) else feats_B
best_X = X_A if best_feats == feats_A else X_B
best_name = "Phương án A (Bỏ Monetary)" if best_feats == feats_A else "Phương án B (Bỏ AvgOrderValue)"
print(f"=> Chốt sử dụng: {best_name}")

df_prepared = pd.DataFrame(best_X, columns=best_feats)
df_prepared.insert(0, "CustomerID", df_prep["CustomerID"])
df_prepared.to_csv("data_prepared.csv", index=False)

pca = PCA()
pca.fit(best_X)
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

print("\n=== 4A. PHƯƠNG SAI GIẢI THÍCH BỞI PCA ===")
for i, (ev, cv) in enumerate(zip(exp_var, cum_var), 1):
    print(f"PC{i}: {ev*100:.2f}% (Tích lũy: {cv*100:.2f}%)")

loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=["PC1", "PC2"],
    index=best_feats
)
print("\n=== 4B. BẢNG LOADINGS CỦA PC1 VÀ PC2 ===")
print(np.round(loadings, 3))
