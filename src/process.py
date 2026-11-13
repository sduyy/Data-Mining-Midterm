import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df_raw = pd.read_csv("data/data_raw_noisy.csv")
df_clean_ref = pd.read_csv("data/data_clean.csv")

df_prep = df_raw.copy()

invalid_age = (df_prep["Age"] < 18) | (df_prep["Age"] > 100)
invalid_recency = (df_prep["Recency"] < 1) | (df_prep["Recency"] > 400)
invalid_discount = (df_prep["DiscountUsageRate"] < 0) | (df_prep["DiscountUsageRate"] > 1)
invalid_apptime = (df_prep["AppTimePerWeek"] < 0) | (df_prep["AppTimePerWeek"] > 168)

df_prep.loc[invalid_age, "Age"] = np.nan
df_prep.loc[invalid_recency, "Recency"] = np.nan
df_prep.loc[invalid_discount, "DiscountUsageRate"] = np.nan
df_prep.loc[invalid_apptime, "AppTimePerWeek"] = np.nan

df_prep["Gender"] = df_prep["Gender"].fillna("Unknown")

tier_map = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
df_prep["Tier_num"] = df_prep["MembershipTier"].map(tier_map)

num_features = [
    "Tier_num", "Age", "Recency", "Frequency", "Monetary", 
    "AvgOrderValue", "AppTimePerWeek", "DiscountUsageRate", 
    "ReturnRate", "SupportTickets"
]

scaler_temp = RobustScaler()
scaled_matrix = scaler_temp.fit_transform(df_prep[num_features])

knn_imputer = KNNImputer(n_neighbors=5, weights="distance", metric="nan_euclidean")
imputed_matrix = knn_imputer.fit_transform(scaled_matrix)

df_prep[num_features] = scaler_temp.inverse_transform(imputed_matrix)
df_prep = df_prep.drop(columns=["Tier_num"])

df_prep["Age"] = np.round(df_prep["Age"]).astype(int)
df_prep["Recency"] = np.maximum(1, np.round(df_prep["Recency"])).astype(int)
df_prep["Frequency"] = np.round(df_prep["Frequency"]).astype(int)
df_prep["SupportTickets"] = np.round(df_prep["SupportTickets"]).astype(int)
df_prep["DiscountUsageRate"] = np.round(np.clip(df_prep["DiscountUsageRate"], 0.0, 1.0), 2)
df_prep["AppTimePerWeek"] = np.round(np.maximum(0.0, df_prep["AppTimePerWeek"]), 1)
df_prep["Monetary"] = np.round(np.maximum(0.05, df_prep["Monetary"]), 2)

print(f"Số dòng dồn ở cận dưới (đã tồn tại từ tập dữ liệu nhiễu đầu vào): "
      f"Monetary=0.05 ({(df_prep['Monetary'] == 0.05).sum()} dòng), "
      f"AppTime=0.0 ({(df_prep['AppTimePerWeek'] == 0.0).sum()} dòng)")

df_prep.to_csv("data_preprocessed.csv", index=False)

cols_check = ["Age", "Recency", "Monetary", "AppTimePerWeek", "DiscountUsageRate"]

def build_summary(df_target):
    stats = df_target[cols_check].agg(["mean", "std", "min", "median", "max"]).T
    stats["NaN_count"] = df_target[cols_check].isna().sum()
    return np.round(stats, 4)

print("\n=== TRƯỚC TIỀN XỬ LÝ (data_raw_noisy.csv) ===")
print(build_summary(df_raw))
print("\n=== SAU TIỀN XỬ LÝ (data_preprocessed.csv) ===")
print(build_summary(df_prep))

print("\n=== ĐỐI CHIẾU SAI SỐ KHÔI PHỤC (KNN vs MEDIAN BASELINE) ===")
eval_masks = {
    "Age": df_raw["Age"].isna() | invalid_age,
    "DiscountUsageRate": df_raw["DiscountUsageRate"].isna() | invalid_discount,
    "Recency": invalid_recency,
    "AppTimePerWeek": invalid_apptime
}

for col, mask in eval_masks.items():
    y_true = df_clean_ref.loc[mask, col]
    y_knn = df_prep.loc[mask, col]
    y_med = np.full(mask.sum(), df_raw[col].median())
    
    mae_k, rmse_k = mean_absolute_error(y_true, y_knn), root_mean_squared_error(y_true, y_knn)
    mae_m, rmse_m = mean_absolute_error(y_true, y_med), root_mean_squared_error(y_true, y_med)
    print(f"- {col} ({mask.sum()} ô): KNN [MAE={mae_k:.3f}, RMSE={rmse_k:.3f}] vs "
          f"Median [MAE={mae_m:.3f}, RMSE={rmse_m:.3f}] -> Giảm {(1 - mae_k/mae_m)*100:.1f}% MAE")
