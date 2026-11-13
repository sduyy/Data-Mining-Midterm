import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_prep = pd.read_csv("data/data_preprocessed.csv")

selected_cols = [
    "MembershipTier", "Age", "Recency", "Frequency", "AvgOrderValue",
    "AppTimePerWeek", "DiscountUsageRate", "ReturnRate", "SupportTickets"
]
df_feat = df_prep[selected_cols].copy()
df_feat["MembershipTier"] = df_feat["MembershipTier"].map({"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4})

# Chỉ log1p các biến giảm skewness rõ rệt (bỏ ReturnRate)
for col in ["Recency", "Frequency", "AvgOrderValue"]:
    df_feat[col] = np.log1p(df_feat[col])

X_scaled = StandardScaler().fit_transform(df_feat)
df_prepared = pd.DataFrame(X_scaled, columns=selected_cols)
df_prepared.insert(0, "CustomerID", df_prep["CustomerID"])
df_prepared.to_csv("data_prepared.csv", index=False)
print("Đã cập nhật xong 'data_prepared.csv' (9 đặc trưng chuẩn hóa, bỏ log ReturnRate)!")
