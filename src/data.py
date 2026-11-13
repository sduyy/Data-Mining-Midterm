import numpy as np
import pandas as pd

np.random.seed(24022313)

N = 260
counts = [39, 91, 91, 39]
segments = np.repeat([0, 1, 2, 3], counts)

data_rows = []
for i, seg in enumerate(segments):
    cust_id = f"CUST_{i+1:03d}"
    gender = np.random.choice(["Nam", "Nữ", "Khác"], p=[0.46, 0.50, 0.04])
    
    if seg == 0:
        age = int(np.clip(np.random.normal(38, 7), 24, 62))
        tier = np.random.choice(["Gold", "Platinum"], p=[0.4, 0.6])
        recency = int(np.clip(np.random.exponential(8) + 1, 1, 25))
        freq = int(np.clip(np.random.normal(36, 7), 22, 55))
        aov = np.round(np.random.normal(1.45, 0.3), 2)
        monetary = np.round(freq * aov * np.random.uniform(0.95, 1.05), 2)
        app_time = np.round(np.clip(np.random.normal(6.5, 1.8), 2.0, 12.0), 1)
        discount = np.round(np.clip(np.random.normal(0.20, 0.08), 0.0, 0.45), 2)
        ret_rate = np.round(np.clip(np.random.normal(0.03, 0.02), 0.0, 0.10), 2)
        tickets = int(np.random.poisson(0.5))
        
    elif seg == 1:
        age = int(np.clip(np.random.normal(24, 4), 18, 36))
        tier = np.random.choice(["Bronze", "Silver", "Gold"], p=[0.3, 0.5, 0.2])
        recency = int(np.clip(np.random.normal(18, 10), 1, 45))
        freq = int(np.clip(np.random.normal(24, 6), 12, 42))
        aov = np.round(np.clip(np.random.normal(0.22, 0.06), 0.08, 0.40), 2)
        monetary = np.round(freq * aov * np.random.uniform(0.95, 1.05), 2)
        app_time = np.round(np.clip(np.random.normal(13.5, 2.5), 7.0, 22.0), 1)
        discount = np.round(np.clip(np.random.normal(0.84, 0.08), 0.60, 0.99), 2)
        ret_rate = np.round(np.clip(np.random.normal(0.12, 0.05), 0.0, 0.30), 2)
        tickets = int(np.random.poisson(1.5))
        
    elif seg == 2:
        age = int(np.clip(np.random.normal(32, 8), 20, 58))
        tier = np.random.choice(["Bronze", "Silver"], p=[0.55, 0.45])
        recency = int(np.clip(np.random.normal(55, 20), 15, 110))
        freq = int(np.clip(np.random.normal(9, 3), 4, 18))
        aov = np.round(np.clip(np.random.normal(0.65, 0.2), 0.25, 1.30), 2)
        monetary = np.round(freq * aov * np.random.uniform(0.95, 1.05), 2)
        app_time = np.round(np.clip(np.random.normal(4.2, 1.5), 1.0, 9.0), 1)
        discount = np.round(np.clip(np.random.normal(0.45, 0.15), 0.10, 0.75), 2)
        ret_rate = np.round(np.clip(np.random.normal(0.07, 0.04), 0.0, 0.20), 2)
        tickets = int(np.random.poisson(0.8))
        
    else:
        age = int(np.clip(np.random.normal(30, 9), 18, 60))
        tier = np.random.choice(["Bronze", "Silver"], p=[0.8, 0.2])
        recency = int(np.clip(np.random.normal(230, 45), 140, 350))
        freq = int(np.clip(np.random.normal(2.5, 1.2), 1, 6))
        aov = np.round(np.clip(np.random.normal(0.50, 0.25), 0.15, 1.20), 2)
        monetary = np.round(freq * aov * np.random.uniform(0.95, 1.05), 2)
        app_time = np.round(np.clip(np.random.normal(1.2, 0.7), 0.1, 3.5), 1)
        discount = np.round(np.clip(np.random.normal(0.35, 0.2), 0.0, 0.80), 2)
        ret_rate = np.round(np.clip(np.random.normal(0.25, 0.10), 0.05, 0.55), 2)
        tickets = int(np.random.poisson(2.2))

    data_rows.append([
        cust_id, age, gender, tier, recency, freq, monetary, 
        aov, app_time, discount, ret_rate, tickets
    ])

columns = [
    "CustomerID", "Age", "Gender", "MembershipTier", "Recency", 
    "Frequency", "Monetary", "AvgOrderValue", "AppTimePerWeek", 
    "DiscountUsageRate", "ReturnRate", "SupportTickets"
]

df_clean = pd.DataFrame(data_rows, columns=columns)
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
df_clean["CustomerID"] = [f"CUST_{i+1:03d}" for i in range(N)]

df_clean.to_csv("data_clean.csv", index=False)

df_noisy = df_clean.copy()

df_noisy["AppTimePerWeek"] = np.round(np.maximum(0, df_noisy["AppTimePerWeek"] + np.random.normal(0, 1.5, N)), 1)
df_noisy["Monetary"] = np.round(np.maximum(0.05, df_noisy["Monetary"] + np.random.normal(0, 1.2, N)), 2)
df_noisy["Recency"] = np.maximum(1, (df_noisy["Recency"] + np.random.normal(0, 5, N)).astype(int))

miss_age_idx = np.random.choice(N, size=20, replace=False)      
miss_disc_idx = np.random.choice(N, size=26, replace=False)    
miss_gender_idx = np.random.choice(N, size=14, replace=False) 

df_noisy.loc[miss_age_idx, "Age"] = np.nan
df_noisy.loc[miss_disc_idx, "DiscountUsageRate"] = np.nan
df_noisy.loc[miss_gender_idx, "Gender"] = np.nan

outlier_indices = np.random.choice(N, size=14, replace=False)

df_noisy.loc[outlier_indices[0:3], "Age"] = [148, -4, 162]
df_noisy.loc[outlier_indices[3:5], "Recency"] = [4500, 9999]
df_noisy.loc[outlier_indices[5:7], "DiscountUsageRate"] = [1.85, -0.35]
df_noisy.loc[outlier_indices[7:9], "AppTimePerWeek"] = [185.0, 210.5]

whale_idx = outlier_indices[9:14]
df_noisy.loc[whale_idx, "MembershipTier"] = "Platinum"
df_noisy.loc[whale_idx, "Frequency"] = [98, 112, 125, 105, 130]
df_noisy.loc[whale_idx, "AvgOrderValue"] = [1.95, 2.10, 1.88, 2.25, 2.05]
df_noisy.loc[whale_idx, "Monetary"] = np.round(df_noisy.loc[whale_idx, "Frequency"] * df_noisy.loc[whale_idx, "AvgOrderValue"], 2)
df_noisy.loc[whale_idx, "Recency"] = [1, 2, 1, 3, 2]

df_noisy.to_csv("data/data_raw_noisy.csv", index=False)

print("Đã xuất thành công 'data_clean.csv' và 'data_raw_noisy.csv'!")
print("Số lượng giá trị thiếu trên từng cột:\n", df_noisy.isnull().sum()[df_noisy.isnull().sum() > 0])
print("Các dòng Outliers được tiêm vào (index):", sorted(outlier_indices.tolist()))
