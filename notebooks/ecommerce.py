# %% 1. Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

os.makedirs("data/processed", exist_ok=True)
RAW = "data/raw/data.csv"

# %% 2. load
df = pd.read_csv(RAW, encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
print(df.shape)
print(df.isna().sum())
# %% 3. Data Cleaning
n_raw = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]      # membuang transaksi yang dibatalkan
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]            # membuang qty/harga tidak valid
non_product = r"^(POST|D|M|DOT|C2|BANK CHARGES|CRUK|S|B|AMAZONFEE|PADS|gift.*)$"
df = df[~df["StockCode"].astype(str).str.match(non_product, case=False)]

df["Description"] = df["Description"].str.strip().str.upper()
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDay"] = df["InvoiceDate"].dt.normalize()
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
print(f"Baris: {n_raw:,} -> {len(df):,}")

# %% 4. EDA
monthly = df.groupby("YearMonth").agg(
    Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
monthly["AOV"] = monthly["Revenue"] / monthly["Orders"]
print(monthly.round(2))

country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
print((country / country.sum()).head(10).round(3))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
monthly["Revenue"].plot(kind="bar", ax=ax[0], title="Monthly Revenue")
country.head(10).plot(kind="barh", ax=ax[1], title="Top 10 Country by Revenue").invert_yaxis()
plt.tight_layout(); plt.show()

# %% 5. RFM
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
).reset_index()

feat = ["Recency", "Frequency", "Monetary"]
X = np.log1p(rfm[feat])
for c in feat:                                   # outlier treatment IQR (clipping)
    q1, q3 = X[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    X[c] = X[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
Xs = StandardScaler().fit_transform(X)

# %% 6. KMeans clustering
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertia, marker="o"); ax[0].set_title("Elbow")
ax[1].plot(ks, sil, marker="o"); ax[1].set_title("Silhouette")
plt.show()


# %% 7. K-Means final (k = 3 : champion, loyal, at risk)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
rfm["Cluster"] = km.labels_

# skor centroid: recency rendah baik, frequency & monetary tinggi baik
c = km.cluster_centers_
score = -c[:, 0] + c[:, 1] + c[:, 2]
order = np.argsort(score)[::-1]
label_map = {order[0]: "Champions", order[1]: "Loyal", order[2]: "At Risk"}
rfm["Segment"] = rfm["Cluster"].map(label_map)

seg = rfm.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum"),
)
seg["CustomerShare"] = seg["Customers"] / seg["Customers"].sum()
seg["RevenueShare"] = seg["TotalRevenue"] / seg["TotalRevenue"].sum()
print(seg.round(3))   # catat angka Champions untuk README

# %% 8. Market Basket (Apriori, pasar UK, top 150 produk)
uk = df[df["Country"] == "United Kingdom"]
top_items = uk["Description"].value_counts().head(150).index
b = uk[uk["Description"].isin(top_items)]
basket = b.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack(fill_value=0) > 0

freq = apriori(basket, min_support=0.015, use_colnames=True)
try:
    rules = association_rules(freq, num_itemsets=len(basket), metric="lift", min_threshold=1.2)
except TypeError:
    rules = association_rules(freq, metric="lift", min_threshold=1.2)

rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]] \
    .sort_values("lift", ascending=False)
print(rules.head(10))

# %% 9. save result
cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "InvoiceDay", "UnitPrice", "CustomerID", "Country", "Revenue"]
df[cols].to_csv("data/processed/transactions_clean.csv", index=False, encoding="utf-8-sig")
rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]] \
    .to_csv("data/processed/rfm_segments.csv", index=False, encoding="utf-8-sig")
rules.to_csv("data/processed/basket_rules.csv", index=False, encoding="utf-8-sig")
