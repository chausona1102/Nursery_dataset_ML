import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dữ liệu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
df = pd.read_csv(url, names=columns)
df = df[~df['class'].isin(['very_recom', 'recommend'])]

# Mã hóa
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("class", axis=1)
y = df["class"]

# Tạo dict lưu accuracy theo từng k
k_range = list(range(3, 30, 2))
k_acc_all_runs = {k: [] for k in k_range}

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        k_acc_all_runs[k].append(acc)

# Tính trung bình accuracy cho từng k
k_avg_acc = {k: np.mean(acc_list) for k, acc_list in k_acc_all_runs.items()}
optimal_k = max(k_avg_acc, key=k_avg_acc.get)
optimal_acc = k_avg_acc[optimal_k]

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(k_avg_acc.keys(), k_avg_acc.values(), marker='o', color='b', linestyle='-', linewidth=1, markersize=6)
plt.xlabel("K Value (Number of Neighbors)")
plt.ylabel("Average Accuracy over 10 runs (%)")
plt.title("KNN: Accuracy vs. K Value (average of 10 runs)")
plt.grid(True)

plt.axvline(optimal_k, color='r', linestyle='--', linewidth=1)
plt.annotate(f'Optimal K = {optimal_k}\nAccuracy = {optimal_acc:.2f}%',
             xy=(optimal_k, optimal_acc),
             xytext=(optimal_k + 2, optimal_acc - 3),
             arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"))

plt.tight_layout()
plt.show()


# Danh sách các metric cần kiểm tra
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

print("So sánh độ chính xác theo từng metric với K=7:")
for metric in metrics:
    model = KNeighborsClassifier(n_neighbors=7, metric=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"- {metric.capitalize():<10}: Accuracy = {acc:.4f}")