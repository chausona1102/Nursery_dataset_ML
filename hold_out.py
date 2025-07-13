import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Tải dữ liệu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
df = pd.read_csv(url, names=columns)
df = df[~df['class'].isin(['very_recom', 'recommend'])]

# Mã hóa
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("class", axis=1)
y = df["class"]
class_labels = label_encoders["class"].classes_

# Khởi tạo kết quả
results = {
    "Decision Tree": [],
    "Random Forest": [],
    "KNN": []
}
conf_matrices = {
    "Decision Tree": np.zeros((len(class_labels), len(class_labels)), dtype=np.float64),
    "Random Forest": np.zeros((len(class_labels), len(class_labels)), dtype=np.float64),
    "KNN": np.zeros((len(class_labels), len(class_labels)), dtype=np.float64)
}

# Huấn luyện 10 lần
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=15, criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    results["Decision Tree"].append(accuracy_score(y_test, dt_preds))
    conf_matrices["Decision Tree"] += confusion_matrix(y_test, dt_preds, labels=range(len(class_labels)))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["Random Forest"].append(accuracy_score(y_test, rf_preds))
    conf_matrices["Random Forest"] += confusion_matrix(y_test, rf_preds, labels=range(len(class_labels)))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    results["KNN"].append(accuracy_score(y_test, knn_preds))
    conf_matrices["KNN"] += confusion_matrix(y_test, knn_preds, labels=range(len(class_labels)))

# Vẽ Confusion Matrix trung bình
for model in conf_matrices:
    avg_cm = conf_matrices[model] / 10
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f"Confusion Matrix trung bình - {model}")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.show()

# Trung bình accuracy
print("Độ chính xác trung bình sau 10 lần chạy:")
for model in results:
    acc = np.mean(results[model])
    print(f"{model}: {acc:.4f}")

# Mô hình tốt nhất
best_model = max(results, key=lambda k: np.mean(results[k]))
print(f"\nThuật toán tốt nhất là: {best_model}")

# Biểu đồ cột độ chính xác
avg_accuracies = [np.mean(results[model]) for model in results]
model_names = list(results.keys())

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, avg_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0.7, 1.0)
plt.ylabel("Độ chính xác trung bình")
plt.title("So sánh độ chính xác của các mô hình (10 lần chạy)")
for bar, acc in zip(bars, avg_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{acc:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()