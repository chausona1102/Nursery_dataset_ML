import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load và xử lý dữ liệu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
df = pd.read_csv(url, names=columns)
df = df[~df['class'].isin(['very_recom', 'recommend'])]

# Encode các cột dạng text sang số
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('class', axis=1).values
y = df['class'].values

# 2. K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. Khởi tạo kết quả
results = {
    "Decision Tree": [],
    "Random Forest": [],
    "KNN": []
}

# 4. Huấn luyện với KFold
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=15, criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    results["Decision Tree"].append(dt_acc)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    results["Random Forest"].append(rf_acc)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)
    results["KNN"].append(knn_acc)

# 5. In kết quả trung bình
print("Độ chính xác trung bình qua 10 Fold:")
for model in results:
    print(f"{model}: {np.mean(results[model]):.4f}")

# 6. Vẽ biểu đồ cột trung bình độ chính xác của từng mô hình
avg_accuracies = [np.mean(results[m]) for m in results]
model_names = list(results.keys())

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, avg_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0.7, 1.0)
plt.ylabel("Độ chính xác trung bình")
plt.title("So sánh độ chính xác trung bình của các mô hình (10-Fold)\n")
for bar, acc in zip(bars, avg_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{acc:.4f}",
             ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# 8. Hàm vẽ biểu đồ cột từng fold riêng biệt cho từng mô hình
def plot_fold_bar(model_name, accuracies, color):
    plt.figure(figsize=(10, 5))
    folds = [f"Fold {i+1}" for i in range(10)]
    bars = plt.bar(folds, [acc * 100 for acc in accuracies], color=color)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{acc * 100:.2f}",
                 ha='center', fontsize=9)

    plt.ylim(0, 100)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} - 10-Fold Cross Validation Accuracy\n")
    plt.tight_layout()
    plt.show()

# 9. Vẽ từng biểu đồ cột riêng
plot_fold_bar("Decision Tree", results["Decision Tree"], color='skyblue')
plot_fold_bar("Random Forest", results["Random Forest"], color='lightgreen')
plot_fold_bar("KNN", results["KNN"], color='salmon')
