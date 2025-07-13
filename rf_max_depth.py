import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load và xử lý dữ liệu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
df = pd.read_csv(url, names=columns)
df = df[~df['class'].isin(['very_recom', 'recommend'])]

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("class", axis=1)
y = df["class"]

# Các siêu tham số
n_values = [50, 100, 200]
max_depth_values = [5, 10, 15, 20]

test_accuracies = []
x_labels = []

# Huấn luyện và tính độ chính xác trung bình 10 lần
for n_estimators in n_values:
    for max_depth in max_depth_values:
        acc_list = []
        for seed in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            acc_list.append(acc)

        avg_acc = np.mean(acc_list)
        test_accuracies.append(avg_acc)
        x_labels.append(f"{n_estimators}-{max_depth}")

# Tìm điểm có độ chính xác cao nhất
max_acc = max(test_accuracies)
max_idx = test_accuracies.index(max_acc)
best_label = x_labels[max_idx]

# Vẽ biểu đồ và đánh dấu điểm tốt nhất
plt.figure(figsize=(12, 6))
plt.plot(range(len(test_accuracies)), test_accuracies, marker='o', color='darkorange', label='Avg Test Accuracy')
plt.xticks(range(len(x_labels)), x_labels, rotation=45)
plt.xlabel('n_estimators - max_depth')
plt.ylabel('Accuracy (%)')
plt.title('Random Forest Accuracy vs. n_estimators & max_depth')
plt.grid(True)

# Đánh dấu điểm tốt nhất
plt.scatter(max_idx, max_acc, color='red', zorder=5)
plt.annotate(f"Best: {best_label}\n{max_acc:.2f}%", 
             xy=(max_idx, max_acc), 
             xytext=(max_idx, max_acc+1),
             ha='center',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.legend()
plt.tight_layout()
plt.show()

# In ra kết quả tốt nhất
print(f"Độ chính xác cao nhất: {max_acc:.2f}% với n_estimators={best_label}")