import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

# Khởi tạo danh sách độ sâu
max_depths = [10, 15, 20, 25]

# Mảng lưu độ chính xác trung bình
gini_acc = []
entropy_acc = []

# Chạy với 10 lần train/test khác nhau
for depth in max_depths:
    gini_scores = []
    entropy_scores = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        # Gini
        clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        clf_gini.fit(X_train, y_train)
        y_pred_gini = clf_gini.predict(X_test)
        acc_gini = accuracy_score(y_test, y_pred_gini) * 100
        gini_scores.append(acc_gini)

        # Entropy
        clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        clf_entropy.fit(X_train, y_train)
        y_pred_entropy = clf_entropy.predict(X_test)
        acc_entropy = accuracy_score(y_test, y_pred_entropy) * 100
        entropy_scores.append(acc_entropy)

    # Lưu trung bình độ chính xác sau 10 lần
    gini_acc.append(np.mean(gini_scores))
    entropy_acc.append(np.mean(entropy_scores))

# Vẽ biểu đồ kết quả
plt.figure(figsize=(8, 6))
plt.plot(max_depths, gini_acc, label='Gini', marker='o')
plt.plot(max_depths, entropy_acc, label='Entropy', marker='s')

plt.xlabel("Max Depth")
plt.ylabel("Average Accuracy over 10 runs (%)")
plt.title("Decision Tree: Gini vs Entropy Accuracy theo Max Depth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()