import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Bước 1: Tải và tiền xử lý dữ liệu
columns = [
    'parents', 'has_nurs', 'form', 'children', 'housing',
    'finance', 'social', 'health', 'class'
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
data = pd.read_csv(url, header=None, names=columns)

# Gán label encoder cho từng cột
encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop("class", axis=1)
y = data["class"]


f1_scores = []
estimators = range(50, 350, 50)
def find_best_n_estimators(X, y):
    for n in estimators:
        f1_list = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            model = RandomForestClassifier(n_estimators=n, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            f1_list.append(f1)
        f1_scores.append(np.mean(f1_list))
        best_f1 = max(f1_scores)
        print(f" F1-score tốt nhất: {best_f1:.4f}")
    # In giá trị tốt nhất
    best_n = estimators[np.argmax(f1_scores)]
    print(f" n_estimators tốt nhất: {best_n}")

# Vẽ biểu đồ
# plt.plot(estimators, f1_scores, marker='o')
# plt.xlabel('n_estimators')
# plt.ylabel('F1-score (macro)')
# plt.title('Chọn n_estimators tốt nhất')
# plt.grid(True)
# plt.show()

def test_find_best_n_estimators():
    find_best_n_estimators(X,y)

if __name__ == "__main__":
    find_best_n_estimators(X, y)