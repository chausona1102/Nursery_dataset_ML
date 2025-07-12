import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_and_save_knn_euclidean():
    # Bước 1: Tải và xử lý dữ liệu
    columns = [
        'parents', 'has_nurs', 'form', 'children', 'housing',
        'finance', 'social', 'health', 'class'
    ]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    data = pd.read_csv(url, header=None, names=columns)

    # Loại bỏ các lớp ít xuất hiện
    data = data[~data['class'].isin(['very_recom', 'recommend'])]

    # Mã hóa từng cột bằng LabelEncoder và lưu lại
    encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop('class', axis=1)
    y = data['class']

    # Bước 2: Khởi tạo các biến theo dõi
    best_f1 = 0
    best_model = None
    best_encoders = None

    accs, f1_macros, f1_weighteds = [], [], []

    # Bước 3: Huấn luyện và đánh giá trong 10 lần
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )

        model = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        accs.append(acc)
        f1_macros.append(f1_macro)
        f1_weighteds.append(f1_weighted)

        print(f"Lần {i+1}: Accuracy={acc:.4f}, F1 Macro={f1_macro:.4f}, F1 Weighted={f1_weighted:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model
            best_encoders = encoders.copy()

    # Bước 4: In kết quả trung bình
    print("\n== Kết quả trung bình ==")
    print(f"Accuracy trung bình     : {np.mean(accs):.4f}")
    print(f"F1 Macro trung bình     : {np.mean(f1_macros):.4f}")
    print(f"F1 Weighted trung bình  : {np.mean(f1_weighteds):.4f}")

    # Bước 6: Lưu mô hình và encoder
    if best_model:
        joblib.dump(best_model, "knn_euclidean.pkl")
        joblib.dump(best_encoders, "label_encoders_best.pkl")
        print("\nMô hình tốt nhất đã được lưu vào 'knn_euclidean.pkl'")

# Chạy chương trình
if __name__ == "__main__":
    train_and_save_knn_euclidean()