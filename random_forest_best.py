import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_best_rf():
    # Tải dữ liệu
    columns = [
        'parents', 'has_nurs', 'form', 'children', 'housing',
        'finance', 'social', 'health', 'class'
    ]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    data = pd.read_csv(url, header=None, names=columns)

    # Loại bỏ các lớp ít xuất hiện
    data = data[~data['class'].isin(['very_recom', 'recommend'])]

    # Encode dữ liệu
    encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop('class', axis=1)
    y = data['class']

    best_acc = 0
    best_model = None
    best_encoders = None
    accuracies = []

    # Huấn luyện và đánh giá qua 10 lần
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Lần {i+1}: Accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_encoders = encoders.copy()

    print(f"\nTrung bình Accuracy sau 10 lần chạy: {sum(accuracies)/len(accuracies):.4f}")

    # Lưu mô hình tốt nhất
    if best_model is not None:
        joblib.dump(best_model, "random_forest_best.pkl")
        joblib.dump(best_encoders, "label_encoders_best.pkl")
        print("Mô hình Random Forest tốt nhất đã được lưu vào 'random_forest_best.pkl'")

if __name__ == "__main__":
    train_and_save_best_rf()