import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

def train_and_save_best_rf():
    # Tải dữ liệu
    columns = [
        'parents', 'has_nurs', 'form', 'children', 'housing',
        'finance', 'social', 'health', 'class'
    ]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    data = pd.read_csv(url, header=None, names=columns)

    data = data[~data['class'].isin(['very_recom', 'recommend'])]

    # Encode các cột
    encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop('class', axis=1)
    y = data['class']

    best_f1 = 0
    best_model = None
    best_encoders = None
    accuracies = []
    f1_macros = []
    f1_weighteds = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )
        model = RandomForestClassifier(
            max_depth=15,
            n_estimators=200,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = model.score(X_test, y_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        accuracies.append(acc)
        f1_macros.append(f1_macro)
        f1_weighteds.append(f1_weighted)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model
            best_encoders = encoders.copy()
        print(f"Lần {i+1}: acc={acc:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")

    print(f"\nTrung bình acc: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Trung bình f1_macro: {sum(f1_macros)/len(f1_macros):.4f}")
    print(f"Trung bình f1_weighted: {sum(f1_weighteds)/len(f1_weighteds):.4f}")

    if best_model is not None:
        joblib.dump(best_model, "random_forest_best.pkl")
        joblib.dump(best_encoders, "label_encoders_best.pkl")
        print("Đã lưu mô hình Random Forest tốt nhất vào random_forest_best.pkl")

# test ham train_and_save_best_rf()
def test_train_and_save_best_rf():
    # Tải dữ liệu
    columns = [
        'parents', 'has_nurs', 'form', 'children', 'housing',
        'finance', 'social', 'health', 'class'
    ]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    data = pd.read_csv(url, header=None, names=columns)

    # Encode các cột
    encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop('class', axis=1)
    y = data['class']

    best_f1 = 0
    best_model = None
    best_encoders = None
    accuracies = []
    f1_macros = []
    f1_weighteds = []

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
        acc = model.score(X_test, y_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        accuracies.append(acc)
        f1_macros.append(f1_macro)
        f1_weighteds.append(f1_weighted)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model
            best_encoders = encoders.copy()
        print(f"Lần {i+1}: acc={acc:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")

    print(f"\nTrung bình acc: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Trung bình f1_macro: {sum(f1_macros)/len(f1_macros):.4f}")
    print(f"Trung bình f1_weighted: {sum(f1_weighteds)/len(f1_weighteds):.4f}")

    if best_model is not None:
        joblib.dump(best_model, "random_forest_best.pkl")
        joblib.dump(best_encoders, "label_encoders_best.pkl")
        print("Đã lưu mô hình Random Forest tốt nhất vào random_forest_best.pkl")


if __name__ == "__main__":
    train_and_save_best_rf()
