from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Load Breast Cancer Wisconsin dataset
data = load_breast_cancer()

X = data.data
y = data.target


# Random train-test split WITHOUT fixing random_state
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)


# Random Forest WITHOUT fixing random_state
model = RandomForestClassifier(
    n_estimators=100
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("=" * 50)
print("REPRODUCIBILITY EXPERIMENT")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")