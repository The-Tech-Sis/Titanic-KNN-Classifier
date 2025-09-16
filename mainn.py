# Titanic Survival Prediction with KNN and Model Comparison

# --- Introduction ---
# Objective: Predict survival on the Titanic dataset using KNN and compare with other models.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load Data ---
data = pd.read_csv("Titanic/titanic.csv")
print(data.head())

# --- Preprocessing ---
def preprocess_data(df):
    df = df.copy()
    
    # Drop irrelevant columns
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # Handle missing values
    df.fillna({"Embarked": "S"}, inplace=True)
    
    # Fill missing ages based on Pclass median
    age_fill_map = df.groupby("Pclass")["Age"].median().to_dict()
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)

    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)
    
    return df

processed_data = preprocess_data(data)

# --- Features & Target ---
X = processed_data.drop(columns=["Survived"])
y = processed_data["Survived"]

print(X)
print(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Column Transformation ---
numeric_features = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "FareBin", "AgeBin"]
categorical_features = ["Sex", "Embarked", "Pclass", "IsAlone"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --- Model Pipelines ---
knn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier())
])

logreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# --- Hyperparameter Tuning for KNN ---
param_grid = {
    "model__n_neighbors": range(1, 21),
    "model__metric": ["euclidean", "manhattan", "minkowski"],
    "model__weights": ["uniform", "distance"]
}

knn_grid = GridSearchCV(knn_pipeline, param_grid, cv=5, n_jobs=-1)
knn_grid.fit(X_train, y_train)

best_knn = knn_grid.best_estimator_
print("Best KNN Params:", knn_grid.best_params_)

# --- Train Other Models ---
logreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# --- Evaluation Function ---
def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"\n{name} Performance:")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return acc

# --- Evaluate Models ---
knn_acc = evaluate_model(best_knn, X_test, y_test, "KNN")
logreg_acc = evaluate_model(logreg_pipeline, X_test, y_test, "Logistic Regression")
rf_acc = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")

# --- Accuracy Comparison ---
results = pd.DataFrame({
    "Model": ["KNN", "Logistic Regression", "Random Forest"],
    "Accuracy": [knn_acc, logreg_acc, rf_acc]
})

sns.barplot(data=results, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()