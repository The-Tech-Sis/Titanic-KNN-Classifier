#imports

import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns 

data = pd.read_csv("Titanic/titanic.csv")
data.info()
print(data.isnull().sum())
# print(data)


# Data Cleaning and Feature Engineering
def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    #df["Fare"] = df(["Fare"].median("Fare"), inplace=True)
    df.fillna({"Embarked": "S"}, inplace=True)
    df.drop(columns=["Embarked"], inplace=True)

    fill_missing_ages(df)
  
  
    # Convert Gender
    df["Sex"] = df["Sex"].map({"male":1, "female":0})
     
    
    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)
    
    return df


# Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
            
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row ["Age"], axis=1)
    
data = preprocess_data(data)
print(data.isnull().sum())

# Create Features / Target Variables (Make Flashcrads)
x = data.drop(columns=["Survived"])
y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# ML Preprocessing
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Hyperparameter Tuning - KNN
def tune_model(x_train, y_train):
    param_grid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }
    
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, error_score='raise')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_


best_model = tune_model(x_train, y_train)


#Pediction and Evaluation
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)
    accuracy= accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confustion Matrix: ')
print(matrix)