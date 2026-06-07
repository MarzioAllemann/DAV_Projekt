import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score


def load_and_split_data(test_size=0.2, random_state=42):
    df = pd.read_csv('Data/df_mice.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, X, y


def train_knn(X_train, y_train):
    pipe_knn = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid_knn = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    grid_knn = GridSearchCV(
        pipe_knn,
        param_grid_knn,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_knn.fit(X_train, y_train)

    best_model = grid_knn.best_estimator_

    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    return best_model, grid_knn.best_params_, grid_knn.best_score_, train_sizes, train_scores, test_scores


def train_rf(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_rf.fit(X_train, y_train)

    best_model = grid_rf.best_estimator_

    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    return best_model, grid_rf.best_params_, grid_rf.best_score_, train_sizes, train_scores, test_scores


def train_rf_with_rfecv(X_train, y_train, min_features_to_select=10):
    param_grid_rf = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_rf.fit(X_train, y_train)
    best_params = grid_rf.best_params_
    
    final_model = RandomForestClassifier(**best_params, random_state=42)
    
    rfecv = RFECV(
        estimator=final_model,
        step=1,
        cv=5,
        scoring='accuracy',
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )
    
    rfecv.fit(X_train, y_train)
    
    X_train_selected = rfecv.transform(X_train)
    
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train_selected, y_train)
    
    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X_train_selected,
        y_train,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    return best_model, best_params, grid_rf.best_score_, train_sizes, train_scores, test_scores, rfecv, X_train_selected


def compare_models_mcnemar(modelA, modelB, X_test, y_test, alpha=0.05):
    y_pred_A = modelA.predict(X_test)
    y_pred_B = modelB.predict(X_test)

    acc_A = accuracy_score(y_test, y_pred_A)
    acc_B = accuracy_score(y_test, y_pred_B)

    correct_A = (y_pred_A == y_test)
    correct_B = (y_pred_B == y_test)

    n00 = np.sum(correct_A & correct_B)
    n01 = np.sum(correct_A & ~correct_B)
    n10 = np.sum(correct_B & ~correct_A)
    n11 = np.sum(~correct_A & ~correct_B)

    contingency = np.array([[n00, n01], [n10, n11]])

    result = mcnemar(contingency, exact=False, correction=True)

    if result.pvalue < alpha:
        better = "Model A" if acc_A > acc_B else "Model B"
        interpretation = f"H0 abgelehnt — {better} ist signifikant besser."
    else:
        interpretation = "H0 nicht abgelehnt — kein signifikanter Unterschied."

    return {
        "acc_A": acc_A,
        "acc_B": acc_B,
        "contingency": contingency,
        "chi2": result.statistic,
        "pvalue": result.pvalue,
        "interpretation": interpretation
    }


def compare_models_mcnemar_with_rfecv(knn_model, rfecv_model, X_test, X_test_selected, y_test, alpha=0.05):
    y_pred_knn = knn_model.predict(X_test)
    y_pred_rfecv = rfecv_model.predict(X_test_selected)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_rfecv = accuracy_score(y_test, y_pred_rfecv)

    correct_knn = (y_pred_knn == y_test)
    correct_rfecv = (y_pred_rfecv == y_test)

    n00 = np.sum(correct_knn & correct_rfecv)
    n01 = np.sum(correct_knn & ~correct_rfecv)
    n10 = np.sum(correct_rfecv & ~correct_knn)
    n11 = np.sum(~correct_knn & ~correct_rfecv)

    contingency = np.array([[n00, n01], [n10, n11]])

    result = mcnemar(contingency, exact=False, correction=True)

    if result.pvalue < alpha:
        better = "KNN" if acc_knn > acc_rfecv else "RFECV-RandomForest"
        interpretation = f"H0 abgelehnt — {better} ist signifikant besser."
    else:
        interpretation = "H0 nicht abgelehnt — kein signifikanter Unterschied."

    return {
        "acc_knn": acc_knn,
        "acc_rfecv": acc_rfecv,
        "contingency": contingency,
        "chi2": result.statistic,
        "pvalue": result.pvalue,
        "interpretation": interpretation
    }