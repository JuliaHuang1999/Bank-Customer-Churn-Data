import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def xgb_classifier(X, y, kf, n_of_features):
    # Initialize the model
    model_xgb = xgb.XGBClassifier(tree_method="hist", enable_categorical=True)

    # Compute cross-validated predictions
    y_pred = cross_val_predict(model_xgb, X, y, cv=kf)

    # Compute cross-validated accuracy scores
    cv_scores = cross_val_score(model_xgb, X, y, cv=kf, scoring='accuracy')

    # Print results
    #  print('XGBoost hist tree')
    #  print(f"Number of features: {n_of_features}")
    #  print(f"Accuracy scores for the 5 folds: {cv_scores}")
    
    print(f"XGBoost Hist Tree Mean Accuracy: {np.mean(cv_scores):.4f}")
    #  print(f"Standard deviation: {np.std(cv_scores):.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y, y_pred))


   # XGBoost hist tree
   # Number of features: 9
   # Accuracy scores for the 5 folds: [0.85784745 0.85238682 0.8534179  0.86381254 0.85007047]
   # Mean accuracy: 0.8555
   # Standard deviation: 0.0049
    