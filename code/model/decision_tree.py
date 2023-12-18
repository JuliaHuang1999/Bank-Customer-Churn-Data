import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def decision_tree(X_train, y_train, X_test, y_test, top_N_features):
   # Select the top features
   selected_features = X_train.columns[top_N_features]

   # Create a decision tree classifier
   clf = DecisionTreeClassifier()

   # Train the classifier
   clf.fit(X_train[selected_features], y_train)

   # Make predictions on the test set
   y_pred = clf.predict(X_test[selected_features])

   # Evaluate the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy}")
   print("\nClassification Report:\n", classification_report(y_test, y_pred))


   # # Select categorical columns
   # categorical_cols = ['occupation']

   # # Select the remaining non-categorical columns
   # non_categorical_cols = [col for col in selected_features if col not in categorical_cols]

   # # Create transformers for categorical and non-categorical columns
   # categorical_transformer = Pipeline(steps=[
   #    ('onehot', OneHotEncoder(handle_unknown='ignore'))
   # ])

   # preprocessor = ColumnTransformer(
   #    transformers=[
   #       ('cat', categorical_transformer, categorical_cols),
   #       ('num', 'passthrough', non_categorical_cols)
   #    ])

   # # Initialize the Decision Tree model within a pipeline
   # dt_model = Pipeline(steps=[
   #    ('preprocessor', preprocessor),
   #    ('classifier', DecisionTreeClassifier(random_state=42))
   # ])

   # # Train the Decision Tree model
   # dt_model.fit(X_train[selected_features], y_train)

   # # Make predictions on the test set
   # y_pred_dt = dt_model.predict(X_test[selected_features])

   # # Evaluate performance
   # accuracy_dt = accuracy_score(y_test, y_pred_dt)

   # print(f"Decision Tree Accuracy: {accuracy_dt}")
   # print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))
   # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

   return