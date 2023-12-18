import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def select_feature(df):
   ''' column names
    'customer_id', 'vintage', 'age', 'gender', 'dependents', 'occupation',
       'city', 'customer_nw_category', 'branch_code', 'current_balance',
       'previous_month_end_balance', 'average_monthly_balance_prevQ',
       'average_monthly_balance_prevQ2', 'current_month_credit',
       'previous_month_credit', 'current_month_debit', 'previous_month_debit',
       'current_month_balance', 'previous_month_balance', 'churn',
       'last_transaction', 'last_transaction_days_ago'
   '''

   X = df.drop(columns=['churn'])
   y = df['churn']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Use XGBoost Classifier to Select Top N Features
   model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True)
   model.fit(X_train, y_train)

   # Get feature importances
   xgb_importances = model.feature_importances_

   # # Plot
   # plt.figure(figsize=(21, 8))
   # xgb.plot_importance(model, max_num_features=14, importance_type='gain')

   # # Save the word cloud as an image file
   # filename = f'./plots/XGB_feature_importance.png'
   # plt.savefig(filename, bbox_inches='tight', dpi=1200)
   # plt.show()

   # Get the top 10 feature indices
   sorted_idx = np.argsort(xgb_importances)[::-1]
   n_of_features = 5
      #14 for random forest (Random Forest Accuracy: 0.8693357597816197) #add diff not higher
      #11 for xgb (XGBoost Hist Tree Mean Accuracy: 0.8573)
      #10 for decision tree (Decision Tree Accuracy: 0.7997094606863991) #add diff not higher
   top_N_features = sorted_idx[:n_of_features]
    
   # Drop rows with NaN values in the 'city' column
   X_train = X_train.dropna(subset=['city', 'occupation', 'last_transaction_days_ago'])
   y_train = y_train[X_train.index]  # Update y_train accordingly

   X_test = X_test.dropna(subset=['city', 'occupation', 'last_transaction_days_ago'])
   y_test = y_test[X_test.index]  # Update y_test accordingly

   # Create a k-fold cross-validator object (e.g., 5-fold)
   kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   # Keep only top20 features in the original dataset
   X = X.iloc[:, top_N_features]
   # print(X_train.iloc[:, top_N_features].isnull().sum())

   # # XGBClassifier
   # xgb_classifier(X, y, kf, n_of_features)

   # # decision tree
   # decision_tree(X_train, y_train, X_test, y_test, top_N_features)

   # # random forest
   # random_forest(X_train, y_train, X_test, y_test, top_N_features)

   return 


from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

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


def random_forest(X_train, y_train, X_test, y_test, top_N_features):
   # Select the top features
   selected_features = X_train.columns[top_N_features]

   random_forest_model = RandomForestClassifier(n_estimators=300, random_state=42)
   random_forest_model.fit(X_train[selected_features], y_train)
   y_pred = random_forest_model.predict(X_test[selected_features])
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

   # # Initialize the Random Forest model within a pipeline
   # rf_model = Pipeline(steps=[
   #    ('preprocessor', preprocessor),
   #    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
   # ])

   # # Train the Random Forest model on the entire training set
   # rf_model.fit(X_train[selected_features], y_train)

   # # Make predictions on the test set
   # y_pred_rf = rf_model.predict(X_test[selected_features])

   # # Evaluate performance
   # accuracy_rf = accuracy_score(y_test, y_pred_rf)

   # print(f"Random Forest Accuracy: {accuracy_rf}")
   # print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
   # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))