import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

import numpy as np
import os
import sys

def select_feature(df):

    # # drop relatively irrelevant columns
    # df = df.drop(['host_listings_count', 'host_total_listings_count', 'minimum_nights_avg_ntm','maximum_minimum_nights','minimum_nights','minimum_minimum_nights', 'property_type',
    #             'calculated_host_listings_count_shared_rooms','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','maximum_nights_avg_ntm','cnt_amenities'], axis=1)

    # # convert host_response_time from categorical variable to numerical one
    # df['host_response_time'] = df['host_response_time'].replace({
    #     'within an hour': 1,
    #     'within a few hours': 5,
    #     'within a day': 12,
    #     'a few days or more': 48
    # })

    # df['host_response_time'].value_counts()

    # categorical_cols = [ 'neighbourhood_cleansed',
    #                     # 'property_type',
    #                     'room_type','has_availability']  # add more if needed
    # # categorical_cols = ['host_response_time', 'neighbourhood_cleansed', 'property_type', 'room_type']  # add more if needed
    # for col in categorical_cols:
    #     df[col] = df[col].astype('category')


    ''' column names
    'customer_id', 'vintage', 'age', 'gender', 'dependents', 'occupation',
       'city', 'customer_nw_category', 'branch_code', 'current_balance',
       'previous_month_end_balance', 'average_monthly_balance_prevQ',
       'average_monthly_balance_prevQ2', 'current_month_credit',
       'previous_month_credit', 'current_month_debit', 'previous_month_debit',
       'current_month_balance', 'previous_month_balance', 'churn',
       'last_transaction', 'last_transaction_days_ago'
    '''

    X = df.drop(columns=['customer_id', 'churn', 'last_transaction'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use XGBoost Classifier to Select Top N Features
    model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True)
    model.fit(X_train, y_train)

    # Get feature importances
    xgb_importances = model.feature_importances_

   #  # Plot
   #  xgb.plot_importance(model, max_num_features=8, importance_type='gain')
   #  plt.show()

    # Get the top 10 feature indices
    sorted_idx = np.argsort(xgb_importances)[::-1]
    top_8_features = sorted_idx[:10]
   #  print(X_train.columns[top_8_features])
    
   # Create a k-fold cross-validator object (e.g., 5-fold)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   # Keep only top20 features in the original dataset
    X = X.iloc[:, top_8_features]

   # Initialize the model
    model_xgb = xgb.XGBClassifier(tree_method="hist", enable_categorical=True)

   # Compute cross-validated accuracy scores
    cv_scores = cross_val_score(model_xgb, X, y, cv=kf, scoring='accuracy')

   # Print results
    print('XGBoost hist tree')
    print(f"Accuracy scores for the 5 folds: {cv_scores}")
    print(f"Mean accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard deviation: {np.std(cv_scores):.4f}")

   # XGBoost hist tree
   # Accuracy scores for the 5 folds: [0.85819975 0.84956843 0.85482734 0.85923185 0.85112755]
   # Mean accuracy: 0.8546
   # Standard deviation: 0.0038
