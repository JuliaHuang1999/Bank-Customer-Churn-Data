import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict


def select_feature(df):
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

   # Keep only top features in the original dataset
   X = X.iloc[:, top_N_features]
   # print(X_train.iloc[:, top_N_features].isnull().sum())

   return X, y, kf, n_of_features, X_train, y_train, X_test, y_test, top_N_features