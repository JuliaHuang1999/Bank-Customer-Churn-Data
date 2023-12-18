import pandas as pd
import warnings
warnings.filterwarnings(action = 'ignore')

from code.data_cleaning import clean_data
from code.feature_selection import select_feature
from code.model.xgb_classifier import xgb_classifier
from code.model.decision_tree import decision_tree
from code.model.random_forest import random_forest

# Import data
data = pd.read_csv('data/archive/churn_prediction.csv')

# Clean data
cleaned_df = clean_data(data)
X, y, kf, n_of_features, X_train, y_train, X_test, y_test, top_N_features = select_feature(cleaned_df)

# XGB Classifier
xgb_classifier(X, y, kf, n_of_features)

# Decision tree
decision_tree(X_train, y_train, X_test, y_test, top_N_features)

# Random forest
random_forest(X_train, y_train, X_test, y_test, top_N_features)