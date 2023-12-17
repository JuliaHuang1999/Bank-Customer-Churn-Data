# importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(action = 'ignore')

from code.data_cleaning import clean_data
from code.feature_selection import select_feature

#importing data
data = pd.read_csv('data/archive/churn_prediction.csv')
cleaned_df = clean_data(data)
select_feature(cleaned_df)
# print(cleaned_df.head())