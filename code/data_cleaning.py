# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings(action = 'ignore')

def clean_data(data):

    # converting int to category data type
    data['churn'] = data['churn'].astype('category')
    data['branch_code'] = data['branch_code'].astype('category')
    data['customer_nw_category'] = data['customer_nw_category'].astype('category')

    # converting "dependents" and "city" to their respective types
    data['dependents'] = data['dependents'].astype('Int64')
    data['city'] = data['city'].astype('category')

    # typecasting "gender" and "occupation" to category type
    data['gender'] = data['gender'].astype('category')
    data['occupation'] = data['occupation'].astype('category')


    ## Transform 'last_transaction' from date to int (days to the latest transcation date)
    data['last_transaction'] = pd.to_datetime(data['last_transaction'], errors='coerce')

    # Find the latest date
    latest_date = data['last_transaction'].max()

    # Calculate the difference in days and transform the data, leaving NaT as NaN
    data['last_transaction_days_ago'] = (latest_date - data['last_transaction']).dt.days
    data['last_transaction_days_ago'] = data['last_transaction_days_ago'].astype('Int64')


    # # standard deviation factor
    # factor = 3

    # # copying current_month
    # cm_data = data['current_balance','current_month_credit','current_month_debit','current_month_balance']

    # # filtering using standard deviation (not considering obseravtions > 3* standard deviation)
    # cm_data = cm_data[cm_data['current_balance'] < factor*cm_data['current_balance'].std()]
    # cm_data = cm_data[cm_data['current_month_credit'] < factor*cm_data['current_month_credit'].std()]
    # cm_data = cm_data[cm_data['current_month_debit'] < factor*cm_data['current_month_debit'].std()]
    # cm_data = cm_data[cm_data['current_month_balance'] < factor*cm_data['current_month_balance'].std()]

    # # checking how many points removed
    # len(data), len(cm_data)

    # # finding number of missing values in every variable
    # print(data.isnull().sum())

    # data = data.drop(columns = ['last_transaction'])

    # # first 5 instances using "head()" function
    # data.head().to_clipboard()
    return data