import pandas as pd
import warnings
warnings.filterwarnings(action = 'ignore')

def clean_data(data):

    # converting int to category data type
    data['churn'] = data['churn'].astype('bool')
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


    data['credit_difference'] = data['current_month_credit'] - data['previous_month_credit']
    data['debit_difference'] = data['current_month_debit'] - data['previous_month_debit']
    data['balance_difference'] = data['current_month_balance'] - data['previous_month_balance']

    # standard deviation factor
    factor = 3

    # filtering using standard deviation (not considering obseravtions > 3* standard deviation)
    data = data[data['current_balance'] < factor*data['current_balance'].std()]
    data = data[data['current_month_credit'] < factor*data['current_month_credit'].std()]
    data = data[data['current_month_debit'] < factor*data['current_month_debit'].std()]
    data = data[data['current_month_balance'] < factor*data['current_month_balance'].std()]

    # checking how many points removed
    # print(len(data), len(data)) #28382 27113

    # # finding number of missing values in every variable
    # print(data.isnull().sum())

    data = data.drop(columns = ['last_transaction', 'customer_id'])

    return data