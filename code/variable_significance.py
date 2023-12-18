import pandas as pd
import scipy.stats as stats

def test_significance(df):
    
    # Handle missing values
    df = df.dropna()

    # Separate categorical and numerical variables
    categorical_vars = df.select_dtypes(include='category').columns
    numeric_vars = df.select_dtypes(include=['number', 'bool']).columns

    # Chi-Square Test for Categorical Variables
    chi_square_results = {}
    for cat_var in categorical_vars:
        df[cat_var] = df[cat_var].astype(str)  # Ensure categorical variables are strings
        cross_tab = pd.crosstab(df[cat_var], df['churn'])
        chi2, p, _, _ = stats.chi2_contingency(cross_tab)
        
        # Add stars based on the level of significance
        stars = ""
        if p < 0.001:
            stars = "（＊＊＊）"
        elif p < 0.01:
            stars = "（＊＊）"
        elif p < 0.05:
            stars = "（＊）"

        chi_square_results[cat_var] = {'Chi2': chi2, 'P-value': p, 'Significance': stars}

    # T-Test for Numeric Variables
    t_test_results = {}
    for num_var in numeric_vars:
        try:
            df[num_var] = pd.to_numeric(df[num_var], errors='raise', downcast='float')  # Convert to numeric, handle non-numeric values
            churned = df[df['churn'] == 1][num_var]
            not_churned = df[df['churn'] == 0][num_var]
            t_stat, p_val = stats.ttest_ind(churned, not_churned, equal_var=False, nan_policy='omit')
            
            # Determine the sign
            sign = "＋" if t_stat > 0 else "－"
            
            # Add stars based on the level of significance
            stars = ""
            if p_val < 0.001:
                stars = "（＊＊＊）"
            elif p_val < 0.01:
                stars = "（＊＊）"
            elif p_val < 0.05:
                stars = "（＊）"

            t_test_results[num_var] = {'T-Statistic': t_stat, 'P-value': p_val, 'Significance': stars, 'Sign': sign}
        except ValueError as e:
            print(f"Error processing {num_var}: {e}")

    # Print results
    # Print results
    print("Chi-Square Test Results:")
    for cat_var, result in chi_square_results.items():
        print(f"{cat_var}: {result['Significance']} {result.get('Sign', '')} ")

    print("\nT-Test Results:")
    for num_var, result in t_test_results.items():
        print(f"{num_var}: {result['Significance']} {result['Sign']} ")

    # # Handle missing values
    # df = df.dropna()

    # # Separate categorical and numerical variables
    # categorical_vars = df.select_dtypes(include='category').columns
    # numeric_vars = df.select_dtypes(include=['number', 'bool']).columns

    # # Chi-Square Test for Categorical Variables
    # chi_square_results = {}
    # for cat_var in categorical_vars:
    #     df[cat_var] = df[cat_var].astype(str)  # Ensure categorical variables are strings
    #     cross_tab = pd.crosstab(df[cat_var], df['churn'])
    #     chi2, p, _, _ = stats.chi2_contingency(cross_tab)
        
    #     # Add stars based on the level of significance
    #     stars = ""
    #     if p < 0.001:
    #         stars = "***"
    #     elif p < 0.01:
    #         stars = "**"
    #     elif p < 0.05:
    #         stars = "*"

    #     chi_square_results[cat_var] = {'Chi2': chi2, 'P-value': p, 'Significance': stars}

    # # T-Test for Numeric Variables
    # t_test_results = {}
    # for num_var in numeric_vars:
    #     try:
    #         df[num_var] = pd.to_numeric(df[num_var], errors='raise', downcast='float')  # Convert to numeric, handle non-numeric values
    #         churned = df[df['churn'] == 1][num_var]
    #         not_churned = df[df['churn'] == 0][num_var]
    #         t_stat, p_val = stats.ttest_ind(churned, not_churned, equal_var=False, nan_policy='omit')
            
    #         # Add stars based on the level of significance
    #         stars = ""
    #         if p_val < 0.001:
    #             stars = "***"
    #         elif p_val < 0.01:
    #             stars = "**"
    #         elif p_val < 0.05:
    #             stars = "*"

    #         t_test_results[num_var] = {'T-Statistic': t_stat, 'P-value': p_val, 'Significance': stars}
    #     except ValueError as e:
    #         print(f"Error processing {num_var}: {e}")

    # # Print results
    # print("Chi-Square Test Results:")
    # for cat_var, result in chi_square_results.items():
    #     print(f"{cat_var}: Chi2 = {result['Chi2']}, P-value = {result['P-value']} {result['Significance']}")

    # print("\nT-Test Results:")
    # for num_var, result in t_test_results.items():
    #     print(f"{num_var}: T-Statistic = {result['T-Statistic']}, P-value = {result['P-value']} {result['Significance']}")
