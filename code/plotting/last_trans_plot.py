import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def last_trans_plot(df):
    # Group by the "churn" column
    grouped_data = df.groupby('churn')

    # Plot KDE plots for each group
    plt.figure(figsize=(10, 6))
    for name, group in grouped_data:
        sns.kdeplot(data=group['last_transaction_days_ago'], label=f'Churn: {name}', shade=True)

    # Set x-axis limits to match the actual data range
    plt.xlim(-2, 420)

    # Set plot labels and title
    plt.xlabel('Last Transaction Days Ago')
    plt.ylabel('Density')
    plt.title('Distribution of Time Since Last Transaction (Days), Grouped by Churn Status')

    # Add legend
    plt.legend()

    # Save the word cloud as an image file
    filename = f'./plots/Distribution of Time Since Last Transaction.png'
    plt.savefig(filename, bbox_inches='tight', dpi=1200)

    # Show the plot
    plt.show()