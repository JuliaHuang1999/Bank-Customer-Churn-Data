import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Group by 'churn' and calculate value counts for 'occupation'
grouped_df = df.groupby('churn')['occupation'].value_counts(normalize=True).unstack(fill_value=0) * 100

# Use a different color palette
sns.set_palette("pastel")

# Plot the bar chart
ax = grouped_df.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set labels and title
ax.set_ylabel('Percentage')
ax.set_xlabel('Churn')
ax.set_title('Occupation distribution by Churn')

# Move legend outside the bars
ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add annotations to display percentage values on the bars (only for values > 1)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if height > 1:
        ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center')

# Save the word cloud as an image file
filename = f'./plots/Occupation distribution by Churn.png'
plt.savefig(filename, bbox_inches='tight', dpi=1200)

# Show the plot
plt.show()
