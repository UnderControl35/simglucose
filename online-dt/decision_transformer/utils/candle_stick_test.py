import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Simulate data based on the plot's appearance
# Each method (BB-CD, BB, PID, PID-IF, PPO-RNN) has percentages for each state
np.random.seed(42)  # For reproducibility

# Simulated data: 10 samples per method per state
data = {
    'Method': [],
    'State': [],
    'Percentage': []
}

methods = ['BB-CD', 'BB', 'PID', 'PID-IF', 'PPO-RNN']
states = ['Euglycemia', 'Hyperglycemia', 'Hypoglycemia']

# Approximate percentages based on the plot
for method in methods:
    for state in states:
        if state == 'Euglycemia':
            if method == 'BB-CD':
                percentages = np.random.normal(65, 5, 10)  # Mean ~65%
            elif method == 'BB':
                percentages = np.random.normal(70, 5, 10)  # Mean ~70%
            elif method == 'PID':
                percentages = np.random.normal(68, 5, 10)  # Mean ~68%
            elif method == 'PID-IF':
                percentages = np.random.normal(72, 5, 10)  # Mean ~72%
            else:  # PPO-RNN
                percentages = np.random.normal(75, 5, 10)  # Mean ~75%
        elif state == 'Hyperglycemia':
            if method == 'BB-CD':
                percentages = np.random.normal(35, 5, 10)  # Mean ~35%
            elif method == 'BB':
                percentages = np.random.normal(25, 5, 10)  # Mean ~25%
            elif method == 'PID':
                percentages = np.random.normal(30, 5, 10)  # Mean ~30%
            elif method == 'PID-IF':
                percentages = np.random.normal(20, 5, 10)  # Mean ~20%
            else:  # PPO-RNN
                percentages = np.random.normal(15, 5, 10)  # Mean ~15%
        else:  # Hypoglycemia
            if method == 'BB-CD':
                percentages = np.random.normal(0, 1, 10)  # Mean ~0%
            elif method == 'BB':
                percentages = np.random.normal(5, 2, 10)  # Mean ~5%
            elif method == 'PID':
                percentages = np.random.normal(2, 1, 10)  # Mean ~2%
            elif method == 'PID-IF':
                percentages = np.random.normal(8, 2, 10)  # Mean ~8%
            else:  # PPO-RNN
                percentages = np.random.normal(10, 2, 10)  # Mean ~10%
        
        # Ensure percentages are between 0 and 100
        percentages = np.clip(percentages, 0, 100)
        
        data['Method'].extend([method] * 10)
        data['State'].extend([state] * 10)
        data['Percentage'].extend(percentages)

# Create DataFrame
df = pd.DataFrame(data)

# Define colors matching the plot
colors = {
    'BB-CD': '#66C2A5',    # Teal
    'BB': '#8DA0CB',       # Purple
    'PID': '#FC8D62',      # Pink
    'PID-IF': '#E78AC3',   # Orange
    'PPO-RNN': '#A6D854'   # Yellow
}

# Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create boxplot
sns.boxplot(x='State', y='Percentage', hue='Method', data=df,
            palette=colors, width=0.8, showfliers=False)

# Overlay individual points (outliers)
sns.stripplot(x='State', y='Percentage', hue='Method', data=df,
              palette=colors, dodge=True, size=4, alpha=0.6, jitter=True)

# Customize plot
plt.xlabel('Glycemic State', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.title('Percentage of Time in Glycemic States by Method', fontsize=14, pad=20)

# Adjust legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:5], labels[:5], title='Method', loc='upper right')

plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics:")
print(df.groupby(['State', 'Method'])['Percentage'].describe())