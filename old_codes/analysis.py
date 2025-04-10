import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from simglucose.analysis.risk import risk_index

# Constants
DATA_PATH = '/home/guleserhocam/VS_Projects/simglucose/dataset/T1DatasetAnalysis/BB/output_noisy/adolescent#001/adolescent#001_combined_seed.pkl'
HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180
SAMPLE_SIZE = 100  # Number of data points per sample for distribution

def load_data(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as handle:
            return pkl.load(handle)
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def calculate_risk_metrics(observations, horizon=1):
    """Calculate risk metrics (LBGI, HBGI, risk index) for a list of observations."""
    observations = np.array(observations).flatten()
    risk_i, lbgi, hbgi = [], [], []
    for obs in observations:
        x, y, z = risk_index([obs], horizon=horizon)
        lbgi.append(x)
        hbgi.append(y)
        risk_i.append(z)
    return risk_i, lbgi, hbgi

def process_all_trajectories(data, hypo_threshold, hyper_threshold):
    """Process all trajectories as a single dataset and return a DataFrame."""
    all_observations = []
    for traj in data:
        observations = np.array(traj['observations']).flatten()
        all_observations.extend(observations)
    
    df = pd.DataFrame({'observations': all_observations})
    risk_i, lbgi, hbgi = calculate_risk_metrics(df['observations'])
    df['risk_i'] = risk_i
    df['LBGI'] = lbgi
    df['HBGI'] = hbgi
    
    # Categorize glycemic states
    def categorize_state(glucose):
        if glucose < hypo_threshold:
            return 'Hypoglycemia'
        elif glucose <= hyper_threshold:
            return 'Euglycemia'
        return 'Hyperglycemia'
    
    df['State'] = df['observations'].apply(categorize_state)
    return df

def prepare_glycemic_state_data(df, hypo_threshold, hyper_threshold, sample_size):
    """Prepare data for box plot by calculating time in each glycemic state across random samples."""
    data = {'State': [], 'Percentage': []}
    observations = df['observations'].values
    
    n_samples = max(1, len(observations) // sample_size)
    for _ in range(n_samples):
        sample = np.random.choice(observations, size=min(sample_size, len(observations)), replace=False)
        hypo_percent = (sample < hypo_threshold).sum() / len(sample) * 100
        hyper_percent = (sample > hyper_threshold).sum() / len(sample) * 100
        eugly_percent = 100 - hypo_percent - hyper_percent
        
        data['State'].extend(['Hypoglycemia', 'Euglycemia', 'Hyperglycemia'])
        data['Percentage'].extend([hypo_percent, eugly_percent, hyper_percent])
    
    return pd.DataFrame(data)

def prepare_risk_candlestick_data(df):
    """Prepare candlestick data for risk index grouped by glycemic state."""
    candlestick_data = pd.DataFrame({
        'Open': df.groupby('State')['risk_i'].first(),
        'High': df.groupby('State')['risk_i'].max(),
        'Low': df.groupby('State')['risk_i'].min(),
        'Close': df.groupby('State')['risk_i'].last()
    }).reindex(['Hypoglycemia', 'Euglycemia', 'Hyperglycemia']).dropna()
    return candlestick_data

def plot_combined_charts(df, hypo_threshold, hyper_threshold, sample_size):
    """Create a figure with two subplots: box plot and candlestick chart."""
    # Define colors for each state
    colors = {
        'Hypoglycemia': '#8DA0CB',    # Purple
        'Euglycemia': '#66C2A5',      # Teal
        'Hyperglycemia': '#FC8D62'    # Pink
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    
    # Set figure title
    fig.suptitle('Figure 1', fontsize=16, y=1.05)
    
    # Prepare data for box plot
    plot_df = prepare_glycemic_state_data(df, hypo_threshold, hyper_threshold, sample_size)
    
    # Box plot (left subplot)
    sns.boxplot(x='State', y='Percentage', hue='State', data=plot_df,
                palette=colors, width=0.5, showfliers=False, legend=False,
                order=['Hypoglycemia', 'Euglycemia', 'Hyperglycemia'], ax=ax1)
    
    sns.stripplot(x='State', y='Percentage', hue='State', data=plot_df,
                  palette=colors, size=4, alpha=0.2, jitter=True, legend=False,
                  order=['Hypoglycemia', 'Euglycemia', 'Hyperglycemia'], ax=ax1)
    
    ax1.set_xlabel('Glycemic State', fontsize=12)
    ax1.set_ylabel('Percentage', fontsize=12)
    ax1.set_title('Percentage of Time in Glycemic States', fontsize=14, pad=20)
    
    # Set grid: horizontal lines only
    ax1.grid(True, which='major', axis='y', linestyle='-', color='gray', alpha=0.7)
    ax1.grid(False, which='major', axis='x')
    
    # Prepare data for candlestick chart
    candlestick_data = prepare_risk_candlestick_data(df)
    
    # Candlestick chart (right subplot)
    width = 0.6
    for i, (state, row) in enumerate(candlestick_data.iterrows()):
        color = colors[state]
        # Wick
        ax2.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
        # Body
        ax2.bar(i, abs(row['Close'] - row['Open']), width, 
                bottom=min(row['Open'], row['Close']),
                color=color, edgecolor='black', label=state)
    
    ax2.set_xticks(range(len(candlestick_data)))
    ax2.set_xticklabels(candlestick_data.index, rotation=45)
    ax2.set_xlabel('Glycemic State', fontsize=12)
    ax2.set_ylabel('Risk Index', fontsize=12)
    ax2.set_title('Risk Index Candlestick Chart by Glycemic State', fontsize=14, pad=20)
    ax2.legend()
    
    # Set grid: horizontal lines only
    ax2.grid(True, which='major', axis='y', linestyle='-', color='gray', alpha=0.7)
    ax2.grid(False, which='major', axis='x')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    data = load_data(DATA_PATH)
    if not data:
        return
    
    # Process all trajectories as a single dataset
    result = process_all_trajectories(data, HYPO_THRESHOLD, HYPER_THRESHOLD)
    
    # Set display options
    pd.set_option('float_format', '{:.3f}'.format)
    pd.set_option("display.max_rows", len(result))
    
    # Print summary statistics
    print("Summary Statistics:")
    print(result.describe())
    
    # Plot combined charts
    plot_combined_charts(result, HYPO_THRESHOLD, HYPER_THRESHOLD, SAMPLE_SIZE)

if __name__ == "__main__":
    main()