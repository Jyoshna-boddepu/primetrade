import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style
sns.set_theme(style="whitegrid")

# Paths
DATA_DIR = r"c:\Users\thota\Documents\primetrade\data"
SENTIMENT_FILE = os.path.join(DATA_DIR, "fear_greed_index.csv")
TRADER_FILE = os.path.join(DATA_DIR, "historical_data.csv")
OUTPUT_DIR = os.path.join(r"c:\Users\thota\Documents\primetrade", "output")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading datasets...")
# Sentiment Data
sentiment_df = pd.read_csv(SENTIMENT_FILE)
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Trader Data - Loading with caution due to size
# We'll parse dates and handle types
trader_df = pd.read_csv(TRADER_FILE)
print(f"Trader data loaded: {len(trader_df)} trades")

# Data Cleaning
print("Cleaning data...")
# Convert Timestamp to datetime. We'll use the 'Timestamp' column which seems to be unix ms or the 'Timestamp IST'
# Based on previous view, 'Timestamp' (col 16) looks like 1.73E+12 which is unix ms.
# 'Timestamp IST' (col 7) is "DD-MM-YYYY HH:MM"
trader_df['date'] = pd.to_datetime(trader_df['Timestamp IST'], dayfirst=True).dt.date
trader_df['date'] = pd.to_datetime(trader_df['date']) # Convert back to datetime object for merging

# Convert PnL and Size to numeric
trader_df['Closed PnL'] = pd.to_numeric(trader_df['Closed PnL'], errors='coerce').fillna(0)
trader_df['Size USD'] = pd.to_numeric(trader_df['Size USD'], errors='coerce').fillna(0)

# Merge Sentiment with Trader Data
print("Merging datasets...")
# Aggregate trader data by date
daily_trader = trader_df.groupby('date').agg(
    total_pnl=('Closed PnL', 'sum'),
    trade_count=('Account', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    total_volume=('Size USD', 'sum')
).reset_index()

# Calculate Win Rate
daily_trader['win_rate'] = trader_df[trader_df['Closed PnL'] > 0].groupby('date')['Account'].count() / daily_trader['trade_count']
daily_trader['win_rate'] = daily_trader['win_rate'].fillna(0)

# Merge
merged_df = pd.merge(daily_trader, sentiment_df, on='date', how='inner')

if merged_df.empty:
    print("WARNING: Merged dataframe is empty. Checking date overlap...")
    print(f"Sentiment dates: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    print(f"Trader dates: {daily_trader['date'].min()} to {daily_trader['date'].max()}")
else:
    print(f"Merged data contains {len(merged_df)} overlapping days.")

# Analysis & Visualization
print("Generating insights...")

# 1. PnL by Sentiment Classification
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_df, x='classification', y='total_pnl', order=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
plt.title('Total Daily PnL vs Market Sentiment')
plt.ylabel('Total PnL (USD)')
plt.savefig(os.path.join(OUTPUT_DIR, 'pnl_vs_sentiment.png'))

# 2. Win Rate by Sentiment
plt.figure(figsize=(12, 6))
sns.barplot(data=merged_df, x='classification', y='win_rate', ci=None, order=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
plt.title('Average Trader Win Rate vs Market Sentiment')
plt.ylabel('Win Rate (%)')
plt.savefig(os.path.join(OUTPUT_DIR, 'winrate_vs_sentiment.png'))

# 3. Trade Count vs Sentiment
plt.figure(figsize=(12, 6))
sns.barplot(data=merged_df, x='classification', y='trade_count', ci=None, order=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
plt.title('Daily Trade Count vs Market Sentiment')
plt.ylabel('Total Trades')
plt.savefig(os.path.join(OUTPUT_DIR, 'activity_vs_sentiment.png'))

# 4. Correlation Matrix
cols_to_corr = ['total_pnl', 'trade_count', 'win_rate', 'total_volume', 'value']
corr = merged_df[cols_to_corr].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (value = Fear/Greed Index)')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))

# Summary Statistics
summary = merged_df.groupby('classification').agg({
    'total_pnl': ['mean', 'std', 'sum'],
    'win_rate': 'mean',
    'trade_count': 'mean'
}).round(2)

summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_stats.csv'))

print("Analysis complete. Check the 'output' folder for charts and stats.")
