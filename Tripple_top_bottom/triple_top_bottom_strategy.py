import pandas as pd
import numpy as np

# Load data
print("Loading data...")
data = pd.read_csv('/Users/pranaygaurav/Downloads/AlgoTrading/Devin_Group_Strategy/test_technical_indicator_strategy/test_triple_top_bottom_strategy/data_2020_2023_5min.csv', parse_dates=['datetime'])


def find_triple_bottom_patterns(data):
    trades = []
    print("Analyzing data for triple bottom patterns...")
    data['date'] = data['datetime'].dt.date
    for date, group in data.groupby('date'):
        print(f"Processing data for {date}...")
        start_time = min(pd.to_datetime('00:00:00').time(), pd.to_datetime('05:30:00').time(), group['datetime'].dt.time.min())
        end_time = pd.to_datetime('23:55:00').time()
        filtered_group = group[(group['datetime'].dt.time >= start_time) & (group['datetime'].dt.time <= end_time)]
        lows = filtered_group['low'].reset_index(drop=True)

        i = 0
        while i < len(filtered_group) - 2:
            if (lows.iloc[i], lows.iloc[i+1]) and (lows.iloc[i+1], lows.iloc[i+2]):
                high1 = filtered_group['high'].iloc[i+1]
                high2 = filtered_group['high'].iloc[i+2]
                max_high = max(high1, high2)
                # Ensure that the loop doesn't exceed DataFrame bounds
                for j in range(i+3, len(filtered_group)):
                    if j >= len(filtered_group):
                        break  # Prevents 'index out-of-bounds' error
                    if filtered_group['close'].iloc[j] > max_high * 1.05:
                        trades.append({
                            'type': 'long',
                            'entry_time': filtered_group['datetime'].iloc[i+3],
                            'entry_price': filtered_group['close'].iloc[i+3],
                            'close_time': filtered_group['datetime'].iloc[j],
                            'close_price': filtered_group['close'].iloc[j],
                            'resistance': max_high
                        })
                        i = j  # Move to the index after this trade
                        break  # Exit this for-loop after adding a trade
            i += 1  # Increment to check the next potential pattern

    return trades



def find_triple_top_patterns(data):
    trades = []
    print("Analyzing data for triple top patterns...")
    # Group data by date for daily analysis
    data['date'] = data['datetime'].dt.date
    for date, group in data.groupby('date'):
        print(f"Processing data for {date}...")
        start_time = min(pd.to_datetime('00:00:00').time(), pd.to_datetime('05:30:00').time(), group['datetime'].dt.time.min())
        end_time = pd.to_datetime('23:55:00').time()
        filtered_group = group[(group['datetime'].dt.time >= start_time) & (group['datetime'].dt.time <= end_time)]
        highs = filtered_group['high'].reset_index(drop=True)  # Resetting index for safe access

        i = 0
        while i < len(highs) - 2:
            if (highs.iloc[i], highs.iloc[i+1]) and (highs.iloc[i+1], highs.iloc[i+2]):
                low1 = filtered_group['low'].iloc[i+1]
                low2 = filtered_group['low'].iloc[i+2]
                min_low = min(low1, low2)
                for j in range(i+3, len(filtered_group)):
                    if filtered_group['close'].iloc[j] < min_low * 0.95:  # Trigger a trade if the price falls below 95% of the minimum low
                        trades.append({
                            'type': 'short',
                            'entry_time': filtered_group['datetime'].iloc[i+3],
                            'entry_price': filtered_group['close'].iloc[i+3],
                            'close_time': filtered_group['datetime'].iloc[j],  # Record the close time
                            'close_price': filtered_group['close'].iloc[j],  # Record the close price
                            'support': min_low
                        })
                        i = j  # Skip to after this trade
                        break
            i += 1

    return trades





triple_tops = find_triple_top_patterns(data)
triple_bottoms = find_triple_bottom_patterns(data)
all_trades = triple_tops + triple_bottoms  # Combine lists of trade dicts
trades_df = pd.DataFrame(all_trades)
print("Saving results to CSV...")
# trades_df = pd.DataFrame(triple_bottoms)
trades_df.to_csv('/Users/pranaygaurav/Downloads/AlgoTrading/Devin_Group_Strategy/test_technical_indicator_strategy/test_triple_top_bottom_strategy/trades.csv', index=False)
print("Results saved successfully to 'trades.csv'.")
