import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta
from backtesting.lib import crossover

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/8_multiema_startegy.log'
logging.basicConfig(level=logging.DEBUG, filename=log_file_path, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
coloredlogs.install(level='DEBUG',
                    fmt='%(asctime)s - %(levelname)s - %(message)s',
                    level_styles={'info': {'color': 'green'},
                                  'debug': {'color': 'white'},
                                  'error': {'color': 'red'}})

# Function to load and prepare data
def load_and_prepare_data(csv_file_path):
    try:
        logging.info("Loading data from CSV")
        data = pd.read_csv(csv_file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        logging.info("Data loading and preparation complete")
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise

# Function to calculate indicators on the daily timeframe
def calculate_daily_indicators(daily_data):
    try:
     
        # Calculate EMAs
        daily_data['ema8'] = ta.ema(daily_data['Close'], length=8)
        daily_data['ema21'] = ta.ema(daily_data['Close'], length=21)
        daily_data['ema50'] = ta.ema(daily_data['Close'], length=50)
        daily_data['ema200'] = ta.ema(daily_data['Close'], length=200)

        # Condition: All short-term EMAs must be above the 200-period EMA
        daily_data['all_above_200'] = (daily_data['ema8'] > daily_data['ema200']) & \
                                      (daily_data['ema21'] > daily_data['ema200']) & \
                                      (daily_data['ema50'] > daily_data['ema200'])

        logging.info(f"Data after indicator calculation: {daily_data}")
        return daily_data.dropna()
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise

# Function to generate buy/sell signals based on the strategy logic
def generate_signals(daily_data):
    try:
        logging.info("Generating signals based on strategy logic")

        # Initialize the 'signal' column
        daily_data['signal'] = 0

        # Loop through each row in the DataFrame, starting from index 1
        for i in range(1, len(daily_data)):
            # Access values directly using iloc and column names
            prev_ema8 = daily_data.iloc[i - 1]['ema8']
            prev_ema21 = daily_data.iloc[i - 1]['ema21']
            current_ema8 = daily_data.iloc[i]['ema8']
            current_ema21 = daily_data.iloc[i]['ema21']
            all_above_200 = daily_data.iloc[i]['all_above_200']

            # Check buy condition
            buy_condition = (prev_ema8 < prev_ema21) and \
                            (current_ema8 > current_ema21) and \
                            all_above_200

            # Check sell condition
            sell_condition = (prev_ema8 > prev_ema21) and \
                             (current_ema8 < current_ema21)

            # Assign signals based on conditions
            if buy_condition:
                daily_data.loc[daily_data.index[i], 'signal'] = 1
            elif sell_condition:
                daily_data.loc[daily_data.index[i], 'signal'] = -1
            # No action is taken if neither condition is met (signal remains 0)

        logging.info(f"Data after Signal generation {daily_data}")
        return daily_data

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise



# Define the strategy class
class MultiEMAStrategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing strategy")

            
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

    def next(self):
        try:
            
            # Check for signals and execute trades based on signal value
            if self.data.signal[-1] == 1:
                logging.debug(f"Buy signal detected, close={self.data.Close[-1]}")
                self.buy()

            elif self.data.signal[-1] == -1:
                logging.debug(f"Sell signal detected, close={self.data.Close[-1]}")
                self.position.close()

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise

# Load and prepare data
try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_1h_data_2023_2024/converted_sorted_btc_1h_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    # logging.info(f"Minute data:\n{minute_data.head(20)}")

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
  
 

    bt = Backtest(daily_signals, MultiEMAStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")
        # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/8_MultiEMAStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
