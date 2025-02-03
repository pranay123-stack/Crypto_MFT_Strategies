import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta
from backtesting.lib import crossover

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/10_ema9_wma30_strategy.log'

# Configure basic logging to file
logging.basicConfig(level=logging.DEBUG, filename=log_file_path, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up colored logs for console output
coloredlogs.install(level='DEBUG',
                    fmt='%(asctime)s - %(levelname)s - %(message)s',
                    level_styles={
                        'info': {'color': 'green'},
                        'debug': {'color': 'white'},
                        'error': {'color': 'red'},
                    })

# Function to load and prepare data from a CSV file
def load_and_prepare_data(csv_file_path):
    try:
        logging.info("Loading data from CSV")
        data = pd.read_csv(csv_file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        logging.info("Data loading and preparation complete")
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise

# Function to calculate indicators on the daily timeframe
def calculate_daily_indicators(daily_data, length_ema=9, length_wma=30, fast_length=12, slow_length=26, macd_length=9):
    try:
        
        # Calculate EMA, WMA, and MACD
        daily_data['ema9'] = ta.ema(daily_data['Close'], length=length_ema)
        daily_data['wma30'] = ta.wma(daily_data['Close'], length=length_wma)
        macd = ta.macd(daily_data['Close'], fast=fast_length, slow=slow_length, signal=macd_length)
        daily_data['macd_line'] = macd['MACD_12_26_9']
        daily_data['signal_line'] = macd['MACDs_12_26_9']

        # Additional Indicators
        daily_data['sma200'] = ta.sma(daily_data['Close'], length=200)
        daily_data['ema21'] = ta.ema(daily_data['Close'], length=21)
        daily_data['vwap'] = ta.vwap(daily_data['High'], daily_data['Low'], daily_data['Close'], daily_data['Volume'])

        logging.info("Daily indicator calculation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise


# Function to generate buy/sell signals based on the strategy logic
def generate_signals(daily_data):
    try:
        logging.info("Generating signals based on strategy logic")

        # Initialize columns for signals and counts
        daily_data['buy_signal'] = 0
        daily_data['below_ema9_count'] = 0
        daily_data['below_wma30_count'] = 0
        daily_data['macd_bearish_cross'] = 0

        # Initialize counters
        below_ema9_count = 0
        below_wma30_count = 0

        # Loop through the DataFrame, starting from index 1
        for i in range(1, len(daily_data)):
            # Check EMA9/WMA30 crossover and MACD confirmation
            prev_ema9 = daily_data['ema9'].iloc[i - 1]
            prev_wma30 = daily_data['wma30'].iloc[i - 1]
            current_ema9 = daily_data['ema9'].iloc[i]
            current_wma30 = daily_data['wma30'].iloc[i]
            macd_line = daily_data['macd_line'].iloc[i]
            signal_line = daily_data['signal_line'].iloc[i]

            # Check for EMA9/WMA30 crossover
            buy_signal = (prev_ema9 < prev_wma30) and (current_ema9 > current_wma30) and \
                         (macd_line > signal_line)

            if buy_signal:
                daily_data.at[daily_data.index[i], 'buy_signal'] = 1

            # Compute consecutive counts for closes below EMA9 and WMA30
            if daily_data['Close'].iloc[i] < current_ema9:
                below_ema9_count += 1
            else:
                below_ema9_count = 0

            if daily_data['Close'].iloc[i] < current_wma30:
                below_wma30_count += 1
            else:
                below_wma30_count = 0

            daily_data.at[daily_data.index[i], 'below_ema9_count'] = below_ema9_count
            daily_data.at[daily_data.index[i], 'below_wma30_count'] = below_wma30_count

            # Check for MACD bearish crossover
            prev_macd_line = daily_data['macd_line'].iloc[i - 1]
            prev_signal_line = daily_data['signal_line'].iloc[i - 1]
            macd_bearish_cross = (prev_macd_line > prev_signal_line) and (macd_line < signal_line)

            if macd_bearish_cross:
                daily_data.at[daily_data.index[i], 'macd_bearish_cross'] = 1

            # Exit conditions
            exit_condition1 = (below_ema9_count >= 2) and (below_wma30_count >= 1)
            exit_condition2 = macd_bearish_cross

            if exit_condition1 or exit_condition2:
                daily_data.at[daily_data.index[i], 'signal'] = -1
            elif daily_data['buy_signal'].iloc[i] == 1:
                daily_data.at[daily_data.index[i], 'signal'] = 1
            else:
                daily_data.at[daily_data.index[i], 'signal'] = 0

        logging.info("Signal generation complete")
        return daily_data

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise



# Define the strategy class
class EMA9WMA30Strategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing strategy")


            logging.info("Strategy initialization complete")
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

    def next(self):
        try:
            # logging.debug(f"Processing bar: {self.data.index[-1]} with signal {self.data.signal[-1]} at price {self.data.Close[-1]}")
            # Check for signals and execute trades based on signal value
            if self.data.signal[-1] == 1:
                logging.debug(f"Buy signal detected, close={self.data.Close[-1]}")
                self.buy()

            elif self.data.signal[-1] == -1 :
                logging.debug(f"Sell signal detected, close={self.data.Close[-1]}")
                self.position.close()

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise

# Load and prepare data
try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    # logging.info(f"Minute data:\n{minute_data.head(20)}")

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
   
    # Run the backtest
    bt = Backtest(daily_signals, EMA9WMA30Strategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")
    
    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/10_EMA9WMA30Strategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
