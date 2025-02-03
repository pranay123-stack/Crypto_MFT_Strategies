import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta
from backtesting.lib import crossover

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/9_ema_sma_crossover_strategy.log'

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
def calculate_daily_indicators(daily_data, ema_length=9, sma30_length=30, sma50_length=50, sma200_length=200, sma325_length=325):
    try:
     
        # Calculate EMA and SMAs
        daily_data['ema'] = ta.ema(daily_data['Close'], length=ema_length)
        daily_data['sma30'] = ta.sma(daily_data['Close'], length=sma30_length)
        daily_data['sma50'] = ta.sma(daily_data['Close'], length=sma50_length)
        daily_data['sma200'] = ta.sma(daily_data['Close'], length=sma200_length)
        daily_data['sma325'] = ta.sma(daily_data['Close'], length=sma325_length)

        logging.info("Daily indicator calculation complete")
        return daily_data
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
            # Access values directly using iloc
            prev_ema = daily_data.iloc[i - 1]['ema']
            prev_sma30 = daily_data.iloc[i - 1]['sma30']
            current_ema = daily_data.iloc[i]['ema']
            current_sma30 = daily_data.iloc[i]['sma30']
            prev_sma50 = daily_data.iloc[i - 1]['sma50']
            current_sma50 = daily_data.iloc[i]['sma50']

            # Buy Signal Condition
            buy_signal = (prev_ema < prev_sma30) and (current_ema > current_sma30)

            # Sell Signal Condition
            sell_signal = (prev_sma30 < prev_ema and current_sma30 > current_ema) or \
                          (prev_sma50 < prev_ema and current_sma50 > current_ema)

            # Assign signals based on conditions
            if buy_signal:
                daily_data.loc[daily_data.index[i], 'signal'] = 1
            elif sell_signal:
                daily_data.loc[daily_data.index[i], 'signal'] = -1
            # No action is taken if neither condition is met (signal remains 0)

        logging.info("Signal generation complete")
        return daily_data

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise



# Define the strategy class
class EMASMACrossoverStrategy(Strategy):
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
  
    
  
    bt = Backtest(daily_signals, EMASMACrossoverStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")
        # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/9_EMASMACrossoverStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")
except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
