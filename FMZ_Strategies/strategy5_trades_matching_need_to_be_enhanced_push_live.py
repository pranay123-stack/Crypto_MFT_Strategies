import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import logging
import coloredlogs
import pandas_ta as ta

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/5_bb_strategy.log'

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
def calculate_daily_indicators(daily_data, length=20, mult=2.0, ma_type='SMA'):
    try:
   

        logging.info("Calculating Bollinger Bands on daily data")
        
        # Moving average calculation based on selected type
        if ma_type == 'SMA':
            daily_data['basis'] = ta.sma(daily_data['Close'], length)
        elif ma_type == 'EMA':
            daily_data['basis'] = ta.ema(daily_data['Close'], length)
        elif ma_type == 'SMMA (RMA)':
            daily_data['basis'] = ta.rma(daily_data['Close'], length)
        elif ma_type == 'WMA':
            daily_data['basis'] = ta.wma(daily_data['Close'], length)
        elif ma_type == 'VWMA':
            daily_data['basis'] = ta.vwma(daily_data['Close'], length)

        # Calculate the Bollinger Bands
        daily_data['std_dev'] = daily_data['Close'].rolling(window=length).std()
        daily_data['upper'] = daily_data['basis'] + (daily_data['std_dev'] * mult)
        daily_data['lower'] = daily_data['basis'] - (daily_data['std_dev'] * mult)

        daily_data.dropna(inplace=True)

        logging.info(f"Daily indicator calculation complete\n{daily_data.head(20)}")
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise

# Function to generate buy/sell signals based on the strategy logic
def generate_signals(daily_data):
    try:
        logging.info("Generating signals based on strategy logic")

        # Initialize columns to store conditions
        daily_data['long_condition'] = False
        daily_data['short_condition'] = False
        daily_data['signal'] = 0

        # Iterate through daily data to find crossovers
        for i in range(1, len(daily_data)):
            # Long condition: Close crosses above upper band
            if (daily_data['Close'].iloc[i-1] < daily_data['upper'].iloc[i-1] and
                daily_data['Close'].iloc[i] > daily_data['upper'].iloc[i]):
                daily_data.at[daily_data.index[i], 'long_condition'] = True
            
            # Short condition: Close crosses below lower band
            if (daily_data['Close'].iloc[i-1] > daily_data['lower'].iloc[i-1] and
                daily_data['Close'].iloc[i] < daily_data['lower'].iloc[i]):
                daily_data.at[daily_data.index[i], 'short_condition'] = True

        # Generate signals
        daily_data['signal'] = np.where(daily_data['long_condition'], 1,
                                        np.where(daily_data['short_condition'], -1, 0))

        logging.info("Signal generation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise



# Define the strategy class
class BBStrategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing strategy")

            self.entry_price = None
            logging.info("Strategy initialization complete")
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

    def next(self):
        try:
            # Check if there is a long position
            if self.position.is_long:
                if self.data.signal[-1] == -1:  # If there's a sell signal
                    logging.debug(f"Sell signal detected, closing long position at close={self.data.Close[-1]}")
                    self.position.close()  # Close the long position

            # Check if there is a short position
            elif self.position.is_short:
                if self.data.signal[-1] == 1:  # If there's a buy signal
                    logging.debug(f"Buy signal detected, closing short position at close={self.data.Close[-1]}")
                    self.position.close()  # Close the short position

            # Handle buy (long) signal
            if self.data.signal[-1] == 1:
                if not self.position.is_long:  # Open long position only if not already in one
                    logging.debug(f"Buy signal detected, opening long position at close={self.data.Close[-1]}")
                    self.entry_price = self.data.Close[-1]
                    self.buy()  # Enter long position

            # Handle sell (short) signal
            elif self.data.signal[-1] == -1:
                if not self.position.is_short:  # Open short position only if not already in one
                    logging.debug(f"Sell signal detected, opening short position at close={self.data.Close[-1]}")
                    self.entry_price = self.data.Close[-1]
                    self.sell()  # Enter short position

            

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise


# Load and prepare data
try:
    data_path ='/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
  

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
   
 
    bt = Backtest(daily_signals, BBStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")
        # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/5_BBStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
