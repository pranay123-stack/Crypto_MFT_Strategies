import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/20_ TrendStructureBreakStrategy_strategy.log'

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


def calculate_daily_indicators(daily_data, fast_length=9, slow_length=21, order_block_threshold=0.1, fvg_threshold=0.5, atr_length=14):
    logging.debug("Calculating daily indicators")

   
    # Calculate moving averages using pandas_ta
    daily_data['Fast_MA'] = ta.sma(daily_data['Close'], length=fast_length)
    daily_data['Slow_MA'] = ta.sma(daily_data['Close'], length=slow_length)
    
    # Determine trend
    daily_data['Bullish_Trend'] = daily_data['Fast_MA'] > daily_data['Slow_MA']
    daily_data['Bearish_Trend'] = daily_data['Fast_MA'] < daily_data['Slow_MA']

    # Break of Structure (BOS)
    daily_data['Highest_High'] = daily_data['High'].rolling(window=10).max()
    daily_data['Lowest_Low'] = daily_data['Low'].rolling(window=10).min()
    daily_data['Bullish_BOS'] = (daily_data['Bullish_Trend']) & (daily_data['Close'] > daily_data['Highest_High'])
    daily_data['Bearish_BOS'] = (daily_data['Bearish_Trend']) & (daily_data['Close'] < daily_data['Lowest_Low'])

    # Order Block Identification
    daily_data['Order_Block_High'] = np.where(daily_data['Bullish_BOS'], daily_data['Highest_High'], np.nan)
    daily_data['Order_Block_Low'] = np.where(daily_data['Bullish_BOS'], daily_data['Close'] * (1 - order_block_threshold / 100), np.nan)
    daily_data['Order_Block_High'] = np.where(daily_data['Bearish_BOS'], daily_data['Close'] * (1 + order_block_threshold / 100), daily_data['Order_Block_High'])
    daily_data['Order_Block_Low'] = np.where(daily_data['Bearish_BOS'], daily_data['Lowest_Low'], daily_data['Order_Block_Low'])

    # Fair Value Gap (FVG)
    daily_data['FVG_High'] = np.where(daily_data['Bullish_BOS'], daily_data['High'], np.nan)
    daily_data['FVG_Low1'] = np.where(daily_data['Bullish_BOS'], daily_data['High'] * (1 - fvg_threshold / 100), np.nan)
    daily_data['FVG_Low2'] = np.where(daily_data['Bullish_BOS'], daily_data['High'] * (1 - fvg_threshold / 100 * 2), np.nan)
    daily_data['FVG_Low1'] = np.where(daily_data['Bearish_BOS'], daily_data['Low'] * (1 + fvg_threshold / 100), daily_data['FVG_Low1'])
    daily_data['FVG_Low2'] = np.where(daily_data['Bearish_BOS'], daily_data['Low'] * (1 + fvg_threshold / 100 * 2), daily_data['FVG_Low2'])

    # Calculate ATR using pandas_ta
    daily_data['ATR'] = ta.atr(daily_data['High'], daily_data['Low'], daily_data['Close'], length=atr_length)

    logging.debug("Daily indicators calculated successfully")
    return daily_data


def generate_signals(daily_data):
    logging.debug("Generating signals based on strategy logic")

    # Initialize 'Signal' column with NaNs
    daily_data['signal'] = 0

    # Loop through each row of the DataFrame
    for i in range(1, len(daily_data)):
        # Check long entry condition
        if (daily_data['Fast_MA'].iloc[i] > daily_data['Slow_MA'].iloc[i] and
            daily_data['Fast_MA'].iloc[i-1] <= daily_data['Slow_MA'].iloc[i-1]):
            daily_data['signal'].iloc[i] = 1
        
        # Check short entry condition
        elif (daily_data['Fast_MA'].iloc[i] < daily_data['Slow_MA'].iloc[i] and
              daily_data['Fast_MA'].iloc[i-1] >= daily_data['Slow_MA'].iloc[i-1]):
            daily_data['signal'].iloc[i] = -1

    logging.debug("Signals generated successfully")
    return daily_data







# Define the strategy class
class TrendStructureBreakStrategy(Strategy):
    def init(self):
     
        self.atr_multiplier = 5.5

    def next(self):
        if self.data.signal[-1] == 1:
            if self.position.is_short:
                self.position.close()
            self.buy()

            if self.position.is_long:
                self.position.close()
            self.sell()
        

  



# Load and prepare data
try:
    data_path =  '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
  

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
    
    bt = Backtest(daily_signals,  TrendStructureBreakStrategy , cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/20_ TrendStructureBreakStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
