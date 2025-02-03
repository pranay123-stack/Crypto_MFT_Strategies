import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/16_TAMMY_V2_strategy.log'

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



# 2. Indicator Calculation
def calculate_daily_indicators(daily_data, fast_len=14, slow_len=100, atr_length=10):
    try:
        logging.info("Resampling data to daily timeframe for indicator calculation")
      
        logging.info("Calculating SMAs and ATR on daily data")

        # Calculate SMAs
        daily_data['fast_sma'] = ta.sma(daily_data['Close'], length=fast_len)
        daily_data['slow_sma'] = ta.sma(daily_data['Close'], length=slow_len)
        
        # True Range calculation
        tr = pd.concat([
            daily_data['High'] - daily_data['Low'],
            (daily_data['High'] - daily_data['Close'].shift(1)).abs(),
            (daily_data['Low'] - daily_data['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        # Calculate ATR as the SMA of the True Range
        daily_data['atr'] = ta.sma(tr, length=atr_length)

        logging.info("Indicator calculation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise


# 3. Generate Signals
def generate_signals(daily_data, risk_per_trade=2.0):
    try:
        logging.info("Generating trading signals")

        # Initialize signals and stop loss columns
        daily_data['signal'] = 0


        # Loop through the data to generate signals
        for i in range(1, len(daily_data)):
            # Check for a buy signal
            if (daily_data['Close'].iloc[i] > daily_data['slow_sma'].iloc[i]) and \
               (daily_data['Close'].iloc[i-1] <= daily_data['slow_sma'].iloc[i-1]):
                daily_data.at[daily_data.index[i], 'signal'] = 1

            # Check for a sell signal
            elif (daily_data['Close'].iloc[i] < daily_data['fast_sma'].iloc[i]) and \
                 (daily_data['Close'].iloc[i-1] >= daily_data['fast_sma'].iloc[i-1]):
                daily_data.at[daily_data.index[i], 'signal'] = -1

      
        logging.info(f"Signal generation complete\n{daily_data.head(20)}")
        return daily_data
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise


# Define the Trading Strategy
class TAMMY_V2(Strategy):
 
  

    def init(self):
        logging.info("Initializing strategy")
        self.stop_loss = None
        self.risk_per_trade = 2.0
      

    def next(self):
        # Calculate stop loss dynamically
        atr_value = self.data.atr[-1]  # Use ATR from the latest available data
        
        # Check if we need to enter a long position
        if self.data.signal[-1] == 1:
            # Calculate and set the stop loss for the new long position
            self.stop_loss = self.data.Close[-1] - atr_value * (self.risk_per_trade / 100)
            self.buy()

        # Manage the existing position
        if self.position.is_long:
            # Check if the stop loss condition is met
            if self.data.Close[-1] <= self.stop_loss:
                self.position.close()
            
            # Alternatively, check for a sell signal to close the position
            elif self.data.signal[-1] == -1:
                self.position.close()

# Load and prepare data
try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    
    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
   
    bt = Backtest(daily_signals, TAMMY_V2, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/16_TAMMY_V2_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
