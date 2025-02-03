import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/12_ichimoku_strategy.log'

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







def calculate_daily_indicators(df):
    # Calculate Ichimoku components
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    displacement = 26

    df['tenkan_sen'] = (df['High'].rolling(window=tenkan_period).max() + df['Low'].rolling(window=tenkan_period).min()) / 2
    df['kijun_sen'] = (df['High'].rolling(window=kijun_period).max() + df['Low'].rolling(window=kijun_period).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
    df['senkou_span_b'] = (df['High'].rolling(window=senkou_b_period).max() + df['Low'].rolling(window=senkou_b_period).min()) / 2
    df['senkou_span_b'] = df['senkou_span_b'].shift(displacement)
    df['chikou_span'] = df['Close'].shift(-displacement)

    # Replace NaN values
    df.fillna(method='bfill', inplace=True)
    return df


def generate_signals(df):
    # Define conditions
    df['long_condition'] = (df['tenkan_sen'] > df['kijun_sen']) & (df['Close'] > df['senkou_span_a']) & (df['Close'] > df['senkou_span_b'])
    df['short_condition'] = (df['tenkan_sen'] < df['kijun_sen']) & (df['Close'] < df['senkou_span_a']) & (df['Close'] < df['senkou_span_b'])

  

    df['signal'] = 0
    df.loc[df['long_condition'], 'signal'] = 1
    df.loc[df['short_condition'], 'signal'] = -1
    
    return df


# Define the strategy class
class IchimokuStrategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing strategy")
        
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            logging.info("Strategy initialization complete")
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

   

    def next(self):
        try:
            current_time = self.data.index[-1]  # Get the current timestamp of the candle

            # Initialize the stop_loss_percentage and take_profit_percentage values
            stop_loss_percentage = 0.05  # 5% Stop Loss
            take_profit_percentage = 0.10  # 10% Take Profit

            # Check for long condition
            if self.data.signal[-1]==1:
                logging.debug(f"Long condition met, close={self.data.Close[-1]}")
                self.entry_price = self.data.Close[-1]
                self.stop_loss = self.entry_price * (1 - stop_loss_percentage)  # 5% Stop Loss
                self.take_profit = self.entry_price * (1 + take_profit_percentage)  # 10% Take Profit
                self.buy()
             
            # Check for short condition
            elif self.data.signal[-1]==-1:
                logging.debug(f"Short condition met, close={self.data.Close[-1]}")
                self.entry_price = self.data.Close[-1]
                self.stop_loss = self.entry_price * (1 + stop_loss_percentage)  # 5% Stop Loss
                self.take_profit = self.entry_price * (1 - take_profit_percentage)  # 10% Take Profit
                
                # If currently in a long position, close it before selling short
                if self.position.is_long:
                    self.position.close()
                    logging.info(f"Closed Long position at {self.data.Close[-1]} due to short signal")
                
           
            # Exit strategy based on stop loss and take profit for long positions
            if self.position.is_long:
                if self.data.Close[-1] <= self.stop_loss:
                    self.position.close()
                    logging.info(f"Closed Long position at {self.data.Close[-1]} due to hitting stop loss")
                elif self.data.Close[-1] >= self.take_profit:
                    self.position.close()
                    logging.info(f"Closed Long position at {self.data.Close[-1]} due to hitting take profit")

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise

# Load and prepare data
try:
    data_path =  '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
 

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
   
    bt = Backtest(daily_signals, IchimokuStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")
        # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/12_IchimokuStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")