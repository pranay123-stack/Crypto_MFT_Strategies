import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/6_kc_pullback_strategy.log'

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
def calculate_daily_indicators(daily_data, atr_length=35, atr_multiplier=5.5, kc_length=20, kc_multiplier=6.0, ema_length=280):
    try:
      
        # Calculate ATR
        daily_data['atr'] = ta.atr(daily_data['High'], daily_data['Low'], daily_data['Close'], length=atr_length)
        
        # Calculate Keltner Channel
        daily_data['kc_basis'] = ta.sma(daily_data['Close'], length=kc_length)
        daily_data['kc_range'] = daily_data['atr'] * kc_multiplier
        daily_data['upper_kc'] = daily_data['kc_basis'] + daily_data['kc_range']
        daily_data['lower_kc'] = daily_data['kc_basis'] - daily_data['kc_range']
        
        # Calculate EMA
        daily_data['ema'] = ta.ema(daily_data['Close'], length=ema_length)

            # Drop NaN values resulting from indicator calculations
        daily_data.dropna(inplace=True)
        

        logging.info(f"Daily indicator calculation complete\n{daily_data.head(20)}")
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise

def generate_signals(daily_data, candle_lookback=120):
    try:
        logging.info("Generating signals based on strategy logic")

        # Function to check if Keltner Channel was touched within the lookback period
        def was_kc_touched(data, direction):
            touched = False
            # Ensure we don't go out of bounds
            max_lookback = min(candle_lookback, len(data) - 1)
            for i in range(1, max_lookback + 1):
                if direction == "long" and data['High'].iloc[-i] >= data['upper_kc'].iloc[-i]:
                    touched = True
                    break
                if direction == "short" and data['Low'].iloc[-i] <= data['lower_kc'].iloc[-i]:
                    touched = True
                    break
            return touched

        # Check for middle line touch by wick
        daily_data['middle_line_touched'] = (daily_data['High'] >= daily_data['kc_basis']) & (daily_data['Low'] <= daily_data['kc_basis'])

        # Generate long and short conditions based on the custom function and other criteria
        daily_data['long_condition'] = daily_data.apply(lambda row: 
            was_kc_touched(daily_data, "long") and 
            row['middle_line_touched'] and 
            row['Close'] > row['ema'], axis=1)

        daily_data['short_condition'] = daily_data.apply(lambda row: 
            was_kc_touched(daily_data, "short") and 
            row['middle_line_touched'] and 
            row['Close'] < row['ema'], axis=1)

        # Generate signals
        daily_data['signal'] = np.where(daily_data['long_condition'], 1,
                                        np.where(daily_data['short_condition'], -1, 0))

        logging.info("Signal generation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise



class KCPullbackStrategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing strategy")
            self.atr_multiplier = 5.5
            self.entry_price = None
            self.prev_atr = None
            logging.info("Strategy initialization complete")
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

    def next(self):
        try:


            # Entry Conditions
            if self.data.signal[-1] == 1:
                logging.debug(f"Long signal detected, close={self.data.Close[-1]}")
                self.entry_price = self.data.Close[-1]
                self.prev_atr = self.data.atr[-2]
                long_stop_loss = self.entry_price - self.atr_multiplier * self.prev_atr

                self.buy(sl=long_stop_loss)
                logging.info(f"Entered long position at {self.entry_price} with stop loss at {long_stop_loss}")

            elif self.data.signal[-1] == -1:
                logging.debug(f"Short signal detected, close={self.data.Close[-1]}")
                self.entry_price = self.data.Close[-1]
                self.prev_atr = self.data.atr[-2]
                short_stop_loss = self.entry_price + self.atr_multiplier * self.prev_atr

                self.sell(sl=short_stop_loss)
                logging.info(f"Entered short position at {self.entry_price} with stop loss at {short_stop_loss}")

            # Exit Conditions
            if self.position.is_long and self.data.High[-1] >= self.data.upper_kc[-1]:
                self.position.close()
                logging.info(f"Closing long position at {self.data.Close[-1]} due to high >= upper KC")

            elif self.position.is_short and self.data.Low[-1] <= self.data.lower_kc[-1]:
                self.position.close()
                logging.info(f"Closing short position at {self.data.Close[-1]} due to low <= lower KC")

           
      

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise

# Load and prepare data
try:
    data_path ='/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    # logging.info(f"Minute data:\n{minute_data.head(20)}")

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
    bt = Backtest(daily_signals, KCPullbackStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")
    
    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/6_KCPullbackStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
