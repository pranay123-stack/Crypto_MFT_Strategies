import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/4_retracement_strategy.log'
logging.basicConfig(level=logging.DEBUG, filename=log_file_path, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
coloredlogs.install(level='DEBUG',
                    fmt='%(asctime)s - %(levelname)s - %(message)s',
                    level_styles={
                        'info': {'color': 'green'},
                        'debug': {'color': 'white'},
                        'error': {'color': 'red'},
                    })

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

def calculate_daily_indicators(data):


    # Calculate the 50-period Simple Moving Average (SMA)
    data['ma_50'] = ta.sma(data['Close'], length=50)

    # Calculate the 200-period Simple Moving Average (SMA)
    data['ma_200'] = ta.sma(data['Close'], length=200)


    # Define lengths
    len_21 = 21
    len_50 = 50
    len_9 = 9
    
    # Initialize Fibonacci levels with NaN
    data['fib_50_level'] = np.nan
    data['fib_786_level'] = np.nan
    
    # Calculate retracement levels only when close > MA_200 and close > MA_50
    condition = (data['Close'] > data['ma_200']) & (data['Close'] > data['ma_50'])
    
    data.loc[condition, 'retrace_21_high'] = data['High'].rolling(window=len_21).max()
    data.loc[condition, 'retrace_21_low'] = data['Low'].rolling(window=len_21).min()
    data.loc[condition, 'retrace_21_mid'] = (data['retrace_21_high'] + data['retrace_21_low']) / 2

    data.loc[condition, 'retrace_50_high'] = data['High'].rolling(window=len_50).max()
    data.loc[condition, 'retrace_50_low'] = data['Low'].rolling(window=len_50).min()
    data.loc[condition, 'retrace_50_mid'] = (data['retrace_50_high'] + data['retrace_50_low']) / 2

    data.loc[condition, 'retrace_9_high'] = data['High'].rolling(window=len_9).max()
    data.loc[condition, 'retrace_9_low'] = data['Low'].rolling(window=len_9).min()
    data.loc[condition, 'retrace_9_mid'] = (data['retrace_9_high'] + data['retrace_9_low']) / 2
    
    # Calculate the Fibonacci levels only for the filtered rows
    data.loc[condition, 'fib_50_level'] = (data['retrace_21_mid'] + data['retrace_50_mid'] + data['retrace_9_mid']) / 3
        # Apply the calculation for 'fib_786_level' directly to the DataFrame
    data.loc[condition, 'fib_786_level'] = (
        (data.loc[condition, 'retrace_21_high'] + data.loc[condition, 'retrace_50_high'] + data.loc[condition, 'retrace_9_high']) / 3 -
        ((data.loc[condition, 'retrace_21_high'] + data.loc[condition, 'retrace_50_high'] + data.loc[condition, 'retrace_9_high'] - 
        data.loc[condition, 'retrace_21_low'] - data.loc[condition, 'retrace_50_low'] - data.loc[condition, 'retrace_9_low']) * 0.786)
    )
    return data.dropna()
def generate_signals(daily_data):
    try:
        logging.info("Generating signals based on strategy logic")
        daily_data['long_condition'] = (
            (daily_data['Close'] > daily_data['ma_200']) &
            (daily_data['Close'] > daily_data['ma_50']) &
            (daily_data['Close'] <= daily_data['fib_50_level'])
        )

        daily_data['signal'] = np.where(daily_data['long_condition'], 1, 0)
        logging.info("Signal generation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise

class RetracementStrategy(Strategy):
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
            latest_close = self.data.Close[-1]
            latest_signal = self.data.signal[-1]
            # logging.debug(f"Processing bar: {self.data.index[-1]} with signal {latest_signal} at price {latest_close}")

            if latest_signal == 1 :
                logging.debug(f"Buy signal detected, close={latest_close}")
                self.entry_price = latest_close
                risk_reward_ratio = 2.0
                take_profit_level = self.entry_price + (self.entry_price - self.data.fib_786_level[-1]) * risk_reward_ratio
                stop_loss_level = self.data.fib_786_level[-1]
                # if self.position:
                #      self.position.close()

                self.buy(sl=stop_loss_level,tp= take_profit_level)

                    
                       
                   
           
        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise

# Process the data
try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
    logging.info(f" daily_signals {daily_signals}")
   
    bt = Backtest(daily_signals, RetracementStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    # bt.plot()

    logging.info(stats)
    logging.info("Backtest complete")

    trades = stats['_trades']
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/4_RetracementStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
