import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import logging
import coloredlogs
import pandas_ta as ta

#refer strategy 21 -for seeing buy and sell function same like

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/2_ComboStrategy.log'

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
        data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        logging.info("Data loading and preparation complete")
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise




def calculate_daily_indicators(df):
    # Calculate the required indicators
    df['SMA01'] = ta.sma(df['Close'], length=3)
    df['SMA02'] = ta.sma(df['Close'], length=8)
    df['SMA03'] = ta.sma(df['Close'], length=10)
    df['EMA01'] = ta.ema(df['Close'], length=5)
    df['EMA02'] = ta.ema(df['Close'], length=3)
    df['OHLC'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0

    # Ensure no NaN values from indicator calculations
    df.dropna(inplace=True)

    return df

def generate_signals(df):
    # Initialize signal columns
    df['signal'] = 0

    # Entry01 conditions
    df['Cond01'] = df['Close'] < df['SMA03']
    df['Cond02'] = df['Close'] <= df['SMA01']
    df['Cond03'] = df['Close'].shift(1) > df['SMA01'].shift(1)
    df['Cond04'] = df['Open'] > df['EMA01']
    df['Cond05'] = df['SMA02'] < df['SMA02'].shift(1)
    
    df['Entry01'] = df['Cond01'] & df['Cond02'] & df['Cond03'] & df['Cond04'] & df['Cond05']

    # Entry02 conditions
    df['Cond06'] = df['Close'] < df['EMA02']
    df['Cond07'] = df['Open'] > df['OHLC']
    df['Cond08'] = df['Volume'] <= df['Volume'].shift(1)

    # Fixing Cond09
    shifted_open = df['Open'].shift(1)
    shifted_close = df['Close'].shift(1)
    
    df['Cond09'] = (df['Close'] < shifted_open.combine(shifted_close, min)) | (df['Close'] > shifted_open.combine(shifted_close, max))
    
    df['Entry02'] = df['Cond06'] & df['Cond07'] & df['Cond08'] & df['Cond09']

    # Generate signals
    df['buy_condition'] = (df['Entry01'] | df['Entry02']).astype(int)

    # Update signal column where buy_condition is True (set to 1 for buy signal)
    df.loc[df['buy_condition'] == 1, 'signal'] = 1

    return df




class ComboStrategy(Strategy):
    def init(self):
        # Initialize variables
        self.BarsSinceEntry = None
        self.MaxProfitCount = 0  # Initialize MaxProfitCount as 0
        self.MaxBars = 10  # Maximum bars to hold position
        self.position_avg_price = None  # Track average price of the position

    def next(self):
        # Initialize BarsSinceEntry if it's the first bar of the strategy
        if self.BarsSinceEntry is None:
            self.BarsSinceEntry = 0

        # Cond00: Check if no position is open
        Cond00 = self.position.size == 0

        # Update BarsSinceEntry
        if Cond00:
            self.BarsSinceEntry = 0  # Reset BarsSinceEntry if no position
        else:
            # Increment BarsSinceEntry if there is an open position
            self.BarsSinceEntry += 1

          # Update BarsSinceEntry
        if Cond00:
             self.MaxProfitCount = 0  # Reset BarsSinceEntry if no position
        else:
              # If the current close price is greater than the average entry price and BarsSinceEntry > 1
            if self.data.Close[-1] > self.position_avg_price and self.BarsSinceEntry > 1:
                self.MaxProfitCount += 1  # Increment MaxProfitCount
                logging.info(f"MaxProfitCount incremented: {self.MaxProfitCount}")

           

        # Check if we should enter a position based on signals
        if self.data.signal[-1] == 1 and self.position.size == 0:
            self.buy(size=1)
            self.position_avg_price = self.data.Close[-1]  # Store the entry price
           

        # Exit the position if BarsSinceEntry exceeds MaxBars or MaxProfitCount exceeds threshold
        if (self.BarsSinceEntry-1) >= self.MaxBars or self.MaxProfitCount >= 5:
            self.position.close()
            logging.info(f"Position closed at {self.data.Close[-1]}")


   

try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    # logging.info(f"Minute data:\n{minute_data.head(20)}")

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
   
    bt = Backtest(daily_signals, ComboStrategy, cash=1000000, commission=.002)
    stats = bt.run()
  
    logging.info(stats)
    logging.info("Backtest complete")

    trades = stats['_trades']
    trades.to_csv('/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/2_ComboStrategy_trades.csv')

except Exception as e:
    logging.error(f"Error during strategy execution: {e}")
    raise
