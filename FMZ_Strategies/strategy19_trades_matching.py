import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
from backtesting.lib import crossover
import logging
import coloredlogs
import numpy as np
import pandas_ta as ta
import logging
import pandas as pd
import pandas_ta as ta


def load_and_prepare_data(csv_file_path):
    try:
        
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
       
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise



def calculate_daily_indicators(df):
    try:
        # KDJ calculation (using highest, lowest, and simple moving averages)
        kdj_length = 9
        kdj_signal = 3

        kdj_highest = df['High'].rolling(window=kdj_length).max()
        kdj_lowest = df['Low'].rolling(window=kdj_length).min()
        kdj_rsv = 100 * (df['Close'] - kdj_lowest) / (kdj_highest - kdj_lowest)
        df['K'] = kdj_rsv.rolling(window=kdj_signal).mean()
        df['D'] = df['K'].rolling(window=kdj_signal).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        # Moving Average calculation
        ma_length = 20
        df['MA'] = df['Close'].rolling(window=ma_length).mean()

        # Drop NaN rows created by rolling windows
        df.dropna(inplace=True)

        return df

    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise



def generate_signals(df):
    try:
        logging.info("Generating signals")

        # Initialize signal column
        df['signal'] = 0

        # Define KDJ overbought and oversold levels
        kdj_overbought = 80
        kdj_oversold = 20

        # Moving Average crossovers
        df['ma_cross_up'] = (df['Close'] > df['MA']) & (df['Close'].shift(1) <= df['MA'].shift(1))
        df['ma_cross_down'] = (df['Close'] < df['MA']) & (df['Close'].shift(1) >= df['MA'].shift(1))

        # Generate Buy (Long) and Sell (Short) signals
        df.loc[(df['J'] <= kdj_oversold) & df['ma_cross_up'], 'signal'] = 1  # Buy
        df.loc[(df['J'] >= kdj_overbought) & df['ma_cross_down'], 'signal'] = -1  # Sell

        return df

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise





class KDJMAStrategy(Strategy):
    def init(self):
        self.entry_price = None

    def next(self):
        if self.data.signal[-1] == 1:
            if self.position:
                    if self.position.is_short:
                        self.position.close()
            self.buy()
           
                        
          
        elif self.data.signal[-1] == -1 :
            if self.position:
                    if self.position.is_long:
                        self.position.close()
                        
            self.sell()
            
         


# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)


    daily_data = calculate_daily_indicators(minute_data)


    daily_signals = generate_signals(daily_data)

        # Log rows where signal column is 1
    filtered_df = daily_signals[daily_signals['signal'] == 1]

    logging.info(f"Rows with signal column equal to 1:\n{filtered_df}")
        

    # Run backtest
    bt = Backtest(daily_signals, KDJMAStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    print(stats)
  

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/19_ KDJMAStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)



except Exception as e:
    logging.error(f"Error in main script execution: {e}")

