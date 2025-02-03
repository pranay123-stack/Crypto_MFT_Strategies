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

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/15_VWAPMTFStockStrategy.log'

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
    # Calculate VWAP
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['sumSrc'] = (df['ohlc4'] * df['Volume']).cumsum()
    df['sumVol'] = df['Volume'].cumsum()
    df['vwapW'] = df['sumSrc'] / df['sumVol']
    
    # Custom calculation of source
    df['h'] = np.power(df['High'], 2) / 2
    df['l'] = np.power(df['Low'], 2) / 2
    df['o'] = np.power(df['Open'], 2) / 2
    df['c'] = np.power(df['Close'], 2) / 2
    df['source'] = np.sqrt((df['h'] + df['l'] + df['o'] + df['c']) / 4)
    
    # Moving Average and Range calculation
    length = 27
    mult = 0
    df['ma'] = df['source'].rolling(window=length).mean()
    df['range'] = df['High'] - df['Low']
    df['rangema'] = df['range'].rolling(window=length).mean()
    df['upper'] = df['ma'] + df['rangema'] * mult
    df['lower'] = df['ma'] - df['rangema'] * mult
    
    return df




def generate_signals(df):
    # Signal conditions based on indicator crossovers and VWAP conditions
    df['crossUpper'] = (df['source'] > df['upper']) & (df['source'].shift(1) <= df['upper'].shift(1))
    df['crossLower'] = (df['source'] < df['lower']) & (df['source'].shift(1) >= df['lower'].shift(1))
    
    df['bprice'] = np.where(df['crossUpper'], df['High'] + 0.01, np.nan)
    df['bprice'] = df['bprice'].fillna(method='ffill')
    
    df['sprice'] = np.where(df['crossLower'], df['Low'] - 0.01, np.nan)
    df['sprice'] = df['sprice'].fillna(method='ffill')
    
    df['crossBcond'] = df['crossUpper']
    df['crossBcond'] = np.where(df['crossBcond'].isna(), False, df['crossBcond'])
    
    df['crossScond'] = df['crossLower']
    df['crossScond'] = np.where(df['crossScond'].isna(), False, df['crossScond'])
    
    df['cancelBcond'] = df['crossBcond'] & ((df['source'] < df['ma']) | (df['High'] >= df['bprice']))
    df['cancelScond'] = df['crossScond'] & ((df['source'] > df['ma']) | (df['Low'] <= df['sprice']))
    
    # Long and short conditions based on VWAP
    df['longCondition'] = (df['Close'] > df['vwapW'])
    df['shortCondition'] = (df['Close'] < df['vwapW'])
    
    df['signal'] = 0
    df.loc[df['crossUpper'], 'signal'] = 1
    df.loc[df['crossLower'], 'signal'] = -1
    
    return df



from backtesting import Strategy

class VWAPMTFStockStrategy(Strategy):
    def init(self):
       # Initialize the strategy with a 27-day moving average and a 10-day range multiplier
        self.entry_price = None
        
     
    def next(self):
     
        # Long entry logic
        if self.data.signal == 1 :
            self.buy(stop=self.data.bprice[-1])

            # Long entry logic
        if self.data.signal == -1:
            if self.position.is_long:
                self.position.close()
              


     




# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    # logging.info(f"Minute data:\n{minute_data.head(20)}")

    daily_data = calculate_daily_indicators(minute_data)
    logging.info(f"Daily data:\n{daily_data.tail(20)}")

    daily_signals = generate_signals(daily_data)
    filtered_generated_signals_on_resampled_Data = daily_signals[(daily_signals['signal'] == 1) | (daily_signals['signal'] == -1)]
    logging.info(f"======= data WITH SIGNAL 1 OR -1 ======= :\n{filtered_generated_signals_on_resampled_Data}")



    # Run backtest
    bt = Backtest(daily_signals,  VWAPMTFStockStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
  

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/15_VWAPMTFStockStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)



except Exception as e:
    logging.error(f"Error in main script execution: {e}")

