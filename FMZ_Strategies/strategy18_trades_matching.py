import pandas as pd
import numpy as np
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
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/18_BollingerBandsStochasticRSI.log'

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
    # Bollinger Bands calculation
    length = 20
    mult = 2.0
    df['basis'] = df['Close'].rolling(window=length).mean()
    df['stddev'] = df['Close'].rolling(window=length).std()
    df['dev'] = mult * df['stddev']
    df['upper'] = df['basis'] + df['dev']
    df['lower'] = df['basis'] - df['dev']

    # Stochastic RSI calculation
    rsi_length = 14
    stoch_length = 14
    smooth_k = 3
    smooth_d = 3
    df['rsi'] = ta.rsi(df['Close'], length=rsi_length)

  

    # Calculate Stochastic RSI using pandas_ta
    stoch_rsi_df = ta.stochrsi(df['Close'], length=rsi_length, rsi_length=stoch_length, k=smooth_k, d=smooth_d)
    print(stoch_rsi_df)
    # Add Stochastic RSI (%K and %D) to the DataFrame
    df['stoch_k'] = stoch_rsi_df['STOCHRSIk_14_14_3_3']
    df['stoch_d'] = stoch_rsi_df['STOCHRSId_14_14_3_3']
    return df



def generate_signals(df):
    upper_limit = 90
    lower_limit = 10

    # Conditions for Bearish and Bullish entries
    df['Bear'] = (df['Close'].shift(1) > df['upper'].shift(1)) & (df['Close'] < df['upper']) & \
                 (df['stoch_k'].shift(1) > upper_limit) & (df['stoch_d'].shift(1) > upper_limit)
    
    df['Bull'] = (df['Close'].shift(1) < df['lower'].shift(1)) & (df['Close'] > df['lower']) & \
                 (df['stoch_k'].shift(1) < lower_limit) & (df['stoch_d'].shift(1) < lower_limit)

    # Generating signals based on Bull and Bear conditions
    df['signal'] = 0
    df.loc[df['Bear'], 'signal'] = 1  # Enter Short
    df.loc[df['Bull'], 'signal'] = -1   # Enter Long

    return df


from backtesting import Strategy

class BollingerBandsStochasticRSI(Strategy):
    def init(self):
        # Initializing indicators here if needed
        pass

    def next(self):
        # Access the signal generated from daily data
        signal = self.data.signal[-1]
        current_price = self.data.Close[-1]
        
        if signal == 1:
            logging.debug(f"Buy signal detected, executing buy at close={current_price}")
            self.entry_price = current_price
            if self.position:
                if self.position.is_short:
                    logging.debug("Closing short position before opening long")
                    self.position.close()
                elif self.position.is_long:
                    logging.debug("Already in long position, no action needed")
                    return
            self.buy()  # Enter Long position
          
        

           
               
        elif signal == -1:  # Bearish signal
            if self.position:
                if self.position.is_long:
                    logging.debug("Closing long position before opening short")
                    self.position.close()
                elif self.position.is_short:
                    logging.debug("Already in short position, no action needed")
                    return
            self.sell()  # Enter Short position





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
    bt = Backtest(daily_signals,  BollingerBandsStochasticRSI, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
  

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/18_BollingerBandsStochasticRSI_backtest_trades.csv'
    trades.to_csv(trades_csv_path)



except Exception as e:
    logging.error(f"Error in main script execution: {e}")

