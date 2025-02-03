import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/13_PolynomialRegression.log'

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
        logging.info("Loading and preparing data.")
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
        logging.info("Data loading and preparation complete.")
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise




def calculate_daily_indicators(daily_data, factor=5, atr_period=10):
    try:
        # Calculate Supertrend
        supertrend = ta.supertrend(daily_data['High'], daily_data['Low'], daily_data['Close'], length=10, multiplier=5)
        print("supertrend", supertrend)
        daily_data['Supertrend'] =   supertrend['SUPERT_10_5.0']
        daily_data['Direction'] =   supertrend['SUPERTd_10_5.0']
              # Drop rows where 'Supertrend' or 'Direction' are NaN
   
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise




def generate_signals(daily_data):
    try:
        logging.info("Generating signals based on Supertrend")

        # Initialize signal column
        daily_data['signal'] = 0
        daily_data['signal_direction'] = 0
      

        for i in range(3, len(daily_data)):
            # Get current and previous values
            current_supertrend = daily_data['Supertrend'].iloc[i]
            previous_supertrend_2 = daily_data['Supertrend'].iloc[i - 2]
            previous_supertrend_3 = daily_data['Supertrend'].iloc[i - 3]
            current_direction = daily_data['Direction'].iloc[i]

            if current_direction < 0:
                if current_supertrend > previous_supertrend_2:
                    daily_data.at[daily_data.index[i], 'signal'] = 1  # Buy signal (long)
                    daily_data.at[daily_data.index[i], 'signal_direction'] = 1
              
            elif current_direction > 0:
                if current_supertrend < previous_supertrend_3:
                    daily_data.at[daily_data.index[i], 'signal'] = -1  # Sell signal (short)
                    daily_data.at[daily_data.index[i], 'signal_direction'] = -1
                
        logging.info("Signal generation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise






class SupertrendStrategy(Strategy):
    def init(self):
        # Initialization if required (e.g., to save values for later use)
        pass

    def next(self):
        # Access the last row of the data
        current_signal = self.data.signal[-1]
        current_signaldirection = self.data.signal_direction[-1]

        if  current_signal ==1:
               self.buy()

        if  current_signal ==-1:
               self.sell()
        
        if current_signaldirection <0 :
            if self.position.is_short :
                self.position.close()

        
        if current_signaldirection >0 : 
            if self.position.is_long :
                self.position.close()

        






  


  

       


      
                    

# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
  
    # Calculate daily indicators
    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
    logging.info(f"============Daily indicators========\n {daily_data}")
  

  
    # Perform backtest
    bt = Backtest(daily_signals, SupertrendStrategy, cash=1000000, commission=0.002)
    stats = bt.run()
    logging.info(f"Backtest complete with stats:\n{stats}")

    # Save trades to CSV
    trades = stats['_trades']
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/13_PolynomialRegression_backtest_trades.csv'
    trades.to_csv(trades_csv_path, index=False)
    logging.info(f"Trades saved to {trades_csv_path}")
except Exception as e:
    logging.error(f"Critical error in the main script execution: {e}")
    raise
