import pandas as pd
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/23_SupertrendStrategyStrategy_strategy.log'

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



def calculate_daily_indicators(daily_data):
    try:
        logging.info("Calculating daily indicators")

     
        # Supertrend parameters
        length1, factor1 = 7, 3
        length2, factor2 = 14, 2
        length3, factor3 = 21, 1

        # Calculate Supertrend
        supertrend1 = ta.supertrend(daily_data['High'], daily_data['Low'], daily_data['Close'], length=length1, multiplier=factor1)
        supertrend2 = ta.supertrend(daily_data['High'], daily_data['Low'], daily_data['Close'], length=length2, multiplier=factor2)
        supertrend3 = ta.supertrend(daily_data['High'], daily_data['Low'], daily_data['Close'], length=length3, multiplier=factor3)

        daily_data['Supertrend1'] = supertrend1['SUPERT_7_3.0']
        daily_data['Supertrend2'] = supertrend2['SUPERT_14_2.0']
        daily_data['Supertrend3'] = supertrend3['SUPERT_21_1.0']

        logging.info("Daily indicators calculated successfully.")
        return daily_data

    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise



def generate_signals(daily_data):
    try:
        logging.info("Generating trading signals")

        # Initialize 'signal' column with zeros
        daily_data['signal'] = 0

        # Iterate over rows in the DataFrame, starting from the second row
        for i in range(1, len(daily_data)):
            close_price = daily_data['Close'].iloc[i]
            prev_close = daily_data['Close'].iloc[i - 1]
            supertrend1 = daily_data['Supertrend1'].iloc[i]
            prev_supertrend1 = daily_data['Supertrend1'].iloc[i - 1]
            supertrend2 = daily_data['Supertrend2'].iloc[i]
            prev_supertrend2 = daily_data['Supertrend2'].iloc[i - 1]
            supertrend3 = daily_data['Supertrend3'].iloc[i]
            prev_supertrend3 = daily_data['Supertrend3'].iloc[i - 1]

            # Buy Signal Condition: Close price crosses above Supertrend
            if (
                (prev_close <= prev_supertrend1 and close_price > supertrend1) or
                (prev_close <= prev_supertrend2 and close_price > supertrend2) or
                (prev_close <= prev_supertrend3 and close_price > supertrend3)
            ):
                daily_data.at[daily_data.index[i], 'signal'] = 1  # Buy signal

            # Sell Signal Condition: Close price crosses below Supertrend
            elif (
                (prev_close >= prev_supertrend1 and close_price < supertrend1) or
                (prev_close >= prev_supertrend2 and close_price < supertrend2) or
                (prev_close >= prev_supertrend3 and close_price < supertrend3)
            ):
                daily_data.at[daily_data.index[i], 'signal'] = -1  # Sell signal

        logging.info("Signals generated successfully.")
        return daily_data

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise






class SupertrendStrategy(Strategy):
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
            current_signal = self.data.signal[-1]
            current_price = self.data.Close[-1]
            
       
            # Handle buy signal
            if current_signal == 1:
                logging.debug(f"Buy signal detected, executing buy at close={current_price}")
                self.entry_price = current_price
                self.buy()
               

            # Handle sell signal
            elif current_signal == -1:
                logging.debug(f"Sell signal detected, executing sell at close={current_price}")
                self.entry_price = current_price
                if self.position.is_long:
                      self.position.close()
                
              
                

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise




# Main script execution
try:
    # Load and prepare data
    data_path =  '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
  

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
 
    # Run backtest
    bt = Backtest(daily_signals, SupertrendStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/23_SupertrendStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in main script execution: {e}")
