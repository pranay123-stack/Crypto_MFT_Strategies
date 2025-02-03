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
from datetime import time

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/1_GoldenHarmonyBreakoutStrategy.log'

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
        # Calculate EMAs, HMA, and SMA using pandas_ta
        daily_data['fast_ema'] = ta.ema(daily_data['Close'], length=9)
        daily_data['slow_ema'] = ta.ema(daily_data['Close'], length=21)
        daily_data['ema_200'] = ta.ema(daily_data['Close'], length=200)
        daily_data['hma_300'] = ta.hma(daily_data['Close'], length=300)
        daily_data['ma_18'] = ta.sma(daily_data['Close'], length=18)

        # Initialize columns for Fibonacci levels
        daily_data['fib_618'] = np.nan
        daily_data['fib_65'] = np.nan

        # Initialize variables for low, high, and a flag for the first crossover
        low = np.nan
        high = np.nan
        first_crossover = False

        # Loop through data to calculate indicators and Fibonacci levels
        for i in range(1, len(daily_data)):
            # Get current and previous values of EMAs
            prev_fast_ema = daily_data['fast_ema'].iloc[i - 1]
            curr_fast_ema = daily_data['fast_ema'].iloc[i]
            prev_slow_ema = daily_data['slow_ema'].iloc[i - 1]
            curr_slow_ema = daily_data['slow_ema'].iloc[i]

            # Check if fast EMA crosses above slow EMA (crossover)
            if prev_fast_ema < prev_slow_ema and curr_fast_ema > curr_slow_ema:
                if not first_crossover:  # If this is the first crossover
                    # Initialize low and high at the first crossover
                    low = daily_data['Close'].iloc[i] if np.isnan(low) else low
                    high = daily_data['Close'].iloc[i] if np.isnan(high) else high
                    first_crossover = True
                else:  # Update low and high after the first crossover
                    low = min(low, daily_data['Close'].iloc[i]) if not np.isnan(low) else daily_data['Close'].iloc[i]
                    high = max(high, daily_data['Close'].iloc[i]) if not np.isnan(high) else daily_data['Close'].iloc[i]
            
            # Check if fast EMA crosses below slow EMA (crossunder)
            elif prev_fast_ema > prev_slow_ema and curr_fast_ema < curr_slow_ema:
                # Reset low, high, and the first crossover flag
                low = np.nan
                high = np.nan
                first_crossover = False

            # Calculate Fibonacci levels if low and high are set
            if not np.isnan(low) and not np.isnan(high):
                daily_data.at[daily_data.index[i], 'fib_618'] = high - (high - low) * 0.618
                daily_data.at[daily_data.index[i], 'fib_65'] = high - (high - low) * 0.65

        return daily_data

    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise




def generate_signals(daily_data):
    try:
        daily_data['signal'] = 0  # Initialize signal column

        # Loop through data to check conditions for buy and sell signals
        for i in range(1, len(daily_data)):
            # Previous and current Close prices
            prev_close = daily_data['Close'].iloc[i - 1]
            curr_close = daily_data['Close'].iloc[i]

            # Previous and current Fibonacci 0.618 level
            prev_fib_618 = daily_data['fib_618'].iloc[i - 1]
            curr_fib_618 = daily_data['fib_618'].iloc[i]

            # Check for Buy Signal: Close price crosses above the Fibonacci 0.618 level
            if not np.isnan(curr_fib_618) and not np.isnan(prev_fib_618) and prev_close < prev_fib_618 and curr_close > curr_fib_618:
                daily_data.at[daily_data.index[i], 'signal'] = 1

            # Check for Sell Signal: Close price crosses below the Fibonacci 0.618 level
            elif not np.isnan(curr_fib_618) and not np.isnan(prev_fib_618) and prev_close > prev_fib_618 and curr_close < curr_fib_618:
                daily_data.at[daily_data.index[i], 'signal'] = -1

        return daily_data

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise




class GoldenHarmonyBreakoutStrategy(Strategy):
    def init(self):
        try:
            
          
          
            self.entry_price = None
            self.position_open_date = None  # To track the date when the position was opened


           
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise


    def next(self):
        try:
            current_signal = self.data.signal[-1] 
            current_price = self.data.Close[-1]

            current_datetime = self.data.index[-1]
            current_date = current_datetime.date()
            current_time_only = current_datetime.time()

            # Get the index of the current bar
            current_bar_index = self.data.index.get_loc(current_datetime)

            # # Check if there is a next bar
            # if self.position:
            #     if current_bar_index + 1 < len(self.data.index):
            #         next_datetime = self.data.index[current_bar_index + 1]
            #         next_date = next_datetime.date()
            #         if next_date > current_date:
            #             # This is the last bar of the day
            #             self.position.close()
            #             logging.info(f"Closing position at the last bar of the day: {current_datetime}")
            #             self.position_open_date = None
                    
            #     else:
            #         # This is the last bar in the data
            #         self.position.close()
            #         logging.info(f"Closing position at the last bar of the data: {current_datetime}")
            #         self.position_open_date = None
        

         
        
            # Handle buy signal
            if current_signal == 1:
                if self.position:
                    if self.position.is_short:
                        self.position.close()
                        
                         
                self.buy()
                self.entry_price = current_price
                self.position_open_date = current_date
            

        
          
                            
           
            # Handle sell signal
            if current_signal == -1:
                if self.position:
                    if self.position.is_long:
                        self.position.close()
                        

                    
                self.sell()
                self.entry_price = current_price
                self.position_open_date = current_date
            
                            

        

                  
             


                            
           
          

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise





# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    # logging.info(f"Minute data:\n{minute_data.head(20)}")

    daily_data = calculate_daily_indicators(minute_data)
    logging.info(f"Daily data:\n{daily_data.tail(20)}")

    daily_signals = generate_signals(daily_data)
        # Filter the DataFrame
    filtered_generated_signals_on_resampled_Data = daily_signals[
        ((daily_signals['signal'] == 1) | (daily_signals['signal'] == -1)) &
        (daily_signals.index.date == pd.to_datetime('2023-01-24').date())
    ]

        # Log the filtered data
    logging.info(f"======= DATA WITH SIGNAL 1 OR -1 ON 2023-01-24 =======\n{filtered_generated_signals_on_resampled_Data}")



    
      # Run backtest
    bt = Backtest(
    daily_signals,
    GoldenHarmonyBreakoutStrategy,
    cash=100000,
    commission=0.002,
    exclusive_orders=False,
    trade_on_close = False




    )

    stats = bt.run()
    logging.info(stats)
  

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/1_GoldenHarmonyBreakoutStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)



except Exception as e:
    logging.error(f"Error in main script execution: {e}")

