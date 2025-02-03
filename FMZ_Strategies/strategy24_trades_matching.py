import pandas as pd
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/24_VolatilityBreakoutStrategy_strategy.log'

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
    # ATR calculation
    df['tr'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift()), 
                                     abs(df['Low'] - df['Close'].shift())))
        # ATR Calculation using pandas_ta
    atrLength = 14
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=atrLength)


    # Bollinger Bands calculation
    df['basis'] = df['Close'].rolling(window=20).mean()
    df['deviation'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['basis'] + (2 * df['deviation'])
    df['lower_band'] = df['basis'] - (2 * df['deviation'])

        # RSI Calculation using pandas_ta
    rsiLength = 14
    df['rsi'] = ta.rsi(df['Close'], length=rsiLength)


    # MACD Calculation using pandas_ta
    macdShortLength = 12
    macdLongLength = 26
    macdSignalSmoothing = 9
    macd = ta.macd(df['Close'], fast=macdShortLength, slow=macdLongLength, signal=macdSignalSmoothing)
    df['macd_line'] = macd['MACD_12_26_9']
    df['signal_line'] = macd['MACDs_12_26_9']


    return df






def generate_signals(df):
    df['long_condition'] = (df['Close'].shift(1) > df['upper_band'].shift(1)) & (df['rsi'] > 50) & (df['macd_line'] > df['signal_line'])
    df['short_condition'] = (df['Close'].shift(1) < df['lower_band'].shift(1)) & (df['rsi'] < 50) & (df['macd_line'] < df['signal_line'])

    # Reversed strategy logic
    df['signal'] = 0
    df.loc[df['long_condition'], 'signal'] = -1  # Sell (short) signal
    df.loc[df['short_condition'], 'signal'] = 1   # Buy (long) signal

    return df




class VolatilityBreakoutStrategy(Strategy):
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
            if current_signal == -1:
                logging.debug(f"Buy signal detected, executing sell at close={current_price}")
                self.entry_price = current_price
                if self.position.is_long:
                     self.position.close()
                self.sell()  
                

            # Handle sell signal
            elif current_signal == 1:
                logging.debug(f"Sell signal detected, executing buy at close={current_price}")
                self.entry_price = current_price
                if self.position.is_short:
                     self.position.close()
                self.buy()  
               


            
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
    bt = Backtest(daily_signals, VolatilityBreakoutStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/24_VolatilityBreakoutStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in main script execution: {e}")
