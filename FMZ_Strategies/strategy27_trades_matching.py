import pandas as pd
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/27_BollingerMacdRsiStrategy_strategy.log'

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
        
        # Drop rows with missing values
        data = data.dropna()
        
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise

# Function to calculate indicators on the DataFrame
def calculate_daily_indicators(df, bb_length=20, bb_multiplier=2.0, macd_fast=12, macd_slow=26, macd_signal=9, rsi_length=14):
    try:
        logging.info("Calculating Bollinger Bands, MACD, and RSI on data")

        # Bollinger Bands
        df['bb_basis'] = ta.sma(df['Close'], length=bb_length)
        df['bb_dev'] = bb_multiplier * ta.stdev(df['Close'], length=bb_length)
        df['bb_upper'] = df['bb_basis'] + df['bb_dev']
        df['bb_lower'] = df['bb_basis'] - df['bb_dev']

        # MACD
        macd = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        print("macd: ", macd)
        df['macd_line']=macd['MACD_12_26_9']
        df['signal_line']=macd['MACDs_12_26_9']
        df['macd_hist'] =macd['MACDh_12_26_9']

        # RSI
        df['rsi'] = ta.rsi(df['Close'], length=rsi_length)

        # Drop NaN values
        df.dropna(inplace=True)
        logging.info("Indicator calculation complete")
        return df
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise


# Function to generate buy/sell signals
def generate_signals(df, rsi_oversold=30, rsi_overbought=70):
    try:
        logging.info("Generating buy/sell signals")

        # Buy signal: price below lower Bollinger band, MACD line > signal line, RSI < oversold level
        df['buy_signal'] = (df['Close'] < df['bb_lower']) & (df['macd_line'] > df['signal_line']) & (df['rsi'] < rsi_oversold)

        # Sell signal: price above upper Bollinger band, MACD line < signal line, RSI > overbought level
        df['sell_signal'] = (df['Close'] > df['bb_upper']) & (df['macd_line'] < df['signal_line']) & (df['rsi'] > rsi_overbought)

        # Create the 'signal' column
        # 1 for buy, -1 for sell, and 0 for no signal
        df['signal'] = 0  # Default to no signal
        df.loc[df['buy_signal'], 'signal'] = 1  # Buy signal
        df.loc[df['sell_signal'], 'signal'] = -1  # Sell signal

        logging.info("Signal generation complete")
        return df
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise


# Define the strategy class
class BollingerMacdRsiStrategy(Strategy):
    def init(self):
        self.entry_price = None

    def next(self):
        # Buy signal
        if self.data.signal[-1] == 1:
            if self.position:
                    if self.position.is_short:
                        self.position.close()
            self.buy()

        # Sell signal
        elif self.data.signal[-1] == -1:
            if self.position:
                    if self.position.is_long:
                        self.position.close()
            self.sell()




# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_1h_data_2023_2024/converted_sorted_btc_1h_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)

    logging.info(f"daily_signals\n{daily_signals}")
    
    # Run backtest
    bt = Backtest(daily_signals, BollingerMacdRsiStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/27_BollingerMacdRsiStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in main script execution: {e}")