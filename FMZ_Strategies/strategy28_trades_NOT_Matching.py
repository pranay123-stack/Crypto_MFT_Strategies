import pandas as pd
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/28_StochRSIMoveStrategy_strategy.log'

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





import pandas_ta as ta

def calculate_daily_indicators(df):
    try:
        logging.info("Calculating RSI, Stochastic RSI, and price change")

        lookback_period = 24  # Lookback period in bars for 30min timeframe
        rsi_length = 14  # RSI Length
        stoch_length = 14  # Stochastic RSI Length
        k = 3  # Stochastic %K
        d = 3  # Stochastic %D
        big_move_threshold = 2.5 / 100  # Big Move Threshold as percentage

        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], length=rsi_length)

        # Calculate Stochastic RSI
        stoch_rsi = ta.stochrsi(df['RSI'], length=stoch_length)

    
      
    
        df['StochRSI_K'] = ta.sma(stoch_rsi['STOCHRSIk_14_14_3_3'], length=k)
        df['StochRSI_D'] = ta.sma(df['StochRSI_K'], length=d)

        # Calculate percent price change from 12 hours ago (lookback period)
        df['Price_12hrs_Ago'] = df['Close'].shift(lookback_period - 1)
        df['Percent_Change'] = abs(df['Close'] - df['Price_12hrs_Ago']) / df['Price_12hrs_Ago']

        # # Drop NaN rows
        # df.dropna(inplace=True)

        logging.info("Indicator calculation complete")
        return df
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise


def generate_signals(df):
    try:
        logging.info("Generating signals based on Stoch RSI and price moves")

        # Initialize signal column
        df['signal'] = 0

        big_move_threshold = 2.5 / 100  # Big Move Threshold as percentage

        for i in range(len(df)):
            # Check conditions for entering long or short
            if (df['Percent_Change'].iloc[i] >= big_move_threshold) and (df['StochRSI_K'].iloc[i] < 3 or df['StochRSI_D'].iloc[i] < 3):
                df.at[df.index[i], 'signal'] = 1  # Long signal
            elif (df['Percent_Change'].iloc[i] >= big_move_threshold) and (df['StochRSI_K'].iloc[i] > 97 or df['StochRSI_D'].iloc[i] > 97):
                df.at[df.index[i], 'signal'] = -1  # Short signal

        logging.info("Signal generation complete")
        return df.dropna()
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise





class StochRSIMoveStrategy(Strategy):
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
            # Buy signal
            if self.data.signal[-1] == 1 :
                logging.debug(f"Long entry signal at close price {self.data.Close[-1]}")
                self.entry_price = self.data.Close[-1]
                if self.position:
                    if self.position.is_short:
                        self.position.close()
                        
                self.buy()

            # Sell/Short signal
            if self.data.signal[-1] == -1 :
                logging.debug(f"Short entry signal at close price {self.data.Close[-1]}")
                self.entry_price = self.data.Close[-1]
                if self.position:
                    if self.position.is_long:
                        self.position.close()
                        
                self.sell()

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise






# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)

    logging.info(f"daily_signals\n{daily_signals}")
    
    # Run backtest
    bt = Backtest(daily_signals, StochRSIMoveStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/28_StochRSIMoveStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in main script execution: {e}")
