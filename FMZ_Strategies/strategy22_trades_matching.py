import pandas as pd
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/22_VWAPStrategyStrategy_strategy.log'

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





def calculate_daily_indicators(df, cumulative_period=14):
    """
    Calculate VWAP and any other required indicators on the daily timeframe.
    """
    try:
        # Calculate typical price
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate typical price * volume
        df['Typical_Price_Volume'] = df['Typical_Price'] * df['Volume']
        
        # Calculate cumulative sums for typical price * volume and volume
        df['Cumulative_Typical_Price_Volume'] = df['Typical_Price_Volume'].rolling(window=cumulative_period).sum()
        df['Cumulative_Volume'] = df['Volume'].rolling(window=cumulative_period).sum()

        # Calculate VWAP
        df['VWAP'] = df['Cumulative_Typical_Price_Volume'] / df['Cumulative_Volume']

        # Drop any rows with NaN values (resulting from rolling calculations)
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Error in calculate_daily_indicators: {e}")
        raise



def generate_signals(df):
    """
    Generate buy/sell signals based on VWAP crossover.
    """
    try:
        # Initialize signal column
        df['signal'] = 0

        # Generate signals based on VWAP crossover
        df['long_condition'] = (df['Close'] > df['VWAP']) & (df['Close'].shift(1) < df['VWAP'].shift(1))
        df['short_condition'] = (df['Close'] < df['VWAP']) & (df['Close'].shift(1) > df['VWAP'].shift(1))

        # Set signal = 1 for long, -1 for short
        df.loc[df['long_condition'], 'signal'] = 1
        df.loc[df['short_condition'], 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error in generate_signals: {e}")
        raise






class VWAPStrategy(Strategy):
    def init(self):
        """
        Initialize any strategy-related variables or calculations.
        """
        self.entry_price = None
        self.long_profit_target = None
        self.short_profit_target = None


    def next(self):
        """
        Execute the strategy on each new bar (price data update).
        """
        try:
            # Long entry condition
            if self.data.signal[-1] == 1  :
                self.buy()
                if self.position:
                    if self.position.is_short:
                        self.position.close()
                self.entry_price = self.data.Close[-1]
                self.long_profit_target = self.entry_price * 1.03

            # Short entry condition
            elif self.data.signal[-1] == -1  :
                self.sell()
                if self.position:
                    if self.position.is_long:
                        self.position.close()
                self.entry_price = self.data.Close[-1]
                self.short_profit_target = self.entry_price * 0.97

            # Managing long position (take profit)
            if self.position.is_long:
                  
                if self.data.Close[-1] >= self.long_profit_target:
                    self.position.close()

             
            # Managing short position (take profit)
            elif self.position.is_short:
              
                if self.data.Close[-1] <= self.short_profit_target:
                    self.position.close()

        except Exception as e:
            print(f"Error in next function: {e}")
            raise




# Main script execution
try:
    # Load and prepare data
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
  

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
 
    # Run backtest
    bt = Backtest(daily_signals, VWAPStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/22_VWAPStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in main script execution: {e}")
