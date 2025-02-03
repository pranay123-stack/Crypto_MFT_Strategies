import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs


# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/21_TRMUSStrategy_strategy.log'

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
        # Parameters
        maLength = 50
        lengthATR = 14
        multiplier = 1.5
        length = 20

        # Calculate Simple Moving Average (SMA)
        daily_data['ma'] = daily_data['Close'].rolling(window=maLength).mean()

        # Calculate ATR using pandas_ta
        daily_data['atr'] = ta.atr(daily_data['High'], daily_data['Low'], daily_data['Close'], length=lengthATR)

        # Calculate Alpha Trend levels
        daily_data['upperLevel'] = daily_data['Close'] + (multiplier * daily_data['atr'])
        daily_data['lowerLevel'] = daily_data['Close'] - (multiplier * daily_data['atr'])

        # Initialize Alpha Trend with NaN
        daily_data['alphaTrend'] = np.nan

 
        # Calculate Alpha Trend
        for i in range(1, len(daily_data)):
            current_close = daily_data['Close'].iloc[i]
            current_upper = daily_data['upperLevel'].iloc[i]
            current_lower = daily_data['lowerLevel'].iloc[i]
            prev_alpha_trend = daily_data['alphaTrend'].iloc[i - 1]

            if pd.isna(prev_alpha_trend):
                daily_data.at[daily_data.index[i], 'alphaTrend'] = current_close
            elif current_close > daily_data['lowerLevel'].iloc[i - 1]:
                daily_data.at[daily_data.index[i], 'alphaTrend'] = max(prev_alpha_trend, current_lower)
            elif current_close < daily_data['upperLevel'].iloc[i - 1]:
                daily_data.at[daily_data.index[i], 'alphaTrend'] = min(prev_alpha_trend, current_upper)
            else:
                daily_data.at[daily_data.index[i], 'alphaTrend'] = prev_alpha_trend

        # Calculate highest and lowest close over the specified window
        daily_data['highestClose'] = daily_data['Close'].rolling(window=length).max()
        daily_data['lowestClose'] = daily_data['Close'].rolling(window=length).min()

        return daily_data

    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise


def generate_signals(daily_data):
    try:
        # Initialize the 'signal' column with zeros
        daily_data['signal'] = 0

        # Loop through each row of the DataFrame
        for i in range(1, len(daily_data)):
            # Calculate the buy signal
            if (daily_data['Close'].iloc[i] > daily_data['highestClose'].iloc[i-1] and
                daily_data['Close'].iloc[i-1] <= daily_data['highestClose'].iloc[i-1] and
                daily_data['Close'].iloc[i] > daily_data['ma'].iloc[i] and
                daily_data['Close'].iloc[i] > daily_data['alphaTrend'].iloc[i]):
                daily_data['signal'].iloc[i] = 1
            
            # Calculate the sell signal
            elif (daily_data['Close'].iloc[i] < daily_data['lowestClose'].iloc[i-1] and
                  daily_data['Close'].iloc[i-1] >= daily_data['lowestClose'].iloc[i-1] and
                  daily_data['Close'].iloc[i] < daily_data['ma'].iloc[i] and
                  daily_data['Close'].iloc[i] < daily_data['alphaTrend'].iloc[i]):
                daily_data['signal'].iloc[i] = -1

        return daily_data
    
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise


class TRMUSStrategy(Strategy):
    stop_loss_perc = 0.02
    take_profit_perc = 0.04

    def init(self):
         logging.info("Initializing strategy")

    def next(self):
        try:
            if self.data.signal[-1] == 1:

                if self.position:
                        if self.position.is_short:
                            logging.debug("Closing short position before opening long")
                            self.position.close()
                        elif self.position.is_long:
                            logging.debug("Already in long position, no action needed")
                            return
                self.buy(
                    stop=self.data['Close'][-1] * (1 - self.stop_loss_perc),
                    limit=self.data['Close'][-1] * (1 + self.take_profit_perc)
                )
            elif self.data.signal[-1] == -1:
                if self.position:
                    if self.position.is_long:
                        logging.debug("Closing long position before opening short")
                        self.position.close()
                    elif self.position.is_short:
                        logging.debug("Already in short position, no action needed")
                        return
                self.sell(
                    stop=self.data['Close'][-1] * (1 + self.stop_loss_perc),
                    limit=self.data['Close'][-1] * (1 - self.take_profit_perc)
                )
        except Exception as e:
            logging.error(f"Error in TRMUSStrategy next: {e}")
            raise


# Load and prepare data
try:
    data_path =  '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)

    bt = Backtest(daily_signals, TRMUSStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/21_TRMUSStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
