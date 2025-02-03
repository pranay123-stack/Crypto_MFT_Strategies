import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import coloredlogs
import pandas_ta as ta

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/14_pullback_strategy.log'
logging.basicConfig(level=logging.DEBUG, filename=log_file_path, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

coloredlogs.install(level='DEBUG',
                    fmt='%(asctime)s - %(levelname)s - %(message)s',
                    level_styles={
                        'info': {'color': 'green'},
                        'debug': {'color': 'white'},
                        'error': {'color': 'red'},
                    })


def load_and_prepare_data(csv_file_path):
    try:
        logging.info("Loading data from CSV")
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
        logging.info("Data loading and preparation complete")
        return data
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}")
        raise


def calculate_daily_indicators(daily_data, ma_length1=200, ma_length2=13, too_deep=0.27, too_thin=0.03):
    try:
        daily_data['ma1'] = ta.sma(daily_data['Close'], ma_length1)
        daily_data['ma2'] = ta.sma(daily_data['Close'], ma_length2)

        daily_data['too_deep'] = (daily_data['ma2'] / daily_data['ma1'] - 1) < too_deep
        daily_data['too_thin'] = (daily_data['ma2'] / daily_data['ma1'] - 1) > too_thin

        logging.info("Daily indicator calculation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_daily_indicators: {e}")
        raise


def generate_signals(daily_data, stop_loss=0.07):
    try:
        logging.info("Generating signals based on strategy logic")

        # Initialize the 'signal' column
        daily_data['signal'] = 0

        # Iterate over each row to apply the logic
        for i in range(1, len(daily_data)):
            # Directly access values for the current and previous row
            buy_condition = (daily_data.iloc[i]['Close'] > daily_data.iloc[i]['ma1']) and \
                            (daily_data.iloc[i]['Close'] < daily_data.iloc[i]['ma2']) and \
                            daily_data.iloc[i]['too_deep'] and \
                            daily_data.iloc[i]['too_thin']
            
            close_condition1 = (daily_data.iloc[i]['Close'] > daily_data.iloc[i]['ma2']) and \
                               (daily_data.iloc[i]['Close'] < daily_data.iloc[i - 1]['Low'])

            # Use NaN for stop_distance if position_size is 0
            stop_distance = np.nan
            if daily_data.iloc[i].get('stop_distance') is not np.nan:
                stop_distance = (daily_data.iloc[i - 1]['Close'] - daily_data.iloc[i]['Close']) / daily_data.iloc[i]['Close']
            
            close_condition2 = stop_distance > stop_loss if not np.isnan(stop_distance) else False

            if buy_condition:
                daily_data.at[daily_data.index[i], 'signal'] = 1
            elif close_condition1 or close_condition2:
                daily_data.at[daily_data.index[i], 'signal'] = -1

        logging.info("Signal generation complete")
        return daily_data
    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise


class PullbackStrategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing strategy")
            self.entry_price = None
            self.stop_loss = None
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

    def next(self):
        try:
            current_price = self.data.Close[-1]
            
            # Check for buy signal
            if self.data.signal[-1] == 1:
                logging.debug(f"Buy signal detected, close={current_price}")
                self.entry_price = current_price
                self.stop_loss = self.entry_price * (1 - 0.07)  # 7% Stop Loss
                self.buy()
            
            # Check for close conditions including stop-loss
            if self.position.is_long:
                stop_distance = (self.entry_price - current_price) / current_price
                
                # Check for stop-loss condition
                if stop_distance > 0.07:
                    logging.debug(f"Stop-loss triggered at {current_price}. Closing position.")
                    self.position.close()
                # Check for breakout condition
                elif self.data.signal[-1] == -1:
                    logging.debug(f"Close signal detected, close={current_price}")
                    self.position.close()

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise


try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
   
    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)
  
    bt = Backtest(daily_signals, PullbackStrategy, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")

    trades = stats['_trades']
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/14_PullbackStrategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
