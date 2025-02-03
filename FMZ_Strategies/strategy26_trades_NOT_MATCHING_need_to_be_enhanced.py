import pandas as pd
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import pandas_ta as ta
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/26_Fukuiz_Trading_Strategy_strategy.log'

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



def calculate_daily_indicators(daily_data):
    def rma(series, length):
        # Calculate the running moving average (RMA) of the series
        alpha = 1 / length
        rma = series.ewm(alpha=alpha, adjust=False).mean()
        return rma

    def rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = rma(gain, length)
        avg_loss = rma(loss, length)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Calculate RSI for short and long lengths
    daily_data['RSI_Short'] = rsi(daily_data['Close'], 24)
    daily_data['RSI_Long'] = rsi(daily_data['Close'], 100)

    logging.info(f"data after indicator calculation\n{daily_data}")

    return daily_data



def generate_signals(daily_data, lbR=5, lbL=5, rangeUpper=60, rangeLower=5, plotBull=True, plotBear=True):
    daily_data['signal'] = 0
    

    
    osc = daily_data['RSI_Short']
    osc_long = daily_data['RSI_Long']
    
    bullCond = osc > osc_long
    bearCond = osc < osc_long
    
    # Generate Bullish and Bearish Conditions
    for i in range(lbR + lbL, len(daily_data)):
        window = daily_data.iloc[i - lbL - lbR:i + 1]
        osc_window = window['RSI_Short']
        
        # Pivot detection
        plFound = osc_window[lbL] == osc_window.min()
        phFound = osc_window[lbL] == osc_window.max()
        
        oscHL = osc_window.iloc[-lbR] > osc_window.iloc[-(lbR + lbL)]
        priceLL = daily_data['Low'].iloc[i - lbR] < daily_data['Low'].iloc[i - (lbR + lbL)]
        bullCond_current = plotBull and priceLL and oscHL and plFound
        
        oscLH = osc_window.iloc[-lbR] < osc_window.iloc[-(lbR + lbL)]
        priceHH = daily_data['High'].iloc[i - lbR] > daily_data['High'].iloc[i - (lbR + lbL)]
        bearCond_current = plotBear and priceHH and oscLH and phFound
        
        if bullCond_current:
            daily_data.at[daily_data.index[i], 'signal'] = 1
        elif bearCond_current:
            daily_data.at[daily_data.index[i], 'signal'] = -1


    
    logging.info(f"data after signal calculation\n{daily_data}")

    return daily_data



class Fukuiz_Trading_Strategy(Strategy):
    def init(self):
        try:
            logging.info("Initializing Fukuiz Trading Strategy")
            self.entry_price = None
            logging.info("Initialization complete")
        except Exception as e:
            logging.error(f"Error in init method: {e}")
            raise

    def next(self):
      
        try:
            current_signal = self.data.signal[-1]
            current_price = self.data.Close[-1]
            # logging.debug(f"Processing bar: {self.data.index[-1]} with signal {current_signal} at price {current_price}")

            # Handle buy signal
            if current_signal==1:
                logging.debug(f"Buy signal detected,  at close={current_price}")
                self.entry_price = current_price
                if self.position.is_short:
                    self.position.close()
                self.buy()  


            
                   # Handle buy signal
            if current_signal==-1:
                logging.debug(f"sell signal detected,g at close={current_price}")
                self.entry_price = current_price
                if self.position.is_long:
                    self.position.close()
                self.buy() 


            

         
            logging.info("Next method processing complete")

        except Exception as e:
            logging.error(f"Error in next method: {e}")
            raise






# Main script execution
try:
    # Load and prepare data
    data_path =  '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_1h_data_2023_2024/converted_sorted_btc_1h_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
    

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)

    logging.info(f"daily_signals\n{daily_signals}")
    
    # Run backtest
    bt = Backtest(daily_signals, Fukuiz_Trading_Strategy, cash=1000000, commission=.002)
    stats = bt.run()
    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/26_Fukuiz_Trading_Strategy_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in main script execution: {e}")