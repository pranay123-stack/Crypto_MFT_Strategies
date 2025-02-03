import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import logging
import coloredlogs

# Set up logging
log_file_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/17_EMARSI_Cross_strategy.log'

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
# Function to load and prepare data from a CSV file
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

# 2. Indicator Calculation
def calculate_daily_indicators(daily_data, rsi_period=7, ema_period=50, atr_length=14):
    try:
        logging.info("Calculating indicators")

      

          # Calculate RSI excluding the latest close at each index
        daily_data['RSI'] = ta.rsi(daily_data['Close'].shift(1), length=rsi_period)
        
        
        # Calculate EMA for the entire series
        daily_data['EMA'] = ta.ema(daily_data['Close'], length=ema_period)
        
        # Calculate ATR for the entire series
        daily_data['ATR'] = ta.atr(daily_data['High'], daily_data['Low'], daily_data['Close'], length=atr_length)
        
        daily_data.dropna(inplace=True)
        logging.info(f"Indicator calculation complete\n{daily_data.head(20)}")
        return daily_data
    except Exception as e:
        logging.error(f"Error in calculate_indicators: {e}")
        raise




def generate_signals(data):
    try:
        logging.info("Generating trading signals")

        # Initialize signals
        data['signal'] = 0

        # Loop through the data to apply conditions
        for i in range(1, len(data)):
            # Define buyFlag and sellFlag
            buyFlag = data['EMA'].iloc[i] > data['Close'].iloc[i]
            sellFlag = data['EMA'].iloc[i] < data['Close'].iloc[i]
            
            # Define green and red candles
            green_candle = data['Close'].iloc[i] > data['Close'].iloc[i-1]
            red_candle = data['Close'].iloc[i] < data['Close'].iloc[i-1]

            # Define RSI conditions for buying and selling
            buyRsiFlag = data['RSI'].iloc[i] < 20
            sellRsiFlag = data['RSI'].iloc[i] > 80

            # Buy signal: EMA > Close, RSI < 20, green candle, and no open trades
            if buyFlag and buyRsiFlag and green_candle:
                data['signal'].iloc[i] = 1  # Long entry signal
            
            # Sell signal: EMA < Close, RSI > 80, red candle, and no open trades
            elif sellFlag and sellRsiFlag and red_candle:
                data['signal'].iloc[i] = -1  # Short entry signal
        
        logging.info(f"Signal generation complete\n{data.head(20)}")
        return data

    except Exception as e:
        logging.error(f"Error in generate_signals: {e}")
        raise

# Define the Trading Strategy
class EMARSI_Cross(Strategy):
    
    def init(self):
        logging.info("Initializing strategy")
      
    def next(self):
        atr_value = self.data.ATR[-1]  # ATR value at the current candle
        candle_body = abs(self.data.Close[-1] - self.data.Open[-1])  # Calculate the candle body size

       

        # Long Trade Conditions
        if self.data.signal[-1] == 1:
                # Calculate stop loss distance based on ATR and candle body
            slDist = atr_value + candle_body
            # Calculate stop loss and take profit levels
            stop_loss = self.data.Close[-1] - slDist  # Stop loss for long position
            take_profit = self.data.Close[-1] + (1.2 * slDist)  # Take profit for long position
            
            logging.info(f"Long Entry: Stop Loss = {stop_loss}, Take Profit = {take_profit}")
            if self.position:
                    if self.position.is_short:
                        self.position.close()
            
            # Execute the buy order with stop loss and take profit
            # self.buy(sl=stop_loss, tp=take_profit)
            self.buy(stop=stop_loss, limit=take_profit)

        # Short Trade Conditions
        elif self.data.signal[-1] == -1:
             # Calculate stop loss distance based on ATR and candle body
            slDist = atr_value + candle_body
            # Calculate stop loss and take profit levels
            stop_loss = self.data.High[-1] + slDist  # Stop loss for short position
            take_profit = self.data.High[-1] - (1.2 * slDist)  # Take profit for short position
            
            logging.info(f"Short Entry: Stop Loss = {stop_loss}, Take Profit = {take_profit}")
            if self.position:
                    if self.position.is_long:
                        self.position.close()
            
            # Execute the sell order with stop loss and take profit
            # self.sell(sl=stop_loss, tp=take_profit)
            self.sell(stop=stop_loss, limit=take_profit)


      
# Load and prepare data
try:
    data_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/0.Dataset/btc_day_data_2023_2024/converted_sorted_btc_day_data_2023_2024.csv'
    minute_data = load_and_prepare_data(data_path)
   

    daily_data = calculate_daily_indicators(minute_data)
    daily_signals = generate_signals(daily_data)

    bt = Backtest(daily_signals, EMARSI_Cross, cash=1000000, commission=.002)
    stats = bt.run()

    logging.info(stats)
    logging.info("Backtest complete")

    # Convert the trades to a DataFrame
    trades = stats['_trades']  # Accessing the trades DataFrame from the stats object

    # Save the trades to a CSV file
    trades_csv_path = '/Users/pranaygaurav/Downloads/AlgoTrading/2cents_capital/indicator_strategies_backtest/Tradebook/17_EMARSI_Cross_backtest_trades.csv'
    trades.to_csv(trades_csv_path)

    logging.info(f"Trades have been exported to {trades_csv_path}")

except Exception as e:
    logging.error(f"Error in backtest execution: {e}")
