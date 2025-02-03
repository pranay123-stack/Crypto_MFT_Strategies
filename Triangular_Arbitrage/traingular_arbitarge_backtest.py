import ccxt
import pandas as pd
import math
from datetime import datetime
from backtesting import Backtest, Strategy
import time
# Initialize Binance client
binance = ccxt.binance()

# Function to download historical data for a given symbol
def download_historical_data(symbol, timeframe='1h', limit=1000):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Download data for BTC/USDT, ETH/USDT, and ETH/BTC
df_btc_usdt = download_historical_data('BTC/USDT')
df_eth_usdt = download_historical_data('ETH/USDT')
df_eth_btc = download_historical_data('ETH/BTC')


# Rename columns to match the expected format by `backtesting.py`
df_btc_usdt.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df_eth_usdt.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df_eth_btc.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Mock the current price retrieval to use the latest data from the DataFrame
def fetch_current_ticker_price(symbol):
    if symbol == 'BTC/USDT':
        return df_btc_usdt['Close'].iloc[-1]
    elif symbol == 'ETH/USDT':
        return df_eth_usdt['Close'].iloc[-1]
    elif symbol == 'ETH/BTC':
        return df_eth_btc['Close'].iloc[-1]
    else:
        return None

# Include your functions here (check_buy_buy_sell, check_profit_loss, etc.)


def check_if_float_zero(value):
    return math.isclose(value, 0.0, abs_tol=1e-3)

def check_buy_buy_sell(scrip1, scrip2, scrip3,initial_investment):
    
    ## SCRIP1
    investment_amount1 = initial_investment
    current_price1 = fetch_current_ticker_price(scrip1)
    final_price = 0
    scrip_prices = {}
    
    if current_price1 is not None:
        buy_quantity1 = round(investment_amount1 / current_price1, 8)
        time.sleep(1)
        ## SCRIP2
        investment_amount2 = buy_quantity1     
        current_price2 = fetch_current_ticker_price(scrip2)
        if current_price2 is not None:
            buy_quantity2 = round(investment_amount2 / current_price2, 8)
            time.sleep(1)
            ## SCRIP3
            investment_amount3 = buy_quantity2     
            current_price3 = fetch_current_ticker_price(scrip3)
            if current_price3 is not None:
                sell_quantity3 = buy_quantity2
                final_price = round(sell_quantity3 * current_price3,3)
                scrip_prices = {scrip1 : current_price1, scrip2 : current_price2, scrip3 : current_price3}
                
    return final_price, scrip_prices



def check_profit_loss(total_price_after_sell,initial_investment,transaction_brokerage, min_profit):
    apprx_brokerage = transaction_brokerage * initial_investment/100 * 3
    min_profitable_price = initial_investment + apprx_brokerage + min_profit
    profit_loss = round(total_price_after_sell - min_profitable_price,3)
    return profit_loss






class TriangularArbitrageStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        # For simplicity, we're using fixed values for these parameters
        initial_investment = 10000
        transaction_brokerage = 0.1  # in percentage
        min_profit = 10  # in the currency of initial_investment

        # Choose the symbols for the triangular arbitrage cycle
        scrip1, scrip2, scrip3 = 'BTC/USDT', 'ETH/USDT', 'ETH/BTC'

        # Perform the triangular arbitrage logic
        final_price, _ = check_buy_buy_sell(scrip1, scrip2, scrip3, initial_investment)

        # Calculate profit or loss
        profit_loss = check_profit_loss(final_price, initial_investment, transaction_brokerage, min_profit)

        if profit_loss > 0:
            print(f"PROFIT-{datetime.now().strftime('%H:%M:%S')}:" \
                  f"BUY_BUY_SELL, {scrip1}, {scrip2}, {scrip3}, Profit: {round(final_price - initial_investment, 3)}")

# As backtesting.py expects a single DataFrame, we'll arbitrarily choose one for the backtest
# This is a simplification, as triangular arbitrage would typically involve simultaneous prices from multiple pairs
data = df_btc_usdt

bt = Backtest(data, TriangularArbitrageStrategy, cash=10000, commission=.001)
stats = bt.run()
print(stats)
bt.plot()
