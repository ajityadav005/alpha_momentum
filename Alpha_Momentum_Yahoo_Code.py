
"""

Requirements:
1. NSE 500 constituents list downloaded from NSE website  -> ind_nifty500list.csv
2. India 10-Year Bond Yield Historical Data from investing.com -> risk_free_rate.csv
"""

# Imports 
import pandas as pd
from datetime import date
from pathlib import Path
import yfinance as yf
import statsmodels.api as sm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Data Config 
DATA_DIR = Path(r"C:\Users\ajiya\OneDrive - Multi-Act\Projects\Yahoo Momentum Automation")
NSE_LIST_PATH   = DATA_DIR / "ind_nifty500list.csv"
RISK_FREE_PATH  = DATA_DIR / "India 10-Year Bond Yield Historical Data.csv"

PORTFOLIO_SIZE = 25   # set to 10 or 25

# Universe 
data_list = pd.read_csv(NSE_LIST_PATH)
data_list['Symbol'] = data_list['Symbol'] + '.NS'
symbols = list(data_list['Symbol'])

# Download helper functions

def download_single_stock(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        return symbol, data
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return symbol, pd.DataFrame()

def fetch_nse_500_data(symbols, start_date, end_date):
    all_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_single_stock, symbol, start_date, end_date): symbol
            for symbol in symbols
        }
        for i, future in enumerate(as_completed(futures), 1):
            symbol, data = future.result()
            print(i, symbol)
            all_data[symbol] = data
    return all_data

# Dates
today = date.today()
start_date = pd.Timestamp(today) - pd.DateOffset(years=1)
end_date   = pd.Timestamp(today)

# Download price data 
nse_500_data = fetch_nse_500_data(symbols, start_date, end_date)

successful = {k: v for k, v in nse_500_data.items() if not v.empty}
if len(successful) < 10:
    raise ValueError(
        f"Only {len(successful)} stocks downloaded successfully. "
        "Check your internet connection or yfinance rate limits."
    )

# Build merged price matrix

merged_data = None

for symbol, data in successful.items():
    if data.empty:
        continue

    close = data[['Close']].copy()

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(0)

    close.rename(columns={'Close': symbol}, inplace=True)

    merged_data = close if merged_data is None else merged_data.merge(
        close, how='outer', left_index=True, right_index=True
    )

if merged_data is None or merged_data.empty:
    raise ValueError("Price matrix is empty after merging. No data to process.")

if isinstance(merged_data.columns, pd.MultiIndex):
    merged_data.columns = merged_data.columns.get_level_values(-1)

merged_data.index = pd.to_datetime(merged_data.index)
merged_data = merged_data.ffill().bfill()

#Risk-free rate 

risk_free = pd.read_csv(RISK_FREE_PATH)
risk_free['daily_rf'] = (1 + risk_free['Price'] / 100) ** (1 / 365) - 1
risk_free = risk_free.set_index('Date')
risk_free.index = pd.to_datetime(risk_free.index, format="%d-%m-%Y")

if risk_free.index.isna().any():
    raise ValueError(
        "Some risk-free rate dates could not be parsed. "
        "Check the date format in the CSV (expected DD-MM-YYYY)."
    )

# Market (benchmark) data
market_raw = yf.download("^CRSLDX", start=start_date, end=end_date, auto_adjust=True)

if isinstance(market_raw.columns, pd.MultiIndex):
    market_raw.columns = market_raw.columns.get_level_values(0)

market_data = market_raw[['Close']].copy()
market_data = market_data[market_data.index >= start_date]
market_data = market_data.pct_change()

# Combine market returns + risk-free rate
risk_free_filtered = risk_free[risk_free.index >= start_date].copy()

market_rf_data = pd.merge(
    market_data, risk_free_filtered,
    left_index=True, right_index=True, how='left'
)
market_rf_data = market_rf_data[['Close', 'daily_rf']]
market_rf_data = market_rf_data.ffill()   

# Price matrix for analysis 
price = merged_data.copy()

# Alpha calculation
def calculate_alpha(df, start_date, end_date, market_rf_data):
    """
    Run CAPM OLS regressions for every stock in df over [start_date, end_date].
    Returns a DataFrame sorted by Alpha (descending) with NaN rows dropped.

    """
    obj = df[(df.index >= start_date) & (df.index <= end_date)]
    stock_returns = obj.pct_change().iloc[1:]   # skip the leading NaN row

    alpha_df = pd.DataFrame(index=stock_returns.columns, columns=['Alpha', 'Beta'])

    for st in stock_returns.columns:
        stock = stock_returns[[st]].dropna()
        if len(stock) < 30:   # require at least 30 observations for a meaningful regression
            continue

        reg = pd.merge(stock, market_rf_data, left_index=True, right_index=True, how='inner')
        reg['y'] = reg[st]      - reg['daily_rf']
        reg['x'] = reg['Close'] - reg['daily_rf']

        y = reg['y']
        X = sm.add_constant(reg[['x']])
        results = sm.OLS(y, X).fit()

        alpha_df.loc[st, 'Alpha'] = results.params['const']
        alpha_df.loc[st, 'Beta']  = abs(results.params['x'])

    alpha_df = alpha_df.dropna()
    alpha_df = alpha_df.sort_values(by='Alpha', ascending=False)
    return alpha_df

# Run momentum screen 
date_str = today.strftime("%Y-%m-%d")
print(f"Processing date: {date_str}")

df = price.ffill()   

sorted_df = calculate_alpha(df, start_date, end_date, market_rf_data)

# Filter stocks with poor 1-month returns

one_month_ago = pd.to_datetime(date_str) + pd.offsets.MonthEnd(-1)
past_one_month_data = df[(df.index <= date_str) & (df.index > one_month_ago)]

if len(past_one_month_data) >= 2:
    one_month_ret = (
        past_one_month_data.tail(1).values / past_one_month_data.head(1).values - 1
    )
    one_month_ret_series = pd.Series(
        one_month_ret[0], index=past_one_month_data.columns
    )
    # Stocks with insufficient data get NaN — drop them explicitly
    insufficient_data = one_month_ret_series[one_month_ret_series.isna()].index
    bad_stocks = one_month_ret_series[one_month_ret_series <= -0.05].index
    bad_stocks = bad_stocks.union(insufficient_data)
else:
    print("Warning: not enough data for 1-month return filter; skipping filter.")
    bad_stocks = pd.Index([])

updated_sorted_df = sorted_df.loc[~sorted_df.index.isin(bad_stocks)]
updated_sorted_df = updated_sorted_df[updated_sorted_df['Alpha']>0].copy()

# Portfolio selection 
if PORTFOLIO_SIZE == 10:
    top_ids = list(updated_sorted_df.index[:10])

elif PORTFOLIO_SIZE == 25:
    top_ids       = list(updated_sorted_df.index[:10])
    next_top      = updated_sorted_df.iloc[10:40].sort_values(by='Beta')
    next_top      = next_top.head(15).sort_values('Alpha', ascending=False)
    top_ids      += list(next_top.index)

else:
    raise ValueError(f"PORTFOLIO_SIZE must be 10 or 25, got {PORTFOLIO_SIZE}")

final_stocks_list = data_list[data_list['Symbol'].isin(top_ids)].reset_index(drop=True)

print(f"\nFinal {PORTFOLIO_SIZE}-stock portfolio:")
print(final_stocks_list)