import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path
import yfinance as yf
import statsmodels.api as sm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
def validate_risk_free_file(
    risk_free_file,
    analysis_date,
    lookback_years
):

    try:

        rf_check = pd.read_csv(risk_free_file)

        required_cols = ['Date', 'Price']

        missing_cols = [
            col
            for col in required_cols
            if col not in rf_check.columns
        ]

        if missing_cols:

            st.sidebar.error(
                f"Risk-Free file is missing required column(s): "
                f"{', '.join(missing_cols)}"
            )
            st.stop()

        rf_check['Date'] = pd.to_datetime(
            rf_check['Date'],
            format="%d-%m-%Y"
        )

        rf_start = rf_check['Date'].min()
        rf_end = rf_check['Date'].max()

        required_start = (
            pd.Timestamp(analysis_date)
            - pd.DateOffset(years=lookback_years)
        )

        required_end = pd.Timestamp(analysis_date)

        if rf_start > required_start + pd.Timedelta(days=5):

            st.sidebar.error(
                f"Risk-free data must start on or before "
                f"{required_start.date()}.\n\n"
                f"Uploaded file starts on "
                f"{rf_start.date()}."
            )

            st.stop()

        if rf_end < required_end - pd.Timedelta(days=5):

            st.sidebar.error(
                f"Risk-free data must extend until "
                f"{required_end.date()}.\n\n"
                f"Uploaded file ends on "
                f"{rf_end.date()}."
            )

            st.stop()

        risk_free_file.seek(0)

        return True

    except Exception as e:

        st.sidebar.error(
            f"Invalid Risk-Free file: {str(e)}"
        )

        st.stop()

st.set_page_config(
    page_title="Alpha Momentum Portfolio",
    page_icon="📈",
    layout="wide"
)
lookback_years = st.sidebar.number_input(
    "Lookback Period (Years)",
    min_value=1,
    max_value=60,
    value=1,
    step=1
)

analysis_date = st.sidebar.date_input(
    "Portfolio Date",
    value=date.today(),
    max_value=date.today(),
    help="Select the date for which alpha rankings should be calculated."
)

st.title("📈 Alpha Momentum Portfolio")
st.caption(
    f"NSE 500 · CAPM Alpha Momentum Strategy "
    f"With {lookback_years}-Year Look-back Period"
)

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
st.sidebar.header("Upload Required Files")


with st.sidebar.expander("📄 View Required File Format"):

    st.write("NSE Constituents File")

    st.dataframe(
        pd.DataFrame({
            "Symbol": ["RELIANCE", "TCS", "INFY"],
            "Company Name": [
                "Reliance Industries",
                "Tata Consultancy Services",
                "Infosys"
            ]
        }),
        hide_index=True
    )

    st.write("Risk-Free Rate File")

    st.dataframe(
        pd.DataFrame({
            "Date": [
                "31-05-2025",
                "01-06-2025",
                "02-06-2025"
            ],
            "Price": [6.75, 6.74, 6.73]
        }),
        hide_index=True
    )

    st.caption(
        "Risk-free file must contain Date and Price columns, "
        "use DD-MM-YYYY format, and cover at least the last 12 months."
    )

nse_file = st.sidebar.file_uploader(
    "Upload NSE Constituents CSV",
    type=['csv']
)

if nse_file is not None:

    try:

        nse_check = pd.read_csv(nse_file)

        required_cols = ['Symbol']

        missing_cols = [
            col
            for col in required_cols
            if col not in nse_check.columns
        ]

        if missing_cols:

            st.sidebar.error(
                f"Missing required column(s): "
                f"{', '.join(missing_cols)}"
            )
            st.stop()

        if nse_check.empty:

            st.sidebar.error(
                "NSE file contains no data."
            )
            st.stop()

        if nse_check['Symbol'].isna().all():

            st.sidebar.error(
                "Symbol column is empty."
            )
            st.stop()

        nse_file.seek(0)

    except Exception as e:

        st.sidebar.error(
            f"Invalid NSE file: {str(e)}"
        )
        st.stop()




risk_free_file = st.sidebar.file_uploader(
    "Upload Risk-Free Rate CSV",
    type=['csv']
)
if risk_free_file is not None:

    validate_risk_free_file(
        risk_free_file,
        analysis_date,
        lookback_years
    )

if nse_file is None or risk_free_file is None:

    st.info(
        "Please upload both files to continue."
    )

    st.stop()


# ═══════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════

def download_single_stock(symbol, start_date, end_date):

    try:

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        return symbol, data

    except Exception:

        return symbol, pd.DataFrame()


def fetch_nse_500_data(symbols, start_date, end_date):

    all_data = {}

    progress_bar = st.progress(0)

    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=5) as executor:

        futures = {

            executor.submit(
                download_single_stock,
                symbol,
                start_date,
                end_date
            ): symbol

            for symbol in symbols

        }

        total = len(futures)

        for i, future in enumerate(as_completed(futures), 1):

            symbol, data = future.result()

            all_data[symbol] = data

            progress_bar.progress(i / total)

            status_text.text(
                f"Downloading stocks... {i}/{total}"
            )

    progress_bar.empty()

    status_text.empty()

    return all_data

# ═══════════════════════════════════════════════════════
# ALPHA CALCULATION
# ═══════════════════════════════════════════════════════

def calculate_alpha(df, start_date, end_date, market_rf):

    obj = df[
        (df.index >= start_date) &
        (df.index <= end_date)
    ]

    stock_returns = obj.pct_change().iloc[1:]

    alpha_df = pd.DataFrame(
        index=stock_returns.columns,
        columns=['Alpha', 'Beta']
    )

    progress_text = st.empty()

    total = len(stock_returns.columns)

    for i, st_sym in enumerate(stock_returns.columns, 1):

        progress_text.text(
            f"Running regressions... {i}/{total}"
        )

        stock = stock_returns[[st_sym]].dropna()

        if len(stock) < 30:
            continue

        try:

            reg = pd.merge(
                stock,
                market_rf,
                left_index=True,
                right_index=True,
                how='inner'
            )

            reg['y'] = reg[st_sym] - reg['daily_rf']

            reg['x'] = reg['Close'] - reg['daily_rf']

            y = reg['y']

            X = sm.add_constant(reg[['x']])

            results = sm.OLS(y, X).fit()

            alpha_df.loc[st_sym, 'Alpha'] = results.params['const']

            alpha_df.loc[st_sym, 'Beta'] = abs(results.params['x'])

        except Exception:
            continue

    progress_text.empty()

    alpha_df = alpha_df.dropna()

    alpha_df = alpha_df.sort_values(
        by='Alpha',
        ascending=False
    )

    return alpha_df

# ═══════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=86400)

def run_engine(lookback_years , analysis_date):

    today = analysis_date

    start_date = pd.Timestamp(today) - pd.DateOffset(years=lookback_years)

    end_date = pd.Timestamp(today)

    # Universe

    data_list = pd.read_csv(nse_file)

    data_list['Symbol'] = data_list['Symbol'] + '.NS'

    symbols = list(data_list['Symbol'])

    # Download data

    nse_500_data = fetch_nse_500_data(
        symbols,
        start_date,
        end_date
    )

    successful = {
        k: v
        for k, v in nse_500_data.items()
        if not v.empty
    }

    if len(successful) < 10:

        raise ValueError(
            "Too few stocks downloaded successfully."
        )

    # Price matrix

    close_prices = []

    for symbol, data in successful.items():

        close = data[['Close']].copy()

        if isinstance(close.columns, pd.MultiIndex):

            close.columns = close.columns.get_level_values(0)

        close.columns = [symbol]

        close_prices.append(close)

    merged_data = pd.concat(close_prices, axis=1)

    merged_data.index = pd.to_datetime(merged_data.index)

    merged_data = merged_data.ffill().bfill()

    # Risk-free rate

    risk_free = pd.read_csv(risk_free_file)

    risk_free['daily_rf'] = (
        (1 + risk_free['Price'] / 100) ** (1 / 365)
    ) - 1

    risk_free = risk_free.set_index('Date')

    risk_free.index = pd.to_datetime(
        risk_free.index,
        format="%d-%m-%Y"
    )

    risk_free_filtered = risk_free[
        risk_free.index >= start_date
    ].copy()

    # Benchmark

    market_raw = yf.download(
        "^CRSLDX",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if isinstance(market_raw.columns, pd.MultiIndex):

        market_raw.columns = (
            market_raw.columns.get_level_values(0)
        )

    market_data = market_raw[['Close']].copy()

    market_data = market_data.pct_change()

    rf_df = risk_free_filtered[['daily_rf']].copy()

    market_rf = pd.merge(

        market_data,
        rf_df,

        left_index=True,
        right_index=True,

        how='left'

    )

    market_rf = market_rf.ffill().bfill()

    # Alpha

    sorted_df = calculate_alpha(
        merged_data,
        start_date,
        end_date,
        market_rf
    )

    # One-month filter

    date_str = today.strftime("%Y-%m-%d")

    one_month_ago = (
        pd.to_datetime(date_str) +
        pd.offsets.MonthEnd(-1)
    )

    past_month = merged_data[
        (merged_data.index <= date_str) &
        (merged_data.index > one_month_ago)
    ]

    if len(past_month) >= 2:

        one_month_ret_series = (

            past_month.iloc[-1] /
            past_month.iloc[0]

        ) - 1

        bad_stocks = one_month_ret_series[
            (one_month_ret_series <= -0.05) |
            (one_month_ret_series.isna())
        ].index

    else:

        one_month_ret_series = pd.Series(dtype=float)

        bad_stocks = pd.Index([])

    updated_sorted_df = sorted_df.loc[
        ~sorted_df.index.isin(bad_stocks)
    ]

    updated_sorted_df = updated_sorted_df[
        updated_sorted_df['Alpha'] > 0
    ].copy()

    return (
        updated_sorted_df,
        sorted_df,
        one_month_ret_series,
        data_list,
        date_str
    )

# ═══════════════════════════════════════════════════════
# ENGINE EXECUTION
# ═══════════════════════════════════════════════════════

with st.spinner("Running momentum engine..."):

    (
        updated_sorted_df,
        sorted_df,
        one_month_ret_series,
        data_list,
        date_str
    ) = run_engine(
        lookback_years,
        analysis_date
    )
# ═══════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════

c1, c2 = st.columns(2)

c1.metric("Date", date_str)

c2.metric(
    "Stocks After Filter",
    len(updated_sorted_df)
)

st.divider()

# ═══════════════════════════════════════════════════════
# PORTFOLIO SIZE
# ═══════════════════════════════════════════════════════

max_portfolio_size = len(updated_sorted_df)

portfolio_size = st.number_input(
    "Enter Portfolio Size",
    min_value=1,
    max_value=max_portfolio_size,
    value=min(25, max_portfolio_size),
    step=1
)

# ═══════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════

top_ids = list(updated_sorted_df.index[:portfolio_size])


final_stocks = data_list[
    data_list['Symbol'].isin(top_ids)
].reset_index(drop=True)

final_stocks['Alpha'] = final_stocks['Symbol'].map(
    lambda s: sorted_df.loc[s, 'Alpha']
)

final_stocks['Beta'] = final_stocks['Symbol'].map(
    lambda s: sorted_df.loc[s, 'Beta']
)

final_stocks['1M Return'] = final_stocks['Symbol'].map(
    lambda s: one_month_ret_series.get(s, np.nan)
)

final_stocks = final_stocks.sort_values(
    by='Alpha',
    ascending=False
)

final_stocks.index += 1

# ═══════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════

st.subheader(
    f"Final {portfolio_size}-Stock Portfolio"
)

display_df = final_stocks.copy()

display_df = display_df.reset_index(drop=True)

display_df.index = display_df.index + 1

display_df.index.name = 'Rank'

display_df['Alpha'] = (
    display_df['Alpha']
    .astype(float)
    .map('{:+.6f}'.format)
)

display_df['Beta'] = (
    display_df['Beta']
    .astype(float)
    .map('{:.4f}'.format)
)

company_col = next(
    (
        c for c in display_df.columns
        if c.lower() in [
            'company name',
            'name of company',
            'company'
        ]
    ),
    None
)

cols = ['Symbol']

if company_col:

    cols.append(company_col)

cols += ['Alpha', 'Beta']

display_df = display_df[cols]

st.dataframe(
    display_df,
    use_container_width=True,
    height=min(
        900,
        35 * (len(display_df) + 1) + 10
    )
)

# ═══════════════════════════════════════════════════════
# DOWNLOAD CSV
# ═══════════════════════════════════════════════════════

csv = display_df.to_csv(index=True)

st.download_button(
    label="⬇️ Download Portfolio CSV",
    data=csv,
    file_name=(
        f"momentum_portfolio_"
        f"{portfolio_size}_{date_str}.csv"
    ),
    mime="text/csv"
)

# ═══════════════════════════════════════════════════════
# REFRESH BUTTON
# ═══════════════════════════════════════════════════════

if st.button("🔄 Refresh Data"):

    st.cache_data.clear()

    st.rerun()