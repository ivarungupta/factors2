import requests
import pandas as pd
import numpy as np
import time
import pickle
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from technical_factors import technical_factor_calculations
from financial_factors import financial_factor_calculations, FinancialFactorCalculator

# Initialize the FMP data fetcher
api_key = "bEiVRux9rewQy16TXMPxDqBAQGIW8UBd"
base_url = "https://financialmodelingprep.com/api/v3"

start_date = "1996-01-01"
end_date = "2024-12-31"

# Create directory for storing CSV files if it doesn't exist
output_dir = "sp500_constituents"
os.makedirs(output_dir, exist_ok=True)

# Fetch current S&P 500 constituents
endpoint = f"{base_url}/sp500_constituent"
params = {
    "apikey": api_key
}

response = requests.get(endpoint, params=params)
if response.status_code == 200:
    constituents = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(constituents)

    # Save to CSV with current date
    current_date = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"sp500_constituents_{current_date}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved S&P 500 constituents to {output_file}")

    # Print summary
    print(f"\nTotal companies in S&P 500: {len(df)}")
    print("\nFirst few companies:")
    print(df[['symbol', 'name', 'sector']].head())
else:
    print(f"Error fetching data: {response.status_code}")

# Get the most recent file from the sp500_constituents directory
constituent_files = glob.glob(os.path.join(output_dir, "sp500_constituents_*.csv"))
if not constituent_files:
    raise FileNotFoundError("No S&P 500 constituents files found")
most_recent_file = max(constituent_files, key=os.path.getctime)

# Read the most recent CSV file
df = pd.read_csv(most_recent_file)  # Use the most recent file instead of hardcoded date

# Extract symbols column
symbols = df['symbol'].tolist()

# Save to pickle file
with open("sp500_symbols.pickle", "wb") as f:
    pickle.dump(symbols, f)
print(f"Saved {len(symbols)} symbols to sp500_symbols.pickle")

# Load and print the symbols from pickle file
with open("sp500_symbols.pickle", "rb") as f:
    loaded_symbols = pickle.load(f)

print("Loaded symbols from pickle file:")
print(f"Total symbols: {len(loaded_symbols)}")
print("\nFirst 20 symbols:")
print(loaded_symbols[:20])

tickers_considered = loaded_symbols[:20]

print(tickers_considered)

#downloading the stock data from tickers_considered and filter out the
# Create directory for storing individual CSV files
output_dir = "stock_prices"
os.makedirs(output_dir, exist_ok=True)

insufficient_data_stocks = []

# Function to fetch historical prices
def fetch_historical_prices(symbol):
    endpoint = f"{base_url}/historical-price-full/{symbol}"
    params = {
        "from": start_date,
        "to": end_date,
        #"to": datetime.now().strftime("%Y-%m-%d")
        "apikey": api_key
    }

    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            return df
    return None


# Fetch data for each ticker
for i, symbol in enumerate(tickers_considered):
    print(f'\n[{i+1}/{len(tickers_considered)}] Fetching close price for {symbol}...')
    df = fetch_historical_prices(symbol)

    if df is not None:
        # Save to individual CSV file
        csv_path = os.path.join(output_dir, f"{symbol}_prices.csv")
        df.to_csv(csv_path, index=False)

        if len(df) < 630:  # Check if data points are less than 630
            insufficient_data_stocks.append(symbol)
            print(f"Warning: {symbol} has only {len(df)} data points")
        else:
            print(f"Saved {len(df)} rows for {symbol}")

    # Add a small delay to avoid hitting API rate limits
    time.sleep(0.5)

# Print stocks with insufficient data
print("\nStocks with insufficient data (less than 630 rows):")
for stock in insufficient_data_stocks:
    print(f"{stock}: {len(pd.read_csv(os.path.join(output_dir, f'{stock}_prices.csv')))} rows")

# Create final_tickers by removing insufficient data stocks
final_tickers = [ticker for ticker in tickers_considered if ticker not in insufficient_data_stocks]

# Save final_tickers to pickle
with open("final_tickers.pickle", "wb") as f:
    pickle.dump(final_tickers, f)

print(f"\nOriginal number of tickers: {len(tickers_considered)}")
print(f"Number of tickers with insufficient data: {len(insufficient_data_stocks)}")
print(f"Final number of tickers: {len(final_tickers)}")
print(final_tickers)

# Print summary of saved files
print("\nSummary of saved files:")
print(f"Total CSV files created: {len(os.listdir(output_dir))}")
print(f"Files are saved in: {os.path.abspath(output_dir)}")

# Load final_tickers from pickle file
with open("final_tickers.pickle", "rb") as f:
    final_tickers = pickle.load(f)

# Create an empty list to store all data
all_data = []

# Read each stock's CSV file and append to the list
for symbol in final_tickers:
    # Read the CSV file
    file_path = os.path.join("stock_prices", f"{symbol}_prices.csv")
    df = pd.read_csv(file_path)

    # Add symbol column
    df['symbol'] = symbol

    # Append to the list
    all_data.append(df)

# Combine all data into a single DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Sort by date and symbol
combined_df = combined_df.sort_values(['date', 'symbol'], ascending=[False, True])
# Reorder columns to put 'symbol' as the second column
columns = combined_df.columns.tolist()
columns.remove('symbol')
columns.remove('date')
new_column_order = ['date', 'symbol'] + columns
combined_df = combined_df[new_column_order].reset_index(drop=True)

# Print summary statistics
print("\nSummary of the data:")
print(f"Total number of rows: {len(combined_df)}")
print(f"Number of unique stocks: {combined_df['symbol'].nunique()}")
print(f"Date range: from {combined_df['date'].min()} to {combined_df['date'].max()}")

# Display the first few rows
print("First few rows of the combined data:")
print(combined_df.head())

combined_df = combined_df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

#Check the data for any stock
combined_df[combined_df["symbol"]=="WSM"].sort_values('date', ascending=False)

# Create a copy of the DataFrame
combined_df2 = combined_df.copy()

# Convert date to datetime and set multi-index
combined_df2['date'] = pd.to_datetime(combined_df2['date'])
combined_df2.set_index(['date', 'symbol'], inplace=True)

# Sort the index
combined_df2.sort_index(ascending=[False, True], inplace=True)

print(combined_df2)  # Replaced display() with print()

# Get unique tickers from combined_df2
unique_tickers = combined_df2.index.get_level_values('symbol').unique()

print(len(unique_tickers))

# Create a copy of combined_df and store the factors
combined_df3 = combined_df.reset_index()
combined_df3 = combined_df3.drop(combined_df3.columns[0], axis=1)
# Sort the DataFrame by date in ascending order (oldest to newest)
combined_df3['date'] = pd.to_datetime(combined_df3['date'])
combined_df3 = combined_df3.sort_values(['symbol', 'date'])

# Calculate risk-free rates
risk_free_rate_annual = 0.045
risk_free_rate_20 = (1 + risk_free_rate_annual) ** (20 / 252) - 1
risk_free_rate_60 = (1 + risk_free_rate_annual) ** (60 / 252) - 1

factor_columns = list(technical_factor_calculations.keys())

# Add all factor columns with NaN values
for col in factor_columns:
    combined_df3[col] = np.nan

# Get unique stock symbols
symbols = combined_df3['symbol'].unique()

# Process each stock separately
for symbol in symbols:
    try:
        # Get data for current symbol
        stock_data = combined_df3[combined_df3['symbol'] == symbol].copy()

        # Sort data in ascending order (oldest to newest) for calculations
        stock_data = stock_data.sort_values('date', ascending=True)

        # Calculate returns first
        stock_data['returns'] = stock_data['close'].pct_change()

        # Calculate all technical factors
        for factor_name, calc_function in technical_factor_calculations.items():
            try:
                stock_data[factor_name] = calc_function(stock_data)
            except Exception as e:
                print(f"Error calculating {factor_name} for {symbol}: {str(e)}")

        # Forward fill any remaining NaN values (up to 5 days)
        for col in factor_columns:
            stock_data[col] = stock_data[col].ffill(limit=5)

        # Sort back to descending order (newest to oldest) before updating
        stock_data = stock_data.sort_values('date', ascending=False)

        # Update the original DataFrame with calculated values
        combined_df3.loc[combined_df3['symbol'] == symbol, factor_columns] = stock_data[factor_columns]

    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        continue

# Sort back to descending order (newest to oldest) if needed
combined_df3 = combined_df3.sort_values(['symbol', 'date'], ascending=[True, False])



##Calculate Beta
# Sort combined_df3 from old date to new date
combined_df3 = combined_df3.sort_values(['symbol', 'date'], ascending=[True, True])

# Calculate stock returns for all stocks
combined_df3['stock_returns'] = combined_df3.groupby('symbol')['close'].pct_change()

# Get benchmark data and calculate returns
benchmark = '^GSPC'  # S&P 500 ETF benchmark

def get_benchmark_data():
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{benchmark}?serietype=line&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and "historical" in data:
            return [(entry["date"], entry["close"]) for entry in data["historical"] if start_date <= entry["date"] <= end_date]
    except Exception as e:
        print(f"Error fetching benchmark data: {str(e)}")
    return []

# Fetch benchmark data
benchmark_data = get_benchmark_data()
benchmark_df = pd.DataFrame(benchmark_data, columns=['date', 'close'])
benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
benchmark_df.set_index('date', inplace=True)
benchmark_df.sort_index(ascending=True, inplace=True)

# Calculate benchmark returns
benchmark_df['market_returns'] = benchmark_df['close'].pct_change()

# Merge benchmark returns with combined_df3
#drop any existing market_returns column if it exists
if 'market_returns' in combined_df3.columns:
    combined_df3 = combined_df3.drop('market_returns', axis=1)

combined_df3 = combined_df3.merge(
    benchmark_df['market_returns'].reset_index(),
    on='date',
    how='left'
)

# Calculate beta for all dates
window = 60  # 60-day rolling window

# Group by symbol and calculate rolling beta
for symbol in combined_df3['symbol'].unique():
    try:
        # Get data for current symbol
        symbol_data = combined_df3[combined_df3['symbol'] == symbol].copy()

        # Calculate rolling beta
        rolling_beta = (symbol_data['stock_returns'].rolling(window=window, min_periods=30)
                       .cov(symbol_data['market_returns']) /
                       symbol_data['market_returns'].rolling(window=window, min_periods=30).var())

        # Update beta values in combined_df3
        combined_df3.loc[combined_df3['symbol'] == symbol, 'Beta'] = rolling_beta.values

    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        continue

# Forward fill any remaining NaN values (up to 5 days)
combined_df3['Beta'] = combined_df3.groupby('symbol')['Beta'].ffill(limit=5)

# Sort back to descending order (newest to oldest)
combined_df3 = combined_df3.sort_values(['symbol', 'date'], ascending=[True, False]).reset_index(drop=True)

print("\nFirst few rows of the combined DataFrame with factors:")
print(combined_df3.round(4))

print("\nNumber of NaN Values for each factor:")
print(combined_df3[factor_columns].isna().sum())

# Function to display NaN values for each stock
def display_stock_nan_values(df):
    """
    Display total NaN values for each stock.

    Args:
        df: DataFrame containing the factors
    """
    # Get all factor columns
    factor_columns = [
        'GrowthRate', 'Momentum', 'Beta',
        'VOL60', 'DAVOL60', 'VOSC', 'VMACD', 'ATR42',
        'Variance60', 'Skewness60', 'Kurtosis60', 'SharpeRatio20', 'SharpeRatio60',
        'ROC60', 'Volume1Q', 'TRIX30', 'Price1Q', 'PLRC36',
        'MACD60', 'boll_up', 'boll_down', 'MFI42'
    ]

    # Group by symbol and count NaN values
    nan_counts = df.groupby('symbol')[factor_columns].apply(lambda x: x.isna().sum())

    # Calculate total NaN values for each stock
    nan_counts['Total_NaN'] = nan_counts.sum(axis=1)

    # Sort by total NaN values in descending order
    nan_counts = nan_counts.sort_values('Total_NaN', ascending=False)

    # Display results
    print("\nNaN values for each stock:")
    print(nan_counts)

# Call the function
display_stock_nan_values(combined_df3)

daily_tech_factors = combined_df3[['symbol', 'date'] + factor_columns].reset_index(drop=True)
print(daily_tech_factors.round(4))

#daily_tech_factors.to_csv('daily_tech_factors.csv', index = False)

# Convert date to datetime if not already
daily_tech_factors['date'] = pd.to_datetime(daily_tech_factors['date'])

quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='QE').strftime("%Y-%m-%d").tolist()
quarterly_dates = pd.to_datetime(quarterly_dates)

# Resample daily data to quarterly
quarterly_tech_factors = daily_tech_factors.copy()
quarterly_tech_factors['date'] = pd.to_datetime(quarterly_tech_factors['date'])

# Group by symbol and resample to quarterly frequency
quarterly_tech_factors = (quarterly_tech_factors
    .set_index(['symbol', 'date'])
    .groupby('symbol')
    .resample('QE', level='date')
    .last()
    .reset_index()
)

# Filter for only the dates in our quarterly_dates list
quarterly_tech_factors = quarterly_tech_factors[
    quarterly_tech_factors['date'].isin(quarterly_dates)
]

# Sort by symbol and date in descending order (latest first)
quarterly_tech_factors = quarterly_tech_factors.sort_values(['symbol', 'date'], ascending=[True, False])

# Drop the oldest quarter for each stock
quarterly_tech_factors = quarterly_tech_factors.groupby('symbol').apply(
    lambda x: x.iloc[:-1]).reset_index(drop=True)



# Display results
print("\nDate range in quarterly technical factors:")
print("Earliest date:", quarterly_tech_factors['date'].min())
print("Latest date:", quarterly_tech_factors['date'].max())

print("\nFirst few rows of quarterly technical factors:")
print(quarterly_tech_factors.head().round(4))

# Check for any missing values
print("\nMissing values in quarterly technical factors:")
print(quarterly_tech_factors.isna().sum())

#Save the Quarterly Tech Factors File
print("\nQuarterly Technical Factors are saved to 'quarterly_tech_factors.csv'")
quarterly_tech_factors.to_csv('quarterly_tech_factors.csv', index = False)

quarterly_tech_factors

"""#FACTORS FROM FINANCIAL DATA - QUARTERLY"""

#Fetching Financial Statements
# Load final_tickers
with open("final_tickers.pickle", "rb") as f:
    final_tickers = pickle.load(f)

# Load combined_df2 to determine the date range (if needed)
# Assuming combined_df2 already exists
start_date = combined_df2.index.get_level_values('date').min().strftime('%Y-%m-%d')
end_date = combined_df2.index.get_level_values('date').max().strftime('%Y-%m-%d')
print(f"Fetching financials from {start_date} to {end_date}")

# Create directory to store individual financial statements as CSV
os.makedirs("financials_csv", exist_ok=True)

# Function to fetch financial data for a symbol
def fetch_financial_data(symbol, statement_type):
    url = f"{base_url}/{statement_type}/{symbol}"
    params = {
        "period": "quarter",
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        if not df.empty:
            df['symbol'] = symbol
            df['statement_type'] = statement_type
        return df
    else:
        print(f"Failed to fetch {statement_type} for {symbol} | Status Code: {response.status_code}")
        return pd.DataFrame()

# Main loop to fetch all financials and save to CSV
financial_data = {}

for i, symbol in enumerate(final_tickers):
    print(f"\n[{i+1}/{len(final_tickers)}] Fetching financials for {symbol}...")

    income_df = fetch_financial_data(symbol, "income-statement")
    balance_df = fetch_financial_data(symbol, "balance-sheet-statement")
    cashflow_df = fetch_financial_data(symbol, "cash-flow-statement")

    # Optional: Filter by date range if needed
    for merged_df2_with_ttm in [income_df, balance_df, cashflow_df]:
        if not merged_df2_with_ttm.empty and 'date' in merged_df2_with_ttm.columns:
            merged_df2_with_ttm['date'] = pd.to_datetime(merged_df2_with_ttm['date'])
            merged_df2_with_ttm.dropna(subset=['date'], inplace=True)

    income_df = income_df[income_df['date'].between(start_date, end_date)]
    balance_df = balance_df[balance_df['date'].between(start_date, end_date)]
    cashflow_df = cashflow_df[cashflow_df['date'].between(start_date, end_date)]

    # Save CSVs
    if not income_df.empty:
        income_df.to_csv(f"financials_csv/{symbol}_income_statement.csv", index=False)
    if not balance_df.empty:
        balance_df.to_csv(f"financials_csv/{symbol}_balance_sheet.csv", index=False)
    if not cashflow_df.empty:
        cashflow_df.to_csv(f"financials_csv/{symbol}_cash_flow_statement.csv", index=False)

    # Save to dictionary
    financial_data[symbol] = {
        "income-statement": income_df,
        "balance-sheet-statement": balance_df,
        "cash-flow-statement": cashflow_df
    }

    time.sleep(0.5)  # API rate limit

# # Optionally, save the financial data to a pickle file as well
# with open("financial_statements_quarterly.pickle", "wb") as f:
#     pickle.dump(financial_data, f)

#print("\nAll financial statements saved to 'financials_csv' and 'financial_statements_quarterly.pickle'.")
print("\nAll financial statements saved to 'financials_csv'.")

def print_financial_statement(symbol, statement_type, start_date=None, end_date=None):
    """
    Print financial statement for a given stock and period.

    Args:
        symbol (str): Stock symbol
        statement_type (str): Type of statement ('income', 'balance', or 'cashflow')
        start_date (str): Start date in 'YYYY-MM-DD' format (optional)
        end_date (str): End date in 'YYYY-MM-DD' format (optional)
    """
    # Validate statement type
    valid_statements = {
        "income": "income_statement",
        "balance": "balance_sheet",
        "cashflow": "cash_flow_statement"
    }

    if statement_type.lower() not in valid_statements:
        print(f" Invalid statement type. Please choose from: {', '.join(valid_statements.keys())}")
        return

    # Build file path
    file_path = f"financials_csv/{symbol}_{valid_statements[statement_type.lower()]}.csv"

    try:
        # Load the CSV
        df = pd.read_csv(file_path, parse_dates=["date"])

        # Print basic information
        print(f"\n{statement_type.upper()} Statement for {symbol}")
        print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total number of periods: {len(df)}")

        # Filter by date if provided
        if start_date or end_date:
            if start_date:
                df = df[df["date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["date"] <= pd.to_datetime(end_date)]
            print(f"Filtered period: {start_date or 'beginning'} to {end_date or 'end'}")

        # Print the data
        #print(f"\n Statement data:")
        print(df.sort_values('date', ascending=False))

        print(f"\nAvailable columns:")
        print(df.columns.tolist())

    except FileNotFoundError:
        print(f" File not found: {file_path}")
        print("Please check if the stock symbol is correct and the file exists.")

    except Exception as e:
        print(f" Error: {e}")


# Example usage:
print_financial_statement("APO", "income", "2023-01-01", "2024-12-31")

def load_financial_statements(csv_dir="financials_csv"):
    """
    Load and combine financial statements for all stocks into three separate dataframes.

    Args:
        csv_dir (str): Directory containing financial statement CSV files

    Returns:
        tuple: (income_df_all, balance_df_all, cashflow_df_all) containing combined financial statements
    """
    # Predefine the selected columns
    balance_cols = ['date', 'symbol', 'reportedCurrency', 'cik', 'totalAssets', 'totalCurrentAssets',
                    'totalLiabilities', 'totalCurrentLiabilities', 'cashAndCashEquivalents',
                    'inventory', 'totalDebt', 'retainedEarnings', 'totalStockholdersEquity']

    income_cols = ['date', 'symbol', 'reportedCurrency', 'cik', 'revenue', 'grossProfit', 'netIncome', 'eps',
                   'ebitda', 'weightedAverageShsOut']

    cashflow_cols = ['date', 'symbol', 'reportedCurrency', 'cik', 'operatingCashFlow']

    # Initialize empty lists to collect data
    income_dfs = []
    balance_dfs = []
    cashflow_dfs = []

    # Get list of all available stock symbols from the directory
    available_symbols = set()
    for file in os.listdir(csv_dir):
        if file.endswith('_income_statement.csv'):
            symbol = file.split('_')[0]
            available_symbols.add(symbol)

    print(f"Found {len(available_symbols)} stocks with financial statements")

    # Loop through all available tickers and read their statements
    for symbol in sorted(available_symbols):
        try:
            # Load income statement
            income_path = os.path.join(csv_dir, f"{symbol}_income_statement.csv")
            if os.path.exists(income_path):
                df_income = pd.read_csv(income_path, parse_dates=["date"])
                df_income = df_income[income_cols]
                income_dfs.append(df_income)

            # Load balance sheet
            balance_path = os.path.join(csv_dir, f"{symbol}_balance_sheet.csv")
            if os.path.exists(balance_path):
                df_balance = pd.read_csv(balance_path, parse_dates=["date"])
                df_balance = df_balance[balance_cols]
                balance_dfs.append(df_balance)

            # Load cash flow statement
            cashflow_path = os.path.join(csv_dir, f"{symbol}_cash_flow_statement.csv")
            if os.path.exists(cashflow_path):
                df_cashflow = pd.read_csv(cashflow_path, parse_dates=["date"])
                df_cashflow = df_cashflow[cashflow_cols]
                cashflow_dfs.append(df_cashflow)

        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")

    # Concatenate dataframes
    income_df_all = pd.concat(income_dfs, ignore_index=True)
    balance_df_all = pd.concat(balance_dfs, ignore_index=True)
    cashflow_df_all = pd.concat(cashflow_dfs, ignore_index=True)

    # Sort all dataframes by date and symbol
    for df in [income_df_all, balance_df_all, cashflow_df_all]:
        df.sort_values(['date', 'symbol'], ascending=[False, True], inplace=True)

    # Print summary information
    # print("\nCombined Financial Statements Summary:")
    # print(f"Income Statement: {income_df_all.shape[0]} rows, {income_df_all.shape[1]} columns")
    # print(f"Balance Sheet: {balance_df_all.shape[0]} rows, {balance_df_all.shape[1]} columns")
    # print(f"Cash Flow Statement: {cashflow_df_all.shape[0]} rows, {cashflow_df_all.shape[1]} columns")
    # print(f"Income DF shape: {income_df_all.shape}")
    # print(f"Balance DF shape: {balance_df_all.shape}")
    # print(f"Cashflow DF shape: {cashflow_df_all.shape}")

    print("\nDate Range:")
    print(f"From: {min(income_df_all['date'].min(), balance_df_all['date'].min(), cashflow_df_all['date'].min())}")
    print(f"To: {max(income_df_all['date'].max(), balance_df_all['date'].max(), cashflow_df_all['date'].max())}")

    print("\nNumber of unique stocks in each statement:")
    print(f"Income Statement: {income_df_all['symbol'].nunique()}")
    print(f"Balance Sheet: {balance_df_all['symbol'].nunique()}")
    print(f"Cash Flow Statement: {cashflow_df_all['symbol'].nunique()}")

    return income_df_all, balance_df_all, cashflow_df_all

# Load the financial statements
income_df_all, balance_df_all, cashflow_df_all = load_financial_statements()

# Optional: Save the combined dataframes to CSV files
# income_df.to_csv('combined_income_statements.csv', index=False)
# balance_df.to_csv('combined_balance_sheets.csv', index=False)
# cashflow_df.to_csv('combined_cashflow_statements.csv', index=False)

income_df_all.columns

# Merge the dataframes
merged_df = income_df_all.merge(balance_df_all, on=["date", "symbol", "reportedCurrency", "cik"], how="inner")  # Using inner join to ensure we only keep rows with data in all statements

merged_df = merged_df.merge(cashflow_df_all, on=["date", "symbol", "reportedCurrency", "cik"], how="inner")

# Sort the merged dataframe
merged_df = merged_df.sort_values(['date', 'symbol'], ascending=[False, True])

# Print information about the merged dataframe
print("Merged Financial Statements Summary:")
print(f"Total number of rows: {len(merged_df)}")
print(f"Total number of columns: {len(merged_df.columns)}")
print(f"Number of unique stocks: {merged_df['symbol'].nunique()}")
#print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

#Save to CSV
merged_df.to_csv('merged_financial_data.csv', index=False)

print("\nFirst few rows of merged dataframe:")
print(merged_df.head())

merged_df.columns

def calculate_ttm(df, columns):
    """
    Calculate TTM (Trailing Twelve Months) for specified columns in the DataFrame.
    TTM is computed by summing the values of the most recent quarter + the previous 3 quarters.

    Args:
        df (pd.DataFrame): DataFrame containing financial data
        columns (list): List of column names to calculate TTM for

    Returns:
        pd.DataFrame: DataFrame with TTM columns added
    """
    # Create a copy to avoid modifying the original dataframe
    df_ttm = df.copy()

    # Verify all columns exist in the dataframe
    missing_cols = [col for col in columns if col not in df_ttm.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")

    # Calculate TTM for each column
    for column in columns:
        ttm_values = []
        ttm_column = f'{column}_ttm'

        for symbol, group in df_ttm.groupby('symbol'):
            # Sort by date descending to ensure most recent is first
            group = group.sort_values('date', ascending=False)
            ttm = [None] * len(group)

            # Calculate TTM for each quarter
            for i in range(len(group) - 3):
                # Sum the latest quarter + previous 3 quarters
                ttm_sum = group.iloc[i:i+4][column].sum()
                ttm[i] = ttm_sum

            # Assign TTM values back to the dataframe
            df_ttm.loc[group.index, ttm_column] = ttm

    return df_ttm

# Columns to calculate TTM for
ttm_columns = [
    'netIncome',
    'operatingCashFlow',
    'revenue',
    'grossProfit',
    'eps',
    'ebitda'
]

# Create list of TTM column names
ttm_columns_ttm = [f"{col}_ttm" for col in ttm_columns]

try:
    # Apply TTM calculation
    merged_df_with_ttm = calculate_ttm(merged_df, ttm_columns)

    # Drop rows where any of the TTM columns are NA
    merged_df_with_ttm = merged_df_with_ttm.dropna(subset=ttm_columns_ttm)

    # Print summary information
    print("\nTTM Calculation Summary:")
    print(f"Original number of rows: {len(merged_df)}")
    print(f"Rows after TTM calculation: {len(merged_df_with_ttm)}")
    print(f"Number of rows removed: {len(merged_df) - len(merged_df_with_ttm)}")
    print(f"Number of unique stocks: {merged_df_with_ttm['symbol'].nunique()}")
    print(f"Date range: {merged_df_with_ttm['date'].min()} to {merged_df_with_ttm['date'].max()}")

    # Print TTM columns added
    print("\nTTM columns added:")
    for col in ttm_columns_ttm:
        print(f"- {col}")

    # Save to CSV
    merged_df_with_ttm.to_csv('merged_financial_data_with_ttm.csv', index=False)

    # Display sample data
    print("\nSample data:")
    print(merged_df_with_ttm)

except Exception as e:
    print(f"Error during TTM calculation: {str(e)}")

merged_df_with_ttm.columns

len(merged_df_with_ttm['symbol'].unique())

def get_previous_price_data(df_price, target_date, symbol, max_lookback=5):
    """
    Get the most recent price data before or on the target date for a stock.

    Args:
        df_price (pd.DataFrame): DataFrame containing price data
        target_date (pd.Timestamp): Target date to find price for
        symbol (str): Stock symbol
        max_lookback (int): Maximum number of days to look back

    Returns:
        pd.Series: Price data (open, high, low, close, volume) or None if not found
    """
    # Reset index to make date and symbol regular columns
    df_price_reset = df_price.reset_index()

    # Get price data for the symbol up to target date
    symbol_data = df_price_reset[df_price_reset['symbol'] == symbol]
    symbol_data = symbol_data[symbol_data['date'] <= target_date]

    if not symbol_data.empty:
        # Sort by date descending to get most recent first
        symbol_data = symbol_data.sort_values('date', ascending=False)

        # Look back up to max_lookback days
        for i in range(min(max_lookback, len(symbol_data))):
            price_data = symbol_data.iloc[i]
            if not price_data[['open', 'high', 'low', 'close', 'volume']].isna().any():
                return price_data[['open', 'high', 'low', 'close', 'volume']]

    return None

# Create a copy of merged_df_with_ttm to work with
merged_df_with_ttm_and_price = merged_df_with_ttm.copy()

# Initialize price columns
price_columns = ['open', 'high', 'low', 'close', 'volume']
for col in price_columns:
    merged_df_with_ttm_and_price[col] = None

# Get unique dates and symbols from merged_df_with_ttm
unique_dates = merged_df_with_ttm_and_price['date'].unique()
unique_symbols = merged_df_with_ttm_and_price['symbol'].unique()

# For each date and symbol combination
for date in unique_dates:
    for symbol in unique_symbols:
        # Get the row index for this date and symbol
        mask = (merged_df_with_ttm_and_price['date'] == date) & (merged_df_with_ttm_and_price['symbol'] == symbol)
        if mask.any():
            # Get previous price data
            price_data = get_previous_price_data(combined_df2, date, symbol)

            if price_data is not None:
                # Update the price columns
                for col in price_columns:
                    merged_df_with_ttm_and_price.loc[mask, col] = price_data[col]

# Print summary information
print("\nMerged Data Summary:")
print(f"Total rows: {len(merged_df_with_ttm_and_price)}")
print(f"Rows with price data: {merged_df_with_ttm_and_price[price_columns].notna().all(axis=1).sum()}")
print(f"Rows missing price data: {merged_df_with_ttm_and_price[price_columns].isna().any(axis=1).sum()}")
# Print date range
print("\nDate range:")
print(f"From: {merged_df_with_ttm_and_price['date'].min()}")
print(f"To: {merged_df_with_ttm_and_price['date'].max()}")

# Display sample data
print("\nSample data:")
print(merged_df_with_ttm_and_price)

# Save to CSV
merged_df_with_ttm_and_price.to_csv('merged_financial_data_with_ttm_and_price.csv', index=False)

# Find rows where close price is missing in merged_with_prices
missing_close_prices = merged_df_with_ttm_and_price[merged_df_with_ttm_and_price['close'].isna()]
print(f"Total number of rows with missing close prices: {len(missing_close_prices)}")
print("\nStocks with missing close prices:")
print(missing_close_prices)

def check_stock_price(symbol, date):
    """
    Check the close price of a stock on a specific date.

    Args:
        symbol (str): Stock symbol
        date (str): Date in 'YYYY-MM-DD' format
    """
    # Convert date to datetime
    check_date = pd.to_datetime(date)

    # Get price data for the stock
    stock_data = combined_df2.xs(symbol, level='symbol')

    # Try to get the exact date
    try:
        price = stock_data.loc[check_date, 'close']
        print(f"\nClose price for {symbol} on {date}:")
        print(f"Exact date match: ${price:.2f}")
    except KeyError:
        # If exact date not found, get the most recent previous date
        try:
            previous_dates = stock_data.index[stock_data.index <= check_date]
            if len(previous_dates) > 0:
                most_recent_date = previous_dates[-1]
                price = stock_data.loc[most_recent_date, 'close']
                print(f"\nClose price for {symbol} on {date}:")
                print(f"No exact date match found.")
            else:
                print(f"\nNo price data available for {symbol} on or before {date}")
        except Exception as e:
            print(f"\nError finding price data: {str(e)}")

# Example usage:
# Check price for a specific stock and date
symbol = "EXE"
date = "2020-12-31"
check_stock_price(symbol, date)

# Drop rows with missing close prices
merged_df_with_ttm_and_price = merged_df_with_ttm_and_price.dropna(subset=['close'])

# Print confirmation of the cleaning
print(f"Rows dropped: {len(missing_close_prices)}")
print(f"Remaining rows: {len(merged_df_with_ttm_and_price)}")

print(merged_df_with_ttm_and_price['symbol'].nunique()) #no of unique tickers
#len(merged_price_financial['symbol'].unique())

print(len(merged_df_with_ttm_and_price.columns)) #no of columns

merged_df_with_ttm_and_price

merged_df_with_ttm_and_price.columns

# Create a copy of the DataFrame
financial_factor_df = merged_df_with_ttm_and_price.copy()

# Add new factor columns with NaN values
for col in financial_factor_calculations.keys():
    financial_factor_df[col] = pd.Series(np.nan, index=financial_factor_df.index, dtype='float64')

# Calculate factors for each stock
for symbol in financial_factor_df['symbol'].unique():
    stock_data = financial_factor_df[financial_factor_df['symbol'] == symbol].sort_values('date')

    for factor_name, calc_function in financial_factor_calculations.items():
        try:
            result = calc_function(stock_data)
            if result is not None:
                financial_factor_df.loc[stock_data.index, factor_name] = result
            else:
                print(f"Error: Required columns for {factor_name} not found for {symbol}")
        except Exception as e:
            print(f"Error calculating {factor_name} for {symbol}: {str(e)}")

# Handle any remaining invalid values
for col in financial_factor_calculations.keys():
    financial_factor_df[col] = (
        financial_factor_df[col]
        .replace([np.inf, -np.inf], np.nan)
        .astype('float64')
    )

# Display sample results
print("\nSample results:")

print(financial_factor_df.head())

#Number of Null Values for Each Factor
financial_factor_df[financial_factor_calculations.keys()].isna().sum()

#Stockwise Null Values
factor_list = list(financial_factor_calculations.keys())
stock_wise_nulls = financial_factor_df.groupby('symbol')[factor_list].apply(lambda x: x.isna().sum())
# Add total null values column
stock_wise_nulls['Total_Null_Values'] = stock_wise_nulls.sum(axis=1)
stock_wise_nulls = stock_wise_nulls.sort_values('Total_Null_Values', ascending=False)

print(stock_wise_nulls)

# Print factors for a specific stock and period
factor_df2 = financial_factor_df[['symbol', 'date', 'netIncome_ttm', 'net_profit_ttm', 'operatingCashFlow_ttm'] + list(financial_factor_calculations.keys())]

def print_stock_factors(factor_df2, symbol, start_date=None, end_date=None):
    """
    Print factors for a specific stock and date range

    Parameters:
    - factor_df: DataFrame containing the factors
    - symbol: Stock symbol to filter
    - start_date: Start date for filtering (optional)
    - end_date: End date for filtering (optional)
    """
    # Filter for the specific stock
    stock_data = factor_df2[factor_df2['symbol'] == symbol].copy()

    # Convert date column to datetime if it's not already
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # Apply date filters if provided
    if start_date:
        stock_data = stock_data[stock_data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        stock_data = stock_data[stock_data['date'] <= pd.to_datetime(end_date)]

    # Sort by date
    stock_data = stock_data.sort_values('date', ascending=False)

    # Select relevant columns
    #display_columns = ['date'] + factor_columns

    print(f"\nFactors for {symbol}:")
    print("=" * 100)
    #print(stock_data[display_columns].to_string())
    print(stock_data.to_string())
    print("=" * 100)

# Example usage:
# Print factors for a specific stock and period
print_stock_factors(factor_df2.round(4), 'DECK', '2023-01-01', '2024-12-31')

#Save the Quarterly Factors
quarterly_financial_factors = financial_factor_df[['symbol', 'date'] + list(financial_factor_calculations.keys())]
print(quarterly_financial_factors.head())

quarterly_financial_factors.to_csv('quarterly_financial_factors.csv', index=False)

print(len(quarterly_financial_factors.columns))
print(len(quarterly_tech_factors.columns))

quarterly_financial_factors.columns

quarterly_dates

# First, create a function to map dates to quarter ends
def map_to_quarter_end(date):
    date = pd.to_datetime(date)
    year = date.year
    month = date.month

    if 1 <= month <= 3:
        return pd.Timestamp(f"{year}-03-31")
    elif 4 <= month <= 6:
        return pd.Timestamp(f"{year}-06-30")
    elif 7 <= month <= 9:
        return pd.Timestamp(f"{year}-09-30")
    else:  # 10 <= month <= 12
        return pd.Timestamp(f"{year}-12-31")

# Create a copy of quarterly_financial_factors and map its dates
financial_factors_mapped = quarterly_financial_factors.copy()
financial_factors_mapped['mapped_date'] = financial_factors_mapped['date'].apply(map_to_quarter_end)

# Create a copy of quarterly_tech_factors
tech_factors_mapped = quarterly_tech_factors.copy()
tech_factors_mapped['mapped_date'] = tech_factors_mapped['date']

# Merge the DataFrames using the mapped dates
quarterly_merged_factors = pd.merge(
    financial_factors_mapped,
    tech_factors_mapped,
    on=['symbol', 'mapped_date'],
    how='inner',
    suffixes=('_financial', '_tech')
)

# Drop the original date columns and rename mapped_date to date
quarterly_merged_factors = quarterly_merged_factors.drop(['date_financial', 'date_tech'], axis=1)
quarterly_merged_factors = quarterly_merged_factors.rename(columns={'mapped_date': 'date'})

# Sort by symbol and date (latest first)
quarterly_merged_factors = quarterly_merged_factors.sort_values(['symbol', 'date'], ascending=[True, False]).reset_index(drop=True)

columns = quarterly_merged_factors.columns.tolist()
columns.remove('date')
columns = ['date'] + columns

# Reorder the columns
quarterly_merged_factors = quarterly_merged_factors[columns]

# Display results
print("\nFirst few rows of merged factors:")
print(quarterly_merged_factors.head())

print("\nAll quarterly merged factors are saved to 'qaurterly_merged_factors.csv'")
quarterly_merged_factors.to_csv('quarterly_merged_factors.csv', index=False)

print("Shape of merged factors:", quarterly_merged_factors.shape)
# Show the date range
print("\nDate range in merged factors:")
print("Earliest date:", quarterly_merged_factors['date'].min())
print("Latest date:", quarterly_merged_factors['date'].max())

# Check for any missing values
print("\nMissing values in Quarterly Merged Factors:")
print(quarterly_merged_factors.isna().sum())

quarterly_merged_factors.columns

unique_dates = sorted(quarterly_merged_factors['date'].unique(), reverse=True)
for date in unique_dates:
    print(f"{date.strftime('%Y-%m-%d')} - Quarter: Q{(date.month-1)//3 + 1}")

#Stock wise NaN Values in Quarterly_Marged_factors
def analyze_NaN_values(factor_df):
    factor_columns = [col for col in factor_df.columns if col not in ['date', 'symbol']]

    # Calculate total NaN values per stock
    stock_nan_totals = factor_df.groupby('symbol')[factor_columns].apply(
        lambda x: x.isna().sum().sum()
    ).reset_index()
    stock_nan_totals.columns = ['Symbol', 'Total NaN Values']

    # Sort by total NaN values in descending order
    stock_nan_totals = stock_nan_totals.sort_values('Total NaN Values', ascending=False)

    # Calculate detailed statistics for each factor
    detailed_stats = []
    for symbol in factor_df['symbol'].unique():
        stock_data = factor_df[factor_df['symbol'] == symbol]

        for col in factor_columns:
            nan_count = stock_data[col].isna().sum()
            total_count = len(stock_data)
            percentage = (nan_count / total_count) * 100

            if percentage > 0:  # Only include factors with missing values
                detailed_stats.append({
                    'Symbol': symbol,
                    'Factor Name': col,
                    'Total Values': total_count,
                    'NaN Values': nan_count,
                    'Percentage NaN': f"{percentage:.2f}%"
                })

    return stock_nan_totals, pd.DataFrame(detailed_stats)

def print_stock_wise_stats(factor_df):
    # Get statistics
    total_nan_stats, detailed_stats = analyze_NaN_values(factor_df)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("=" * 100)
    print("\nTotal NaN Values per Stock (Sorted by Missing Values):")
    print(total_nan_stats.to_string(index=False))
    print("\n")

    # Print detailed statistics if available
    if not detailed_stats.empty:
        print("=== DETAILED STATISTICS ===")
        print("=" * 100)
        print("\nDetailed Factor-wise Missing Values:")
        print(detailed_stats.sort_values(['Symbol', 'Factor Name']).to_string(index=False))

    else:
        print("\nNo stocks have missing factor values!")

# Print stock-wise statistics
print_stock_wise_stats(quarterly_merged_factors)

def print_rows_with_nan_factors(factor_df):
    """
    Print rows that have NaN values in any of the factor columns
    """
    # Get all columns except 'date' and 'symbol'
    factor_columns = [col for col in factor_df.columns if col not in ['date', 'symbol']]

    # Find rows with any NaN values in factor columns
    nan_rows = factor_df[factor_df[factor_columns].isna().any(axis=1)]

    if not nan_rows.empty:
        print("\nRows with Missing Factor Values:")
        print("=" * 100)

        # Sort by symbol and date (latest first)
        nan_rows = nan_rows.sort_values(['symbol', 'date'], ascending=[True, False])

        # Display the rows
        print(nan_rows.to_string())
        print("=" * 100)

        # Print summary of how many rows have NaN values
        print(f"\nTotal number of rows with missing factors: {len(nan_rows)}")

        # Save to CSV
        nan_rows_csv_path = 'rows_with_missing_factors.csv'
        nan_rows.to_csv(nan_rows_csv_path, index=False)
        print(f"\nRows with missing factors saved to: {nan_rows_csv_path}")
    else:
        print("\nNo rows have missing factor values!")

# Print rows with NaN factors
print_rows_with_nan_factors(quarterly_merged_factors)

# Calculate Quarterly Returns
combined_df_return_4 = combined_df.copy()

# Ensure date is datetime
combined_df_return_4['date'] = pd.to_datetime(combined_df_return_4['date'])

# Set date and symbol as index
combined_df_return_4.set_index(['date', 'symbol'], inplace=True)

# Get close prices and unstack to have symbols as columns
close_prices = combined_df_return_4['close'].unstack(level=-1)

# Resample to quarterly frequency and get last price of each quarter
quarterly_prices = close_prices.resample('QE').last()  # Get last price of each quarter

# Calculate percentage change between current quarter and previous quarter
quarterly_returns = quarterly_prices.pct_change().dropna(axis=0, how='all')  # This calculates (current - previous) / previous
quarterly_returns['factor'] = 'quarterly_return'

# Reset index to get date as a column and Melt the dataframe to get it in long format
quarterly_returns = quarterly_returns.reset_index().melt(
    id_vars=['date', 'factor'],
    var_name='symbol',
    value_name='Quarterly_Return'
)

# Pivot the reshaped data back to have 'date' and 'factor' as index and symbols as columns
quarterly_returns = quarterly_returns.pivot(
    index=['date', 'factor'],
    columns='symbol',
    values='Quarterly_Return'
)

# Sort by date in descending order (most recent first)
quarterly_returns = quarterly_returns.sort_index(level='date', ascending=False)

# Display the first few rows of quarterly returns
print("Quarterly Returns:")
print(quarterly_returns.head())

# Save to CSV
quarterly_returns.to_csv('quarterly_returns.csv')

# Calculate percentage change and shift forward to get next quarter's returns
next_quarter_returns = quarterly_prices.pct_change().dropna(axis=0, how='all')
next_quarter_returns = next_quarter_returns.shift(1)  # Shift forward to get next quarter's returns
next_quarter_returns.dropna(axis=0, how='all', inplace=True)
next_quarter_returns['factor'] = 'next_quarter_return'

# Reset index to get date as a column and Melt the dataframe to get it in long format
next_quarter_returns = next_quarter_returns.reset_index().melt(
    id_vars=['date', 'factor'],
    var_name='symbol',
    value_name='next_quarter_return'
)

# Pivot the reshaped data back to have 'date' and 'factor' as index and symbols as columns
next_quarter_returns = next_quarter_returns.pivot(
    index=['date', 'factor'],
    columns='symbol',
    values='next_quarter_return'
)

# Sort by date in descending order (most recent first)
next_quarter_returns = next_quarter_returns.sort_index(level='date', ascending=False)

# Display the first few rows of next quarter returns
print("Next Quarter's Returns:")
print(next_quarter_returns.head())

# Save to CSV
next_quarter_returns.to_csv('next_quarter_returns.csv')

# Convert quarterly_merged_factors to pivot format
quarterly_merged_factors = quarterly_merged_factors.reset_index()
quarterly_merged_factors = quarterly_merged_factors.melt(
    id_vars=['date', 'symbol'],
    var_name='factor',
    value_name='value'
)
quarterly_merged_factors = quarterly_merged_factors.pivot_table(
    index=['date', 'factor'],
    columns='symbol',
    values='value'
)

# 2. Merge all three datasets
combined_quarterly_data = pd.concat([
    quarterly_merged_factors,  # Factor data
    quarterly_returns,         # Current quarter returns
    next_quarter_returns       # Next quarter returns
])

# Sort the data by date (most recent first) and factor
combined_quarterly_data = combined_quarterly_data.sort_index(level=['date', 'factor'], ascending=[False, True])

# Save to CSV
#combined_quarterly_data.to_csv('combined_quarterly_data.csv', index=False)

# Check shape and columns of combined_data
print(f"Shape of combined_data: {combined_quarterly_data.shape}")
print(f"Columns in combined_data: {combined_quarterly_data.columns.tolist()}")
print(f"Start Date: {combined_quarterly_data.index.get_level_values('date').max()}")
print(f"End Date: {combined_quarterly_data.index.get_level_values('date').min()}")
print(f"Index levels in combined_data: {combined_quarterly_data.index.names}")

# Display the first few rows of combined data
print("Combined Quarterly Data:")
print(combined_quarterly_data.head().round(4))

# Drop the 'index' factor from the DataFrame
combined_quarterly_data = combined_quarterly_data[combined_quarterly_data.index.get_level_values(1) != 'index']

# Print factors to verify the change
print("\nFactors in the Dataset:")
print("=" * 50)

# Get unique factors from the index
factors = combined_quarterly_data.index.get_level_values(1).unique()

# Print each factor on a new line with numbering
for i, factor in enumerate(factors, 1):
    print(f"{i}. {factor}")

print("=" * 50)
print(f"Total number of factors: {len(factors)}")

# Display the first few rows to verify the format
print("First few rows of Combined Quartely Data:")
print(combined_quarterly_data.head().round(4))

# Save the result to a CSV file
print(f"Combined Quarterly Data saved to {'combined_quarterly_data.csv'}")
combined_quarterly_data.to_csv('combined_quarterly_data.csv')



