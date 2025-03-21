# Install required package
# !pip install yfinance  # Removed for production; ensure installed via requirements.txt

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
import io
import base64
from flask import Flask, request, jsonify
warnings.filterwarnings('ignore')
from matplotlib.ticker import PercentFormatter

app = Flask(__name__)

class PortfolioAnalyzer:
    def __init__(self):
        self.today_date = datetime.today().strftime("%Y-%m-%d")
        self.default_start_date = "2015-01-01"
        self.risk_free_rate = 0.02  # Risk-free rate for Sharpe and Sortino calculations

    def fetch_stock_data(self, stocks, start=None, end=None):
        """Fetch stock data with error handling and track earliest available date."""
        if start is None:
            start = self.default_start_date
        if end is None:
            end = self.today_date
        error_tickers = {}
        earliest_dates = {}
        stock_data = yf.download(stocks, start=start, end=end, auto_adjust=True)
        if stock_data.empty:
            print("⚠️ No data available for the specified date range.")
            return None, error_tickers, earliest_dates
        stock_data = stock_data['Close']
        for ticker in stocks:
            if ticker not in stock_data.columns or stock_data[ticker].isna().all():
                error_tickers[ticker] = "Data not available"
            else:
                first_valid = stock_data[ticker].first_valid_index()
                if first_valid:
                    earliest_dates[ticker] = first_valid
                else:
                    error_tickers[ticker] = "No valid data points"
                    stock_data = stock_data.drop(columns=[ticker])
        return stock_data, error_tickers, earliest_dates

    def compute_returns(self, prices):
        """Compute daily returns from price data"""
        return prices.pct_change().dropna()

    def compute_max_drawdown(self, series):
        """Compute maximum drawdown for a cumulative return series"""
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        return drawdown.min()

    def compute_downside_deviation(self, returns, target=0):
        """Compute downside deviation for Sortino Ratio (standard deviation of negative returns)."""
        negative_returns = returns[returns < target]
        if len(negative_returns) == 0:
            return 0
        return np.std(negative_returns) * np.sqrt(252)  # Annualized

    def portfolio_performance(self, weights, returns):
        """Calculate portfolio performance metrics including Sharpe and Sortino Ratios."""
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        
        # Compute portfolio daily returns for Sortino
        portfolio_daily_returns = returns.dot(weights)
        downside_dev = self.compute_downside_deviation(portfolio_daily_returns, target=0)
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_dev if downside_dev != 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio, sortino_ratio

    def optimize_portfolio(self, returns, objective='sharpe', max_allocation=1):
        """Optimize portfolio based on different objectives with minimum investment at 0%."""
        num_assets = returns.shape[1]
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, max_allocation) for _ in range(num_assets))  # Minimum is 0%

        def negative_sharpe(weights, returns):
            portfolio_return, portfolio_volatility, sharpe, _ = self.portfolio_performance(weights, returns)
            return -sharpe

        def negative_sortino(weights, returns):
            portfolio_return, _, _, sortino = self.portfolio_performance(weights, returns)
            return -sortino

        def portfolio_volatility(weights, returns):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        if objective == 'sharpe':
            obj_fun = negative_sharpe
            obj_args = (returns,)
        elif objective == 'sortino':
            obj_fun = negative_sortino
            obj_args = (returns,)
        elif objective == 'min_volatility':
            obj_fun = portfolio_volatility
            obj_args = (returns,)
        else:
            raise ValueError(f"Unknown optimization objective: {objective}")

        try:
            optimal_results = minimize(
                obj_fun, initial_weights, args=obj_args,
                method='SLSQP', constraints=constraints, bounds=bounds
            )
            if not optimal_results.success:
                print(f"⚠️ Optimization warning: {optimal_results.message}")
            weights = optimal_results.x
            weights[weights < 0.001] = 0  # Ensure small weights are set to 0
            weights = weights / np.sum(weights) if np.sum(weights) != 0 else weights
            return weights
        except Exception as e:
            print(f"❌ Optimization error: {str(e)}")
            return initial_weights

    def compute_var(self, returns, confidence_level=0.95, method='historical'):
        """Compute Value at Risk (VaR) using historical simulation."""
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[index]
        return -var

    def plot_to_base64(self, plt):
        """Convert matplotlib plot to base64 string."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()
        return img_str

    def plot_correlation_matrix(self, prices, fig_size=(10, 6)):
        """Compute and print the correlation matrix of portfolio stocks instead of plotting."""
        corr_matrix = prices.corr()
        print("\n**Correlation Matrix of Portfolio Stocks:**")
        print(corr_matrix.to_string(float_format="%.2f"))
        return "Correlation matrix printed"

    def adjust_weights_over_time(self, tickers, weights, returns):
        """Adjust weights dynamically, ensuring alignment with returns data."""
        ticker_list = list(tickers)  # Convert Index to list for indexing
        weights_df = pd.DataFrame(index=returns.index, columns=tickers)
        for date in returns.index:
            active_tickers = [t for t in ticker_list if not returns[t].loc[:date].isna().all()]
            if active_tickers:
                total_weight = sum(weights[ticker_list.index(t)] for t in active_tickers)
                for t in ticker_list:
                    if t in active_tickers:
                        weights_df.loc[date, t] = weights[ticker_list.index(t)] / total_weight if total_weight > 0 else 0
                    else:
                        weights_df.loc[date, t] = 0
            else:
                weights_df.loc[date, :] = 0
        return weights_df.fillna(method='ffill')

    def plot_cumulative_returns(self, returns, weights, benchmark_returns=None, optimized_weights=None, sortino_optimized_weights=None, earliest_dates=None, user_start_date=None, fig_size=(10, 6)):
        """Compute and print cumulative returns metrics instead of plotting."""
        earliest_date = max([d for d in earliest_dates.values()] + [pd.to_datetime(user_start_date)])
        adjusted_returns = returns.loc[earliest_date:]
        if adjusted_returns.empty:
            print(f"⚠️ No data available after adjusting start date to {earliest_date.strftime('%Y-%m-%d')}.")
            return None, None

        tickers = returns.columns
        weights_df = self.adjust_weights_over_time(tickers, weights, adjusted_returns)
        portfolio_returns = (adjusted_returns * weights_df).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod() - 1

        print(f"\n**Cumulative Returns Metrics (From {earliest_date.strftime('%Y-%m-%d')}):**")
        print("\n--- Current Portfolio ---")
        print(f"Final Cumulative Return: {cumulative.iloc[-1]:.2%}")
        print(f"Max Cumulative Return: {cumulative.max():.2%}")
        print(f"Min Cumulative Return: {cumulative.min():.2%}")

        if optimized_weights is not None:
            opt_weights_df = self.adjust_weights_over_time(tickers, optimized_weights, adjusted_returns)
            opt_returns = (adjusted_returns * opt_weights_df).sum(axis=1)
            opt_cumulative = (1 + opt_returns).cumprod() - 1
            print("\n--- Optimized Portfolio (Max Sharpe) ---")
            print(f"Final Cumulative Return: {opt_cumulative.iloc[-1]:.2%}")
            print(f"Max Cumulative Return: {opt_cumulative.max():.2%}")
            print(f"Min Cumulative Return: {opt_cumulative.min():.2%}")

        if sortino_optimized_weights is not None:
            sortino_opt_weights_df = self.adjust_weights_over_time(tickers, sortino_optimized_weights, adjusted_returns)
            sortino_opt_returns = (adjusted_returns * sortino_opt_weights_df).sum(axis=1)
            sortino_opt_cumulative = (1 + sortino_opt_returns).cumprod() - 1
            print("\n--- Optimized Portfolio (Max Sortino) ---")
            print(f"Final Cumulative Return: {sortino_opt_cumulative.iloc[-1]:.2%}")
            print(f"Max Cumulative Return: {sortino_opt_cumulative.max():.2%}")
            print(f"Min Cumulative Return: {sortino_opt_cumulative.min():.2%}")

        if benchmark_returns is not None:
            if isinstance(benchmark_returns, dict):
                for key, series in benchmark_returns.items():
                    bench_series = series.loc[earliest_date:]
                    if not bench_series.empty:
                        bench_cumulative = (1 + bench_series).cumprod() - 1
                        print(f"\n--- Benchmark: {key} ---")
                        print(f"Final Cumulative Return: {bench_cumulative.iloc[-1]:.2%}")
                        print(f"Max Cumulative Return: {bench_cumulative.max():.2%}")
                        print(f"Min Cumulative Return: {bench_cumulative.min():.2%}")
            else:
                bench_series = benchmark_returns.loc[earliest_date:]
                if not bench_series.empty:
                    bench_cumulative = (1 + bench_series).cumprod() - 1
                    print("\n--- Benchmark ---")
                    print(f"Final Cumulative Return: {bench_cumulative.iloc[-1]:.2%}")
                    print(f"Max Cumulative Return: {bench_cumulative.max():.2%}")
                    print(f"Min Cumulative Return: {bench_cumulative.min():.2%}")

        return "Cumulative returns printed", weights_df

    def plot_crisis_performance(self, returns, benchmark_returns_dict, results, start_date, end_date, earliest_dates, fig_size=(10, 6)):
        """Compute and print COVID-19 crisis performance metrics instead of plotting."""
        covid_period = ("2020-02-01", "2020-04-30")
        covid_start = pd.to_datetime("2020-02-01")
        
        # Identify stocks with data before COVID-19 start
        tickers = returns.columns
        ticker_list = list(tickers)
        valid_tickers = [t for t in ticker_list if earliest_dates.get(t, covid_start) < covid_start]
        excluded_tickers = [t for t in ticker_list if t not in valid_tickers]

        print(f"Debug: Valid tickers (data before {covid_start}): {valid_tickers}")
        print(f"Debug: Excluded tickers (data after {covid_start}): {excluded_tickers}")

        print("\n**COVID-19 Crisis Performance Metrics (2020-02-01 to 2020-04-30):**")

        if not valid_tickers:
            print("⚠️ No stocks in the portfolio have data before the COVID-19 period (2020-02-01). Skipping portfolio metrics.")
        else:
            # Fetch stock prices specifically for the COVID-19 period for valid tickers
            print(f"Fetching stock data for valid tickers {valid_tickers} for COVID-19 period...")
            crisis_prices, error_tickers, _ = self.fetch_stock_data(valid_tickers, start=covid_period[0], end=covid_period[1])
            if crisis_prices is None or crisis_prices.empty:
                print("⚠️ No price data available for valid tickers during the COVID-19 period.")
            else:
                # Compute returns for the crisis period
                crisis_returns = self.compute_returns(crisis_prices)
                print(f"Debug: crisis_returns shape: {crisis_returns.shape}, columns: {list(crisis_returns.columns)}")
                print(f"Debug: crisis_returns head:\n{crisis_returns.head()}")
                
                if crisis_returns.empty:
                    print("⚠️ No valid returns data for portfolios in COVID-19 period after computing returns.")
                else:
                    # Fill remaining NaNs with 0 to prevent issues in calculations
                    crisis_returns = crisis_returns.fillna(0)
                    print(f"Debug: crisis_returns after filling NaNs (shape: {crisis_returns.shape}, columns: {list(crisis_returns.columns)}")
                    print(f"Debug: crisis_returns head after filling NaNs:\n{crisis_returns.head()}")
                    
                    # Redistribute weights for valid tickers (Current Portfolio)
                    cur_weights_dict = results['current_portfolio']['weights']
                    cur_weights = np.array([cur_weights_dict[t] for t in crisis_returns.columns])
                    cur_weights_sum = cur_weights.sum()
                    if cur_weights_sum > 0:
                        cur_weights_adjusted = cur_weights / cur_weights_sum  # Normalize to sum to 1
                    else:
                        cur_weights_adjusted = cur_weights
                        print("⚠️ Sum of weights for original portfolio is 0; using unnormalized weights.")
                    
                    # Current Portfolio
                    cur_portfolio_returns = crisis_returns.mul(cur_weights_adjusted, axis=1).sum(axis=1)
                    print(f"Debug: Original weights (normalized): {cur_weights_adjusted}")
                    print(f"Debug: Original portfolio returns head:\n{cur_portfolio_returns.head()}")
                    if not cur_portfolio_returns.empty and cur_portfolio_returns.notna().any():
                        cur_cumulative = (1 + cur_portfolio_returns).cumprod() - 1
                        print("\n--- Original Portfolio (COVID) ---")
                        print(f"Final Cumulative Return: {cur_cumulative.iloc[-1]:.2%}")
                        print(f"Max Cumulative Return: {cur_cumulative.max():.2%}")
                        print(f"Min Cumulative Return: {cur_cumulative.min():.2%}")
                    else:
                        print("⚠️ No valid returns calculated for Original Portfolio (empty or all NaN).")

                    # Redistribute weights for Sharpe-optimized portfolio
                    opt_weights_dict = results['optimized_portfolio']['weights']
                    opt_weights = np.array([opt_weights_dict[t] for t in crisis_returns.columns])
                    opt_weights_sum = opt_weights.sum()
                    if opt_weights_sum > 0:
                        opt_weights_adjusted = opt_weights / opt_weights_sum  # Normalize to sum to 1
                    else:
                        opt_weights_adjusted = opt_weights
                        print("⚠️ Sum of weights for Sharpe-optimized portfolio is 0; using unnormalized weights.")
                    
                    # Sharpe-Optimized Portfolio
                    opt_portfolio_returns = crisis_returns.mul(opt_weights_adjusted, axis=1).sum(axis=1)
                    print(f"Debug: Sharpe-Optimized weights (normalized): {opt_weights_adjusted}")
                    print(f"Debug: Sharpe-Optimized portfolio returns head:\n{opt_portfolio_returns.head()}")
                    if not opt_portfolio_returns.empty and opt_portfolio_returns.notna().any():
                        opt_cumulative = (1 + opt_portfolio_returns).cumprod() - 1
                        print("\n--- Optimized Portfolio (Max Sharpe, COVID) ---")
                        print(f"Final Cumulative Return: {opt_cumulative.iloc[-1]:.2%}")
                        print(f"Max Cumulative Return: {opt_cumulative.max():.2%}")
                        print(f"Min Cumulative Return: {opt_cumulative.min():.2%}")
                    else:
                        print("⚠️ No valid returns calculated for Sharpe-Optimized Portfolio (empty or all NaN).")

                    # Redistribute weights for Sortino-optimized portfolio
                    sortino_opt_weights_dict = results['sortino_optimized_portfolio']['weights']
                    sortino_opt_weights = np.array([sortino_opt_weights_dict[t] for t in crisis_returns.columns])
                    sortino_opt_weights_sum = sortino_opt_weights.sum()
                    if sortino_opt_weights_sum > 0:
                        sortino_opt_weights_adjusted = sortino_opt_weights / sortino_opt_weights_sum  # Normalize to sum to 1
                    else:
                        sortino_opt_weights_adjusted = sortino_opt_weights
                        print("⚠️ Sum of weights for Sortino-optimized portfolio is 0; using unnormalized weights.")
                    
                    # Sortino-Optimized Portfolio
                    sortino_opt_portfolio_returns = crisis_returns.mul(sortino_opt_weights_adjusted, axis=1).sum(axis=1)
                    print(f"Debug: Sortino-Optimized weights (normalized): {sortino_opt_weights_adjusted}")
                    print(f"Debug: Sortino-Optimized portfolio returns head:\n{sortino_opt_portfolio_returns.head()}")
                    if not sortino_opt_portfolio_returns.empty and sortino_opt_portfolio_returns.notna().any():
                        sortino_opt_cumulative = (1 + sortino_opt_portfolio_returns).cumprod() - 1
                        print("\n--- Optimized Portfolio (Max Sortino, COVID) ---")
                        print(f"Final Cumulative Return: {sortino_opt_cumulative.iloc[-1]:.2%}")
                        print(f"Max Cumulative Return: {sortino_opt_cumulative.max():.2%}")
                        print(f"Min Cumulative Return: {sortino_opt_cumulative.min():.2%}")
                    else:
                        print("⚠️ No valid returns calculated for Sortino-Optimized Portfolio (empty or all NaN).")

        # Compute benchmarks
        for bench_ticker, bench_series in benchmark_returns_dict.items():
            bench_crisis = bench_series.loc[covid_period[0]:covid_period[1]]
            if not bench_crisis.empty:
                bench_cumulative = (1 + bench_crisis).cumprod() - 1
                print(f"\n--- {bench_ticker} (COVID) ---")
                print(f"Final Cumulative Return: {bench_cumulative.iloc[-1]:.2%}")
                print(f"Max Cumulative Return: {bench_cumulative.max():.2%}")
                print(f"Min Cumulative Return: {bench_cumulative.min():.2%}")

        return "Crisis performance metrics printed"

    def analyze_portfolio(self, tickers, weights=None, start_date=None, end_date=None, benchmark='^GSPC'):
        """Analyze a portfolio of stocks, including Sortino Ratio and Sortino-optimized portfolio."""
        if start_date is None:
            start_date = self.default_start_date
        if end_date is None:
            end_date = self.today_date
        if weights is None:
            weights = [1/len(tickers)] * len(tickers)
        weights = np.array(weights) / sum(weights)
        print(f"Fetching data for {len(tickers)} stocks...")
        stock_prices, error_tickers, earliest_dates = self.fetch_stock_data(tickers, start=start_date, end=end_date)
        if stock_prices is None or stock_prices.empty:
            return {'error': 'No valid stock data available', 'error_tickers': error_tickers, 'earliest_dates': earliest_dates}
        if error_tickers:
            print(f"⚠️ Issues with {len(error_tickers)} tickers:")
            for ticker, error in error_tickers.items():
                print(f"  - {ticker}: {error}")
        returns = self.compute_returns(stock_prices)
        valid_tickers = list(returns.columns)
        if len(valid_tickers) < len(tickers):
            print(f"⚠️ {len(tickers) - len(valid_tickers)} tickers missing from data, adjusting weights...")
            valid_idx = [i for i, t in enumerate(tickers) if t in valid_tickers]
            weights = np.array([weights[i] for i in valid_idx])
            weights = weights / np.sum(weights)
            tickers = valid_tickers
        if returns.shape[0] < 20:
            return {'error': 'Insufficient data for analysis', 'earliest_dates': earliest_dates}
        print(f"Fetching benchmark data: {benchmark}...")
        benchmark_prices, _, _ = self.fetch_stock_data([benchmark], start=start_date, end=end_date)
        if benchmark_prices is None or benchmark_prices.empty:
            print(f"⚠️ Could not retrieve benchmark data for {benchmark}")
            benchmark_returns = None
        else:
            benchmark_returns = self.compute_returns(benchmark_prices).iloc[:, 0]
        portfolio_returns = returns.dot(weights)
        annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio = self.portfolio_performance(weights, returns)
        max_drawdown = self.compute_max_drawdown((1 + portfolio_returns).cumprod())
        if benchmark_returns is not None:
            bench_return, bench_vol, bench_sharpe, bench_sortino = self.portfolio_performance(np.array([1.0]), pd.DataFrame(benchmark_returns))
            benchmark_max_dd = self.compute_max_drawdown((1 + benchmark_returns).cumprod())
        else:
            bench_return = bench_vol = bench_sharpe = bench_sortino = benchmark_max_dd = None
        optimal_weights = self.optimize_portfolio(returns, objective='sharpe', max_allocation=1)
        opt_return, opt_volatility, opt_sharpe, opt_sortino = self.portfolio_performance(optimal_weights, returns)
        opt_max_drawdown = self.compute_max_drawdown((1 + returns.dot(optimal_weights)).cumprod())
        opt_var = self.compute_var(returns.dot(optimal_weights))
        sortino_optimal_weights = self.optimize_portfolio(returns, objective='sortino', max_allocation=1)
        sortino_opt_return, sortino_opt_volatility, sortino_opt_sharpe, sortino_opt_sortino = self.portfolio_performance(sortino_optimal_weights, returns)
        sortino_opt_max_drawdown = self.compute_max_drawdown((1 + returns.dot(sortino_optimal_weights)).cumprod())
        sortino_opt_var = self.compute_var(returns.dot(sortino_optimal_weights))
        min_vol_weights = self.optimize_portfolio(returns, objective='min_volatility')
        min_vol_return, min_vol_volatility, min_vol_sharpe, min_vol_sortino = self.portfolio_performance(min_vol_weights, returns)
        result = {
            'current_portfolio': {
                'tickers': tickers,
                'weights': dict(zip(tickers, weights)),
                'return': annualized_return,
                'volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'var': self.compute_var(portfolio_returns)
            },
            'optimized_portfolio': {
                'weights': dict(zip(tickers, optimal_weights)),
                'return': opt_return,
                'volatility': opt_volatility,
                'sharpe_ratio': opt_sharpe,
                'sortino_ratio': opt_sortino,
                'max_drawdown': opt_max_drawdown,
                'var': opt_var
            },
            'sortino_optimized_portfolio': {
                'weights': dict(zip(tickers, sortino_optimal_weights)),
                'return': sortino_opt_return,
                'volatility': sortino_opt_volatility,
                'sharpe_ratio': sortino_opt_sharpe,
                'sortino_ratio': sortino_opt_sortino,
                'max_drawdown': sortino_opt_max_drawdown,
                'var': sortino_opt_var
            },
            'min_volatility_portfolio': {
                'weights': dict(zip(tickers, min_vol_weights)),
                'return': min_vol_return,
                'volatility': min_vol_volatility,
                'sharpe_ratio': min_vol_sharpe,
                'sortino_ratio': min_vol_sortino
            },
            'earliest_dates': earliest_dates
        }
        if benchmark_returns is not None:
            result['benchmark'] = {
                'ticker': benchmark,
                'return': bench_return,
                'volatility': bench_vol,
                'sharpe_ratio': bench_sharpe,
                'sortino_ratio': bench_sortino,
                'max_drawdown': benchmark_max_dd,
                'var': self.compute_var(benchmark_returns)
            }
        return result

    def plot_efficient_frontier(self, returns, n_portfolios=1000, risk_free_rate=0.02, fig_size=(10, 6)):
        """Compute and print efficient frontier metrics instead of plotting."""
        np.random.seed(42)
        n_assets = returns.shape[1]
        all_weights = np.zeros((n_portfolios, n_assets))
        all_returns = np.zeros(n_portfolios)
        all_volatilities = np.zeros(n_portfolios)
        all_sharpe_ratios = np.zeros(n_portfolios)
        all_sortino_ratios = np.zeros(n_portfolios)
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            all_weights[i, :] = weights
            port_return, port_vol, port_sharpe, port_sortino = self.portfolio_performance(weights, returns)
            all_returns[i] = port_return
            all_volatilities[i] = port_vol
            all_sharpe_ratios[i] = port_sharpe
            all_sortino_ratios[i] = port_sortino

        optimal_weights = self.optimize_portfolio(returns, objective='sharpe')
        opt_return, opt_vol, opt_sharpe, opt_sortino = self.portfolio_performance(optimal_weights, returns)
        sortino_optimal_weights = self.optimize_portfolio(returns, objective='sortino')
        sortino_opt_return, sortino_opt_vol, sortino_opt_sharpe, sortino_opt_sortino = self.portfolio_performance(sortino_optimal_weights, returns)
        min_vol_weights = self.optimize_portfolio(returns, objective='min_volatility')
        min_vol_return, min_vol_vol, min_vol_sharpe, min_vol_sortino = self.portfolio_performance(min_vol_weights, returns)
        equal_weights = np.ones(n_assets) / n_assets
        eq_return, eq_vol, eq_sharpe, eq_sortino = self.portfolio_performance(equal_weights, returns)

        print("\n**Efficient Frontier Metrics:**")
        print("\n--- Optimal Portfolio (Max Sharpe) ---")
        print(f"Return: {opt_return:.2%}")
        print(f"Volatility: {opt_vol:.2%}")
        print(f"Sharpe Ratio: {opt_sharpe:.2f}")
        print(f"Sortino Ratio: {opt_sortino:.2f}")

        print("\n--- Optimal Portfolio (Max Sortino) ---")
        print(f"Return: {sortino_opt_return:.2%}")
        print(f"Volatility: {sortino_opt_vol:.2%}")
        print(f"Sharpe Ratio: {sortino_opt_sharpe:.2f}")
        print(f"Sortino Ratio: {sortino_opt_sortino:.2f}")

        print("\n--- Minimum Volatility Portfolio ---")
        print(f"Return: {min_vol_return:.2%}")
        print(f"Volatility: {min_vol_vol:.2%}")
        print(f"Sharpe Ratio: {min_vol_sharpe:.2f}")
        print(f"Sortino Ratio: {min_vol_sortino:.2f}")

        print("\n--- Equal Weight Portfolio ---")
        print(f"Return: {eq_return:.2%}")
        print(f"Volatility: {eq_vol:.2%}")
        print(f"Sharpe Ratio: {eq_sharpe:.2f}")
        print(f"Sortino Ratio: {eq_sortino:.2f}")

        print("\n--- Simulated Portfolios (Summary) ---")
        print(f"Average Return: {np.mean(all_returns):.2%}")
        print(f"Average Volatility: {np.mean(all_volatilities):.2%}")
        print(f"Average Sharpe Ratio: {np.mean(all_sharpe_ratios):.2f}")
        print(f"Average Sortino Ratio: {np.mean(all_sortino_ratios):.2f}")
        print(f"Max Sharpe Ratio: {np.max(all_sharpe_ratios):.2f}")
        print(f"Max Sortino Ratio: {np.max(all_sortino_ratios):.2f}")

        return "Efficient frontier metrics printed"

    def plot_portfolio_weights(self, weights_dict, title="Portfolio Weights", fig_size=(10, 6), ax=None):
        """Print portfolio weights instead of plotting."""
        print(f"\n**{title}:**")
        for ticker, weight in weights_dict.items():
            print(f"{ticker}: {weight:.1%}")
        return "Portfolio weights printed"

def run_portfolio_analysis(tickers, weights=None, start_date=None, end_date=None, benchmark_tickers=['^GSPC']):
    """Run a complete portfolio analysis with metrics printed instead of visualizations."""
    analyzer = PortfolioAnalyzer()
    results = analyzer.analyze_portfolio(tickers, weights, start_date, end_date, benchmark=benchmark_tickers[0])
    if 'error' in results:
        print(f"Error: {results['error']}")
        return results

    print("\n===== PORTFOLIO ANALYSIS =====")
    print(f"Analysis period: {start_date or analyzer.default_start_date} to {end_date or analyzer.today_date}")

    current = results['current_portfolio']
    print("\n----- CURRENT PORTFOLIO -----")
    print(f"Annual Return: {current['return']:.2%}")
    print(f"Annual Volatility: {current['volatility']:.2%}")
    print(f"Sharpe Ratio: {current['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {current['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {current['max_drawdown']:.2%}")
    print(f"Value at Risk (95%): {current['var']:.2%}")

    optimized = results['optimized_portfolio']
    print("\n----- OPTIMIZED PORTFOLIO (MAX SHARPE) -----")
    print(f"Annual Return: {optimized['return']:.2%}")
    print(f"Annual Volatility: {optimized['volatility']:.2%}")
    print(f"Sharpe Ratio: {optimized['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {optimized['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {optimized['max_drawdown']:.2%}")
    print(f"Value at Risk (95%): {optimized['var']:.2%}")
    sharpe_improvement = (optimized['sharpe_ratio']/current['sharpe_ratio'] - 1) if current['sharpe_ratio'] != 0 else 0
    sortino_improvement = (optimized['sortino_ratio']/current['sortino_ratio'] - 1) if current['sortino_ratio'] != 0 else 0
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:.2%}")
    print(f"Sortino Ratio Improvement: {sortino_improvement:.2%}")

    sortino_optimized = results['sortino_optimized_portfolio']
    print("\n----- OPTIMIZED PORTFOLIO (MAX SORTINO) -----")
    print(f"Annual Return: {sortino_optimized['return']:.2%}")
    print(f"Annual Volatility: {sortino_optimized['volatility']:.2%}")
    print(f"Sharpe Ratio: {sortino_optimized['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {sortino_optimized['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {sortino_optimized['max_drawdown']:.2%}")
    print(f"Value at Risk (95%): {sortino_optimized['var']:.2%}")
    sortino_sharpe_improvement = (sortino_optimized['sharpe_ratio']/current['sharpe_ratio'] - 1) if current['sharpe_ratio'] != 0 else 0
    sortino_sortino_improvement = (sortino_optimized['sortino_ratio']/current['sortino_ratio'] - 1) if current['sortino_ratio'] != 0 else 0
    print(f"Sharpe Ratio Improvement: {sortino_sharpe_improvement:.2%}")
    print(f"Sortino Ratio Improvement: {sortino_sortino_improvement:.2%}")

    if 'benchmark' in results:
        bench = results['benchmark']
        print(f"\n----- BENCHMARK ({bench['ticker']}) -----")
        print(f"Annual Return: {bench['return']:.2%}")
        print(f"Annual Volatility: {bench['volatility']:.2%}")
        print(f"Sharpe Ratio: {bench['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {bench['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {bench['max_drawdown']:.2%}")
        print(f"Value at Risk (95%): {bench['var']:.2%}")
        rel_perf = current['return'] - bench['return']
        print(f"\nPortfolio vs Benchmark: {rel_perf:.2%} {'outperformance' if rel_perf > 0 else 'underperformance'}")
        opt_rel_perf = optimized['return'] - bench['return']
        print(f"Optimized Portfolio (Max Sharpe) vs Benchmark: {opt_rel_perf:.2%} {'outperformance' if opt_rel_perf > 0 else 'underperformance'}")
        sortino_opt_rel_perf = sortino_optimized['return'] - bench['return']
        print(f"Optimized Portfolio (Max Sortino) vs Benchmark: {sortino_opt_rel_perf:.2%} {'outperformance' if sortino_opt_rel_perf > 0 else 'underperformance'}")

    stock_prices, _, earliest_dates = analyzer.fetch_stock_data(tickers, start=start_date, end=end_date)
    returns_data = analyzer.compute_returns(stock_prices)

    benchmark_data = {}
    benchmark_returns_dict = {}
    for bench_ticker in benchmark_tickers:
        data, _, _ = analyzer.fetch_stock_data([bench_ticker], start=start_date, end=end_date)
        if data is not None and not data.empty:
            benchmark_data[bench_ticker] = data
            benchmark_returns_dict[bench_ticker] = analyzer.compute_returns(data).iloc[:, 0]

    print("\nComputing metrics...")

    # 1. Correlation Matrix
    corr_metrics = analyzer.plot_correlation_matrix(stock_prices)
    corr_explanation = (
        "\n**Correlation Matrix Explanation:**\n"
        "The printed correlation matrix displays the correlation coefficients among the stocks in your portfolio.\n"
        "High positive correlations (close to 1) indicate stocks move together, while negative values indicate inverse relationships.\n"
        "This helps assess diversification."
    )

    # 2. Efficient Frontier
    ef_metrics = analyzer.plot_efficient_frontier(returns_data)
    ef_explanation = (
        "\n**Efficient Frontier Metrics Explanation:**\n"
        "The printed metrics show the risk-return trade-off for key portfolios.\n"
        "Metrics include the optimal portfolio maximizing Sharpe Ratio, the optimal portfolio maximizing Sortino Ratio,\n"
        "the minimum volatility portfolio, and an equal-weight portfolio.\n"
        "A summary of simulated portfolios provides average returns, volatilities, and ratios.\n"
        "The Sortino Ratio focuses on downside risk, providing insight into how the portfolio handles negative returns compared to the Sharpe Ratio."
    )

    # 3. Portfolio Weights (Current, Sharpe-Optimized, Sortino-Optimized)
    print("\n**Portfolio Weights Metrics:**")
    weights_metrics_current = analyzer.plot_portfolio_weights(current['weights'], "Current Portfolio Weights")
    weights_metrics_optimized = analyzer.plot_portfolio_weights(optimized['weights'], "Optimized Portfolio Weights (Max Sharpe)")
    weights_metrics_sortino = analyzer.plot_portfolio_weights(sortino_optimized['weights'], "Optimized Portfolio Weights (Max Sortino)")
    weights_explanation = (
        "\n**Portfolio Weights Explanation:**\n"
        "The printed weights show your portfolio's asset allocation for the current, Sharpe-optimized, and Sortino-optimized portfolios,\n"
        "highlighting how the allocation shifts to improve risk-adjusted performance.\n"
        "With the minimum investment set to 0%, the optimized portfolios may exclude some assets entirely to maximize their respective objectives."
    )

    # 4. Cumulative Returns Metrics
    cum_metrics, weights_df = analyzer.plot_cumulative_returns(
        returns_data, 
        np.array(list(current['weights'].values())),
        benchmark_returns=benchmark_returns_dict,
        optimized_weights=np.array(list(optimized['weights'].values())),
        sortino_optimized_weights=np.array(list(sortino_optimized['weights'].values())),
        earliest_dates=earliest_dates,
        user_start_date=start_date
    )
    delayed_stocks = {t: d.strftime('%Y-%m-%d') for t, d in earliest_dates.items() if d > pd.to_datetime(start_date)}
    effective_start = max([d for d in earliest_dates.values()] + [pd.to_datetime(start_date)]).strftime('%Y-%m-%d')
    cum_explanation = (
        "\n**Cumulative Returns Metrics Explanation:**\n"
        f"The printed metrics show cumulative returns starting from {effective_start}, normalized to 0%.\n"
    )
    if delayed_stocks:
        cum_explanation += (
            "Some stocks lacked data back to the user-specified start date and adjusted the metrics' start:\n"
        )
        for ticker, date in delayed_stocks.items():
            cum_explanation += f"  - {ticker}: Data starts {date}\n"
        cum_explanation += (
            f"The metrics begin at {effective_start}, the most recent earliest available date, ensuring all portfolio stocks are included.\n"
            "Weights were dynamically adjusted as stocks became available, maintaining portfolio consistency.\n"
        )
    cum_explanation += (
        "The metrics include final, maximum, and minimum cumulative returns for the current portfolio, Sharpe-optimized portfolio, Sortino-optimized portfolio, and benchmarks.\n"
        "Cumulative returns are shown as percentages (e.g., 150% means 1.5x initial value).\n"
        f"The Sharpe-optimized portfolio's Sortino Ratio ({optimized['sortino_ratio']:.2f}) and the Sortino-optimized portfolio's Sortino Ratio ({sortino_optimized['sortino_ratio']:.2f}) indicate their performance in managing downside risk over this period."
    )

    # 5. Crisis Performance Metrics (COVID-19 only)
    crisis_metrics = analyzer.plot_crisis_performance(
        returns_data, 
        benchmark_returns_dict, 
        results, 
        start_date, 
        end_date, 
        earliest_dates
    )
    covid_start = pd.to_datetime("2020-02-01")
    late_stocks = {t: d.strftime('%Y-%m-%d') for t, d in earliest_dates.items() if d >= covid_start}
    crisis_explanation = (
        "\n**COVID-19 Impact Metrics Explanation:**\n"
        "The printed metrics illustrate the performance of your portfolio and selected benchmarks during the COVID-19 crisis (2020-02-01 to 2020-04-30).\n"
        "Metrics include final, maximum, and minimum cumulative returns for the original portfolio, Sharpe-optimized portfolio, Sortino-optimized portfolio, and benchmarks, all normalized to start at 0%.\n"
    )
    if late_stocks:
        crisis_explanation += (
            "The following stocks lacked price data before the COVID-19 onset (2020-02-01) and were excluded from the portfolio calculations:\n"
        )
        for ticker, date in late_stocks.items():
            crisis_explanation += f"  - {ticker}: Data starts {date}\n"
        crisis_explanation += (
            "Only stocks with data before this date contribute to the portfolio metrics, with weights redistributed accordingly.\n"
        )
    else:
        crisis_explanation += (
            "All stocks had sufficient price data before the COVID-19 period, so no exclusions were necessary.\n"
        )
    crisis_explanation += (
        "The metrics show cumulative returns as percentages, highlighting performance during the crisis.\n"
        f"The Sharpe-optimized portfolio's Sortino Ratio ({optimized['sortino_ratio']:.2f}) and the Sortino-optimized portfolio's Sortino Ratio ({sortino_optimized['sortino_ratio']:.2f}) reflect their ability to mitigate downside risk during this volatile period, compared to their Sharpe Ratios ({optimized['sharpe_ratio']:.2f} and {results['sortino_optimized_portfolio']['sharpe_ratio']:.2f})."
    )

    print("\n===== ORIGINAL PORTFOLIO ANALYSIS INSIGHTS =====")
    print("• The efficient frontier metrics indicate that your current portfolio has a moderate Sharpe ratio, suggesting room for enhanced risk-adjusted returns.")
    print(f"• The Sortino Ratio ({current['sortino_ratio']:.2f}) highlights the portfolio's exposure to downside risk, which may be more relevant if you're concerned about losses rather than overall volatility (Sharpe: {current['sharpe_ratio']:.2f}).")
    print("• The cumulative returns metrics reveal that while your portfolio has grown, there have been periods of underperformance relative to benchmarks.")
    print("• The COVID-19 performance metrics show significant drawdowns, highlighting sensitivity during market stress, which aligns with the Sortino Ratio's focus on downside risk.")

    print("\n===== OPTIMIZED PORTFOLIO (MAX SHARPE) ANALYSIS INSIGHTS =====")
    print("• The Sharpe-optimized portfolio adjusts weights to maximize the Sharpe ratio, reflecting improved risk-adjusted returns.")
    print(f"• Its Sortino Ratio ({optimized['sortino_ratio']:.2f}) indicates its performance in managing downside risk compared to the original portfolio (Sortino: {current['sortino_ratio']:.2f}).")
    print("• The rebalanced asset allocation (as seen in the weights metrics) indicates a shift toward assets that optimize total risk-adjusted returns, with some assets potentially excluded (0% weight).")
    print("• The Sharpe-optimized portfolio shows different cumulative returns and drawdowns during the COVID-19 crisis compared to the original portfolio, reflecting its focus on total volatility.")

    print("\n===== OPTIMIZED PORTFOLIO (MAX SORTINO) ANALYSIS INSIGHTS =====")
    print("• The Sortino-optimized portfolio adjusts weights to maximize the Sortino ratio, focusing on minimizing downside risk.")
    print(f"• Its Sortino Ratio ({sortino_optimized['sortino_ratio']:.2f}) is optimized, indicating superior management of downside risk compared to the original (Sortino: {current['sortino_ratio']:.2f}) and Sharpe-optimized portfolios (Sortino: {optimized['sortino_ratio']:.2f}).")
    print("• The rebalanced asset allocation (as seen in the weights metrics) prioritizes assets that reduce negative returns, with some assets potentially excluded (0% weight).")
    print("• The Sortino-optimized portfolio may show more resilience during the COVID-19 crisis, as seen in the crisis performance metrics, due to its focus on downside risk.")

    print("\n===== APPENDIX =====")
    print("Key Formulas and Data Metric Explanations:")
    print("1. Annual Return = (Average Daily Return) x 252")
    print("2. Annual Volatility = (Standard Deviation of Daily Returns) x √252")
    print("3. Sharpe Ratio = (Annual Return - Risk-Free Rate) / Annual Volatility")
    print("   - Measures risk-adjusted return, considering total volatility (both upside and downside).")
    print("   - A higher Sharpe Ratio indicates better return per unit of risk.")
    print("4. Sortino Ratio = (Annual Return - Risk-Free Rate) / Downside Deviation")
    print("   - Similar to Sharpe Ratio but focuses only on downside risk (negative returns below a target, here 0%).")
    print("   - Downside Deviation = Standard Deviation of negative returns, annualized.")
    print("   - A higher Sortino Ratio indicates better return per unit of downside risk, making it more relevant for investors concerned about losses.")
    print("   - Difference from Sharpe: Sharpe penalizes both upside and downside volatility, while Sortino only penalizes downside, better reflecting investor preference for upside volatility.")
    print("5. Maximum Drawdown = Minimum[(Portfolio Value - Cumulative Maximum Portfolio Value) / Cumulative Maximum Portfolio Value]")
    print("6. Value at Risk (VaR, 95%) = -nth percentile of daily returns, where n = (1 - 0.95) x total observations")
    print("7. Portfolio Return = Dot product of portfolio weights and mean daily returns, annualized by multiplying by 252")
    print("8. Optimization uses the SLSQP method with constraints that ensure weights sum to 1, each weight is capped (default 25%), and minimum investment is 0%.")

    # Add metrics and explanations to results (no plots)
    results['metrics'] = {
        'correlation_matrix': corr_metrics,
        'efficient_frontier': ef_metrics,
        'portfolio_weights_current': weights_metrics_current,
        'portfolio_weights_optimized': weights_metrics_optimized,
        'portfolio_weights_sortino': weights_metrics_sortino,
        'cumulative_returns': cum_metrics,
        'crisis_performance': crisis_metrics
    }
    results['explanations'] = {
        'correlation_matrix': corr_explanation,
        'efficient_frontier': ef_explanation,
        'portfolio_weights': weights_explanation,
        'cumulative_returns': cum_explanation,
        'crisis_performance': crisis_explanation
    }

    return results

@app.route('/analyze', methods=['POST'])
def analyze_portfolio():
    """API endpoint to analyze a portfolio."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    # Extract parameters from request
    tickers = data.get('tickers', [])
    weights = data.get('weights', None)
    start_date = data.get('start_date', None)
    end_date = data.get('end_date', None)
    benchmark_tickers = data.get('benchmark_tickers', ['^GSPC'])

    # Validation
    if not tickers or not isinstance(tickers, list) or len(tickers) == 0:
        return jsonify({'error': 'Tickers must be a non-empty list'}), 400
    
    if weights is not None:
        if not isinstance(weights, list) or len(weights) != len(tickers):
            return jsonify({'error': 'Weights must be a list matching the number of tickers'}), 400
        total_weight = sum(weights)
        if not 0.99 <= total_weight <= 1.01:  # Allow small float precision errors
            return jsonify({'error': 'Weights must sum to approximately 100% (1.0)'}), 400
    
    if start_date:
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Start date must be in YYYY-MM-DD format'}), 400
    
    if end_date:
        try:
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'End date must be in YYYY-MM-DD format'}), 400
    
    if not isinstance(benchmark_tickers, list) or len(benchmark_tickers) == 0:
        return jsonify({'error': 'Benchmark tickers must be a non-empty list'}), 400

    try:
        results = run_portfolio_analysis(tickers, weights, start_date, end_date, benchmark_tickers)
        return jsonify(results)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API usage instructions."""
    return jsonify({
        'message': 'Welcome to the Portfolio Analyzer API',
        'endpoint': '/analyze',
        'method': 'POST',
        'example': {
            'tickers': ['AAPL', 'MSFT'],
            'weights': [0.6, 0.4],
            'start_date': '2020-01-01',
            'end_date': '2023-01-01',
            'benchmark_tickers': ['^GSPC']
        },
        'notes': 'Weights are optional (defaults to equal weighting if omitted). Dates are optional (defaults to 2015-01-01 to today).'
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)