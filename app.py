import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from matplotlib.ticker import PercentFormatter

class PortfolioAnalyzer:
    def __init__(self):
        self.today_date = datetime.now().strftime("%Y-%m-%d")  # Use current date dynamically
        self.default_start_date = (datetime.strptime(self.today_date, "%Y-%m-%d") - timedelta(days=3652)).strftime("%Y-%m-%d")  # 10 years ago
        self.data_cache = {}  # In-memory cache for stock data

    def fetch_treasury_yield(self):
        """Fetch the current 10-year U.S. Treasury yield using ^TNX."""
        try:
            treasury_data = yf.download("^TNX", period="1d", interval="1d")['Close']
            if treasury_data.empty or not isinstance(treasury_data, pd.Series):
                print("Warning: Could not fetch 10-year Treasury yield. Using fallback value of 0.04 (4%).")
                return 0.04
            latest_yield = float(treasury_data.iloc[-1]) / 100  # Explicitly convert to float and scale to decimal
            return latest_yield
        except Exception as e:
            print(f"Error fetching Treasury yield: {e}. Using fallback value of 0.04 (4%).")
            return 0.04

    def fetch_stock_data(self, stocks, start=None, end=None):
        if start is None:
            start = self.default_start_date
        if end is None:
            end = self.today_date
        cache_key = (tuple(sorted(stocks)), start, end)  # Unique key for caching
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]  # Return cached data if available
        error_tickers = {}
        earliest_dates = {}
        try:
            stock_data = yf.download(list(stocks), start=start, end=end, auto_adjust=True)['Close']
            if stock_data.empty:
                print("Warning: No data available for the specified date range.")
                return None, error_tickers, earliest_dates
            for ticker in stocks:
                if ticker not in stock_data.columns or stock_data[ticker].isna().all():
                    error_tickers[ticker] = "Data not available"
                else:
                    first_valid = stock_data[ticker].first_valid_index()
                    earliest_dates[ticker] = first_valid
            self.data_cache[cache_key] = (stock_data, error_tickers, earliest_dates)  # Cache the result
            return stock_data, error_tickers, earliest_dates
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, error_tickers, earliest_dates

    def compute_returns(self, prices):
        return prices.pct_change().dropna()

    def compute_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def compute_sortino_ratio(self, returns, risk_free_rate):
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        annualized_return = returns.mean() * 252
        return (annualized_return - risk_free_rate) / downside_std if downside_std != 0 else 0

    def compute_beta(self, portfolio_returns, benchmark_returns):
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance if benchmark_variance != 0 else 0

    def portfolio_performance(self, weights, returns, risk_free_rate):
        portfolio_returns = returns.dot(weights)
        portfolio_return = portfolio_returns.mean() * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def compute_var(self, returns, confidence_level=0.90):
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return sorted_returns[index] if len(sorted_returns) > 0 else 0

    def optimize_portfolio(self, returns, risk_free_rate, objective='sharpe', min_allocation=0.0, max_allocation=1.0):
        num_assets = returns.shape[1]
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((min_allocation, max_allocation) for _ in range(num_assets))

        def negative_sharpe(weights):
            r, v, s = self.portfolio_performance(weights, returns, risk_free_rate)
            return -s

        def negative_sortino(weights):
            portfolio_returns = returns.dot(weights)
            return -self.compute_sortino_ratio(portfolio_returns, risk_free_rate)

        def max_drawdown(weights):
            portfolio_returns = returns.dot(weights)
            return self.compute_max_drawdown(portfolio_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        def negative_var(weights):
            portfolio_returns = returns.dot(weights)
            return -self.compute_var(portfolio_returns)

        objective_functions = {
            'sharpe': negative_sharpe,
            'sortino': negative_sortino,
            'max_drawdown': max_drawdown,
            'volatility': portfolio_volatility,
            'value_at_risk': negative_var
        }
        obj_fun = objective_functions.get(objective, negative_sharpe)

        try:
            result = minimize(obj_fun, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                print(f"Warning: Optimization failed for {objective}: {result.message}")
                print("Suggestion: Try a different objective (e.g., 'sharpe') or check if your stocks have sufficient data.")
                return initial_weights
            weights = result.x
            weights[weights < 0.001] = 0
            weights /= weights.sum() if weights.sum() != 0 else 1
            return weights
        except Exception as e:
            print(f"Error in optimization: {e}")
            return initial_weights

    def apply_weight_strategy(self, returns, strategy='equal'):
        num_assets = returns.shape[1]
        if strategy == 'equal':
            return np.ones(num_assets) / num_assets
        elif strategy == 'risk_parity':
            volatilities = returns.std() * np.sqrt(252)
            inv_vol = 1 / volatilities
            return inv_vol / inv_vol.sum() if inv_vol.sum() != 0 else np.ones(num_assets) / num_assets
        elif strategy == 'inverse_volatility':
            volatilities = returns.std() * np.sqrt(252)
            inv_vol = 1 / volatilities
            return inv_vol / inv_vol.sum() if inv_vol.sum() != 0 else np.ones(num_assets) / num_assets
        else:
            return np.ones(num_assets) / num_assets

    def print_correlation_matrix(self, prices):
        corr_matrix = prices.corr()
        print("\n===== Correlation Matrix of Portfolio Stocks =====")
        print("Explanation: This table shows how your stocks move together. Values close to 1 mean strong positive correlation, while negative values mean they move oppositely.")
        print(corr_matrix.to_string(float_format="%.2f"))

    def print_efficient_frontier(self, returns, risk_free_rate, n_portfolios=1000):
        """Calculate and print metrics for the efficient frontier and optimized portfolios."""
        np.random.seed(42)  # For reproducibility
        n_assets = returns.shape[1]
        all_weights = np.zeros((n_portfolios, n_assets))
        all_returns = np.zeros(n_portfolios)
        all_volatilities = np.zeros(n_portfolios)
        all_sharpe_ratios = np.zeros(n_portfolios)
        
        # Generate random portfolios
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= weights.sum()
            all_weights[i, :] = weights
            port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
            all_returns[i] = port_return
            all_volatilities[i] = port_vol
            all_sharpe_ratios[i] = port_sharpe

        # Optimize for each specified strategy
        strategies = {
            "Max Sharpe": self.optimize_portfolio(returns, risk_free_rate, "sharpe"),
            "Max Sortino": self.optimize_portfolio(returns, risk_free_rate, "sortino"),
            "Min Max Drawdown": self.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
            "Min Volatility": self.optimize_portfolio(returns, risk_free_rate, "volatility"),
            "Min Value at Risk": self.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
        }
        
        # Compute performance metrics for each strategy
        strategy_metrics = {}
        for name, weights in strategies.items():
            port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
            strategy_metrics[name] = {
                "return": port_return,
                "volatility": port_vol,
                "sharpe": port_sharpe
            }

        # Print the metrics
        print("\n===== Efficient Frontier Metrics =====")
        print("Explanation: These metrics illustrate the risk-return trade-off across thousands of simulated portfolio compositions using your stocks.")
        print("Key optimized portfolios are detailed below:")
        print("- Max Sharpe: Highest risk-adjusted return.")
        print("- Max Sortino: Optimized for downside risk-adjusted return.")
        print("- Min Max Drawdown: Lowest peak-to-trough loss.")
        print("- Min Volatility: Lowest overall risk.")
        print("- Min Value at Risk: Minimized potential daily loss at 90% confidence.")
        print("\nSimulated Portfolios Summary:")
        print(f"Average Annualized Return: {np.mean(all_returns):.2%}")
        print(f"Average Annualized Volatility: {np.mean(all_volatilities):.2%}")
        print(f"Average Sharpe Ratio: {np.mean(all_sharpe_ratios):.2f}")
        print(f"Best Sharpe Ratio in Simulation: {np.max(all_sharpe_ratios):.2f}")
        print("\nOptimized Portfolios Metrics:")
        for name, metrics in strategy_metrics.items():
            print(f"\n{name}:")
            print(f"  Annualized Return: {metrics['return']:.2%}")
            print(f"  Annualized Volatility: {metrics['volatility']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")

    def print_cumulative_returns(self, returns, weights_dict, benchmark_returns, earliest_dates, title="Cumulative Returns"):
        start_date = max(earliest_dates.values()) + timedelta(days=180)
        adjusted_returns = returns.loc[start_date:]
        print(f"\n===== {title} (From {start_date.strftime('%Y-%m-%d')}) =====")
        for label, weights in weights_dict.items():
            portfolio_returns = adjusted_returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod() - 1
            print(f"\n{label}:")
            print(f"  Final Cumulative Return: {cumulative.iloc[-1]:.2%}")
            print(f"  Average Daily Return: {portfolio_returns.mean():.4%}")
            print(f"  Standard Deviation of Daily Returns: {portfolio_returns.std():.4%}")
        for bench_ticker, bench_ret in benchmark_returns.items():
            bench_cum = (1 + bench_ret.loc[start_date:]).cumprod() - 1
            print(f"\n{bench_ticker}:")
            print(f"  Final Cumulative Return: {bench_cum.iloc[-1]:.2%}")
            print(f"  Average Daily Return: {bench_ret.loc[start_date:].mean():.4%}")
            print(f"  Standard Deviation of Daily Returns: {bench_ret.loc[start_date:].std():.4%}")

    def print_crisis_performance(self, returns, weights_dict, benchmark_returns):
        crisis_start = pd.to_datetime("2020-02-01")
        crisis_end = pd.to_datetime("2020-04-30")
        # Find nearest trading days within the returns index
        available_start = returns.index[returns.index >= crisis_start].min()
        available_end = returns.index[returns.index <= crisis_end].max()
        
        if pd.isna(available_start) or pd.isna(available_end) or available_start > available_end:
            print("Warning: Crisis period data not available within the analysis range.")
            return
        
        crisis_returns = returns.loc[available_start:available_end]
        print(f"\n===== COVID-19 Crisis Performance ({available_start.strftime('%Y-%m-%d')} to {available_end.strftime('%Y-%m-%d')}) =====")
        print("Explanation: These metrics show your portfolio’s performance during the COVID-19 crisis.")
        print("- Original Portfolio: Performance based on your initial weights.")
        print("- Optimized Portfolio: Performance with your chosen optimization and weighting.")
        print(f"- Benchmarks ({', '.join(benchmark_returns.keys())}): Market indices for comparison.")
        for label, weights in weights_dict.items():
            portfolio_returns = crisis_returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod() - 1
            print(f"\n{label}:")
            print(f"  Final Cumulative Return: {cumulative.iloc[-1]:.2%}")
            print(f"  Average Daily Return: {portfolio_returns.mean():.4%}")
            print(f"  Standard Deviation of Daily Returns: {portfolio_returns.std():.4%}")
        for bench_ticker, bench_ret in benchmark_returns.items():
            bench_crisis_ret = bench_ret.loc[available_start:available_end]
            bench_cum = (1 + bench_crisis_ret).cumprod() - 1
            print(f"\n{bench_ticker}:")
            print(f"  Final Cumulative Return: {bench_cum.iloc[-1]:.2%}")
            print(f"  Average Daily Return: {bench_crisis_ret.mean():.4%}")
            print(f"  Standard Deviation of Daily Returns: {bench_crisis_ret.std():.4%}")

    def print_historical_strategies(self, tickers, weights_dict, risk_free_rate, hist_returns=None):
        start_date = "2015-03-24"
        end_date = self.today_date
        if hist_returns is None:
            data, _, _ = self.fetch_stock_data(tickers, start_date, end_date)
            if data is None or data.empty:
                print("Warning: Could not fetch historical data for your portfolio.")
                return
            hist_returns = self.compute_returns(data)
        
        # Define optimization strategies, including original portfolio
        strategies = {
            "Original Portfolio": np.array(list(weights_dict.values())),
            "Max Sharpe": self.optimize_portfolio(hist_returns, risk_free_rate, "sharpe"),
            "Max Sortino": self.optimize_portfolio(hist_returns, risk_free_rate, "sortino"),
            "Min Max Drawdown": self.optimize_portfolio(hist_returns, risk_free_rate, "max_drawdown"),
            "Min Volatility": self.optimize_portfolio(hist_returns, risk_free_rate, "volatility"),
            "Min Value at Risk": self.optimize_portfolio(hist_returns, risk_free_rate, "value_at_risk")
        }
        
        # Compute metrics for each strategy
        annual_metrics = {}
        for name, weights in strategies.items():
            portfolio_returns = hist_returns.dot(weights)
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            max_drawdown = self.compute_max_drawdown(portfolio_returns)
            annual_metrics[name] = {
                'Annualized Returns': annual_return,
                'Annualized Volatility': annual_volatility,
                'Max Drawdown': max_drawdown
            }

        # Print the metrics
        print(f"\n===== Historical Performance of Portfolio Strategies (Past Decade: {start_date} to {end_date}) =====")
        print(f"Explanation: These metrics show how your portfolio ({', '.join(tickers)}) would have performed under different strategies:")
        print("- Original Portfolio: Your initial weights, unoptimized.")
        print("- Max Sharpe: Maximizes return per unit of risk.")
        print("- Max Sortino: Optimizes upside return vs. downside risk.")
        print("- Min Max Drawdown: Minimizes the worst loss.")
        print("- Min Volatility: Reduces overall risk.")
        print("- Min Value at Risk: Limits potential daily losses at 90% confidence.")
        print("\nMetrics Explained:")
        print("- Annualized Returns: Average yearly return; higher is better.")
        print("- Annualized Volatility: Yearly risk; lower is more stable.")
        print("- Max Drawdown: Largest loss; less negative is better.")
        print("Compare these to see how optimization improves your original portfolio.")
        for name, metrics in annual_metrics.items():
            print(f"\n{name}:")
            print(f"  Annualized Returns: {metrics['Annualized Returns']:.2%}")
            print(f"  Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
            print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")

    def print_weight_allocation_strategies(self, tickers, weights_dict):
        start_date = "2015-03-24"  # 10 years before 2025-03-23
        end_date = self.today_date  # 2025-03-23
        
        # Fetch portfolio data
        data, _, _ = self.fetch_stock_data(tickers, start_date, end_date)
        if data is None or data.empty:
            print("Warning: Could not fetch historical data for your portfolio.")
            return
        hist_returns = self.compute_returns(data)
        
        # Define weight allocation strategies, including "None" (original weights)
        strategies = {
            "None (Original)": np.array(list(weights_dict.values())),
            "Equal Weight": self.apply_weight_strategy(hist_returns, "equal"),
            "Risk Parity": self.apply_weight_strategy(hist_returns, "risk_parity"),
            "Inverse Volatility": self.apply_weight_strategy(hist_returns, "inverse_volatility")
        }
        
        # Compute metrics for each strategy
        annual_metrics = {}
        for name, weights in strategies.items():
            portfolio_returns = hist_returns.dot(weights)
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            max_drawdown = self.compute_max_drawdown(portfolio_returns)
            annual_metrics[name] = {
                'Annualized Returns': annual_return,
                'Annualized Volatility': annual_volatility,
                'Max Drawdown': max_drawdown
            }

        # Print the metrics
        print(f"\n===== Historical Performance of Weight Allocation Strategies (Past Decade: {start_date} to {end_date}) =====")
        print(f"Explanation: These metrics show how your portfolio ({', '.join(tickers)}) would have performed under different weight allocation strategies:")
        print("- None (Original): Your initial weights, unadjusted.")
        print("- Equal Weight: Distributes capital evenly across your stocks.")
        print("- Risk Parity: Balances risk contribution from each stock.")
        print("- Inverse Volatility: Weights stocks inversely to their volatility.")
        print("\nMetrics Explained:")
        print("- Annualized Returns: Average yearly return; higher is better.")
        print("- Annualized Volatility: Yearly risk; lower is more stable.")
        print("- Max Drawdown: Largest loss; less negative is better.")
        print("Use this to decide if adjusting weights improves your original portfolio.")
        for name, metrics in annual_metrics.items():
            print(f"\n{name}:")
            print(f"  Annualized Returns: {metrics['Annualized Returns']:.2%}")
            print(f"  Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
            print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")
    
    def print_comparison_bars(self, original_metrics, optimized_metrics, benchmark_metrics):
        metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "maximum_drawdown", "value_at_risk"]
        labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Value at Risk (90%)"]
        print("\n===== Portfolio Comparison Metrics =====")
        for metric, label in zip(metrics, labels):
            print(f"\n{label}:")
            print(f"  Original: {original_metrics[metric]:.2f if 'sharpe' in metric else original_metrics[metric]:.2%}")
            print(f"  Optimized: {optimized_metrics[metric]:.2f if 'sharpe' in metric else optimized_metrics[metric]:.2%}")
            if benchmark_metrics:
                for bench, bm in benchmark_metrics.items():
                    print(f"  {bench}: {bm[metric]:.2f if 'sharpe' in metric else bm[metric]:.2%}")

def run_portfolio_analysis():
    analyzer = PortfolioAnalyzer()
    tickers = []
    weights_dict = {}

    # Step 1: Input stocks and optional weights
    print("Let’s build your portfolio. Enter your stocks, ETFs, or indices (e.g., AAPL, SPY, ^GSPC) one by one.")
    while True:
        ticker = input("Enter a stock ticker (or type 'DONE' to finish): ").strip().upper()
        if ticker == "DONE" and tickers:
            break
        if ticker in tickers:
            print("Warning: This ticker has already been added.")
            continue
        try:
            test_data = yf.download(ticker, period="1mo")
            if test_data.empty:
                print(f"Warning: No data found for {ticker}.")
                continue
        except Exception as e:
            print(f"Warning: Error retrieving data for {ticker}: {e}")
            continue
        tickers.append(ticker)
        pct = input(f"Enter percentage for {ticker} (or press Enter for equal weighting later): ").strip()
        try:
            weights_dict[ticker] = float(pct) / 100 if pct else None
        except ValueError:
            print("Warning: Invalid percentage entered. Treating as equal weighting.")
            weights_dict[ticker] = None

    # Assign weights
    if all(w is None for w in weights_dict.values()):
        weights = np.ones(len(tickers)) / len(tickers)
    else:
        total = sum(w for w in weights_dict.values() if w is not None)
        if total > 1:
            print("Warning: Total percentage exceeds 100%. Normalizing weights.")
            weights = np.array([weights_dict[t] for t in tickers])
            weights /= weights.sum()
        elif total == 0:
            weights = np.ones(len(tickers)) / len(tickers)
        else:
            remaining = 1 - total
            num_none = sum(1 for w in weights_dict.values() if w is None)
            weights = np.array([weights_dict[t] if weights_dict[t] is not None else remaining / num_none for t in tickers])
    weights_dict = dict(zip(tickers, weights))

    # Step 2: Start date
    start_date = input(f"Enter the start date for analysis (YYYY-MM-DD, or press Enter for {analyzer.default_start_date}): ") or analyzer.default_start_date
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        print("Warning: Invalid date format. Using default start date.")
        start_date = analyzer.default_start_date
    end_date = analyzer.today_date

    # Step 3: Benchmarks
    benchmark_options = {"1": "^GSPC", "2": "^IXIC", "3": "^DJI", "4": "^RUT", "5": "MSCI"}
    print("\nChoose benchmarks to compare your portfolio against:")
    print("A benchmark is a standard (like a market index) used to evaluate your portfolio’s performance.")
    print("Options: 1. ^GSPC (S&P 500), 2. ^IXIC (NASDAQ), 3. ^DJI (Dow Jones), 4. ^RUT (Russell 2000), 5. MSCI (World Index)")
    bench_choice = input("Enter numbers (e.g., 1,3, or press Enter for ^GSPC): ").strip()
    benchmarks = [benchmark_options.get(c.strip(), "^GSPC") for c in bench_choice.split(",") if c.strip()] or ["^GSPC"]

    # Fetch data and analyze
    # Step 4: Risk Tolerance and Risk-Free Rate
    print("\nWhat’s your risk tolerance? (This helps tailor your portfolio)")
    print("1. Low (prefer stability), 2. Medium (balanced), 3. High (maximize returns)")
    risk_choice = input("Enter your choice (1-3, or press Enter for Medium): ").strip() or "2"
    risk_map = {"1": "low", "2": "medium", "3": "high"}
    risk_tolerance = risk_map.get(risk_choice, "medium")
    
    treasury_yield = analyzer.fetch_treasury_yield()
    print(f"Current 10-year U.S. Treasury yield is {treasury_yield:.4%}.")
    rf_input = input(f"Enter the risk-free rate (e.g., 0.0425 for 4.25%, or press Enter to use {treasury_yield:.4%}): ").strip()
    try:
        risk_free_rate = float(rf_input) if rf_input else treasury_yield
    except ValueError:
        print(f"Warning: Invalid input. Using current 10-year Treasury yield of {treasury_yield:.4%}.")
        risk_free_rate = treasury_yield

    # Fetch data and analyze
    stock_prices, _, earliest_dates = analyzer.fetch_stock_data(tickers, start_date, end_date)
    if stock_prices is None or stock_prices.empty:
        print("Error: No valid stock data available.")
        return
    returns = analyzer.compute_returns(stock_prices)
    if returns.empty:
        print("Error: No valid returns data.")
        return
    portfolio_returns = returns.dot(list(weights_dict.values()))

    benchmark_returns = {}
    benchmark_metrics = {}
    for bench in benchmarks:
        bench_data, _, _ = analyzer.fetch_stock_data([bench], start_date, end_date)
        if bench_data is not None and not bench_data.empty:
            bench_returns = analyzer.compute_returns(bench_data)[bench]
            benchmark_returns[bench] = bench_returns
            benchmark_metrics[bench] = {
                "annual_return": bench_returns.mean() * 252,
                "annual_volatility": bench_returns.std() * np.sqrt(252),
                "sharpe_ratio": analyzer.portfolio_performance(np.array([1.0]), pd.DataFrame(bench_returns), risk_free_rate)[2],
                "maximum_drawdown": analyzer.compute_max_drawdown(bench_returns),
                "value_at_risk": analyzer.compute_var(bench_returns, 0.90)
            }

    original_metrics = {
        "annual_return": portfolio_returns.mean() * 252,
        "annual_volatility": portfolio_returns.std() * np.sqrt(252),
        "sharpe_ratio": analyzer.portfolio_performance(np.array(list(weights_dict.values())), returns, risk_free_rate)[2],
        "maximum_drawdown": analyzer.compute_max_drawdown(portfolio_returns),
        "value_at_risk": analyzer.compute_var(portfolio_returns, 0.90)
    }

    # Portfolio Analysis
    print("\n===== PORTFOLIO ANALYSIS =====")
    print(f"Analysis period: {start_date} to {end_date}")
    print("\n----- Original Portfolio Metrics -----")
    print(f"Annual Return: {original_metrics['annual_return']:.2%}")
    print("Explanation: This is the average yearly return your portfolio earned.")
    print("Hint: Aim for this to beat your benchmark for better growth!")
    print(f"Annual Volatility: {original_metrics['annual_volatility']:.2%}")
    print("Explanation: This measures the risk or variability of your portfolio’s returns.")
    print(f"Sharpe Ratio: {original_metrics['sharpe_ratio']:.2f}")
    print("Explanation: This shows return per unit of risk; higher is better.")
    print(f"Maximum Drawdown: {original_metrics['maximum_drawdown']:.2%}")
    print("Explanation: This is the largest peak-to-trough loss your portfolio experienced.")
    print(f"Value at Risk (90%): {original_metrics['value_at_risk']:.2%}")
    print("Explanation: This estimates the potential loss in a worst-case scenario at 90% confidence.")

    analyzer.print_correlation_matrix(stock_prices)

    print("\n----- Benchmark Metrics -----")
    for bench, metrics in benchmark_metrics.items():
        print(f"\nBenchmark: {bench}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['maximum_drawdown']:.2%}")
        print(f"Value at Risk (90%): {metrics['value_at_risk']:.2%}")

    print("\n----- ISSUES WITH YOUR ORIGINAL PORTFOLIO: WHAT YOU NEED TO KNOW -----")
    print("Now that you understand the metrics, let’s look at potential problems with your portfolio.")
    print("For each issue, I’ll explain what it means, why it’s a concern, and what you can do about it.\n")

    issues = []

    # Issue 1: Annual Return Compared to Benchmark
    for bench_ticker, bench_metrics in benchmark_metrics.items():
        bench_return = bench_metrics['annual_return']
        if original_metrics['annual_return'] < bench_return:
            return_diff = ((bench_return - original_metrics['annual_return']) * 100)  # Absolute difference in percentage points
            issues.append({
                'metric': f'Annual Return vs {bench_ticker}',
                'description': (
                    f"Your portfolio’s Annual Return ({original_metrics['annual_return']:.2%}) is lower than the {bench_ticker} benchmark ({bench_return:.2%}) by {return_diff:.1f} percentage points.\n"
                    "   - What This Means: The benchmark represents the average performance of the market or sector. If your portfolio grows less than the benchmark, you’re missing out on potential gains.\n"
                    "   - Why It’s a Concern: Imagine you’re in a race, and the benchmark is the average runner. If you’re running slower (lower return), you’re not keeping up with what most investors are earning. For example, if you invested $10,000, your portfolio would grow to ${10000 * (1 + original_metrics['annual_return']):.0f} in a year, while the benchmark would grow to ${10000 * (1 + bench_return):.0f}. Over time, this gap can mean a lot less money for your goals, like saving for retirement or a big purchase.\n"
                    "   - What You Can Do: Consider a strategy that focuses on higher returns, like Max Sharpe Ratio, which aims to get you the best growth for the risk you take. You might also look at adding stocks that have historically performed better than the market, such as growth stocks in sectors like technology."
                )
            })
        else:
            print(f"✓ Your Annual Return is higher than the {bench_ticker} benchmark—great job! You’re outperforming this market index.")

    # Issue 2: High Volatility
    if original_metrics['annual_volatility'] > 0.20:
        issues.append({
            'metric': 'Annual Volatility',
            'description': (
                f"Your portfolio’s Annual Volatility ({original_metrics['annual_volatility']:.2%}) is high, above the typical range of 10-20% for a diversified portfolio.\n"
                "   - What This Means: Volatility measures how much your portfolio’s value goes up and down over a year. Think of it like a rollercoaster: high volatility means a wild ride with big ups and downs, which can be stressful and risky.\n"
                "   - Why It’s a Concern: High volatility increases the chance of losing money, especially during market downturns. For example, if the market drops 10%, a high-volatility portfolio might drop 15% or more. If you had $10,000 invested, a 15% drop means a $1,500 loss in value, and it would take a 17.6% gain just to break even. This kind of swing can make it harder to achieve steady growth toward your financial goals.\n"
                "   - What You Can Do: To smooth out the ride, consider a strategy like Min Volatility, which focuses on lowering volatility, potentially bringing it closer to 10-15%. Adding more stable investments, like bonds or low-volatility stocks (e.g., utilities), can also help."
            )
        })
    elif original_metrics['annual_volatility'] > 0.10:
        print(f"⚠ Your Annual Volatility ({original_metrics['annual_volatility']:.2%}) is moderate. It’s within a typical range, but you might still reduce it for more stability if you prefer a smoother investment experience.")
    else:
        print("✓ Your Annual Volatility is low—nice work! Your portfolio is relatively stable.")

    # Issue 3: Low Sharpe Ratio
    if original_metrics['sharpe_ratio'] < 1:
        issues.append({
            'metric': 'Sharpe Ratio',
            'description': (
                f"Your Sharpe Ratio ({original_metrics['sharpe_ratio']:.2f}) is below 1, which is considered poor.\n"
                "   - What This Means: The Sharpe Ratio shows how much return you’re getting for the risk you’re taking. A low Sharpe Ratio means you’re not getting enough reward (returns) for the ups and downs (volatility) in your portfolio.\n"
                "   - Why It’s a Concern: Think of it like buying a car: if you’re paying a high price (taking on a lot of risk) but getting a slow car (low returns), it’s not a good deal. For example, if you’re earning a 5% return but your portfolio swings by 20% (high volatility), you’re taking a lot of risk for a small reward. A higher Sharpe Ratio, like 1.5, would mean better returns for the same level of risk—more bang for your buck.\n"
                "   - What You Can Do: A strategy like Max Sharpe Ratio can help by optimizing your portfolio to get the best return for the risk, potentially pushing your Sharpe Ratio above 1. You might also consider reducing risk (volatility) while maintaining or increasing returns, perhaps by diversifying your investments across different sectors or adding less volatile assets."
            )
        })
    elif original_metrics['sharpe_ratio'] < 2:
        print(f"⚠ Your Sharpe Ratio ({original_metrics['sharpe_ratio']:.2f}) is decent but below 2, which is considered great. You might improve it for better risk-adjusted returns.")
    else:
        print("✓ Your Sharpe Ratio is excellent—well done! You’re getting strong returns for the risk you’re taking.")

    # Issue 4: High Maximum Drawdown
    if original_metrics['maximum_drawdown'] < -0.20:
        recovery_gain = (1 / (1 + original_metrics['maximum_drawdown']) - 1) * 100
        issues.append({
            'metric': 'Maximum Drawdown',
            'description': (
                f"Your Maximum Drawdown ({original_metrics['maximum_drawdown']:.2%}) is concerning, indicating significant historical losses.\n"
                "   - What This Means: Maximum Drawdown shows the largest drop your portfolio experienced from its peak to its lowest point. A drop of {abs(original_metrics['maximum_drawdown']):.2%} means at its worst, your portfolio lost {abs(original_metrics['maximum_drawdown']):.2%} of its value from a previous high.\n"
                "   - Why It’s a Concern: A big drawdown can be hard to recover from. For example, with a {abs(original_metrics['maximum_drawdown']):.2%} drop, if you had $10,000, you’d lose ${10000 * abs(original_metrics['maximum_drawdown']):.0f}, leaving you with ${10000 * (1 + original_metrics['maximum_drawdown']):.0f}. To recover, you’d need a {recovery_gain:.1f}% gain, which could take a long time. This could delay your financial goals, like buying a house or retiring.\n"
                "   - What You Can Do: A strategy like Min Maximum Drawdown can help by focusing on reducing these big losses, potentially keeping drawdowns under 20%. You might also diversify your portfolio more or include assets like bonds or gold that are less likely to drop sharply during market crashes."
            )
        })
    else:
        print("✓ Your Maximum Drawdown is within a reasonable range—good job! Your portfolio hasn’t experienced extreme losses.")

    # Issue 5: High Value at Risk (VaR)
    if original_metrics['value_at_risk'] < -0.05:
        issues.append({
            'metric': 'Value at Risk (VaR, 90%)',
            'description': (
                f"Your Value at Risk ({original_metrics['value_at_risk']:.2%}) is high, indicating a significant potential for daily losses.\n"
                "   - What This Means: VaR estimates the maximum loss you might expect on a typical day, with 90% confidence. A VaR of {abs(original_metrics['value_at_risk']):.2%} means there’s a 10% chance you could lose {abs(original_metrics['value_at_risk']):.2%} or more of your portfolio’s value in a single day.\n"
                "   - Why It’s a Concern: High daily losses can be unsettling and risky. For example, if you have $10,000 invested, a VaR of {abs(original_metrics['value_at_risk']):.2%} means there’s a 10% chance you could lose ${10000 * abs(original_metrics['value_at_risk']):.0f} or more in one day. It’s like knowing there’s a small chance of a big storm hitting your house—you’d want to be prepared.\n"
                "   - What You Can Do: To lower your VaR, consider a strategy like Min Value at Risk, which focuses on reducing downside risk, potentially bringing your VaR closer to 3-4%. You might also reduce your exposure to very volatile stocks and add more stable investments, like bonds or dividend-paying stocks."
            )
        })
    else:
        print("✓ Your Value at Risk is within a reasonable range—nice work! Your portfolio’s potential daily losses are manageable.")

    # Display All Issues
    if issues:
        print("Here are the key issues identified in your portfolio:")
        for i, issue in enumerate(issues, 1):
            print(f"\nIssue {i}: {issue['metric']}")
            print(issue['description'])
    else:
        print("✓ No major issues found with your portfolio—great job! It’s well-balanced based on the metrics analyzed.")

    print("\n===== CUMULATIVE RETURNS OF STRATEGIES =====")
    strategies = {
        "Original Portfolio": np.array(list(weights_dict.values())),
        "Max Sharpe": analyzer.optimize_portfolio(returns, risk_free_rate, "sharpe"),
        "Max Sortino": analyzer.optimize_portfolio(returns, risk_free_rate, "sortino"),
        "Min Max Drawdown": analyzer.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
        "Min Volatility": analyzer.optimize_portfolio(returns, risk_free_rate, "volatility"),
        "Min Value at Risk": analyzer.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
    }
    analyzer.print_cumulative_returns(returns, strategies, benchmark_returns, earliest_dates)

    print("\n===== HIGH-LEVEL INSIGHTS ON EACH STRATEGY =====")
    print("• Max Sharpe: Optimizes risk-adjusted return.")
    print("• Min Volatility: Prioritizes low risk.")
    print("• Equal Weight: Simple diversification.")

    # Moved Historical Performance section here
    print("\n===== HISTORICAL PERFORMANCE OF PORTFOLIO STRATEGIES (PAST DECADE) =====")
    hist_start_date = "2015-03-24"
    hist_data, _, _ = analyzer.fetch_stock_data(tickers, hist_start_date, end_date)
    if hist_data is not None and not hist_data.empty:
        hist_returns = analyzer.compute_returns(hist_data)
        analyzer.print_historical_strategies(tickers, weights_dict, risk_free_rate, hist_returns)
    else:
        print("Warning: Historical data unavailable.")

    print("\n===== OPTIMIZATION OPTIONS =====")
    print("Optimize by: 1. Sharpe Ratio, 2. Sortino Ratio, 3. Maximum Drawdown, 4. Volatility, 5. Value at Risk")
    metric_choice = input("Enter your choice (1-5): ").strip()
    metric_map = {"1": "sharpe", "2": "sortino", "3": "max_drawdown", "4": "volatility", "5": "value_at_risk"}
    opt_metric = metric_map.get(metric_choice, "sharpe")

    min_alloc = input("Enter minimum allocation per stock (e.g., 0.05 for 5%, or press Enter for 0%): ").strip()
    max_alloc = input("Enter maximum allocation per stock (e.g., 0.30 for 30%, or press Enter for 100%): ").strip()
    try:
        min_allocation = float(min_alloc) if min_alloc else 0.0
        max_allocation = float(max_alloc) if max_alloc else 1.0
    except ValueError:
        print("Warning: Invalid allocation input. Using defaults (0% min, 100% max).")
        min_allocation, max_allocation = 0.0, 1.0
    opt_weights = analyzer.optimize_portfolio(returns, risk_free_rate, opt_metric, min_allocation, max_allocation)

    # Weight Allocation Strategies and Choice
    print("\n===== HISTORICAL WEIGHT ALLOCATION STRATEGIES ON NASDAQ =====")
    analyzer.print_weight_allocation_strategies(tickers, weights_dict)

    print("\nWeight allocation: 1. Risk Parity, 2. Equal Weighting, 3. Inverse-Volatility, 4. None")
    weight_choice = input("Enter your choice (1-4): ").strip()
    weight_map = {"1": "risk_parity", "2": "equal", "3": "inverse_volatility", "4": None}
    weight_strategy = weight_map.get(weight_choice, None)

    # Apply optimization and strategy
    opt_weights = analyzer.optimize_portfolio(returns, risk_free_rate, opt_metric)
    if weight_strategy:
        strategy_weights = analyzer.apply_weight_strategy(returns, weight_strategy)
        optimized_weights = (opt_weights + strategy_weights) / 2
    else:
        optimized_weights = opt_weights

    optimized_metrics = {
        "annual_return": analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[0],
        "annual_volatility": analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[1],
        "sharpe_ratio": analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[2],
        "maximum_drawdown": analyzer.compute_max_drawdown(returns.dot(optimized_weights)),
        "value_at_risk": analyzer.compute_var(returns.dot(optimized_weights), 0.90)
    }

    print("\nIssues Addressed by Optimization:")
    if not issues:
        print("No major issues were identified in your original portfolio.")
    else:
        for issue in issues:
            metric = issue['metric']  # Keep original case
            fixed = False
            if "Annual Return vs" in metric:
                bench_ticker = metric.split("vs ")[1].strip()  # Preserve case of ticker
                bench_return = benchmark_metrics[bench_ticker]['annual_return']
                fixed = optimized_metrics['annual_return'] >= bench_return
            elif "Annual Volatility" in metric:
                fixed = optimized_metrics['annual_volatility'] <= 0.20
            elif "Sharpe Ratio" in metric:
                fixed = optimized_metrics['sharpe_ratio'] >= 1
            elif "Maximum Drawdown" in metric:
                fixed = optimized_metrics['maximum_drawdown'] >= -0.20
            elif "Value at Risk" in metric:
                fixed = optimized_metrics['value_at_risk'] >= -0.05
            status = "✓ Fixed" if fixed else "✗ Not Fully Resolved"
            print(f"- {issue['metric']}: {status}")

    print("\n===== COMBINED OPTIMIZED PERFORMANCE =====")
    combined_strategies = {
        "Original Portfolio": np.array(list(weights_dict.values())),
        "Optimized Portfolio": optimized_weights
    }
    analyzer.print_cumulative_returns(returns, combined_strategies, benchmark_returns, earliest_dates, "Optimized vs Original Cumulative Returns")

    # Final comparisons
    print("\n===== OPTIMIZED PORTFOLIO COMPARISONS =====")
    analyzer.print_comparison_bars(original_metrics, optimized_metrics, benchmark_metrics)
    print("Insights:")
    print(f"- Risk: Optimized volatility ({optimized_metrics['annual_volatility']:.2%}) vs Original ({original_metrics['annual_volatility']:.2%}).")
    print(f"- Returns: Optimized return ({optimized_metrics['annual_return']:.2%}) vs Original ({original_metrics['annual_return']:.2%}).")
    start_date_obj = max(earliest_dates.values()) + timedelta(days=180)
    initial_investment = 10000
    opt_cum = (1 + returns.dot(optimized_weights)).cumprod()[-1] * initial_investment
    orig_cum = (1 + returns.dot(list(weights_dict.values()))).cumprod()[-1] * initial_investment
    print(f"- Money Made (from {start_date_obj.strftime('%Y-%m-%d')} with $10,000): Optimized: ${opt_cum:.2f}, Original: ${orig_cum:.2f}")

    if start_date_obj < pd.to_datetime("2020-02-01"):
        print("\n===== COVID-19 PERFORMANCE =====")
        analyzer.print_crisis_performance(returns, combined_strategies, benchmark_returns)

    # Resources and Appendix
    print("\n===== RESOURCES AND APPENDIX =====")
    print("Formulas:")
    print("1. Annual Return = (Average Daily Return) × 252")
    print("2. Annual Volatility = (Standard Deviation of Daily Returns) × √252")
    print("3. Sharpe Ratio = (Annual Return - Risk-Free Rate) / Annual Volatility")
    print("4. Maximum Drawdown = Minimum[(Cumulative Value - Peak Value) / Peak Value]")
    print("5. Value at Risk (90%) = 10th percentile of daily returns")
    print("6. Sortino Ratio = (Annual Return - Risk-Free Rate) / Downside Deviation")
    print("7. Risk Parity = Weights proportional to inverse of risk contribution")
    print("8. Inverse-Volatility = Weights inversely proportional to stock volatility")

if __name__ == "__main__":
    run_portfolio_analysis()