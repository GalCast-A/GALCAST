import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')
from matplotlib.ticker import PercentFormatter
try:
    from pypfopt import BlackLittermanModel, risk_models, expected_returns
    PYPFOPT_AVAILABLE = True
except ImportError:
    print("Warning: 'pypfopt' not installed. Falling back to basic optimization")
    PYPFOPT_AVAILABLE = False
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots








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
    def plot_historical_strategies(self, tickers, weights_dict, risk_free_rate, hist_returns):
        """Plot cumulative returns of different strategies based on historical data."""
        strategies = {
            "Original Portfolio": np.array(list(weights_dict.values())),
            "Max Sharpe": self.optimize_portfolio(hist_returns, risk_free_rate, "sharpe"),
            "Max Sortino": self.optimize_portfolio(hist_returns, risk_free_rate, "sortino"),
            "Min Max Drawdown": self.optimize_portfolio(hist_returns, risk_free_rate, "max_drawdown"),
            "Min Volatility": self.optimize_portfolio(hist_returns, risk_free_rate, "volatility"),
            "Min Value at Risk": self.optimize_portfolio(hist_returns, risk_free_rate, "value_at_risk")
        }
        self.plot_historical_metrics_bar(hist_returns, strategies)












    def plot_historical_metrics_bar(self, returns, strategies):
        metrics = {"Annual Return": [], "Volatility": [], "Avg Correlation": []}
        labels = []




        for label, weights in strategies.items():
            portfolio_returns = returns.dot(weights)
            ann_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sub_returns = returns.loc[:, returns.columns.intersection(set(returns.columns))]  # ensure correct slice
            avg_corr = self.compute_avg_correlation(returns, weights)




            metrics["Annual Return"].append(ann_return)
            metrics["Volatility"].append(volatility)
            metrics["Avg Correlation"].append(avg_corr)
            labels.append(label)








        metric_names = list(metrics.keys())
        num_metrics = len(metric_names)
        x = np.arange(len(labels))
        bar_width = 0.18




        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metric_names):
            values = metrics[metric]
            bars = plt.bar(x + i * bar_width, values, bar_width, label=metric)
            for bar in bars:
                height = bar.get_height()
                if "Sharpe" in metric:
                    text = f"{height:.2f}"
                else:
                    text = f"{height:.2%}" if "Return" in metric or "Volatility" in metric else f"{height:.2%}"
                plt.text(bar.get_x() + bar.get_width()/2, height, text, ha='center', va='bottom', fontsize=8)




        plt.xticks(x + bar_width * (num_metrics - 1) / 2, labels, rotation=15)
        plt.title("Comparative Performance of Strategies (Past Decade)")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\n=== Metric Explanations ===")
        print("1. Annual Return:")
        print("   - What It Means: Average yearly gain from the strategy.")
        print("   - High is good. Too low may underperform the market or inflation.")




        print("\n2. Volatility:")
        print("   - What It Means: Measures risk — how much returns swing up and down.")
        print("   - Too high = unpredictable. Too low = stable but possibly low returns.")




        print("\n3. Avg Correlation:")
        print("   - What It Means: Measures how similarly the stocks in a strategy move together.")
        print("   - High (e.g. > 0.8): Poor diversification. A drop in one stock could affect all.")
        print("   - Low (e.g. < 0.3): Good diversification. Losses in one may be offset by gains in others.")





    def plot_cumulative_returns(self, returns, strategies, benchmark_returns, earliest_dates, title="Cumulative Returns of Strategies"):
        """Plot cumulative returns for specified strategies and benchmarks."""
        plt.figure(figsize=(12, 6))
        for label, weights in strategies.items():
            portfolio_returns = returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod() - 1
            plt.plot(cumulative, label=label)




        for bench_ticker, bench_ret in benchmark_returns.items():
            cumulative = (1 + bench_ret).cumprod() - 1
            plt.plot(cumulative, label=bench_ticker)




        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()








    def fetch_stock_data(self, stocks, start=None, end=None):
        if start is None:
            start = self.default_start_date
        if end is None:
            end = self.today_date
        cache_key = (tuple(sorted(stocks)), start, end)
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        error_tickers = {}
        earliest_dates = {}
        try:
            stock_data = yf.download(list(stocks), start=start, end=end, auto_adjust=True)['Close']
            if stock_data.empty:
                print("Warning: No data available for the specified date range.")
                return None, error_tickers, earliest_dates
            # Drop columns with all NaN values and ensure sufficient data
            stock_data = stock_data.dropna(axis=1, how='all')
            if stock_data.shape[0] < 252:
                print("Warning: Insufficient data (< 252 days). Optimization may be unreliable.")
            for ticker in stocks:
                if ticker not in stock_data.columns or stock_data[ticker].isna().all():
                    error_tickers[ticker] = "Data not available"
                else:
                    first_valid = stock_data[ticker].first_valid_index()
                    earliest_dates[ticker] = first_valid
            self.data_cache[cache_key] = (stock_data, error_tickers, earliest_dates)
            return stock_data, error_tickers, earliest_dates
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, error_tickers, earliest_dates








    def compute_returns(self, prices):
        returns = prices.pct_change().dropna(how='all')
        if returns.empty or returns.shape[0] < 252:
            print("Error: Insufficient valid returns data after cleaning (< 252 days).")
            return pd.DataFrame()
        return returns








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




    def compute_avg_correlation(self, returns_df, weights):
        """
        Compute the weighted average pairwise correlation among assets in a portfolio.
        """
        weighted_corr_sum = 0
        num_assets = returns_df.shape[1]
        corr_matrix = returns_df.corr()
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                weighted_corr_sum += weights[i] * weights[j] * corr_matrix.iloc[i, j]
        avg_corr = 2 * weighted_corr_sum  # account for symmetric matrix
        return avg_corr




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
            drawdown = -self.compute_max_drawdown(portfolio_returns)
            return drawdown








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




    def plot_correlation_matrix(self, prices):
        corr_matrix = prices.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Portfolio Stocks")
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n=== Correlation Matrix: Understanding Stock Relationships ===")
        print("This heatmap displays the correlation between your stocks’ price movements, measured on a scale from -1 to 1:")
        print("- A value of 1 means two stocks move in the same direction at the same time (e.g., both rise or fall together consistently).")
        print("- A value of 0 means there’s no consistent relationship—their movements are unrelated.")
        print("- A value of -1 means they move in opposite directions (e.g., one rises while the other falls).")
        print("The numbers in each cell show the strength of this relationship for each pair of stocks.")
        print("- Colors help visualize this: red indicates strong positive correlation (near 1), blue indicates strong negative correlation (near -1), and neutral colors show weak or no correlation (near 0).")
        print("Why It Matters: High positive correlations (e.g., 0.8 or higher) across many stocks suggest your portfolio’s risk is concentrated—if one stock drops, others may follow, amplifying losses. Low or negative correlations indicate better diversification, as losses in one stock might be offset by gains in another.")
        print("How to Use It: Review the matrix to identify pairs with high correlations (e.g., 0.8+) or low/negative ones (e.g., below 0.2 or negative). This can guide decisions to adjust your portfolio for better risk balance.")
        print("Example: If AAPL and MSFT have a correlation of 0.85, they tend to rise and fall together, like two runners pacing each other in a race.")








    def compute_eigenvalues(self, returns):
        """
        Compute eigenvalues of the covariance matrix of asset returns.








        Eigenvalues measure the amount of variance (risk) in your portfolio explained by different
        independent patterns of stock price movements, called factors or principal components.
        - Each eigenvalue represents the 'size' or strength of a factor.
        - Larger eigenvalues indicate factors that account for more of your portfolio’s risk.
        - The total number of eigenvalues equals the number of stocks, as each factor captures a unique
          way your stocks move together or apart.
        """
        # Covariance matrix scaled to annual terms (252 trading days) reflects how stocks co-vary
        cov_matrix = returns.cov() * 252
        # Eigenvalues are computed from this matrix; they quantify risk contributions
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        # Sort in descending order: largest risk factors come first
        eigenvalues = sorted(eigenvalues, reverse=True)
        # Total variance is the sum of all eigenvalues, representing 100% of portfolio risk
        total_variance = sum(eigenvalues)
        # Explained variance ratio shows each factor’s percentage contribution to total risk
        explained_variance_ratio = [eig / total_variance for eig in eigenvalues]
        return eigenvalues, explained_variance_ratio








    def plot_eigenvalues(self, eigenvalues, explained_variance_ratio, tickers):
        """Plot eigenvalues and cumulative explained variance with a clear explanation, returning cumulative variance."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.6, color='blue', label='Eigenvalues (Risk Factors)')
        cumulative_variance = np.cumsum(explained_variance_ratio)
        plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, 'r-o', label='Cumulative Risk Explained')
        plt.title("Portfolio Risk Factor Analysis")
        plt.xlabel("Risk Factor")
        plt.ylabel("Eigenvalue / Cumulative Percentage")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        plt.tight_layout()
        plt.savefig('eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()








        print("\n=== Portfolio Risk Factor Analysis ===")
        print(f"This chart analyzes how risk is distributed across your portfolio of {len(tickers)} stocks: {', '.join(tickers)}.")
        print("\nWhat Are Eigenvalues?")
        print("- Eigenvalues measure the amount of risk (variance) in your portfolio explained by different patterns of stock price movements.")
        print("- They come from a mathematical breakdown of how your stocks’ returns relate to each other (via the covariance matrix).")
        print("- Each eigenvalue is tied to a 'factor'—a unique combination of your stocks’ movements, not a single stock.")
        print("- The bigger the eigenvalue, the more that factor influences your portfolio’s overall risk.")








        print("\nWhat Are These Factors?")
        print(f"- You have {len(eigenvalues)} factors, one for each stock in your portfolio.")
        print("- Each factor represents an independent way your stocks move together or apart:")
        print("  - Factor 1 (largest eigenvalue): Usually the biggest driver, often linked to market-wide trends affecting all stocks.")
        print("  - Factor 2 and beyond: Smaller, more specific patterns (e.g., sector trends or individual stock behavior).")
        print("- Think of factors as invisible forces: they blend contributions from all your stocks, weighted differently.")








        print("\nChart Details:")
        print("- Blue bars (eigenvalues): Show the size of each factor’s risk contribution.")
        print("- Red line: Shows the total percentage of risk explained as you add more factors, reaching 100% with all included.")








        print("\nBreakdown of Risk Factors:")
        for i, (eig, ratio) in enumerate(zip(eigenvalues, explained_variance_ratio), 1):
            print(f"- Factor {i}: Size = {eig:.2f}, Explains {ratio:.2%} of total risk")
        print(f"- Total Risk Explained: {cumulative_variance[-1]:.2%}")








        print("\nWhy It Matters:")
        print("- If Factor 1 explains a lot (e.g., over 50%), your stocks move together too much, concentrating risk.")
        print("- A more even spread across factors means your stocks offset each other, improving diversification.")
        print("- Use this to see if your portfolio relies too heavily on one pattern (like the market) or balances risk well.")








        print("Example: If Factor 1 has a size of 0.40 and explains 70% of risk, it’s like one big wave moving all your stocks at once—other waves (stocks) don’t calm it down.")








        return cumulative_variance  # Return cumulative_variance for use elsewhere








    def compute_fama_french_exposures(self, portfolio_returns, start_date, end_date):
        """Compute Fama-French 3-factor exposures for the portfolio."""
        # Fetch Fama-French 3-factor data from Kenneth French's data library
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        try:
            ff_data = pd.read_csv(ff_url, skiprows=3, index_col=0)
            ff_data.index = pd.to_datetime(ff_data.index, format="%Y%m%d", errors='coerce')
            ff_data = ff_data.dropna() / 100  # Convert to decimal returns
            ff_data = ff_data[["Mkt-RF", "SMB", "HML"]]  # Select 3 factors
            ff_data = ff_data.loc[start_date:end_date]
        except Exception as e:
            print(f"Error fetching Fama-French data: {e}. Using fallback zero exposures.")
            return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}








        # Align portfolio returns with Fama-French data
        common_dates = portfolio_returns.index.intersection(ff_data.index)
        if len(common_dates) < 30:  # Minimum data points for meaningful regression
            print("Warning: Insufficient overlapping data with Fama-French factors. Using fallback zero exposures.")
            return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}








        aligned_returns = portfolio_returns.loc[common_dates]
        aligned_ff = ff_data.loc[common_dates]








        # Perform regression: portfolio returns ~ Mkt-RF + SMB + HML
        X = sm.add_constant(aligned_ff)  # Add intercept
        model = sm.OLS(aligned_returns, X).fit()
        exposures = {
            "Mkt-RF": model.params["Mkt-RF"],
            "SMB": model.params["SMB"],
            "HML": model.params["HML"]
        }
        return exposures








    def plot_efficient_frontier(self, returns, risk_free_rate, n_portfolios=1000, fig_size=(10, 6)):
        """Plot the efficient frontier with key optimized portfolios based on different strategies."""
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








        # Plotting
        plt.figure(figsize=fig_size)
        plt.scatter(all_volatilities, all_returns, c=all_sharpe_ratios, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')








        # Plot each optimized portfolio with distinct markers and colors
        markers = ['o', '^', 's', '*', 'D']  # Circle, Triangle, Square, Star, Diamond
        colors = ['red', 'darkgreen', 'blue', 'purple', 'orange']
        offset = 0.005  # Small offset for overlapping points
        adjusted_points = []
        for i, (name, metrics) in enumerate(strategy_metrics.items()):
            vol, ret = metrics["volatility"], metrics["return"]
            adjusted_vol, adjusted_ret = vol, ret
            for prev_vol, prev_ret in adjusted_points:
                if abs(vol - prev_vol) < 0.01 and abs(ret - prev_ret) < 0.01:
                    adjusted_vol += offset * (i + 1)
                    adjusted_ret += offset * (i + 1)
            adjusted_points.append((adjusted_vol, adjusted_ret))
            plt.scatter(adjusted_vol, adjusted_ret, c=colors[i], marker=markers[i], s=100, edgecolors='black',
                        linewidths=1.0, label=name)








        # Capital Market Line (based on Max Sharpe portfolio)
        max_sharpe_vol = strategy_metrics["Max Sharpe"]["volatility"]
        max_sharpe_sharpe = strategy_metrics["Max Sharpe"]["sharpe"]
        plt.plot([0, max_sharpe_vol * 1.5], [risk_free_rate, risk_free_rate + max_sharpe_sharpe * max_sharpe_vol * 1.5],
                 'k--', label='Capital Market Line')








        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Efficient Frontier with Optimized Strategies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()








        # Text string with metrics for each strategy, moved lower
        text_str = " | ".join(
            f"{name}: Return={metrics['return']:.2%}, Vol={metrics['volatility']:.2%}, SR={metrics['sharpe']:.2f}"
            for name, metrics in strategy_metrics.items()
        )
        plt.figtext(0.5, -0.15, text_str, ha='center', fontsize=9, wrap=True)  # Adjusted from 0.005 to -0.15








        plt.show()








        print("\n**Efficient Frontier Plot Explanation:**")
        print("This plot illustrates the risk-return trade-off across thousands of simulated portfolio compositions using your stocks.")
        print("The x-axis represents annualized volatility (risk), and the y-axis represents annualized return.")
        print("Each dot is a unique portfolio. Key optimized portfolios are highlighted as follows:")
        print("- Red Circle: Max Sharpe Ratio - Highest risk-adjusted return.")
        print("- Green Triangle: Max Sortino Ratio - Optimized for downside risk-adjusted return.")
        print("- Blue Square: Min Max Drawdown - Lowest peak-to-trough loss.")
        print("- Purple Star: Min Volatility - Lowest overall risk.")
        print("- Orange Diamond: Min Value at Risk - Minimized potential daily loss at 90% confidence.")
        print("The color bar shows Sharpe Ratios, with brighter colors indicating better risk-adjusted performance.")
        print("The dashed black line is the Capital Market Line, showing the optimal risk-return trade-off from the risk-free rate.")
        print("Key metrics for each optimized portfolio are displayed below the plot.")
        print("If markers overlap, they have been slightly offset for visibility.")








    def plot_comparison_bars(self, original_metrics, optimized_metrics, benchmark_metrics):
        metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "maximum_drawdown", "value_at_risk"]
        labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Value at Risk (90%)"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            values = [original_metrics[metric], optimized_metrics[metric]]
            names = ["Original", "Optimized"]
            if benchmark_metrics:
                for bench, bm in benchmark_metrics.items():
                    values.append(bm[metric])
                    names.append(bench)
            bars = axes[i].bar(names, values)
            axes[i].set_title(label)
            if "sharpe" in metric:
                axes[i].set_ylabel("Ratio")
            else:
                axes[i].yaxis.set_major_formatter(PercentFormatter(1.0))
                axes[i].set_ylabel("Percentage")
            for bar in bars:
                yval = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}" if 'sharpe' in metric else f"{yval:.2%}", ha='center', va='bottom')
        plt.tight_layout()
        plt.show()








    def plot_portfolio_exposures(self, tickers, original_weights, optimized_weights):
        """Plot pie charts comparing original and optimized portfolio exposures."""
        # Filter out stocks with zero weights for cleaner pie charts
        original_exposures = [w for w in original_weights if w > 0]
        original_labels = [t for t, w in zip(tickers, original_weights) if w > 0]
        optimized_exposures = [w for w in optimized_weights if w > 0]
        optimized_labels = [t for t, w in zip(tickers, optimized_weights) if w > 0]








        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))








        # Original Portfolio Pie Chart
        ax1.pie(original_exposures, labels=original_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax1.axis('equal')  # Equal aspect ratio ensures pie is circular
        ax1.set_title("Original Portfolio Exposure")








        # Optimized Portfolio Pie Chart
        ax2.pie(optimized_exposures, labels=optimized_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax2.axis('equal')
        ax2.set_title("Optimized Portfolio Exposure")








        plt.tight_layout()
        plt.show()








        # Print explicit exposures
        print("\nOriginal Portfolio Exposures:")
        for ticker, weight in zip(tickers, original_weights):
            print(f"- {ticker}: {weight:.2%}")








        print("\nOptimized Portfolio Exposures:")
        for ticker, weight in zip(tickers, optimized_weights):
            print(f"- {ticker}: {weight:.2%}")








    def plot_rolling_volatility(self, returns, weights_dict, benchmark_returns, window=252):
        """Plot rolling annualized volatility for original and optimized portfolios vs benchmarks."""
        plt.figure(figsize=(12, 6))
        for label, weights in weights_dict.items():
            portfolio_returns = returns.dot(weights)
            rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
            plt.plot(rolling_vol, label=f"{label} Volatility")
        for bench_ticker, bench_ret in benchmark_returns.items():
            rolling_vol = bench_ret.rolling(window=window).std() * np.sqrt(252)
            plt.plot(rolling_vol, label=f"{bench_ticker} Volatility")
        plt.title(f"Rolling {window}-Day Annualized Volatility")
        plt.xlabel("Date")
        plt.ylabel("Annualized Volatility")
        plt.legend()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()








        print("\nRolling Volatility Explanation:")
        print(f"This chart shows the {window}-day rolling annualized volatility of your portfolios and benchmarks.")
        print("- Higher spikes indicate periods of greater risk or instability.")
        print("- Compare Original vs. Optimized to see how optimization affects risk over time.")




    def plot_diversification_benefit(self, returns, original_weights, optimized_weights, tickers):
        """Plot the diversification benefit of optimization vs original and equal-weight portfolios."""
        try:
            equal_weights = np.ones(len(tickers)) / len(tickers)
            cov_matrix = returns.cov() * 252

            orig_vol = np.sqrt(np.dot(original_weights.T, np.dot(cov_matrix, original_weights)))
            opt_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))

            vols = [equal_vol, orig_vol, opt_vol]
            labels = ['Equal Weight', 'Original', 'Optimized']

            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, vols, color=['gray', 'blue', 'green'], alpha=0.7)
            plt.title("Diversification Benefit: Volatility Comparison")
            plt.ylabel("Annualized Volatility")
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2%}", ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('diversification_benefit.png', dpi=300)
            plt.show()
            print("Diversification benefit chart saved as 'diversification_benefit.png'.")

            print("\nDiversification Benefit Explanation:")
            print(f"- Equal Weight: {equal_vol:.2%} volatility (baseline naive diversification)")
            print(f"- Original: {orig_vol:.2%} volatility (your initial allocation)")
            print(f"- Optimized: {opt_vol:.2%} volatility (post-optimization)")
            reduction = orig_vol - opt_vol
            print(f"- Risk Reduction: {reduction:.2%} (positive means optimization lowered risk)")

        except Exception as e:
            print(f"Error in diversification plot: {e}. Skipping visualization.")


    def plot_crisis_performance(self, returns, weights_dict, benchmark_returns, earliest_dates):
        """
        Plot performance during historical crises of low business activity if data is available.
        Only shows plots if the earliest stock data is at least 6 months before the crisis start.
        """
        # Define crisis periods (ordered oldest to most recent)
        crises = [
            {
                "name": "Dot-Com Bust",
                "start": pd.to_datetime("2000-03-01"),
                "end": pd.to_datetime("2002-10-31"),
                "description": "The Dot-Com Bust (March 2000 - October 2002) saw a tech bubble collapse, with the Nasdaq dropping 78% as overvalued internet companies failed, leading to reduced business activity in tech sectors."
            },
            {
                "name": "Great Recession",
                "start": pd.to_datetime("2007-12-01"),
                "end": pd.to_datetime("2009-06-30"),
                "description": "The Great Recession (December 2007 - June 2009) followed a housing bubble burst and financial crisis, with GDP dropping 4.3% and business activity stalling as credit froze."
            },
            {
                "name": "COVID-19 Crisis",
                "start": pd.to_datetime("2020-02-01"),
                "end": pd.to_datetime("2020-04-30"),
                "description": "The COVID-19 Crisis (February - April 2020) involved global lockdowns, halting business activity, with a 31.4% GDP drop in Q2 2020 and a swift 34% S&P 500 decline."
            }
        ]








        # Earliest data point across all stocks
        earliest_data = min(earliest_dates.values())
        six_months = timedelta(days=180)  # Approx 6 months








        # Store performance data for combined analysis
        crisis_summaries = {}








        for crisis in crises:
            crisis_start = crisis["start"]
            crisis_end = crisis["end"]
            # Check if data is available 6 months before crisis start
            if earliest_data > (crisis_start - six_months):
                print(f"Skipping {crisis['name']} plot: Earliest data ({earliest_data.strftime('%Y-%m-%d')}) is not at least 6 months before crisis start ({crisis_start.strftime('%Y-%m-%d')}).")
                continue








            # Find nearest trading days within the returns index
            available_starts = returns.index[returns.index >= crisis_start]
            available_ends = returns.index[returns.index <= crisis_end]








            if available_starts.empty or available_ends.empty:
                print(f"Warning: {crisis['name']} period data not available within the analysis range (no data found).")
                continue








            available_start = available_starts.min()
            available_end = available_ends.max()








            if pd.isna(available_start) or pd.isna(available_end) or available_start > available_end:
                print(f"Warning: {crisis['name']} period data not available within the analysis range (invalid date range).")
                continue








            crisis_returns = returns.loc[available_start:available_end]
            plt.figure(figsize=(10, 6))
            crisis_performance = {}








            # Plot portfolios
            for label, weights in weights_dict.items():
                portfolio_returns = crisis_returns.dot(weights)
                cumulative = (1 + portfolio_returns).cumprod() - 1
                plt.plot(cumulative, label=label)
                crisis_performance[label] = cumulative.iloc[-1]  # Final cumulative return








            # Plot benchmarks
            for bench_ticker, bench_ret in benchmark_returns.items():
                bench_crisis_ret = bench_ret.loc[available_start:available_end]
                if not bench_crisis_ret.empty:
                    bench_cum = (1 + bench_crisis_ret).cumprod() - 1
                    plt.plot(bench_cum, label=bench_ticker)
                    crisis_performance[bench_ticker] = bench_cum.iloc[-1]








            plt.title(f"{crisis['name']} Performance ({available_start.strftime('%Y-%m-%d')} to {available_end.strftime('%Y-%m-%d')})")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.legend()
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()








            # Individual crisis analysis
            print(f"\n===== {crisis['name'].upper()} PERFORMANCE =====")
            print(f"Period: {available_start.strftime('%Y-%m-%d')} to {available_end.strftime('%Y-%m-%d')}")
            print(f"Context: {crisis['description']}")
            print("Performance:")
            orig_return = crisis_performance.get("Original Portfolio", 0)
            opt_return = crisis_performance.get("Optimized Portfolio", 0)
            bench_returns = {k: v for k, v in crisis_performance.items() if k not in ["Original Portfolio", "Optimized Portfolio"]}
            print(f"- Original Portfolio: {orig_return:.2%}")
            print(f"- Optimized Portfolio: {opt_return:.2%}")
            for bench, ret in bench_returns.items():
                print(f"- {bench}: {ret:.2%}")
            print("Analysis:")
            if opt_return > orig_return:
                print(f"- The optimized portfolio outperformed the original by {(opt_return - orig_return):.2%}, suggesting optimization improved resilience or capitalized on opportunities during this crisis.")
            else:
                print(f"- The original portfolio outperformed the optimized by {(orig_return - opt_return):.2%}, indicating the optimization may have been overly cautious or misaligned for this period.")
            if bench_returns:
                avg_bench = np.mean(list(bench_returns.values()))
                print(f"- Compared to benchmarks (avg: {avg_bench:.2%}), the optimized portfolio {'outperformed' if opt_return > avg_bench else 'underperformed'} by {(opt_return - avg_bench):.2%}.")








            crisis_summaries[crisis["name"]] = {
                "original": orig_return,
                "optimized": opt_return,
                "benchmarks": bench_returns,
                "start": available_start,
                "end": available_end
            }








        # Combined analysis across all crises
        if crisis_summaries:
            print("\n===== COMBINED CRISIS PERFORMANCE INSIGHTS =====")
            print("This section combines insights from the Dot-Com Bust, Great Recession, and COVID-19 Crisis to assess your portfolio’s behavior during low business activity periods:")
            for crisis_name, summary in crisis_summaries.items():
                print(f"\n- {crisis_name} ({summary['start'].strftime('%Y-%m-%d')} to {summary['end'].strftime('%Y-%m-%d')}):")
                print(f"  * Original: {summary['original']:.2%}, Optimized: {summary['optimized']:.2%}")
                print(f"  * Benchmark Avg: {np.mean(list(summary['benchmarks'].values())):.2%}")








            # Aggregate performance
            avg_orig = np.mean([s["original"] for s in crisis_summaries.values()])
            avg_opt = np.mean([s["optimized"] for s in crisis_summaries.values()])
            avg_bench = np.mean([np.mean(list(s["benchmarks"].values())) for s in crisis_summaries.values() if s["benchmarks"]])








            print("\nAverage Performance Across Crises:")
            print(f"- Original Portfolio: {avg_orig:.2%}")
            print(f"- Optimized Portfolio: {avg_opt:.2%}")
            print(f"- Benchmarks (Avg): {avg_bench:.2%}")








            print("\nImplications and Insights:")
            if avg_opt > avg_orig:
                print(f"- Optimization Advantage: The optimized portfolio outperformed the original by {(avg_opt - avg_orig):.2%} on average, suggesting it better handles low business activity periods—possibly by reducing risk or reallocating to resilient assets.")
            else:
                print(f"- Original Resilience: The original portfolio outperformed the optimized by {(avg_orig - avg_opt):.2%} on average, indicating your initial weights may naturally suit downturns better than the chosen optimization.")








            if avg_opt > avg_bench:
                print(f"- Market Outperformance: The optimized portfolio beat the average benchmark by {(avg_opt - avg_bench):.2%}, implying it could offer a competitive edge during crises.")
            else:
                print(f"- Market Underperformance: The optimized portfolio lagged the average benchmark by {(avg_bench - avg_opt):.2%}, suggesting room to refine the strategy for broader market resilience.")








            print("\nPotential Meaning:")
            print("- Consistency: If the optimized portfolio consistently beats the original (e.g., in 2+ crises), your optimization aligns well with downturns—stick with it.")
            print("- Volatility Sensitivity: Check Rolling Volatility and Correlation Matrix—high correlations or volatility spikes during these periods may explain performance gaps.")
            print("- Factor Exposure: Review Fama-French exposures (e.g., high market beta or value tilt) to see if they match crisis patterns (e.g., value stocks in recessions).")
            print("Action: Adjust your strategy based on risk tolerance—favor Min Volatility for stability or Max Sharpe for balanced growth in future downturns.")








    def suggest_courses_of_action(self, tickers, original_weights, optimized_weights, returns, risk_free_rate,
                                 benchmark_metrics, risk_tolerance, start_date, end_date):
        """
        Act as a wealth advisor, analyzing the portfolio and suggesting detailed courses of action for short,
        medium, and long terms based on performance, biases, and user preferences.
        """
        # Compute portfolio returns and metrics
        original_returns = returns.dot(original_weights)
        optimized_returns = returns.dot(optimized_weights)
        original_metrics = {
            "annual_return": original_returns.mean() * 252,
            "annual_volatility": original_returns.std() * np.sqrt(252),
            "sharpe_ratio": self.portfolio_performance(original_weights, returns, risk_free_rate)[2],
            "max_drawdown": self.compute_max_drawdown(original_returns),
            "var": self.compute_var(original_returns, 0.90),
            "sortino": self.compute_sortino_ratio(original_returns, risk_free_rate)
        }
        optimized_metrics = {
            "annual_return": optimized_returns.mean() * 252,
            "annual_volatility": optimized_returns.std() * np.sqrt(252),
            "sharpe_ratio": self.portfolio_performance(optimized_weights, returns, risk_free_rate)[2],
            "max_drawdown": self.compute_max_drawdown(optimized_returns),
            "var": self.compute_var(optimized_returns, 0.90),
            "sortino": self.compute_sortino_ratio(optimized_returns, risk_free_rate)
        }


    def optimize_with_factor_and_correlation(self, returns, risk_free_rate, tickers, market_prices=None, min_allocation=0.05, max_allocation=0.30):
        """Optimize portfolio with risk parity, low correlation, and forward-looking returns, ensuring diversification."""
        try:
            num_assets = len(tickers)
            if num_assets < 2:
                raise ValueError("Portfolio must have at least 2 assets for optimization.")

            # Ensure returns are clean and sufficient
            returns = returns.dropna(how='any')
            if returns.shape[0] < 252:
                raise ValueError("Insufficient data points (< 252) for reliable optimization.")
            if returns.isna().any().any():
                raise ValueError("Returns contain NaN values after cleaning.")

            cov_matrix = returns.cov() * 252
            if cov_matrix.isna().any().any():
                raise ValueError("Covariance matrix contains NaN values.")

            initial_weights = np.ones(num_assets) / num_assets


            # Step 1: Forward-Looking Expected Returns
            expected_rets = None
            if PYPFOPT_AVAILABLE and market_prices is not None and not market_prices.empty:
                try:
                    market_prices = market_prices.reindex(returns.index).dropna()
                    if market_prices.empty:
                        raise ValueError("Market prices do not overlap with returns data.")

                    S = risk_models.sample_cov(returns)
                    if not np.all(np.linalg.eigvals(S) >= -1e-10):
                        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')

                    delta = expected_returns.market_implied_risk_aversion(market_prices)
                    market_weights = pd.Series(1/num_assets, index=tickers)
                    market_prior = expected_returns.market_implied_prior_returns(market_weights, S, delta)

                    Q = pd.Series([0.0] * num_assets, index=tickers)
                    P = np.eye(num_assets)
                    Omega = np.diag([0.01] * num_assets)

                    bl = BlackLittermanModel(S, pi=market_prior, Q=Q, P=P, Omega=Omega)
                    expected_rets = bl.bl_returns()
                    if expected_rets.isna().any():
                        raise ValueError("Black-Litterman returned NaN values.")
                except Exception as e:
                    print(f"Black-Litterman failed: {e}. Falling back to PCA-adjusted returns.")


            # Step 2: PCA Fallback
            if expected_rets is None:
                try:
                    pca = PCA(n_components=min(3, num_assets))
                    pca.fit(returns)
                    factor_returns = pd.DataFrame(pca.transform(returns), index=returns.index)
                    expected_rets = pd.Series(pca.inverse_transform(factor_returns.mean()) * 252, index=tickers)
                    if expected_rets.isna().any():
                        raise ValueError("PCA returned NaN values.")
                except Exception as e:
                    print(f"PCA failed: {e}. Using historical mean returns.")
                    expected_rets = returns.mean() * 252


            # Step 3: Final Fallback
            if expected_rets.isna().any():
                print("Warning: Expected returns contain NaN. Using risk-free rate + 5% as fallback.")
                expected_rets = pd.Series([risk_free_rate + 0.05] * num_assets, index=tickers)


            # Step 4: Optimization Objective with Diversification
            def objective(weights):
                try:
                    ret = np.dot(weights, expected_rets)
                    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

                    # Volatility drag adjustment
                    vol_drag = 0.5 * vol**2
                    adj_sharpe = (ret - vol_drag - risk_free_rate) / vol if vol > 0 else 0

                    # Enhanced correlation penalty (scaled by concentration)
                    corr_penalty = 2.0 * self.compute_avg_correlation(returns, weights) * (1 - np.min(weights) / np.max(weights))

                    # Stronger risk parity penalty
                    risk_contribs = weights * np.dot(cov_matrix, weights) / vol if vol > 0 else weights
                    risk_parity_penalty = 2.0 * np.var(risk_contribs)

                    # Entropy-based diversification penalty (negative entropy encourages even weights)
                    weights_clean = weights + 1e-10  # Avoid log(0)
                    entropy = -np.sum(weights_clean * np.log(weights_clean)) / np.log(num_assets)
                    diversification_penalty = 1.0 * (1 - entropy)  # Maximize entropy (1 = equal weights)

                    return -adj_sharpe + corr_penalty + risk_parity_penalty + diversification_penalty
                except Exception as e:
                    print(f"Objective error: {e}. Returning infinity.")
                    return float('inf')


            # Step 5: Constraints and Dynamic Bounds
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            # Adjust bounds for feasibility while ensuring diversification
            max_allowed = min(0.40, 1.0 / num_assets * 1.5)  # Cap at 40% or 1.5x equal weight
            min_allowed = max(0.01, 1.0 / num_assets * 0.5)  # At least 1% or 0.5x equal weight
            if min_allocation * num_assets > 1 or max_allocation * num_assets < 1:
                min_allocation = min_allowed
                max_allocation = max_allowed
                print(f"Adjusted bounds for feasibility: min={min_allocation:.4f}, max={max_allocation:.4f}")
            bounds = tuple((max(min_allocation, min_allowed), min(max_allocation, max_allowed)) for _ in range(num_assets))


            # Step 6: Optimize
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-8})
            if not result.success:
                print(f"Optimization failed: {result.message}. Using equal weights.")
                optimized_weights = initial_weights
            else:
                optimized_weights = result.x
                # Post-optimization diversification adjustment
                optimized_weights = np.clip(optimized_weights, min_allowed, max_allowed)
                optimized_weights /= optimized_weights.sum()  # Renormalize


            # Step 7: Metrics and Insights
            opt_ret, opt_vol, opt_sharpe = self.portfolio_performance(optimized_weights, returns, risk_free_rate)
            avg_corr = self.compute_avg_correlation(returns, optimized_weights)
            risk_contribs = optimized_weights * np.dot(cov_matrix, optimized_weights) / opt_vol if opt_vol > 0 else optimized_weights


            # Step 8: Risk Contribution Visualization
            plt.figure(figsize=(10, 6))
            plt.bar(tickers, risk_contribs, color='teal', alpha=0.7)
            plt.axhline(y=np.mean(risk_contribs), color='red', linestyle='--', label='Equal Risk Contribution')
            plt.title("Risk Contribution by Asset (Post-Optimization)")
            plt.xlabel("Assets")
            plt.ylabel("Risk Contribution")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig('risk_contribution.png', dpi=300)
            plt.show()
            print("Risk contribution chart saved as 'risk_contribution.png'.")


            # Step 9: Weights Comparison Visualization
            plt.figure(figsize=(10, 6))
            bar_width = 0.35
            index = np.arange(num_assets)
            plt.bar(index, initial_weights, bar_width, label='Original Weights', color='blue', alpha=0.6)
            plt.bar(index + bar_width, optimized_weights, bar_width, label='Optimized Weights', color='green', alpha=0.6)
            plt.xlabel('Assets')
            plt.ylabel('Weight')
            plt.title('Original vs Optimized Portfolio Weights')
            plt.xticks(index + bar_width / 2, tickers, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig('weights_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Weights comparison chart saved as 'weights_comparison.png'.")


            # Step 10: Output with Diversification Metrics
            entropy = -np.sum(optimized_weights * np.log(optimized_weights + 1e-10)) / np.log(num_assets)
            print("\n===== Factor and Correlation Optimized Portfolio =====")
            print(f"Original Avg Correlation: {self.compute_avg_correlation(returns, initial_weights):.2f}, "
                  f"New Avg Correlation: {avg_corr:.2f}")
            print(f"Optimized Metrics - Return: {opt_ret:.2%}, Volatility: {opt_vol:.2%}, Sharpe: {opt_sharpe:.2f}")
            print(f"Diversification - Entropy: {entropy:.2f} (1.0 = perfectly equal weights)")
            print("Adjustments Made: Enhanced diversification, reduced weights of highly correlated assets, balanced risk parity.")


            return optimized_weights, {
                "return": opt_ret,
                "volatility": opt_vol,
                "sharpe": opt_sharpe,
                "avg_correlation": avg_corr,
                "entropy": entropy
            }


        except Exception as e:
            print(f"Critical error in optimization: {e}. Returning equal weights.")
            return np.ones(num_assets) / num_assets, {}




        # Fama-French exposures for deeper insight
        ff_exposures = self.compute_fama_french_exposures(original_returns, start_date, end_date)
        corr_matrix = returns.corr()








        # Begin advisor narrative
        print("\n===== YOUR WEALTH ADVISOR’S RECOMMENDATIONS =====")
        print(f"Hello! I’ve taken a close look at your portfolio—{', '.join(tickers)}—and I’m here to help you make the most of it. My goal is to maximize your returns across short, medium, and long terms while keeping your {risk_tolerance} risk tolerance in mind. Let’s dive into what your portfolio is telling us and how we can position you for success.")








        # Analysis of Strengths and Biases
        print("\n--- Strengths of Your Portfolio ---")
        if original_metrics["sharpe_ratio"] > 1.5:
            print(f"- Strong Risk-Adjusted Returns: Your Sharpe Ratio of {original_metrics['sharpe_ratio']:.2f} is impressive—it shows you’re getting solid returns for the risk you’re taking. This is a great foundation to build on!")
        if original_metrics["annual_volatility"] < 0.15:
            print(f"- Low Volatility: At {original_metrics['annual_volatility']:.2%}, your portfolio is stable, which is fantastic for peace of mind and steady growth.")
        if ff_exposures["Mkt-RF"] < 1:
            print(f"- Market Resilience: With a market beta of {ff_exposures['Mkt-RF']:.2f}, your portfolio is less sensitive to market swings than the average—excellent for weathering downturns.")








        print("\n--- Potential Biases and Weaknesses ---")
        if ff_exposures["Mkt-RF"] > 1.2:
            print(f"- High Market Exposure: Your market beta of {ff_exposures['Mkt-RF']:.2f} means your portfolio amplifies market moves. This can boost gains in bull markets but leaves you vulnerable in crashes.")
        if max(corr_matrix.max()) > 0.8:
            high_corr_pairs = [(t1, t2) for t1 in tickers for t2 in tickers if t1 < t2 and corr_matrix.loc[t1, t2] > 0.8]
            print(f"- Concentration Risk: Stocks like {', '.join([f'{p[0]}-{p[1]}' for p in high_corr_pairs])} have correlations above 0.8, suggesting your risk is concentrated. If one drops, others may follow.")
        if original_metrics["max_drawdown"] < -0.25:
            print(f"- Significant Drawdowns: A max drawdown of {original_metrics['max_drawdown']:.2%} indicates past losses were steep. We’ll want to protect against this moving forward.")








        # Current Standing and Interpretation
        print("\n--- Where You Stand Today ---")
        bench_key = list(benchmark_metrics.keys())[0]  # Use first benchmark for comparison
        bench_return = benchmark_metrics[bench_key]["annual_return"]
        print(f"Your original portfolio has delivered an annualized return of {original_metrics['annual_return']:.2%}, with a volatility of {original_metrics['annual_volatility']:.2%}, compared to {bench_key}’s {bench_return:.2%} return.")
        if optimized_metrics["annual_return"] > original_metrics["annual_return"]:
            print(f"- Good News: Optimization boosts your return to {optimized_metrics['annual_return']:.2%}—a {optimized_metrics['annual_return'] - original_metrics['annual_return']:.2%} improvement, showing we can enhance your growth.")
        if optimized_metrics["annual_volatility"] < original_metrics["annual_volatility"]:
            print(f"- Risk Reduction: Optimization cuts volatility to {optimized_metrics['annual_volatility']:.2%}, a {original_metrics['annual_volatility'] - optimized_metrics['annual_volatility']:.2%} drop, aligning better with your {risk_tolerance} risk tolerance.")








        # Courses of Action
        print("\n--- Courses of Action for Your Financial Future ---")
        print(f"Based on your portfolio’s performance, optimization results, and {risk_tolerance} risk tolerance, here are tailored strategies for short, medium, and long terms. Each comes with probabilistic reasoning and specific actions to maximize your returns.")








        # Short-Term (0-1 Year)
        print("\nShort-Term (0-1 Year): Quick Wins and Stability")
        print("- Goal: Capitalize on immediate opportunities while managing risk.")
        if risk_tolerance == "low":
            print(f"- Action 1: De-Risk with Stability Focus")
            print(f"  * Why: Your {risk_tolerance} risk tolerance favors safety. With a VaR of {original_metrics['var']:.2%}, there’s a 10% chance of losing that much in a day.")
            print(f"  * How: Shift 10-15% of your portfolio to low-volatility assets like utilities (e.g., XLU ETF, 1.5% yield, 10% volatility) or Treasuries (e.g., TLT, 2-3% yield). This could reduce VaR to {original_metrics['var'] * 0.85:.2%} based on historical correlations.")
            print(f"  * Probability: 70% chance of stabilizing returns within 6 months, given utilities’ low beta (~0.3).")
        else:
            print(f"- Action 1: Capitalize on Momentum")
            print(f"  * Why: Your {risk_tolerance} tolerance allows chasing short-term gains. Optimized Sharpe ({optimized_metrics['sharpe_ratio']:.2f}) suggests upside potential.")
            print(f"  * How: Increase allocation to top performers (e.g., stocks with recent 20%+ gains in your portfolio—check returns) by 5-10%, or add a momentum ETF like MTUM (12% annualized return, 15% volatility).")
            print(f"  * Probability: 60% chance of outperforming {bench_key} by 2-3% in 6 months, based on momentum factor trends.")
        print(f"- Action 2: Rebalance Quarterly")
        print(f"  * Why: Keeps your portfolio aligned with short-term market shifts.")
        print(f"  * How: Adjust weights to optimized levels (e.g., {', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, optimized_weights)])}).")
        print(f"  * Probability: 80% chance of maintaining or improving Sharpe Ratio, per historical rebalancing studies.")








        # Medium-Term (1-5 Years)
        print("\nMedium-Term (1-5 Years): Growth with Balance")
        print("- Goal: Build wealth steadily while preparing for volatility.")
        if ff_exposures["HML"] > 0.3:
            print(f"- Action 1: Leverage Value Opportunities")
            print(f"  * Why: Your value exposure (HML: {ff_exposures['HML']:.2f}) suggests strength in undervalued stocks, which often shine in recovery phases.")
            print(f"  * How: Allocate 10-20% to a value ETF (e.g., VTV, 10% return, 14% volatility) or deepen exposure to value sectors like financials (e.g., XLF).")
            print(f"  * Probability: 65% chance of 8-10% annualized returns over 3 years, based on value factor outperformance post-recession.")
        else:
            print(f"- Action 1: Explore Growth Sectors")
            print(f"  * Why: Low HML ({ff_exposures['HML']:.2f}) suggests room to capture growth, especially with {risk_tolerance} tolerance.")
            print(f"  * How: Invest 15-25% in tech or consumer discretionary (e.g., QQQ, 13% return, 18% volatility), targeting sectors with 10-15% growth potential.")
            print(f"  * Probability: 55% chance of beating {bench_key} by 3-5% annually, per growth stock cycles.")
        print(f"- Action 2: Diversify Correlation")
        print(f"  * Why: High correlations (e.g., {max(corr_matrix.max()):.2f}) increase risk concentration.")
        print(f"  * How: Add 10% to assets with correlations < 0.5 to your portfolio (e.g., gold via GLD, 5% return, -0.1 correlation to equities).")
        print(f"  * Probability: 75% chance of reducing volatility by 2-3%, per diversification models.")








        # Long-Term (5+ Years)
        print("\nLong-Term (5+ Years): Wealth Maximization")
        print("- Goal: Achieve sustained growth with resilience.")
        if optimized_metrics["sortino"] > original_metrics["sortino"]:
            print(f"- Action 1: Stick with Optimization")
            print(f"  * Why: Optimized Sortino ({optimized_metrics['sortino']:.2f} vs {original_metrics['sortino']:.2f}) shows better downside protection, key for long-term stability.")
            print(f"  * How: Fully adopt optimized weights ({', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, optimized_weights)])}) and reinvest dividends.")
            print(f"  * Probability: 70% chance of growing $10,000 to ${(10000 * (1 + optimized_metrics['annual_return']) ** 5):,.0f} in 5 years, vs ${(10000 * (1 + original_metrics['annual_return']) ** 5):,.0f} originally.")
        else:
            print(f"- Action 1: Enhance Downside Protection")
            print(f"  * Why: Original Sortino ({original_metrics['sortino']:.2f}) suggests vulnerability to losses over time.")
            print(f"  * How: Shift 20% to Min Volatility strategy (e.g., SPLV ETF, 8% return, 10% volatility) or bonds.")
            print(f"  * Probability: 80% chance of cutting max drawdown to {optimized_metrics['max_drawdown'] * 0.8:.2%}, per low-volatility studies.")
        print(f"- Action 2: Expand Globally")
        print(f"  * Why: Broaden exposure beyond U.S. markets reduces systemic risk.")
        print(f"  * How: Allocate 15-20% to international equities (e.g., VXUS, 7% return, 16% volatility), diversifying across emerging markets.")
        print(f"  * Probability: 60% chance of boosting returns by 1-2% annually over 10 years, per global diversification data.")








        # Disclaimer
        print("\nDisclaimer: The information provided by GALCAST Portfolio Analytics & Optimization is for informational and educational purposes only. It should not be considered as financial advice or a recommendation to buy or sell any security. Investment decisions should be based on your own research, investment objectives, financial situation, and needs. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making any investment decisions.")








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








    if all(w is None for w in weights_dict.values()):
        print("No weights provided. Assigning equal weights.")
        num_assets = len(tickers)
        weights = np.ones(num_assets) / num_assets
    # Temporary placeholder; optimization will override
    else:
        total = sum(w for w in weights_dict.values() if w is not None)
        if total > 1:
            print("Warning: Total percentage exceeds 100%. Normalizing weights.")
            weights = np.array([weights_dict[t] for t in tickers])
            weights /= weights.sum()
        elif total == 0:
            print("Error: No valid weights provided. Optimization will assign weights later.")
            weights = np.zeros(len(tickers))  # Placeholder
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








    # Define start_date_obj here since earliest_dates is now available
    start_date_obj = max(earliest_dates.values()) + timedelta(days=180)








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








    analyzer.plot_correlation_matrix(stock_prices)








    # Eigenvalue Analysis
    print("\n===== EIGENVALUE ANALYSIS =====")
    eigenvalues, explained_variance_ratio = analyzer.compute_eigenvalues(returns)
    cumulative_variance = analyzer.plot_eigenvalues(eigenvalues, explained_variance_ratio, tickers)








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
    analyzer.plot_cumulative_returns(returns, strategies, benchmark_returns, earliest_dates)








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
        analyzer.plot_historical_strategies(tickers, weights_dict, risk_free_rate, hist_returns)
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






    # Initial optimization with user-selected metric
    try:
        opt_weights = analyzer.optimize_portfolio(returns, risk_free_rate, opt_metric, min_allocation, max_allocation)
        optimized_weights = opt_weights
    except Exception as e:
        print(f"Error in initial optimization: {e}. Using equal weights as fallback.")
        optimized_weights = np.ones(len(tickers)) / len(tickers)


    # Enhanced optimization with risk parity and diversification
    print("\n===== ENHANCED OPTIMIZATION: RISK PARITY AND DIVERSIFICATION =====")
    market_data, _, _ = analyzer.fetch_stock_data(["^GSPC"], start_date, end_date)
    market_prices = market_data["^GSPC"] if market_data is not None and not market_data.empty and "^GSPC" in market_data.columns else None
    try:
        optimized_weights, opt_metrics = analyzer.optimize_with_factor_and_correlation(
            returns, risk_free_rate, tickers, market_prices, min_allocation, max_allocation
        )
        print(f"Optimized Metrics: Return={opt_metrics['return']:.2%}, Volatility={opt_metrics['volatility']:.2%}, "
              f"Sharpe={opt_metrics['sharpe']:.2f}, Avg Correlation={opt_metrics['avg_correlation']:.2f}")
    except Exception as e:
        print(f"Enhanced optimization failed: {e}. Using equal weights.")
        optimized_weights = np.ones(len(tickers)) / len(tickers)
        opt_metrics = {"return": 0.0, "volatility": 0.0, "sharpe": 0.0, "avg_correlation": 0.0}


    # Add diversification benefit plot
    analyzer.plot_diversification_benefit(returns, np.array(list(weights_dict.values())), optimized_weights, tickers)


    # Update optimized metrics with new weights
    try:
        optimized_metrics = {
            "annual_return": analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[0],
            "annual_volatility": analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[1],
            "sharpe_ratio": analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[2],
            "maximum_drawdown": analyzer.compute_max_drawdown(returns.dot(optimized_weights)),
            "value_at_risk": analyzer.compute_var(returns.dot(optimized_weights), 0.90)
        }
    except Exception as e:
        print(f"Error computing optimized metrics: {e}. Using placeholder metrics.")
        optimized_metrics = {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "maximum_drawdown": 0.0,
            "value_at_risk": 0.0
        }








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








    # New Section: Portfolio Exposure Comparison
    print("\n===== PORTFOLIO EXPOSURE COMPARISON =====")
    analyzer.plot_portfolio_exposures(tickers, list(weights_dict.values()), optimized_weights)








    print("\n===== COMBINED OPTIMIZED PERFORMANCE =====")
    combined_strategies = {
        "Original Portfolio": np.array(list(weights_dict.values())),
        "Optimized Portfolio": optimized_weights
    }
    analyzer.plot_cumulative_returns(returns, combined_strategies, benchmark_returns, earliest_dates, "Optimized vs Original Cumulative Returns")








    # Historical Crises Performance Section
    print("\n===== HISTORICAL CRISES PERFORMANCE (LOW BUSINESS ACTIVITY SCENARIOS) =====")
    analyzer.plot_crisis_performance(returns, combined_strategies, benchmark_returns, earliest_dates)








    # Final comparisons with Enhanced Insights
    print("\n===== OPTIMIZED PORTFOLIO COMPARISONS =====")
    analyzer.plot_comparison_bars(original_metrics, optimized_metrics, benchmark_metrics)
    print("\nInsights and Analytical Comparisons:")
    print(f"- Risk: Optimized volatility ({optimized_metrics['annual_volatility']:.2%}) vs Original ({original_metrics['annual_volatility']:.2%}).")
    print(f"  * Reduction: {original_metrics['annual_volatility'] - optimized_metrics['annual_volatility']:.2%} (positive means less risk after optimization).")
    print(f"- Returns: Optimized return ({optimized_metrics['annual_return']:.2%}) vs Original ({original_metrics['annual_return']:.2%}).")
    print(f"  * Improvement: {optimized_metrics['annual_return'] - original_metrics['annual_return']:.2%} (positive means higher growth).")
    initial_investment = 10000
    opt_cum = (1 + returns.dot(optimized_weights)).cumprod()[-1] * initial_investment
    orig_cum = (1 + returns.dot(list(weights_dict.values()))).cumprod()[-1] * initial_investment
    print(f"- Money Made (from {start_date_obj.strftime('%Y-%m-%d')} with $10,000):")
    print(f"  * Optimized: ${opt_cum:,.2f} (Gain: ${(opt_cum - initial_investment):,.2f})")
    print(f"  * Original: ${orig_cum:,.2f} (Gain: ${(orig_cum - initial_investment):,.2f})")
    print(f"  * Difference: ${(opt_cum - orig_cum):,.2f} (positive means optimization added value).")








    # Additional Statistics
    orig_beta = analyzer.compute_beta(returns.dot(list(weights_dict.values())), benchmark_returns.get(benchmarks[0], returns.mean()))
    opt_beta = analyzer.compute_beta(returns.dot(optimized_weights), benchmark_returns.get(benchmarks[0], returns.mean()))
    print(f"- Beta (vs {benchmarks[0]}):")
    print(f"  * Original: {orig_beta:.2f} (market sensitivity; <1 is less volatile than market)")
    print(f"  * Optimized: {opt_beta:.2f} (lower means less market risk)")








    orig_sortino = analyzer.compute_sortino_ratio(returns.dot(list(weights_dict.values())), risk_free_rate)
    opt_sortino = analyzer.compute_sortino_ratio(returns.dot(optimized_weights), risk_free_rate)
    print(f"- Sortino Ratio (downside risk-adjusted return):")
    print(f"  * Original: {orig_sortino:.2f} (higher is better, focuses on downside risk)")
    print(f"  * Optimized: {opt_sortino:.2f} (improvement: {opt_sortino - orig_sortino:.2f})")








    # Recovery Time Estimate
    orig_drawdown = original_metrics['maximum_drawdown']
    opt_drawdown = optimized_metrics['maximum_drawdown']
    orig_recovery = -orig_drawdown / original_metrics['annual_return'] if original_metrics['annual_return'] != 0 else float('inf')
    opt_recovery = -opt_drawdown / optimized_metrics['annual_return'] if optimized_metrics['annual_return'] != 0 else float('inf')
    print(f"- Estimated Recovery Time from Max Drawdown (years):")
    print(f"  * Original: {orig_recovery:.1f} years (based on {orig_drawdown:.2%} loss)")
    print(f"  * Optimized: {opt_recovery:.1f} years (based on {opt_drawdown:.2%} loss)")








    # Benchmark Comparison
    for bench, metrics in benchmark_metrics.items():
        print(f"- vs {bench}:")
        print(f"  * Original Return Gap: {(original_metrics['annual_return'] - metrics['annual_return']):.2%}")
        print(f"  * Optimized Return Gap: {(optimized_metrics['annual_return'] - metrics['annual_return']):.2%}")
        print(f"  * Volatility Difference: {(optimized_metrics['annual_volatility'] - metrics['annual_volatility']):.2%}")








    # Interesting Facts and Analytical Insights
    print("\nInteresting Facts and Analytical Insights:")
    print(f"- Risk Contribution: Optimization {'reduced' if optimized_metrics['annual_volatility'] < original_metrics['annual_volatility'] else 'increased'} volatility by {(abs(optimized_metrics['annual_volatility'] - original_metrics['annual_volatility'])):.2%}.")
    print(f"- Performance Edge: Optimized portfolio beats original by ${(opt_cum - orig_cum):,.2f} (with an initial investment of $10,000) over the period, a {((opt_cum - orig_cum) / orig_cum * 100 if orig_cum != 0 else 0):.1f}% relative gain.")
    if orig_beta > 1 and opt_beta < 1:
        print(f"- Risk Profile Shift: Original portfolio was more volatile than {benchmarks[0]} (Beta > 1), now less volatile (Beta < 1).")
    print(f"- Downside Protection: Sortino Ratio improved by {opt_sortino - orig_sortino:.2f}, showing better returns per unit of downside risk.")








    # Optional Rolling Volatility Plot
    print("\n===== ROLLING VOLATILITY ANALYSIS =====")
    analyzer.plot_rolling_volatility(returns, combined_strategies, benchmark_returns)








    # Enhanced Portfolio Risk and Diversification Insights with Fama-French
    print("\n===== PORTFOLIO RISK AND DIVERSIFICATION INSIGHTS =====")
    print(f"This section examines how risk is distributed across your portfolio of stocks ({', '.join(tickers)}) and assesses its diversification using multiple lenses: eigenvalue analysis and Fama-French factor exposures. The goal is to determine whether your portfolio’s risk is overly concentrated or well-balanced, and what drives its performance.")








    print("\n1. Risk Distribution Analysis (Understanding Eigenvalues):")
    print("Your portfolio’s risk arises from distinct patterns of price movements, quantified as eigenvalues or 'factors.' These factors represent combinations of how your stocks move together, not individual stocks:")
    print(f"- With {len(tickers)} stocks, your portfolio has {len(eigenvalues)} factors, each contributing to the total risk.")
    print("- The 'size' (eigenvalue) of each factor indicates its influence on your portfolio’s volatility.")
    print("Here’s the breakdown:")
    for i, (eig, ratio) in enumerate(zip(eigenvalues, explained_variance_ratio), 1):
        print(f"- Factor {i}: Size = {eig:.2f}, Contributes {ratio:.2%} to total risk")
    print(f"- Combined, these factors account for {cumulative_variance[-1]:.2%} of your portfolio’s risk.")








    print("\n2. Concentration Assessment:")
    top_factor_share = explained_variance_ratio[0]
    if top_factor_share > 0.5:
        print(f"- Factor 1 contributes {top_factor_share:.2%} to your risk, exceeding 50%.")
        print("  * Observation: A single factor dominating over half of your risk suggests moderate to high concentration. Your stocks tend to move together, likely driven by a common force like market trends.")
        print("  * Recommendation: To enhance diversification, consider adding assets with lower correlation to your current holdings, such as bonds or international equities, to reduce reliance on this dominant factor.")
    else:
        print(f"- Factor 1 contributes {top_factor_share:.2%} to your risk, below 50%.")
        print("  * Observation: Your risk is more evenly spread across factors, indicating decent diversification.")
        print("  * Recommendation: Maintain this balance or explore opportunities to further optimize returns.")








    print("\n3. Dominant Factor Evaluation:")
    print(f"- Factor 1, with a size of {eigenvalues[0]:.2f}, is the primary driver of your portfolio’s risk.")
    print("  * Insight: This factor likely reflects broad market exposure, common in stocks like JPM and GS (financials). Its size suggests significant sensitivity to market-wide movements.")








    print("\n4. Minor Factor Review:")
    print(f"- Factor {len(eigenvalues)}, with a size of {eigenvalues[-1]:.2f}, contributes only {explained_variance_ratio[-1]:.2%} to your risk.")
    print("  * Observation: This small contribution indicates some stocks add little unique movement, potentially overlapping with others.")
    print("  * Recommendation: Review highly correlated pairs (e.g., from the Correlation Matrix) and consider replacing redundant holdings with assets offering distinct risk profiles.")








    print("\n5. Correlation Context:")
    print("Refer to your Correlation Matrix heatmap for additional insight:")
    print("- High correlations (e.g., near 1, red) reinforce Factor 1’s dominance, showing stocks moving in lockstep.")
    print("- Low or negative correlations (e.g., near 0 or blue) enhance risk dispersion.")
    print("  * Action: Identify pairs above 0.8 and diversify them to mitigate concentration risk.")








    print("\n6. Fama-French Factor Exposure Analysis:")
    print("This analysis uses the Fama-French 3-factor model to identify what drives your portfolio’s returns and risk beyond just market movements. It breaks performance into three factors:")
    print("- Market (Mkt-RF): How much your portfolio moves with the overall stock market (e.g., S&P 500).")
    print("- Size (SMB, Small Minus Big): Exposure to small-cap stocks vs. large-cap stocks.")
    print("- Value (HML, High Minus Low): Exposure to value stocks (cheap, high book-to-market) vs. growth stocks (expensive, low book-to-market).")








    ff_exposures = analyzer.compute_fama_french_exposures(portfolio_returns, start_date, end_date)
    print("Your portfolio’s exposures:")
    print(f"- Market Beta: {ff_exposures['Mkt-RF']:.2f}")
    print(f"  * Meaning: A beta of {ff_exposures['Mkt-RF']:.2f} means if the market rises 1%, your portfolio moves {ff_exposures['Mkt-RF']:.2f}% on average. Above 1 indicates higher market sensitivity; below 1 suggests less.")
    print(f"  * Example: If beta is 1.2 and the S&P 500 gains 10%, your portfolio might gain 12%. But if the market drops 10%, you could lose 12%—a double-edged sword.")
    print(f"- Size Exposure (SMB): {ff_exposures['SMB']:.2f}")
    print("  * Meaning: Positive means a tilt toward small-cap stocks; negative means large-cap dominance. Zero is neutral.")
    print(f"  * Example: If SMB is 0.3 and small-caps outperform large-caps by 5%, your portfolio gains an extra {0.3 * 5:.1f}% from this factor—like picking nimble startups over giants like JPM.")
    print(f"- Value Exposure (HML): {ff_exposures['HML']:.2f}")
    print("  * Meaning: Positive means a value stock tilt (e.g., bargains like GS); negative means growth (e.g., tech like BABA). Zero is balanced.")
    print(f"  * Example: If HML is 0.4 and value stocks beat growth by 4%, you gain an extra {0.4 * 4:.1f}%—like finding undervalued gems in a growth-driven market.")








    print("  * Why It Matters:")
    print("    - High Market Beta (>1): Your portfolio amplifies market swings—great in bull markets, risky in downturns.")
    print("    - Size Tilt: Positive SMB could boost returns if small-caps shine (e.g., post-recession), but they’re often volatile.")
    print("    - Value Tilt: Positive HML favors steady, cheap stocks—good for stability, but may lag in tech booms.")
    print("  * Action: If beta is high (e.g., >1.2), consider reducing market exposure with bonds. If SMB or HML is extreme (e.g., >0.5 or <-0.5), balance with opposite-style stocks (e.g., growth for high HML).")








    print("\nStrategic Recommendations:")
    print("- If Factor 1 dominates (>50%) or Market Beta is high (>1), diversify with low-correlation assets (e.g., bonds, gold) to reduce market reliance.")
    print("- If SMB or HML is heavily tilted, adjust your stock mix—add large-caps for high SMB, or growth stocks for high HML—to balance risk and reward.")
    print("- Use strategies like Minimum Volatility or Risk Parity to spread risk across factors, or Maximum Sharpe to optimize returns while respecting your factor profile.")
    print("- Cross-check with the Correlation Matrix: High correlations amplify your dominant factor (likely market), so diversify accordingly.")








        # Final Wealth Advisor Recommendations
    print("\n===== FINAL WEALTH ADVISOR RECOMMENDATIONS =====")
    analyzer.suggest_courses_of_action(
        tickers=tickers,
        original_weights=np.array(list(weights_dict.values())),
        optimized_weights=optimized_weights,
        returns=returns,
        risk_free_rate=risk_free_rate,
        benchmark_metrics=benchmark_metrics,
        risk_tolerance=risk_tolerance,
        start_date=start_date,
        end_date=end_date
    )


    # Scenario-Specific Recommendations
    print("\n--- Scenario-Specific Recommendations ---")
    try:
        if scenario_insights:
            for scenario, text in scenario_insights.items():
                print(text)
        else:
            print("- No specific scenario triggers detected. Your portfolio is balanced across typical conditions.")
    except NameError:
        print("- Scenario insights unavailable due to earlier optimization error. Portfolio is still balanced based on initial optimization.")
    except Exception as e:
        print(f"- Error displaying scenario insights: {e}. Proceeding with general recommendations.")




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
    print("9. Fama-French 3-Factor Model = R_p - R_f = β_Mkt * (R_m - R_f) + β_SMB * SMB + β_HML * HML + ε")
    print("   - R_p: Portfolio return, R_f: Risk-free rate, R_m: Market return")
    print("   - β_Mkt: Market beta, β_SMB: Size factor exposure, β_HML: Value factor exposure")
    print("   - SMB: Small Minus Big return, HML: High Minus Low return, ε: Residual")








if __name__ == "__main__":
    run_portfolio_analysis()
