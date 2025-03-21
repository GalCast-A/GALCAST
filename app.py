# app.py
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
        """Compute downside deviation for Sortino Ratio."""
        negative_returns = returns[returns < target]
        if len(negative_returns) == 0:
            return 0
        return np.std(negative_returns) * np.sqrt(252)

    def portfolio_performance(self, weights, returns):
        """Calculate portfolio performance metrics including Sharpe and Sortino Ratios."""
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        
        portfolio_daily_returns = returns.dot(weights)
        downside_dev = self.compute_downside_deviation(portfolio_daily_returns, target=0)
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_dev if downside_dev != 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio, sortino_ratio

    def optimize_portfolio(self, returns, objective='sharpe', max_allocation=1):
        """Optimize portfolio based on different objectives."""
        num_assets = returns.shape[1]
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, max_allocation) for _ in range(num_assets))

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
            weights = optimal_results.x
            weights[weights < 0.001] = 0
            weights = weights / np.sum(weights) if np.sum(weights) != 0 else weights
            return weights
        except Exception as e:
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
        """Plot correlation matrix and return as base64."""
        corr_matrix = prices.corr()
        plt.figure(figsize=fig_size)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Portfolio Stocks")
        plt.tight_layout()
        return self.plot_to_base64(plt)

    def adjust_weights_over_time(self, tickers, weights, returns):
        """Adjust weights dynamically."""
        ticker_list = list(tickers)
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
        """Plot cumulative returns and return as base64."""
        earliest_date = max([d for d in earliest_dates.values()] + [pd.to_datetime(user_start_date)])
        adjusted_returns = returns.loc[earliest_date:]
        if adjusted_returns.empty:
            return None, None

        plt.figure(figsize=fig_size)
        tickers = returns.columns
        weights_df = self.adjust_weights_over_time(tickers, weights, adjusted_returns)
        portfolio_returns = (adjusted_returns * weights_df).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod() - 1
        plt.plot(cumulative, label='Current Portfolio', linewidth=2)

        if optimized_weights is not None:
            opt_weights_df = self.adjust_weights_over_time(tickers, optimized_weights, adjusted_returns)
            opt_returns = (adjusted_returns * opt_weights_df).sum(axis=1)
            opt_cumulative = (1 + opt_returns).cumprod() - 1
            plt.plot(opt_cumulative, label='Optimized Portfolio (Max Sharpe)', linewidth=2, linestyle='--')

        if sortino_optimized_weights is not None:
            sortino_opt_weights_df = self.adjust_weights_over_time(tickers, sortino_optimized_weights, adjusted_returns)
            sortino_opt_returns = (adjusted_returns * sortino_opt_weights_df).sum(axis=1)
            sortino_opt_cumulative = (1 + sortino_opt_returns).cumprod() - 1
            plt.plot(sortino_opt_cumulative, label='Optimized Portfolio (Max Sortino)', linewidth=2, linestyle='-.')

        if benchmark_returns is not None:
            if isinstance(benchmark_returns, dict):
                for key, series in benchmark_returns.items():
                    bench_series = series.loc[earliest_date:]
                    if not bench_series.empty:
                        bench_cumulative = (1 + bench_series).cumprod() - 1
                        plt.plot(bench_cumulative, label=f'Benchmark: {key}', linewidth=2)
            else:
                bench_series = benchmark_returns.loc[earliest_date:]
                if not bench_series.empty:
                    bench_cumulative = (1 + bench_series).cumprod() - 1
                    plt.plot(bench_cumulative, label='Benchmark', linewidth=2)

        plt.title(f'Cumulative Returns (From {earliest_date.strftime("%Y-%m-%d")})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        return self.plot_to_base64(plt), weights_df

    def plot_crisis_performance(self, returns, benchmark_returns_dict, results, start_date, end_date, earliest_dates, fig_size=(10, 6)):
        """Plot COVID-19 crisis performance and return as base64."""
        covid_period = ("2020-02-01", "2020-04-30")
        covid_start = pd.to_datetime("2020-02-01")
        fig, ax = plt.subplots(figsize=fig_size)
        
        tickers = returns.columns
        ticker_list = list(tickers)
        valid_tickers = [t for t in ticker_list if earliest_dates.get(t, covid_start) < covid_start]
        
        if valid_tickers:
            crisis_prices, error_tickers, _ = self.fetch_stock_data(valid_tickers, start=covid_period[0], end=covid_period[1])
            if crisis_prices is not None and not crisis_prices.empty:
                crisis_returns = self.compute_returns(crisis_prices).fillna(0)
                
                cur_weights_dict = results['current_portfolio']['weights']
                cur_weights = np.array([cur_weights_dict[t] for t in crisis_returns.columns])
                cur_weights_adjusted = cur_weights / cur_weights.sum() if cur_weights.sum() > 0 else cur_weights
                cur_portfolio_returns = crisis_returns.mul(cur_weights_adjusted, axis=1).sum(axis=1)
                if not cur_portfolio_returns.empty and cur_portfolio_returns.notna().any():
                    cur_cumulative = (1 + cur_portfolio_returns).cumprod() - 1
                    ax.plot(cur_cumulative.index, cur_cumulative, label="Original Portfolio (COVID)", linestyle='--', linewidth=2)

                opt_weights_dict = results['optimized_portfolio']['weights']
                opt_weights = np.array([opt_weights_dict[t] for t in crisis_returns.columns])
                opt_weights_adjusted = opt_weights / opt_weights.sum() if opt_weights.sum() > 0 else opt_weights
                opt_portfolio_returns = crisis_returns.mul(opt_weights_adjusted, axis=1).sum(axis=1)
                if not opt_portfolio_returns.empty and opt_portfolio_returns.notna().any():
                    opt_cumulative = (1 + opt_portfolio_returns).cumprod() - 1
                    ax.plot(opt_cumulative.index, opt_cumulative, label="Optimized Portfolio (Max Sharpe, COVID)", linestyle='-.', linewidth=2)

                sortino_opt_weights_dict = results['sortino_optimized_portfolio']['weights']
                sortino_opt_weights = np.array([sortino_opt_weights_dict[t] for t in crisis_returns.columns])
                sortino_opt_weights_adjusted = sortino_opt_weights / sortino_opt_weights.sum() if sortino_opt_weights.sum() > 0 else sortino_opt_weights
                sortino_opt_portfolio_returns = crisis_returns.mul(sortino_opt_weights_adjusted, axis=1).sum(axis=1)
                if not sortino_opt_portfolio_returns.empty and sortino_opt_portfolio_returns.notna().any():
                    sortino_opt_cumulative = (1 + sortino_opt_portfolio_returns).cumprod() - 1
                    ax.plot(sortino_opt_cumulative.index, sortino_opt_cumulative, label="Optimized Portfolio (Max Sortino, COVID)", linestyle=':', linewidth=2)

        for bench_ticker, bench_series in benchmark_returns_dict.items():
            bench_crisis = bench_series.loc[covid_period[0]:covid_period[1]]
            if not bench_crisis.empty:
                bench_cumulative = (1 + bench_crisis).cumprod() - 1
                ax.plot(bench_crisis.index, bench_cumulative, label=f"{bench_ticker} (COVID)", linestyle='-', linewidth=2)

        ax.set_title("COVID-19 Crisis Performance Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return self.plot_to_base64(plt)

    def analyze_portfolio(self, tickers, weights=None, start_date=None, end_date=None, benchmark='^GSPC'):
        """Analyze a portfolio of stocks."""
        if start_date is None:
            start_date = self.default_start_date
        if end_date is None:
            end_date = self.today_date
        if weights is None:
            weights = [1/len(tickers)] * len(tickers)
        weights = np.array(weights) / sum(weights)

        stock_prices, error_tickers, earliest_dates = self.fetch_stock_data(tickers, start=start_date, end=end_date)
        if stock_prices is None or stock_prices.empty:
            return {'error': 'No valid stock data available', 'error_tickers': error_tickers, 'earliest_dates': earliest_dates}
        
        returns = self.compute_returns(stock_prices)
        valid_tickers = list(returns.columns)
        if len(valid_tickers) < len(tickers):
            valid_idx = [i for i, t in enumerate(tickers) if t in valid_tickers]
            weights = np.array([weights[i] for i in valid_idx])
            weights = weights / np.sum(weights)
            tickers = valid_tickers
        
        if returns.shape[0] < 20:
            return {'error': 'Insufficient data for analysis', 'earliest_dates': earliest_dates}

        benchmark_prices, _, _ = self.fetch_stock_data([benchmark], start=start_date, end=end_date)
        benchmark_returns = self.compute_returns(benchmark_prices).iloc[:, 0] if benchmark_prices is not None and not benchmark_prices.empty else None
        
        portfolio_returns = returns.dot(weights)
        annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio = self.portfolio_performance(weights, returns)
        max_drawdown = self.compute_max_drawdown((1 + portfolio_returns).cumprod())
        
        if benchmark_returns is not None:
            bench_return, bench_vol, bench_sharpe, bench_sortino = self.portfolio_performance(np.array([1.0]), pd.DataFrame(benchmark_returns))
            benchmark_max_dd = self.compute_max_drawdown((1 + benchmark_returns).cumprod())
        else:
            bench_return = bench_vol = bench_sharpe = bench_sortino = benchmark_max_dd = None

        optimal_weights = self.optimize_portfolio(returns, objective='sharpe')
        opt_return, opt_volatility, opt_sharpe, opt_sortino = self.portfolio_performance(optimal_weights, returns)
        opt_max_drawdown = self.compute_max_drawdown((1 + returns.dot(optimal_weights)).cumprod())
        opt_var = self.compute_var(returns.dot(optimal_weights))

        sortino_optimal_weights = self.optimize_portfolio(returns, objective='sortino')
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
        """Plot efficient frontier and return as base64."""
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
        
        plt.figure(figsize=fig_size)
        plt.scatter(all_volatilities, all_returns, c=all_sharpe_ratios, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(opt_vol, opt_return, c='red', marker='o', s=100, edgecolors='black', label='Optimal Portfolio (Max Sharpe)')
        plt.scatter(sortino_opt_vol, sortino_opt_return, c='purple', marker='*', s=100, edgecolors='black', label='Optimal Portfolio (Max Sortino)')
        plt.scatter(min_vol_vol, min_vol_return, c='darkgreen', marker='^', s=100, edgecolors='black', label='Minimum Volatility')
        plt.scatter(eq_vol, eq_return, c='blue', marker='s', s=100, edgecolors='black', label='Equal Weight')
        plt.plot([0, opt_vol*1.5], [risk_free_rate, risk_free_rate + opt_sharpe*opt_vol*1.5], 'k--', label='Capital Market Line')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        text_str = (f"Optimal (Sharpe): Return={opt_return:.2%}, Vol={opt_vol:.2%}, SR={opt_sharpe:.2f}, Sortino={opt_sortino:.2f}    |    "
                    f"Optimal (Sortino): Return={sortino_opt_return:.2%}, Vol={sortino_opt_vol:.2%}, SR={sortino_opt_sharpe:.2f}, Sortino={sortino_opt_sortino:.2f}    |    "
                    f"MinVol: Return={min_vol_return:.2%}, Vol={min_vol_vol:.2%}, SR={min_vol_sharpe:.2f}, Sortino={min_vol_sortino:.2f}    |    "
                    f"Equal: Return={eq_return:.2%}, Vol={eq_vol:.2%}, SR={eq_sharpe:.2f}, Sortino={eq_sortino:.2f}")
        plt.figtext(0.5, 0.005, text_str, ha='center', fontsize=9, wrap=True)
        return self.plot_to_base64(plt)

    def plot_portfolio_weights(self, weights_dict, title="Portfolio Weights", fig_size=(10, 6)):
        """Plot portfolio weights and return as base64."""
        tickers = list(weights_dict.keys())
        weights = [weights_dict[ticker] for ticker in tickers]
        plt.figure(figsize=fig_size)
        bars = plt.bar(tickers, weights)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom')
        plt.title(title)
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self.plot_to_base64(plt)

def run_portfolio_analysis(tickers, weights=None, start_date=None, end_date=None, benchmark_tickers=['^GSPC']):
    """Run portfolio analysis and return results with plots as base64."""
    analyzer = PortfolioAnalyzer()
    results = analyzer.analyze_portfolio(tickers, weights, start_date, end_date, benchmark=benchmark_tickers[0])
    if 'error' in results:
        return results

    stock_prices, _, earliest_dates = analyzer.fetch_stock_data(tickers, start=start_date, end=end_date)
    returns_data = analyzer.compute_returns(stock_prices)

    benchmark_data = {}
    benchmark_returns_dict = {}
    for bench_ticker in benchmark_tickers:
        data, _, _ = analyzer.fetch_stock_data([bench_ticker], start=start_date, end=end_date)
        if data is not None and not data.empty:
            benchmark_data[bench_ticker] = data
            benchmark_returns_dict[bench_ticker] = analyzer.compute_returns(data).iloc[:, 0]

    plots = {
        'correlation_matrix': analyzer.plot_correlation_matrix(stock_prices),
        'efficient_frontier': analyzer.plot_efficient_frontier(returns_data),
        'current_weights': analyzer.plot_portfolio_weights(results['current_portfolio']['weights'], "Current Portfolio Weights"),
        'optimized_weights': analyzer.plot_portfolio_weights(results['optimized_portfolio']['weights'], "Optimized Portfolio Weights (Max Sharpe)"),
        'sortino_optimized_weights': analyzer.plot_portfolio_weights(results['sortino_optimized_portfolio']['weights'], "Optimized Portfolio Weights (Max Sortino)"),
        'cumulative_returns': analyzer.plot_cumulative_returns(
            returns_data, 
            np.array(list(results['current_portfolio']['weights'].values())),
            benchmark_returns=benchmark_returns_dict,
            optimized_weights=np.array(list(results['optimized_portfolio']['weights'].values())),
            sortino_optimized_weights=np.array(list(results['sortino_optimized_portfolio']['weights'].values())),
            earliest_dates=earliest_dates,
            user_start_date=start_date
        )[0],
        'crisis_performance': analyzer.plot_crisis_performance(
            returns_data, 
            benchmark_returns_dict, 
            results, 
            start_date, 
            end_date, 
            earliest_dates
        )
    }

    results['plots'] = {k: v for k, v in plots.items() if v is not None}
    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    tickers = data.get('tickers', [])
    weights = data.get('weights', None)
    start_date = data.get('start_date', None)
    end_date = data.get('end_date', None)
    benchmark_tickers = data.get('benchmark_tickers', ['^GSPC'])

    if not tickers or len(tickers) == 0:
        return jsonify({'error': 'No tickers provided'}), 400
    
    if weights and len(weights) != len(tickers):
        return jsonify({'error': 'Weights length must match tickers length'}), 400

    try:
        results = run_portfolio_analysis(tickers, weights, start_date, end_date, benchmark_tickers)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
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
        }
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)