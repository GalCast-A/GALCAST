import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

# Define request models using Pydantic for input validation
class TickerWeight(BaseModel):
    ticker: str
    weight: Optional[float] = None  # Weight as a percentage (e.g., 0.3 for 30%)

class PortfolioRequest(BaseModel):
    tickers: List[TickerWeight]  # List of tickers and their weights
    start_date: Optional[str] = None  # YYYY-MM-DD format
    benchmarks: Optional[List[str]] = None  # List of benchmark tickers (e.g., ["^GSPC"])
    risk_tolerance: Optional[str] = "medium"  # "low", "medium", "high"
    risk_free_rate: Optional[float] = None  # Custom risk-free rate (e.g., 0.0425 for 4.25%)
    optimization_metric: Optional[str] = "sharpe"  # "sharpe", "sortino", "max_drawdown", "volatility", "value_at_risk"
    weight_strategy: Optional[str] = None  # "risk_parity", "equal", "inverse_volatility", None
    min_allocation: Optional[float] = 0.0  # Minimum allocation per stock (e.g., 0.05 for 5%)
    max_allocation: Optional[float] = 1.0  # Maximum allocation per stock (e.g., 0.3 for 30%)

class PortfolioAnalyzer:
    def __init__(self):
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.default_start_date = (datetime.strptime(self.today_date, "%Y-%m-%d") - timedelta(days=3652)).strftime("%Y-%m-%d")
        self.data_cache = {}

    def fetch_treasury_yield(self):
        try:
            treasury_data = yf.download("^TNX", period="1d", interval="1d")['Close']
            if treasury_data.empty or not isinstance(treasury_data, pd.Series):
                return 0.04
            latest_yield = float(treasury_data.iloc[-1]) / 100
            return latest_yield
        except Exception:
            return 0.04

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
                return None, error_tickers, earliest_dates
            for ticker in stocks:
                if ticker not in stock_data.columns or stock_data[ticker].isna().all():
                    error_tickers[ticker] = "Data not available"
                else:
                    first_valid = stock_data[ticker].first_valid_index()
                    earliest_dates[ticker] = first_valid
            self.data_cache[cache_key] = (stock_data, error_tickers, earliest_dates)
            return stock_data, error_tickers, earliest_dates
        except Exception as e:
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
                return initial_weights
            weights = result.x
            weights[weights < 0.001] = 0
            weights /= weights.sum() if weights.sum() != 0 else 1
            return weights
        except Exception:
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
        return corr_matrix.to_dict()

    def print_efficient_frontier(self, returns, risk_free_rate, n_portfolios=1000):
        np.random.seed(42)
        n_assets = returns.shape[1]
        all_weights = np.zeros((n_portfolios, n_assets))
        all_returns = np.zeros(n_portfolios)
        all_volatilities = np.zeros(n_portfolios)
        all_sharpe_ratios = np.zeros(n_portfolios)
        
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= weights.sum()
            all_weights[i, :] = weights
            port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
            all_returns[i] = port_return
            all_volatilities[i] = port_vol
            all_sharpe_ratios[i] = port_sharpe

        strategies = {
            "Max Sharpe": self.optimize_portfolio(returns, risk_free_rate, "sharpe"),
            "Max Sortino": self.optimize_portfolio(returns, risk_free_rate, "sortino"),
            "Min Max Drawdown": self.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
            "Min Volatility": self.optimize_portfolio(returns, risk_free_rate, "volatility"),
            "Min Value at Risk": self.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
        }
        
        strategy_metrics = {}
        for name, weights in strategies.items():
            port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
            strategy_metrics[name] = {
                "return": port_return,
                "volatility": port_vol,
                "sharpe": port_sharpe
            }

        return {
            "simulated_portfolios": {
                "average_annualized_return": float(np.mean(all_returns)),
                "average_annualized_volatility": float(np.mean(all_volatilities)),
                "average_sharpe_ratio": float(np.mean(all_sharpe_ratios)),
                "best_sharpe_ratio": float(np.max(all_sharpe_ratios))
            },
            "optimized_portfolios": strategy_metrics
        }

    def print_cumulative_returns(self, returns, weights_dict, benchmark_returns, earliest_dates, title="Cumulative Returns"):
        start_date = max(earliest_dates.values()) + timedelta(days=180)
        adjusted_returns = returns.loc[start_date:]
        result = {title: {}}
        for label, weights in weights_dict.items():
            portfolio_returns = adjusted_returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod() - 1
            result[title][label] = {
                "final_cumulative_return": float(cumulative.iloc[-1]),
                "average_daily_return": float(portfolio_returns.mean()),
                "std_daily_returns": float(portfolio_returns.std())
            }
        for bench_ticker, bench_ret in benchmark_returns.items():
            bench_cum = (1 + bench_ret.loc[start_date:]).cumprod() - 1
            result[title][bench_ticker] = {
                "final_cumulative_return": float(bench_cum.iloc[-1]),
                "average_daily_return": float(bench_ret.loc[start_date:].mean()),
                "std_daily_returns": float(bench_ret.loc[start_date:].std())
            }
        return result

    def print_crisis_performance(self, returns, weights_dict, benchmark_returns):
        crisis_start = pd.to_datetime("2020-02-01")
        crisis_end = pd.to_datetime("2020-04-30")
        available_start = returns.index[returns.index >= crisis_start].min()
        available_end = returns.index[returns.index <= crisis_end].max()
        
        if pd.isna(available_start) or pd.isna(available_end) or available_start > available_end:
            return {"error": "Crisis period data not available within the analysis range."}
        
        crisis_returns = returns.loc[available_start:available_end]
        result = {
            "COVID-19 Crisis Performance": {
                "period": f"{available_start.strftime('%Y-%m-%d')} to {available_end.strftime('%Y-%m-%d')}",
                "metrics": {}
            }
        }
        for label, weights in weights_dict.items():
            portfolio_returns = crisis_returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod() - 1
            result["COVID-19 Crisis Performance"]["metrics"][label] = {
                "final_cumulative_return": float(cumulative.iloc[-1]),
                "average_daily_return": float(portfolio_returns.mean()),
                "std_daily_returns": float(portfolio_returns.std())
            }
        for bench_ticker, bench_ret in benchmark_returns.items():
            bench_crisis_ret = bench_ret.loc[available_start:available_end]
            bench_cum = (1 + bench_crisis_ret).cumprod() - 1
            result["COVID-19 Crisis Performance"]["metrics"][bench_ticker] = {
                "final_cumulative_return": float(bench_cum.iloc[-1]),
                "average_daily_return": float(bench_crisis_ret.mean()),
                "std_daily_returns": float(bench_crisis_ret.std())
            }
        return result

    def print_historical_strategies(self, tickers, weights_dict, risk_free_rate, hist_returns=None):
        start_date = "2015-03-24"
        end_date = self.today_date
        if hist_returns is None:
            data, _, _ = self.fetch_stock_data(tickers, start_date, end_date)
            if data is None or data.empty:
                return {"error": "Could not fetch historical data for your portfolio."}
            hist_returns = self.compute_returns(data)
        
        strategies = {
            "Original Portfolio": np.array(list(weights_dict.values())),
            "Max Sharpe": self.optimize_portfolio(hist_returns, risk_free_rate, "sharpe"),
            "Max Sortino": self.optimize_portfolio(hist_returns, risk_free_rate, "sortino"),
            "Min Max Drawdown": self.optimize_portfolio(hist_returns, risk_free_rate, "max_drawdown"),
            "Min Volatility": self.optimize_portfolio(hist_returns, risk_free_rate, "volatility"),
            "Min Value at Risk": self.optimize_portfolio(hist_returns, risk_free_rate, "value_at_risk")
        }
        
        annual_metrics = {}
        for name, weights in strategies.items():
            portfolio_returns = hist_returns.dot(weights)
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            max_drawdown = self.compute_max_drawdown(portfolio_returns)
            annual_metrics[name] = {
                'annualized_returns': float(annual_return),
                'annualized_volatility': float(annual_volatility),
                'max_drawdown': float(max_drawdown)
            }

        return {
            "historical_performance": {
                "period": f"{start_date} to {end_date}",
                "tickers": tickers,
                "metrics": annual_metrics
            }
        }

    def print_weight_allocation_strategies(self, tickers, weights_dict):
        start_date = "2015-03-24"
        end_date = self.today_date
        
        data, _, _ = self.fetch_stock_data(tickers, start_date, end_date)
        if data is None or data.empty:
            return {"error": "Could not fetch historical data for your portfolio."}
        hist_returns = self.compute_returns(data)
        
        strategies = {
            "None (Original)": np.array(list(weights_dict.values())),
            "Equal Weight": self.apply_weight_strategy(hist_returns, "equal"),
            "Risk Parity": self.apply_weight_strategy(hist_returns, "risk_parity"),
            "Inverse Volatility": self.apply_weight_strategy(hist_returns, "inverse_volatility")
        }
        
        annual_metrics = {}
        for name, weights in strategies.items():
            portfolio_returns = hist_returns.dot(weights)
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            max_drawdown = self.compute_max_drawdown(portfolio_returns)
            annual_metrics[name] = {
                'annualized_returns': float(annual_return),
                'annualized_volatility': float(annual_volatility),
                'max_drawdown': float(max_drawdown)
            }

        return {
            "weight_allocation_strategies": {
                "period": f"{start_date} to {end_date}",
                "tickers": tickers,
                "metrics": annual_metrics
            }
        }

    def print_comparison_bars(self, original_metrics, optimized_metrics, benchmark_metrics):
        metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "maximum_drawdown", "value_at_risk"]
        labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Value at Risk (90%)"]
        result = {"portfolio_comparison": {}}
        for metric, label in zip(metrics, labels):
            comparison = {
                "original": float(original_metrics[metric]),
                "optimized": float(optimized_metrics[metric])
            }
            if benchmark_metrics:
                for bench, bm in benchmark_metrics.items():
                    comparison[bench] = float(bm[metric])
            result["portfolio_comparison"][label] = comparison
        return result

@app.post("/analyze_portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    analyzer = PortfolioAnalyzer()
    
    # Extract tickers and weights
    tickers = [tw.ticker for tw in request.tickers]
    weights_dict = {tw.ticker: tw.weight for tw in request.tickers}
    if not tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")

    # Validate tickers
    for ticker in tickers:
        try:
            test_data = yf.download(ticker, period="1mo")
            if test_data.empty:
                raise HTTPException(status_code=400, detail=f"No data found for ticker {ticker}.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error retrieving data for {ticker}: {str(e)}")

    # Assign weights
    if all(w is None for w in weights_dict.values()):
        weights = np.ones(len(tickers)) / len(tickers)
    else:
        total = sum(w for w in weights_dict.values() if w is not None)
        if total > 1:
            weights = np.array([weights_dict[t] for t in tickers])
            weights /= weights.sum()
        elif total == 0:
            weights = np.ones(len(tickers)) / len(tickers)
        else:
            remaining = 1 - total
            num_none = sum(1 for w in weights_dict.values() if w is None)
            weights = np.array([weights_dict[t] if weights_dict[t] is not None else remaining / num_none for t in tickers])
    weights_dict = dict(zip(tickers, weights))

    # Validate start date
    start_date = request.start_date or analyzer.default_start_date
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        start_date = analyzer.default_start_date
    end_date = analyzer.today_date

    # Set benchmarks
    benchmarks = request.benchmarks or ["^GSPC"]

    # Set risk tolerance and risk-free rate
    risk_tolerance = request.risk_tolerance
    treasury_yield = analyzer.fetch_treasury_yield()
    risk_free_rate = request.risk_free_rate if request.risk_free_rate is not None else treasury_yield

    # Fetch data
    stock_prices, _, earliest_dates = analyzer.fetch_stock_data(tickers, start_date, end_date)
    if stock_prices is None or stock_prices.empty:
        raise HTTPException(status_code=400, detail="No valid stock data available.")
    returns = analyzer.compute_returns(stock_prices)
    if returns.empty:
        raise HTTPException(status_code=400, detail="No valid returns data.")
    portfolio_returns = returns.dot(list(weights_dict.values()))

    # Fetch benchmark data
    benchmark_returns = {}
    benchmark_metrics = {}
    for bench in benchmarks:
        bench_data, _, _ = analyzer.fetch_stock_data([bench], start_date, end_date)
        if bench_data is not None and not bench_data.empty:
            bench_returns = analyzer.compute_returns(bench_data)[bench]
            benchmark_returns[bench] = bench_returns
            benchmark_metrics[bench] = {
                "annual_return": float(bench_returns.mean() * 252),
                "annual_volatility": float(bench_returns.std() * np.sqrt(252)),
                "sharpe_ratio": float(analyzer.portfolio_performance(np.array([1.0]), pd.DataFrame(bench_returns), risk_free_rate)[2]),
                "maximum_drawdown": float(analyzer.compute_max_drawdown(bench_returns)),
                "value_at_risk": float(analyzer.compute_var(bench_returns, 0.90))
            }

    # Compute original portfolio metrics
    original_metrics = {
        "annual_return": float(portfolio_returns.mean() * 252),
        "annual_volatility": float(portfolio_returns.std() * np.sqrt(252)),
        "sharpe_ratio": float(analyzer.portfolio_performance(np.array(list(weights_dict.values())), returns, risk_free_rate)[2]),
        "maximum_drawdown": float(analyzer.compute_max_drawdown(portfolio_returns)),
        "value_at_risk": float(analyzer.compute_var(portfolio_returns, 0.90))
    }

    # Portfolio Analysis
    result = {
        "analysis_period": f"{start_date} to {end_date}",
        "original_portfolio_metrics": original_metrics,
        "benchmark_metrics": benchmark_metrics,
        "issues": []
    }

    # Identify issues
    for bench_ticker, bench_metrics in benchmark_metrics.items():
        bench_return = bench_metrics['annual_return']
        if original_metrics['annual_return'] < bench_return:
            return_diff = (bench_return - original_metrics['annual_return']) * 100
            result["issues"].append({
                "metric": f"Annual Return vs {bench_ticker}",
                "description": f"Your portfolio’s Annual Return ({original_metrics['annual_return']:.2%}) is lower than the {bench_ticker} benchmark ({bench_return:.2%}) by {return_diff:.1f} percentage points."
            })

    if original_metrics['annual_volatility'] > 0.20:
        result["issues"].append({
            "metric": "Annual Volatility",
            "description": f"Your portfolio’s Annual Volatility ({original_metrics['annual_volatility']:.2%}) is high, above the typical range of 10-20% for a diversified portfolio."
        })

    if original_metrics['sharpe_ratio'] < 1:
        result["issues"].append({
            "metric": "Sharpe Ratio",
            "description": f"Your Sharpe Ratio ({original_metrics['sharpe_ratio']:.2f}) is below 1, which is considered poor."
        })

    if original_metrics['maximum_drawdown'] < -0.20:
        result["issues"].append({
            "metric": "Maximum Drawdown",
            "description": f"Your Maximum Drawdown ({original_metrics['maximum_drawdown']:.2%}) is concerning, indicating significant historical losses."
        })

    if original_metrics['value_at_risk'] < -0.05:
        result["issues"].append({
            "metric": "Value at Risk (VaR, 90%)",
            "description": f"Your Value at Risk ({original_metrics['value_at_risk']:.2%}) is high, indicating a significant potential for daily losses."
        })

    # Correlation Matrix
    result["correlation_matrix"] = analyzer.print_correlation_matrix(stock_prices)

    # Cumulative Returns of Strategies
    strategies = {
        "Original Portfolio": np.array(list(weights_dict.values())),
        "Max Sharpe": analyzer.optimize_portfolio(returns, risk_free_rate, "sharpe"),
        "Max Sortino": analyzer.optimize_portfolio(returns, risk_free_rate, "sortino"),
        "Min Max Drawdown": analyzer.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
        "Min Volatility": analyzer.optimize_portfolio(returns, risk_free_rate, "volatility"),
        "Min Value at Risk": analyzer.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
    }
    result.update(analyzer.print_cumulative_returns(returns, strategies, benchmark_returns, earliest_dates))

    # Efficient Frontier
    result["efficient_frontier"] = analyzer.print_efficient_frontier(returns, risk_free_rate)

    # Historical Performance
    hist_start_date = "2015-03-24"
    hist_data, _, _ = analyzer.fetch_stock_data(tickers, hist_start_date, end_date)
    if hist_data is not None and not hist_data.empty:
        hist_returns = analyzer.compute_returns(hist_data)
        result["historical_strategies"] = analyzer.print_historical_strategies(tickers, weights_dict, risk_free_rate, hist_returns)
    else:
        result["historical_strategies"] = {"error": "Historical data unavailable."}

    # Weight Allocation Strategies
    result["weight_allocation_strategies"] = analyzer.print_weight_allocation_strategies(tickers, weights_dict)

    # Apply optimization and strategy
    opt_weights = analyzer.optimize_portfolio(returns, risk_free_rate, request.optimization_metric, request.min_allocation, request.max_allocation)
    if request.weight_strategy:
        strategy_weights = analyzer.apply_weight_strategy(returns, request.weight_strategy)
        optimized_weights = (opt_weights + strategy_weights) / 2
    else:
        optimized_weights = opt_weights

    optimized_metrics = {
        "annual_return": float(analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[0]),
        "annual_volatility": float(analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[1]),
        "sharpe_ratio": float(analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[2]),
        "maximum_drawdown": float(analyzer.compute_max_drawdown(returns.dot(optimized_weights))),
        "value_at_risk": float(analyzer.compute_var(returns.dot(optimized_weights), 0.90))
    }

    result["optimized_metrics"] = optimized_metrics
    result["optimized_weights"] = dict(zip(tickers, optimized_weights.tolist()))

    # Check if issues are addressed
    issues_addressed = []
    for issue in result["issues"]:
        metric = issue['metric']
        fixed = False
        if "Annual Return vs" in metric:
            bench_ticker = metric.split("vs ")[1].strip()
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
        issues_addressed.append({"metric": metric, "fixed": fixed})
    result["issues_addressed"] = issues_addressed

    # Combined Optimized Performance
    combined_strategies = {
        "Original Portfolio": np.array(list(weights_dict.values())),
        "Optimized Portfolio": optimized_weights
    }
    result.update(analyzer.print_cumulative_returns(returns, combined_strategies, benchmark_returns, earliest_dates, "Optimized vs Original Cumulative Returns"))

    # Final Comparisons
    result.update(analyzer.print_comparison_bars(original_metrics, optimized_metrics, benchmark_metrics))

    # Money Made
    start_date_obj = max(earliest_dates.values()) + timedelta(days=180)
    initial_investment = 10000
    opt_cum = (1 + returns.dot(optimized_weights)).cumprod()[-1] * initial_investment
    orig_cum = (1 + returns.dot(list(weights_dict.values()))).cumprod()[-1] * initial_investment
    result["money_made"] = {
        "start_date": start_date_obj.strftime('%Y-%m-%d'),
        "initial_investment": initial_investment,
        "optimized_value": float(opt_cum),
        "original_value": float(orig_cum)
    }

    # COVID-19 Performance
    if start_date_obj < pd.to_datetime("2020-02-01"):
        result["covid_performance"] = analyzer.print_crisis_performance(returns, combined_strategies, benchmark_returns)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)