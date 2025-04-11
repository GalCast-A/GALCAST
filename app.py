from flask import Flask, request, render_template
from portfolio_analyzer import PortfolioAnalyzer
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form inputs
        tickers = request.form['tickers'].split(',')
        weights = [float(w) for w in request.form['weights'].split(',')]
        start_date = request.form['start_date']
        risk_tolerance = request.form['risk_tolerance']
        benchmarks = request.form['benchmarks'].split(',')
        optimization_metric = request.form['optimization_metric']

        # Basic validation
        if abs(sum(weights) - 1.0) > 0.01:
            return render_template('index.html', error="Weights must sum to 1.0")

        # Initialize PortfolioAnalyzer
        analyzer = PortfolioAnalyzer(tickers, weights, start_date=start_date, risk_tolerance=risk_tolerance)

        # Fetch data and compute metrics
        try:
            stock_data = analyzer.fetch_stock_data()
            returns = analyzer.compute_returns(stock_data)
            metrics = analyzer.compute_portfolio_metrics(returns, stock_data)
            optimized_weights = analyzer.optimize_portfolio(returns, optimization_metric=optimization_metric)
            recommendations = analyzer.suggest_courses_of_action(metrics, optimized_weights)
            benchmark_metrics = analyzer.compare_with_benchmarks(benchmarks)

            return render_template('results.html', 
                                  metrics=metrics, 
                                  optimized_weights=optimized_weights, 
                                  recommendations=recommendations, 
                                  benchmark_metrics=benchmark_metrics)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
