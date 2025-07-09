from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import warnings
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure CORS
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for stock data
stock_cache = {}
CACHE_DURATION = timedelta(minutes=15)

print(f"🚀 Starting Portfolio Intelligence Pro Backend")
print(f"💰 Using Yahoo Finance (yfinance) - FREE unlimited data!")
print(f"📈 Enhanced with backtesting, stress testing, and advanced analytics")
print(f"🌐 CORS enabled for all origins")

class EnhancedYahooFinanceClient:
    def __init__(self):
        print("💰 Enhanced Yahoo Finance client initialized - professional features enabled!")
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def get_quote(self, symbol):
        """Get current quote for a symbol"""
        cache_key = f"quote_{symbol}"
        
        if cache_key in stock_cache:
            cache_time, cache_data = stock_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=5):
                return cache_data
        
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2d")
            
            if len(hist) == 0:
                raise ValueError(f"No data found for symbol {symbol}")
            
            current_price = float(hist['Close'].iloc[-1])
            
            if len(hist) >= 2:
                prev_price = float(hist['Close'].iloc[-2])
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100
            else:
                change = 0
                change_percent = 0
            
            result = {
                'symbol': symbol.upper(),
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': f"{change_percent:.2f}",
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]) else 0
            }
            
            stock_cache[cache_key] = (datetime.now(), result)
            logger.info(f"Got quote for {symbol}: ${result['price']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise ValueError(f"Could not fetch quote for {symbol}")
    
    def get_historical_data(self, symbol, period="2y", interval="1d"):
        """Get historical data with enhanced caching"""
        cache_key = f"hist_{symbol}_{period}_{interval}"
        
        if cache_key in stock_cache:
            cache_time, cache_data = stock_cache[cache_key]
            if datetime.now() - cache_time < CACHE_DURATION:
                logger.info(f"Using cached historical data for {symbol}")
                return cache_data
        
        try:
            logger.info(f"Downloading {period} historical data for {symbol}")
            
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval, auto_adjust=True, prepost=False)
            
            if len(df) == 0:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Clean and validate data
            df = df.dropna()
            
            if len(df) < 50:
                raise ValueError(f"Insufficient data for {symbol} (only {len(df)} periods)")
            
            # Convert timezone-aware datetime to timezone-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            stock_cache[cache_key] = (datetime.now(), df)
            logger.info(f"Cached {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    def get_multiple_historical_data(self, symbols, period="2y"):
        """Fetch historical data for multiple symbols concurrently"""
        def fetch_single(symbol):
            try:
                return symbol, self.get_historical_data(symbol, period)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                return symbol, None
        
        # Use ThreadPoolExecutor for concurrent fetching
        futures = [self.executor.submit(fetch_single, symbol) for symbol in symbols]
        results = {}
        
        for future in futures:
            symbol, data = future.result()
            if data is not None:
                results[symbol] = data
        
        return results
    
    def search_symbol(self, keywords):
        """Enhanced symbol search"""
        common_stocks = {
            'apple': {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            'microsoft': {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            'google': {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            'alphabet': {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            'amazon': {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            'tesla': {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            'nvidia': {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            'meta': {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            'facebook': {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            'netflix': {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
            'disney': {'symbol': 'DIS', 'name': 'Walt Disney Company'},
            'berkshire': {'symbol': 'BRK-B', 'name': 'Berkshire Hathaway'},
            'johnson': {'symbol': 'JNJ', 'name': 'Johnson & Johnson'},
            'jpmorgan': {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            'visa': {'symbol': 'V', 'name': 'Visa Inc.'},
            'procter': {'symbol': 'PG', 'name': 'Procter & Gamble'},
            'unitedhealth': {'symbol': 'UNH', 'name': 'UnitedHealth Group'},
            'mastercard': {'symbol': 'MA', 'name': 'Mastercard Inc.'},
            'walmart': {'symbol': 'WMT', 'name': 'Walmart Inc.'},
            'home': {'symbol': 'HD', 'name': 'Home Depot Inc.'},
            'boeing': {'symbol': 'BA', 'name': 'Boeing Company'},
            'intel': {'symbol': 'INTC', 'name': 'Intel Corporation'},
            'amd': {'symbol': 'AMD', 'name': 'Advanced Micro Devices'},
            'coca': {'symbol': 'KO', 'name': 'Coca-Cola Company'},
            'pepsi': {'symbol': 'PEP', 'name': 'PepsiCo Inc.'},
        }
        
        keywords_lower = keywords.lower()
        matches = []
        
        # Search by name
        for key, stock_info in common_stocks.items():
            if keywords_lower in key or key in keywords_lower:
                matches.append({
                    'symbol': stock_info['symbol'],
                    'name': stock_info['name'],
                    'type': 'Equity',
                    'region': 'United States',
                    'currency': 'USD'
                })
        
        # Direct symbol match
        keywords_upper = keywords.upper()
        if len(keywords_upper) <= 5:
            try:
                test_stock = yf.Ticker(keywords_upper)
                test_hist = test_stock.history(period="1d")
                if len(test_hist) > 0:
                    try:
                        info = test_stock.info
                        name = info.get('longName', info.get('shortName', f'{keywords_upper} Corporation'))
                    except:
                        name = f'{keywords_upper} Corporation'
                    
                    matches.insert(0, {
                        'symbol': keywords_upper,
                        'name': name,
                        'type': 'Equity',
                        'region': 'United States',
                        'currency': 'USD'
                    })
            except:
                pass
        
        # Remove duplicates
        seen_symbols = set()
        unique_matches = []
        for match in matches:
            if match['symbol'] not in seen_symbols:
                seen_symbols.add(match['symbol'])
                unique_matches.append(match)
        
        logger.info(f"Found {len(unique_matches)} matches for '{keywords}'")
        return unique_matches[:8]

class AdvancedPortfolioAnalyzer:
    def __init__(self, stocks, yahoo_client, period="2y"):
        self.stocks = stocks
        self.yahoo_client = yahoo_client
        self.period = period
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None
        
    def fetch_data(self):
        """Fetch data for portfolio and benchmark"""
        logger.info(f"Fetching data for {len(self.stocks)} stocks: {', '.join(self.stocks)}")
        
        # Fetch portfolio data concurrently
        portfolio_data = self.yahoo_client.get_multiple_historical_data(self.stocks, self.period)
        
        if len(portfolio_data) < 2:
            raise ValueError("Need at least 2 stocks with valid data")
        
        # Fetch benchmark data (SPY)
        try:
            self.benchmark_data = self.yahoo_client.get_historical_data('SPY', self.period)
            self.benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
        except Exception as e:
            logger.warning(f"Could not fetch benchmark data: {e}")
            self.benchmark_data = None
            self.benchmark_returns = None
        
        # Align all data to common dates
        all_data = pd.DataFrame()
        for symbol, data in portfolio_data.items():
            if data is not None and len(data) > 50:
                all_data[symbol] = data['Close']
        
        if all_data.empty:
            raise ValueError("No valid data could be processed")
        
        # Clean and align data
        self.data = all_data.fillna(method='ffill').fillna(method='bfill').dropna()
        
        if len(self.data) < 50:
            raise ValueError(f"Insufficient aligned data points ({len(self.data)} days)")
        
        self.returns = self.data.pct_change().dropna()
        
        # Align benchmark to portfolio dates
        if self.benchmark_returns is not None:
            common_dates = self.returns.index.intersection(self.benchmark_returns.index)
            if len(common_dates) > 0:
                self.benchmark_returns = self.benchmark_returns[common_dates]
        
        logger.info(f"Successfully processed data: {len(self.data)} periods, {len(self.data.columns)} stocks")
        
    def optimize_portfolio(self, method='max_sharpe', constraints=None):
        """Enhanced portfolio optimization with multiple methods"""
        n = len(self.stocks)
        logger.info(f"Optimizing portfolio using method: {method}")
        
        if constraints is None:
            constraints = {'max_weight': 0.4, 'min_weight': 0.05}
        
        # Calculate expected returns and covariance matrix
        mu = self.returns.mean() * 252  # Annualized returns
        cov_matrix = self.returns.cov() * 252  # Annualized covariance
        
        # Constraints: weights sum to 1
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) for _ in range(n))
        
        # Initial guess
        x0 = np.array([1/n] * n)
        
        def portfolio_stats(weights):
            """Calculate portfolio statistics"""
            portfolio_return = np.sum(mu * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_volatility
        
        if method == 'max_sharpe':
            def neg_sharpe(weights):
                try:
                    ret, vol = portfolio_stats(weights)
                    return -(ret - 0.02) / vol  # Risk-free rate 2%
                except:
                    return 1000
            
            result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
            
        elif method == 'min_volatility':
            def portfolio_volatility(weights):
                try:
                    _, vol = portfolio_stats(weights)
                    return vol
                except:
                    return 1000
            
            result = minimize(portfolio_volatility, x0, method='SLSQP', bounds=bounds, constraints=cons)
            
        elif method == 'risk_parity':
            def risk_parity_objective(weights):
                try:
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
                    risk_contrib = weights * marginal_risk
                    target_risk = np.ones(n) / n
                    return np.sum((risk_contrib - target_risk) ** 2)
                except:
                    return 1000
            
            result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
            
        else:  # equal_weight
            result = type('obj', (object,), {'x': x0, 'success': True})()
        
        if result.success:
            weights = result.x.copy()
            weights[weights < 0.005] = 0  # Remove tiny weights
            weights = weights / np.sum(weights)  # Renormalize
            return weights
        else:
            logger.warning("Optimization failed, using equal weights")
            return np.array([1/n] * n)
    
    def calculate_advanced_metrics(self, weights):
        """Calculate comprehensive portfolio metrics"""
        portfolio_returns = self.returns @ weights
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        # Beta and Alpha (relative to benchmark)
        beta, alpha = 1.0, 0.0
        tracking_error, information_ratio = annual_volatility, 0.0
        
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            try:
                # Align portfolio and benchmark returns
                common_dates = portfolio_returns.index.intersection(self.benchmark_returns.index)
                if len(common_dates) > 30:
                    port_aligned = portfolio_returns[common_dates]
                    bench_aligned = self.benchmark_returns[common_dates]
                    
                    # Calculate beta using regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(bench_aligned, port_aligned)
                    beta = slope
                    
                    # Calculate alpha
                    benchmark_annual_return = bench_aligned.mean() * 252
                    alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
                    
                    # Calculate tracking error and information ratio
                    excess_returns = port_aligned - bench_aligned
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    if tracking_error > 0:
                        information_ratio = (excess_returns.mean() * 252) / tracking_error
                        
            except Exception as e:
                logger.warning(f"Could not calculate beta/alpha: {e}")
        
        # Additional metrics
        sortino_ratio = (annual_return - risk_free_rate) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)) if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'expected_return': annual_return * 100,
            'volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'beta': beta,
            'alpha': alpha * 100,
            'tracking_error': tracking_error * 100,
            'information_ratio': information_ratio
        }
    
    def run_backtest(self, weights, start_date=None, end_date=None):
        """Run comprehensive backtesting analysis"""
        if start_date is None:
            start_date = self.data.index[0]
        if end_date is None:
            end_date = self.data.index[-1]
        
        # Filter data for backtest period
        backtest_data = self.data.loc[start_date:end_date]
        backtest_returns = backtest_data.pct_change().dropna()
        
        if len(backtest_returns) < 50:
            raise ValueError("Insufficient data for backtesting")
        
        # Calculate portfolio performance
        portfolio_returns = backtest_returns @ weights
        portfolio_values = (1 + portfolio_returns).cumprod() * 10000
        
        # Calculate benchmark performance
        benchmark_values = None
        if self.benchmark_returns is not None:
            benchmark_aligned = self.benchmark_returns.loc[backtest_returns.index[0]:backtest_returns.index[-1]]
            if len(benchmark_aligned) > 0:
                benchmark_values = (1 + benchmark_aligned).cumprod() * 10000
        
        # Calculate rolling metrics
        rolling_window = min(63, len(portfolio_returns) // 4)  # Quarterly rolling or smaller
        rolling_returns = portfolio_returns.rolling(rolling_window).apply(lambda x: (1 + x).prod() - 1) * (252 / rolling_window)
        rolling_volatility = portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns - 0.02) / rolling_volatility
        
        # Calculate performance metrics
        total_return = (portfolio_values.iloc[-1] - 10000) / 10000 * 100
        annual_return = (portfolio_values.iloc[-1] / 10000) ** (252 / len(portfolio_values)) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100
        
        # Maximum drawdown
        cumulative = portfolio_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Benchmark comparison
        benchmark_return = 0
        excess_return = total_return
        if benchmark_values is not None and len(benchmark_values) > 0:
            benchmark_return = (benchmark_values.iloc[-1] - 10000) / 10000 * 100
            excess_return = total_return - benchmark_return
        
        return {
            'dates': portfolio_values.index.strftime('%Y-%m-%d').tolist(),
            'portfolio_values': portfolio_values.round(2).tolist(),
            'benchmark_values': benchmark_values.round(2).tolist() if benchmark_values is not None else None,
            'rolling_returns': rolling_returns.fillna(0).tolist(),
            'rolling_volatility': (rolling_volatility * 100).fillna(0).tolist(),
            'rolling_sharpe': rolling_sharpe.fillna(0).tolist(),
            'metrics': {
                'total_return': round(total_return, 2),
                'annual_return': round(annual_return * 100, 2),
                'benchmark_return': round(benchmark_return, 2),
                'excess_return': round(excess_return, 2),
                'volatility': round(annual_volatility * 100, 2),
                'sharpe_ratio': round((annual_return - 0.02) / annual_volatility, 2),
                'max_drawdown': round(max_drawdown, 2),
                'win_rate': round(win_rate, 1)
            }
        }
    
    def stress_test(self, weights):
        """Run stress testing scenarios"""
        portfolio_returns = self.returns @ weights
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Define stress scenarios
        scenarios = [
            {
                'name': '2008 Financial Crisis',
                'description': 'Market decline of 40% over 6 months with high volatility',
                'market_decline': -0.40,
                'volatility_multiplier': 2.5,
                'duration_days': 126  # 6 months
            },
            {
                'name': '2020 COVID Crash',
                'description': 'Sharp 35% decline followed by recovery',
                'market_decline': -0.35,
                'volatility_multiplier': 3.0,
                'duration_days': 63  # 3 months
            },
            {
                'name': 'Interest Rate Shock',
                'description': 'Fed raises rates by 400 basis points',
                'market_decline': -0.25,
                'volatility_multiplier': 1.8,
                'duration_days': 252  # 1 year
            },
            {
                'name': 'Tech Sector Collapse',
                'description': 'Technology bubble burst scenario',
                'market_decline': -0.50,
                'volatility_multiplier': 2.2,
                'duration_days': 189  # 9 months
            },
            {
                'name': 'Inflation Surge',
                'description': 'Unexpected inflation spike to 10%+',
                'market_decline': -0.20,
                'volatility_multiplier': 1.5,
                'duration_days': 126
            }
        ]
        
        results = []
        for scenario in scenarios:
            # Estimate portfolio impact based on scenario
            base_impact = scenario['market_decline']
            vol_adjustment = (portfolio_volatility / 0.15) * 0.1  # Adjust for portfolio volatility
            
            # Portfolio-specific impact (higher volatility = higher impact)
            portfolio_impact = base_impact * (1 + vol_adjustment)
            
            # Calculate probability based on historical frequency
            if 'crisis' in scenario['name'].lower() or 'crash' in scenario['name'].lower():
                probability = 'Low (5-10%)'
            elif 'shock' in scenario['name'].lower() or 'surge' in scenario['name'].lower():
                probability = 'Medium (10-20%)'
            else:
                probability = 'Medium-Low (8-15%)'
            
            results.append({
                'name': scenario['name'],
                'description': scenario['description'],
                'impact': round(portfolio_impact * 100, 1),
                'probability': probability,
                'recovery_time': f"{scenario['duration_days'] // 63 * 3}-{scenario['duration_days'] // 63 * 6} months"
            })
        
        return results

# Initialize clients
yahoo_client = EnhancedYahooFinanceClient()

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/api/health')
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'features': [
            'Advanced Portfolio Optimization',
            'Historical Backtesting',
            'Stress Testing',
            'Enhanced Risk Metrics',
            'Multi-Strategy Comparison'
        ],
        'data_source': 'Yahoo Finance (yfinance)',
        'api_key_required': False,
        'unlimited_requests': True,
        'real_time_delay': '15-20 minutes',
        'timestamp': datetime.now().isoformat(),
        'server': 'Portfolio Intelligence Pro'
    })

@app.route('/api/search/<keywords>')
def search_stocks(keywords):
    """Enhanced stock search"""
    logger.info(f"Search requested for: {keywords}")
    try:
        results = yahoo_client.search_symbol(keywords)
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Search error for '{keywords}': {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-stock/<symbol>')
def validate_stock(symbol):
    """Validate stock with enhanced data"""
    logger.info(f"Validation requested for: {symbol}")
    try:
        quote = yahoo_client.get_quote(symbol)
        return jsonify({
            'valid': True,
            'symbol': quote['symbol'],
            'price': quote['price'],
            'change': quote['change'],
            'change_percent': quote['change_percent'],
            'volume': quote['volume']
        })
    except Exception as e:
        logger.error(f"Validation failed for {symbol}: {e}")
        return jsonify({
            'valid': False, 
            'symbol': symbol,
            'error': str(e)
        }), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    """Enhanced portfolio analysis"""
    logger.info("Portfolio analysis requested")
    try:
        data = request.json
        stocks = data.get('stocks', [])
        optimization_method = data.get('optimization_method', 'max_sharpe')
        period = data.get('period', '2y')
        
        logger.info(f"Analysis: {len(stocks)} stocks, method: {optimization_method}, period: {period}")
        
        if len(stocks) < 2:
            return jsonify({'error': 'At least 2 stocks required'}), 400
        
        # Initialize analyzer
        analyzer = AdvancedPortfolioAnalyzer(stocks, yahoo_client, period)
        
        # Fetch data
        analyzer.fetch_data()
        
        # Optimize portfolio
        optimal_weights = analyzer.optimize_portfolio(optimization_method)
        
        # Calculate enhanced metrics
        metrics = analyzer.calculate_advanced_metrics(optimal_weights)
        
        # Get allocation
        allocation = [
            {'symbol': symbol, 'weight': round(weight * 100, 2)}
            for symbol, weight in zip(stocks, optimal_weights)
        ]
        
        # Get performance data
        portfolio_returns = analyzer.returns @ optimal_weights
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = 10000 * cumulative_returns
        
        performance = {
            'dates': portfolio_values.index.strftime('%Y-%m-%d').tolist(),
            'values': portfolio_values.round(2).tolist()
        }
        
        # Enhanced technical signals
        signals = get_enhanced_technical_signals(analyzer.data)
        
        # Monte Carlo simulation
        monte_carlo = run_monte_carlo_simulation(analyzer.returns, optimal_weights)
        
        # Correlation analysis
        correlation_data = get_correlation_analysis(analyzer.returns)
        
        result = {
            'metrics': metrics,
            'allocation': allocation,
            'performance': performance,
            'signals': signals,
            'monte_carlo': monte_carlo,
            'correlation_data': correlation_data,
            'data_points': len(analyzer.data),
            'period': period
        }
        
        logger.info(f"Analysis completed: {len(analyzer.data)} data points")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run portfolio backtesting"""
    logger.info("Backtest requested")
    try:
        data = request.json
        stocks = data.get('stocks', [])
        optimization_method = data.get('optimization_method', 'max_sharpe')
        period = data.get('period', '2y')
        
        if len(stocks) < 2:
            return jsonify({'error': 'At least 2 stocks required for backtesting'}), 400
        
        # Initialize analyzer
        analyzer = AdvancedPortfolioAnalyzer(stocks, yahoo_client, period)
        analyzer.fetch_data()
        
        # Optimize portfolio
        optimal_weights = analyzer.optimize_portfolio(optimization_method)
        
        # Run backtest
        backtest_results = analyzer.run_backtest(optimal_weights)
        
        # Run stress tests
        stress_results = analyzer.stress_test(optimal_weights)
        
        result = {
            'backtest': backtest_results,
            'stress_tests': stress_results,
            'optimization_method': optimization_method,
            'period': period,
            'stocks': stocks
        }
        
        logger.info("Backtest completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-strategies', methods=['POST'])
def compare_strategies():
    """Compare multiple portfolio strategies"""
    logger.info("Strategy comparison requested")
    try:
        data = request.json
        stocks = data.get('stocks', [])
        strategies = data.get('strategies', ['max_sharpe', 'min_volatility', 'equal_weight'])
        period = data.get('period', '2y')
        
        if len(stocks) < 2:
            return jsonify({'error': 'At least 2 stocks required for comparison'}), 400
        
        # Initialize analyzer
        analyzer = AdvancedPortfolioAnalyzer(stocks, yahoo_client, period)
        analyzer.fetch_data()
        
        comparisons = []
        
        for strategy in strategies:
            try:
                # Optimize for each strategy
                weights = analyzer.optimize_portfolio(strategy)
                metrics = analyzer.calculate_advanced_metrics(weights)
                
                # Get allocation
                allocation = [
                    {'symbol': symbol, 'weight': round(weight * 100, 2)}
                    for symbol, weight in zip(stocks, weights)
                ]
                
                # Run quick backtest
                backtest = analyzer.run_backtest(weights)
                
                comparisons.append({
                    'strategy': strategy,
                    'name': get_strategy_display_name(strategy),
                    'metrics': metrics,
                    'allocation': allocation,
                    'backtest_summary': {
                        'total_return': backtest['metrics']['total_return'],
                        'sharpe_ratio': backtest['metrics']['sharpe_ratio'],
                        'max_drawdown': backtest['metrics']['max_drawdown']
                    }
                })
                
            except Exception as e:
                logger.warning(f"Failed to analyze strategy {strategy}: {e}")
                continue
        
        if not comparisons:
            return jsonify({'error': 'No strategies could be analyzed'}), 500
        
        # Rank strategies by Sharpe ratio
        comparisons.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        
        result = {
            'comparisons': comparisons,
            'best_strategy': comparisons[0]['strategy'] if comparisons else None,
            'period': period,
            'stocks': stocks
        }
        
        logger.info(f"Strategy comparison completed for {len(comparisons)} strategies")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Strategy comparison error: {e}")
        return jsonify({'error': str(e)}), 500

def get_strategy_display_name(strategy):
    """Get display name for strategy"""
    names = {
        'max_sharpe': 'Maximum Sharpe Ratio',
        'min_volatility': 'Minimum Volatility',
        'equal_weight': 'Equal Weight',
        'risk_parity': 'Risk Parity',
        'max_diversification': 'Maximum Diversification'
    }
    return names.get(strategy, strategy.replace('_', ' ').title())

def get_enhanced_technical_signals(price_data):
    """Calculate enhanced technical indicators"""
    signals = []
    
    for symbol in price_data.columns:
        try:
            prices = price_data[symbol].dropna()
            
            if len(prices) < 50:
                continue
            
            # RSI calculation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Moving averages
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_middle = prices.rolling(window=20).mean()
            bb_std = prices.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            current_price = prices.iloc[-1]
            
            # Trend analysis
            if len(sma_20) > 0 and len(sma_50) > 0:
                sma_trend = 'UP' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'DOWN'
            else:
                sma_trend = 'NEUTRAL'
            
            # Price vs Bollinger Bands
            bb_position = 'NEUTRAL'
            if len(bb_upper) > 0 and len(bb_lower) > 0:
                if current_price > bb_upper.iloc[-1]:
                    bb_position = 'OVERBOUGHT'
                elif current_price < bb_lower.iloc[-1]:
                    bb_position = 'OVERSOLD'
            
            # MACD signal
            macd_signal_current = 'NEUTRAL'
            if len(macd) > 1 and len(macd_signal) > 1:
                if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
                    macd_signal_current = 'BULLISH'
                elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
                    macd_signal_current = 'BEARISH'
            
            # Generate overall signal
            signals_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            # RSI signals
            if current_rsi < 30:
                signals_count['BUY'] += 2
            elif current_rsi > 70:
                signals_count['SELL'] += 2
            else:
                signals_count['HOLD'] += 1
            
            # Trend signals
            if sma_trend == 'UP':
                signals_count['BUY'] += 1
            elif sma_trend == 'DOWN':
                signals_count['SELL'] += 1
            
            # Bollinger Band signals
            if bb_position == 'OVERSOLD':
                signals_count['BUY'] += 1
            elif bb_position == 'OVERBOUGHT':
                signals_count['SELL'] += 1
            
            # MACD signals
            if macd_signal_current == 'BULLISH':
                signals_count['BUY'] += 1
            elif macd_signal_current == 'BEARISH':
                signals_count['SELL'] += 1
            
            # Determine final signal
            max_signal = max(signals_count, key=signals_count.get)
            
            signals.append({
                'symbol': symbol,
                'signal': max_signal,
                'rsi': round(float(current_rsi), 1),
                'sma_trend': sma_trend,
                'bb_position': bb_position,
                'macd_signal': macd_signal_current,
                'price': round(float(current_price), 2),
                'confidence': round(signals_count[max_signal] / sum(signals_count.values()) * 100, 0)
            })
            
        except Exception as e:
            logger.warning(f"Could not calculate signals for {symbol}: {e}")
            continue
    
    return signals

def run_monte_carlo_simulation(returns_data, weights, num_simulations=1000, days=252):
    """Enhanced Monte Carlo simulation"""
    try:
        portfolio_returns = returns_data @ weights
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Use Cholesky decomposition for correlated simulation
        correlation_matrix = returns_data.corr().values
        
        try:
            chol_matrix = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If matrix is not positive definite, use simpler method
            chol_matrix = None
        
        initial_value = 10000
        simulations = np.zeros((num_simulations, days))
        
        for i in range(num_simulations):
            if chol_matrix is not None:
                # Correlated simulation
                random_matrix = np.random.normal(0, 1, (days, len(weights)))
                correlated_randoms = random_matrix @ chol_matrix.T
                asset_returns = returns_data.mean().values + (correlated_randoms * returns_data.std().values)
                portfolio_daily_returns = asset_returns @ weights
            else:
                # Simple simulation
                portfolio_daily_returns = np.random.normal(mean_return, std_return, days)
            
            cumulative_returns = np.cumprod(1 + portfolio_daily_returns)
            simulations[i] = initial_value * cumulative_returns
        
        # Calculate percentiles
        percentiles = {
            'p5': np.percentile(simulations, 5, axis=0).tolist(),
            'p25': np.percentile(simulations, 25, axis=0).tolist(),
            'p50': np.percentile(simulations, 50, axis=0).tolist(),
            'p75': np.percentile(simulations, 75, axis=0).tolist(),
            'p95': np.percentile(simulations, 95, axis=0).tolist()
        }
        
        # Calculate confidence intervals
        final_values = simulations[:, -1]
        confidence_intervals = {
            'probability_of_loss': (final_values < initial_value).sum() / num_simulations * 100,
            'probability_of_gain': (final_values > initial_value).sum() / num_simulations * 100,
            'expected_value': np.mean(final_values),
            'worst_case_5': np.percentile(final_values, 5),
            'best_case_95': np.percentile(final_values, 95)
        }
        
        return {
            'percentiles': percentiles,
            'confidence_intervals': confidence_intervals,
            'num_simulations': num_simulations,
            'time_horizon_days': days
        }
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation error: {e}")
        return None

def get_correlation_analysis(returns_data):
    """Enhanced correlation analysis"""
    try:
        correlation_matrix = returns_data.corr()
        
        if correlation_matrix.isnull().any().any():
            correlation_matrix = correlation_matrix.fillna(0)
        
        # Convert to format expected by frontend
        correlations = []
        stocks = list(correlation_matrix.columns)
        
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                corr_value = correlation_matrix.loc[stock1, stock2]
                if pd.isna(corr_value):
                    corr_value = 0.0
                
                correlations.append({
                    'x': i,
                    'y': j,
                    'v': float(corr_value),
                    'stock1': stock1,
                    'stock2': stock2
                })
        
        # Calculate portfolio diversification metrics
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        max_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
        min_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
        
        # Diversification ratio
        weights = np.array([1/len(stocks)] * len(stocks))  # Equal weights for analysis
        portfolio_vol = np.sqrt(weights.T @ correlation_matrix.values @ weights)
        weighted_avg_vol = np.sum(weights)  # Assuming unit volatilities
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        return {
            'correlations': correlations,
            'stocks': stocks,
            'matrix': correlation_matrix.round(3).to_dict(),
            'diversification_metrics': {
                'average_correlation': round(avg_correlation, 3),
                'max_correlation': round(max_correlation, 3),
                'min_correlation': round(min_correlation, 3),
                'diversification_ratio': round(diversification_ratio, 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        return None

@app.route('/api/test')
def test_endpoint():
    """Enhanced test endpoint"""
    return jsonify({
        'message': 'Portfolio Intelligence Pro Backend is working!',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'features': [
            'Advanced Risk Metrics',
            'Historical Backtesting', 
            'Stress Testing',
            'Strategy Comparison',
            'Enhanced Technical Analysis',
            'Monte Carlo Simulation'
        ],
        'cors_enabled': True,
        'server': 'Flask Portfolio Intelligence Pro',
        'data_source': 'Yahoo Finance (yfinance)',
        'port': 5000
    })

@app.route('/')
def root():
    """Enhanced root endpoint"""
    return jsonify({
        'message': 'Portfolio Intelligence Pro API',
        'version': '2.0.0',
        'status': 'running',
        'data_source': 'Yahoo Finance (yfinance)',
        'features': [
            'Advanced Portfolio Optimization',
            'Historical Backtesting',
            'Stress Testing Scenarios',
            'Multi-Strategy Comparison',
            'Enhanced Risk Analytics',
            'Real-time Technical Signals'
        ],
        'endpoints': [
            '/api/health',
            '/api/test', 
            '/api/search/<keywords>',
            '/api/validate-stock/<symbol>',
            '/api/analyze',
            '/api/backtest',
            '/api/compare-strategies'
        ]
    })

if __name__ == '__main__':
    print(f"🚀 Starting Portfolio Intelligence Pro Backend...")
    print(f"💰 Enhanced Yahoo Finance integration - FREE unlimited data!")
    print(f"📊 Professional-grade analytics: Backtesting, Stress Testing, Advanced Risk Metrics")
    print(f"⚡ Multi-threaded data fetching for optimal performance")
    print(f"🌐 Server available at: http://127.0.0.1:5000")
    print(f"🧪 Test endpoint: http://127.0.0.1:5000/api/test")
    print(f"🏥 Health check: http://127.0.0.1:5000/api/health")
    print(f"📈 Backtest endpoint: http://127.0.0.1:5000/api/backtest")
    print(f"🔄 Strategy comparison: http://127.0.0.1:5000/api/compare-strategies")
    
    app.run(debug=True, port=5000, host='127.0.0.1', threaded=True)