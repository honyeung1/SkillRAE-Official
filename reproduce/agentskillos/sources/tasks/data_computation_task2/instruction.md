I am a risk analyst at an investment firm, and I need to quantify the downside risk of a proposed portfolio for our risk committee. Please build a portfolio risk simulation model. Portfolio: Stock A with 40% weight, 12% return, and 25% vol; Stock B with 35% weight, 8% return, and 18% vol; Bond C with 25% weight, 4% return, and 6% vol. Correlations: A-B=0.6, A-C=-0.2, B-C=0.1. Initial investment: 10M.

Create portfolio_var.xlsx with a 'Simulation' sheet containing 10,000 simulated scenarios accounting for asset correlations, and a 'Risk Metrics' sheet showing expected return, volatility, 95% VaR, 99% VaR, and max drawdown.

Save var_results.json with: expected_return, volatility, var_95, var_99, and max_drawdown — all as decimals, with VaR values negative to indicate losses.