import pandas as pd
import numpy as np

class FinancialFactorCalculator:
    def __init__(self):
        self.factor_calculations = {
            # Quality Factors
            'net_profit_to_total_revenue_ttm': self.calculate_net_profit_to_revenue,
            'roe_ttm': self.calculate_roe,
            'roa_ttm': self.calculate_roa,
            'GMI': self.calculate_gmi,
            'ACCA': self.calculate_acca,
            'debt_to_asset_ratio': self.calculate_debt_to_asset_ratio,
            
            # Value Factors
            'financial_liability': self.calculate_financial_liability,
            'cash_flow_to_price_ratio_ttm': self.calculate_cash_flow_to_price_ratio,
            'price_to_book_ratio': self.calculate_price_to_book_ratio,
            'price_to_sales_ratio_ttm': self.calculate_price_to_sales_ratio,
            'price_to_earning_ratio_ttm': self.calculate_price_to_earning_ratio,
            'total_liability_to_total_asset_ratio': self.calculate_total_liability_to_total_asset_ratio,
            'net_profit_ttm': self.calculate_net_profit,
            'working_capital_ratio': self.calculate_working_capital_ratio,
            'quick_ratio': self.calculate_quick_ratio,
            'debt_to_equity_ratio': self.calculate_debt_to_equity_ratio,
            'operate_cash_flow_to_total_asset_ratio': self.calculate_operate_cash_flow_to_total_asset_ratio,
            'operate_cash_flow_to_total_liabilities_ratio': self.calculate_operate_cash_flow_to_total_liabilities_ratio,
            'operate_cash_flow_to_net_profit_ratio': self.calculate_operate_cash_flow_to_net_profit_ratio,
            'EV_to_operate_cash_flow_ratio': self.calculate_EV_to_operate_cash_flow_ratio,
            'debt_to_EBITDA_ratio': self.calculate_debt_to_EBITDA_ratio,
            
            # Growth Factors
            'EPS_growth_rate_ttm': self.calculate_eps_growth_rate_ttm,
            'PEG_ttm': self.calculate_peg_ttm,
            'net_profit_growth_rate_ttm': self.calculate_net_profit_growth_rate_ttm,
            'revenue_growth_rate_ttm': self.calculate_revenue_growth_rate_ttm,
            'net_asset_growth_rate': self.calculate_net_asset_growth_rate,
            'operate_cash_flow_growth_rate_ttm': self.calculate_operate_cash_flow_growth_rate_ttm,
            
            # Stock Factors
            'net_asset_per_share': self.calculate_net_asset_per_share,
            'net_operate_cash_flow_per_share': self.calculate_net_operate_cash_flow_per_share,
            'retained_earnings_per_share': self.calculate_retained_earnings_per_share,
            'market_cap(size)': self.calculate_market_cap,
            
            # Style Factor
            'liquidity': self.calculate_liquidity
        }

    # Quality Factors
    def calculate_net_profit_to_revenue(self, stock_data):
        """Calculate net profit to total revenue ratio"""
        if 'netIncome_ttm' in stock_data.columns and 'revenue_ttm' in stock_data.columns:
            return (stock_data['netIncome_ttm'].astype('float64').div(
                stock_data['revenue_ttm'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_roe(self, stock_data):
        """Calculate Return on Equity"""
        if 'netIncome_ttm' in stock_data.columns and 'totalStockholdersEquity' in stock_data.columns:
            return (stock_data['netIncome_ttm'].astype('float64').div(
                stock_data['totalStockholdersEquity'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_roa(self, stock_data):
        """Calculate Return on Assets"""
        if 'netIncome_ttm' in stock_data.columns and 'totalAssets' in stock_data.columns:
            stock_data['prev_totalAssets'] = stock_data['totalAssets'].shift(1)
            stock_data['avg_totalAssets'] = ((stock_data['totalAssets'].astype('float64') +
                                            stock_data['prev_totalAssets'].astype('float64')) / 2)
            return (stock_data['netIncome_ttm'].astype('float64').div(
                stock_data['avg_totalAssets'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_gmi(self, stock_data):
        """Calculate Gross Margin Index"""
        if 'grossProfit_ttm' in stock_data.columns and 'revenue_ttm' in stock_data.columns:
            stock_data['gross_margin'] = (stock_data['grossProfit_ttm'].astype('float64').div(
                stock_data['revenue_ttm'].replace(0, np.nan).astype('float64')
            ))
            stock_data['prev_gross_margin'] = stock_data['gross_margin'].shift(1)
            return (stock_data['gross_margin'] - stock_data['prev_gross_margin']).astype('float64')
        return None

    def calculate_acca(self, stock_data):
        """Calculate Accruals to Assets"""
        if all(col in stock_data.columns for col in ['netIncome_ttm', 'operatingCashFlow_ttm', 'totalAssets']):
            return ((stock_data['netIncome_ttm'].astype('float64') -
                    stock_data['operatingCashFlow_ttm'].astype('float64')).div(
                stock_data['totalAssets'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_debt_to_asset_ratio(self, stock_data):
        """Calculate Debt to Asset Ratio"""
        if 'totalDebt' in stock_data.columns and 'totalAssets' in stock_data.columns:
            return (stock_data['totalDebt'].astype('float64').div(
                stock_data['totalAssets'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    # Value Factors
    def calculate_financial_liability(self, stock_data):
        """Calculate financial liability"""
        if 'totalLiabilities' in stock_data.columns:
            return stock_data['totalLiabilities'].astype('float64')
        return None

    def calculate_cash_flow_to_price_ratio(self, stock_data):
        """Calculate cash flow to price ratio"""
        if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'weightedAverageShsOut', 'close']):
            return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
                stock_data['weightedAverageShsOut'].replace(0, np.nan).astype('float64')
            ).div(stock_data['close'].replace(0, np.nan).astype('float64'))).astype('float64')
        return None

    def calculate_price_to_book_ratio(self, stock_data):
        """Calculate price to book ratio"""
        if all(col in stock_data.columns for col in ['close', 'totalStockholdersEquity', 'weightedAverageShsOut']):
            return (stock_data['close'].astype('float64').div(
                stock_data['totalStockholdersEquity'].astype('float64').div(
                    stock_data['weightedAverageShsOut'].replace(0, np.nan).astype('float64')
                )
            )).astype('float64')
        return None

    def calculate_price_to_sales_ratio(self, stock_data):
        """Calculate price to sales ratio"""
        if all(col in stock_data.columns for col in ['close', 'revenue_ttm', 'weightedAverageShsOut']):
            return (stock_data['close'].astype('float64').div(
                stock_data['revenue_ttm'].astype('float64').div(
                    stock_data['weightedAverageShsOut'].replace(0, np.nan).astype('float64')
                )
            )).astype('float64')
        return None

    def calculate_price_to_earning_ratio(self, stock_data):
        """Calculate price to earning ratio"""
        if all(col in stock_data.columns for col in ['close', 'eps_ttm']):
            return (stock_data['close'].astype('float64').div(
                stock_data['eps_ttm'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_total_liability_to_total_asset_ratio(self, stock_data):
        """Calculate total liability to total asset ratio"""
        if all(col in stock_data.columns for col in ['totalLiabilities', 'totalAssets']):
            return (stock_data['totalLiabilities'].astype('float64').div(
                stock_data['totalAssets'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_net_profit(self, stock_data):
        """Calculate net profit"""
        if 'netIncome_ttm' in stock_data.columns:
            return stock_data['netIncome_ttm'].astype('float64')
        return None

    def calculate_working_capital_ratio(self, stock_data):
        """Calculate working capital ratio"""
        if all(col in stock_data.columns for col in ['totalCurrentAssets', 'totalCurrentLiabilities']):
            return (stock_data['totalCurrentAssets'].astype('float64').div(
                stock_data['totalCurrentLiabilities'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_quick_ratio(self, stock_data):
        """Calculate quick ratio"""
        if all(col in stock_data.columns for col in ['totalCurrentAssets', 'inventory', 'totalCurrentLiabilities']):
            return ((stock_data['totalCurrentAssets'].astype('float64') -
                    stock_data['inventory'].astype('float64')).div(
                stock_data['totalCurrentLiabilities'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_debt_to_equity_ratio(self, stock_data):
        """Calculate debt to equity ratio"""
        if all(col in stock_data.columns for col in ['totalLiabilities', 'totalStockholdersEquity']):
            return (stock_data['totalLiabilities'].astype('float64').div(
                stock_data['totalStockholdersEquity'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_operate_cash_flow_to_total_asset_ratio(self, stock_data):
        """Calculate operating cash flow to total asset ratio"""
        if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'totalAssets']):
            return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
                stock_data['totalAssets'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_operate_cash_flow_to_total_liabilities_ratio(self, stock_data):
        """Calculate operating cash flow to total liabilities ratio"""
        if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'totalLiabilities']):
            return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
                stock_data['totalLiabilities'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_operate_cash_flow_to_net_profit_ratio(self, stock_data):
        """Calculate operating cash flow to net profit ratio"""
        if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'netIncome_ttm']):
            return (stock_data['operatingCashFlow_ttm'].astype('float64').div(
                stock_data['netIncome_ttm'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_EV_to_operate_cash_flow_ratio(self, stock_data):
        """Calculate EV to operating cash flow ratio"""
        if all(col in stock_data.columns for col in ['close', 'weightedAverageShsOut', 'totalDebt', 'cashAndCashEquivalents', 'operatingCashFlow_ttm']):
            return ((stock_data['close'].astype('float64') *
                    stock_data['weightedAverageShsOut'].astype('float64') +
                    stock_data['totalDebt'].astype('float64') -
                    stock_data['cashAndCashEquivalents'].astype('float64')).div(
                stock_data['operatingCashFlow_ttm'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    def calculate_debt_to_EBITDA_ratio(self, stock_data):
        """Calculate debt to EBITDA ratio"""
        if all(col in stock_data.columns for col in ['totalDebt', 'ebitda_ttm']):
            return (stock_data['totalDebt'].astype('float64').div(
                stock_data['ebitda_ttm'].replace(0, np.nan).astype('float64')
            )).astype('float64')
        return None

    # Growth Factors
    def calculate_eps_growth_rate_ttm(self, stock_data):
        """Calculate EPS growth rate"""
        if 'eps_ttm' in stock_data.columns:
            return (stock_data['eps_ttm'] / stock_data['eps_ttm'].shift(1) - 1).astype('float64')
        return None

    def calculate_peg_ttm(self, stock_data):
        """Calculate PEG ratio"""
        if all(col in stock_data.columns for col in ['close', 'eps_ttm']):
            pe_ratio = stock_data['close'] / stock_data['eps_ttm']
            eps_ttm_growth = (stock_data['eps_ttm'] / stock_data['eps_ttm'].shift(1) - 1)
            peg_ratio = pe_ratio / eps_ttm_growth.replace(0, np.nan)
            return peg_ratio.astype('float64')
        return None

    def calculate_net_profit_growth_rate_ttm(self, stock_data):
        """Calculate net profit growth rate"""
        if 'netIncome_ttm' in stock_data.columns:
            return (stock_data['netIncome_ttm'] / stock_data['netIncome_ttm'].shift(1) - 1).astype('float64')
        return None

    def calculate_revenue_growth_rate_ttm(self, stock_data):
        """Calculate revenue growth rate"""
        if 'revenue_ttm' in stock_data.columns:
            return (stock_data['revenue_ttm'] / stock_data['revenue_ttm'].shift(1) - 1).astype('float64')
        return None

    def calculate_net_asset_growth_rate(self, stock_data):
        """Calculate net asset growth rate"""
        if 'totalStockholdersEquity' in stock_data.columns:
            return (stock_data['totalStockholdersEquity'] / stock_data['totalStockholdersEquity'].shift(1) - 1).astype('float64')
        return None

    def calculate_operate_cash_flow_growth_rate_ttm(self, stock_data):
        """Calculate operating cash flow growth rate"""
        if 'operatingCashFlow_ttm' in stock_data.columns:
            return (stock_data['operatingCashFlow_ttm'] / stock_data['operatingCashFlow_ttm'].shift(1) - 1).astype('float64')
        return None

    # Stock Factors
    def calculate_net_asset_per_share(self, stock_data):
        """Calculate net asset per share"""
        if all(col in stock_data.columns for col in ['totalStockholdersEquity', 'weightedAverageShsOut']):
            return (stock_data['totalStockholdersEquity'] / stock_data['weightedAverageShsOut']).astype('float64')
        return None

    def calculate_net_operate_cash_flow_per_share(self, stock_data):
        """Calculate net operating cash flow per share"""
        if all(col in stock_data.columns for col in ['operatingCashFlow_ttm', 'weightedAverageShsOut']):
            return (stock_data['operatingCashFlow_ttm'] / stock_data['weightedAverageShsOut']).astype('float64')
        return None

    def calculate_retained_earnings_per_share(self, stock_data):
        """Calculate retained earnings per share"""
        if all(col in stock_data.columns for col in ['retainedEarnings', 'weightedAverageShsOut']):
            return (stock_data['retainedEarnings'] / stock_data['weightedAverageShsOut']).astype('float64')
        return None

    def calculate_market_cap(self, stock_data):
        """Calculate market capitalization"""
        if all(col in stock_data.columns for col in ['close', 'weightedAverageShsOut']):
            return (stock_data['close'] * stock_data['weightedAverageShsOut']).astype('float64')
        return None

    def calculate_liquidity(self, stock_data):
        """Calculate liquidity"""
        if all(col in stock_data.columns for col in ['volume', 'weightedAverageShsOut']):
            return (stock_data['volume'] / stock_data['weightedAverageShsOut']).astype('float64')
        return None

# Create an instance of the calculator
calculator = FinancialFactorCalculator()
financial_factor_calculations = calculator.factor_calculations

# Create an instance of the calculator
calculator = FinancialFactorCalculator()
financial_factor_calculations = calculator.factor_calculations 