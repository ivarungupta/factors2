import pandas as pd
import numpy as np

class TechnicalFactorCalculator:
    def __init__(self):
        self.factor_calculations = {
            # Emotional Factors
            'VOL60': self.calculate_volume_volatility_vol60,
            'DAVOL60': self.calculate_volume_ma_davol60,
            'VOSC': self.calculate_volume_oscillator_vosc,
            'VMACD': self.calculate_volume_macd,
            'ATR42': self.calculate_atr42,
            
            # Momentum Factors
            'ROC60': self.calculate_rate_of_change_roc60,
            'Volume1Q': self.calculate_volume_quarterly_1q,
            'TRIX30': self.calculate_trix30,
            'Price1Q': self.calculate_price_quarterly_price1q,
            'PLRC36': self.calculate_price_level_ratio_PLRC36,
            
            # Risk Factors
            'Variance60': self.calculate_variance_60,
            'Skewness60': self.calculate_skewness60,
            'Kurtosis60': self.calculate_kurtosis60,
            'SharpeRatio20': lambda df: self.calculate_sharpe_ratio(df, 20, 0.045),
            'SharpeRatio60': lambda df: self.calculate_sharpe_ratio(df, 60, 0.045),
            
            # Technical Factors
            'MACD60': self.calculate_macd_60,
            'boll_up': lambda df: self.calculate_bollinger_bands(df)['boll_up'],
            'boll_down': lambda df: self.calculate_bollinger_bands(df)['boll_down'],
            'MFI42': self.calculate_mfi_42,
            
            # Style Factors
            'GrowthRate': self.calculate_growth_rate,
            'Momentum': self.calculate_momentum
        }

    # Emotional Factors
    def calculate_volume_volatility_vol60(self, df, window=60):
        """Calculate 60-day volume volatility."""
        return df['close'].pct_change().rolling(window=window).std()

    def calculate_volume_ma_davol60(self, df, window=60):
        """Calculate 60-day average volume."""
        return df['volume'].rolling(window=window).mean()

    def calculate_volume_oscillator_vosc(self, df, window=60):
        """Calculate volume oscillator."""
        volume_ma = df['volume'].rolling(window=window).mean()
        return df['volume'] - volume_ma

    def calculate_volume_macd(self, df, fast_span=36, slow_span=78):
        """Calculate volume MACD."""
        fast_ma = df['volume'].ewm(span=fast_span).mean()
        slow_ma = df['volume'].ewm(span=slow_span).mean()
        return fast_ma - slow_ma

    def calculate_atr42(self, df, window=42):
        """Calculate 42-day Average True Range."""
        return (df['high'] - df['low']).rolling(window=window).mean()

    # Momentum Factors
    def calculate_rate_of_change_roc60(self, df, window=60):
        """Calculate 60-day rate of change."""
        return (df['close'] - df['close'].shift(window)) / df['close'].shift(window)

    def calculate_volume_quarterly_1q(self, df, window=60):
        """Calculate quarterly volume."""
        return df['volume'].rolling(window=window).sum()

    def calculate_trix30(self, df, span=30):
        """Calculate TRIX indicator."""
        triple_ema = df['close'].ewm(span=span).mean().ewm(span=span).mean().ewm(span=span).mean()
        return triple_ema.pct_change(periods=1)

    def calculate_price_quarterly_price1q(self, df, window=60):
        """Calculate quarterly price change."""
        return df['close'] - df['close'].shift(window)

    def calculate_price_level_ratio_PLRC36(self, df, window=36):
        """Calculate price level ratio."""
        return df['close'].rolling(window=window).mean() / df['close'].shift(window) - 1

    # Risk Factors
    def calculate_variance_60(self, df, window=60):
        """Calculate 60-day variance of returns."""
        return df['close'].pct_change().rolling(window=window).var()

    def calculate_skewness60(self, df, window=60):
        """Calculate 60-day skewness of returns."""
        return df['close'].pct_change().rolling(window=window).skew()

    def calculate_kurtosis60(self, df, window=60):
        """Calculate 60-day kurtosis of returns."""
        return df['close'].pct_change().rolling(window=window).kurt()

    def calculate_sharpe_ratio(self, df, window, risk_free_rate):
        """Calculate Sharpe ratio for given window."""
        returns = df['close'].pct_change()
        annualized_returns = returns.rolling(window=window).mean() * 252
        annualized_std = returns.rolling(window=window).std() * np.sqrt(252)
        return (annualized_returns - risk_free_rate) / annualized_std

    # Technical Factors
    def calculate_macd_60(self, df, fast_span=36, slow_span=78):
        """Calculate MACD indicator."""
        fast_ma = df['close'].ewm(span=fast_span).mean()
        slow_ma = df['close'].ewm(span=slow_span).mean()
        return fast_ma - slow_ma

    def calculate_bollinger_bands(self, df, window=60, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        return {
            'boll_up': rolling_mean + (rolling_std * num_std),
            'boll_down': rolling_mean - (rolling_std * num_std)
        }

    def calculate_mfi_42(self, df, window=42):
        """Calculate Money Flow Index."""
        typical_price = (df['close'] + df['high'] + df['low']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        return 100 - (100 / (1 + money_flow_ratio))

    # Style Factors
    def calculate_growth_rate(self, df, window=252):
        """Calculate growth rate using log returns."""
        return np.log(df['close'] / df['close'].shift(window))

    def calculate_momentum(self, df, window=252):
        """Calculate momentum as price ratio."""
        return df['close'] / df['close'].shift(window)

# Create an instance of the calculator
calculator = TechnicalFactorCalculator()
technical_factor_calculations = calculator.factor_calculations 