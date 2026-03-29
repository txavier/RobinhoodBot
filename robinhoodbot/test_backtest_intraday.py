#!/usr/bin/env python3
"""
Unit and integration tests for backtest_intraday.py.

These tests capture the EXACT behavior of the current backtest engine so that
performance optimizations can be validated against bit-identical results.

Run:  pytest test_backtest_intraday.py -v
"""

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

from backtest_intraday import (
    IntradayTradingStrategy,
    IntradayBacktester,
    Position,
    Trade,
    TradeType,
    SellReason,
    MarketCondition,
    generate_market_data,
    generate_golden_cross_intraday,
    generate_intraday_data,
    is_trading_day,
    next_trading_day,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hourly_df(prices, start_date=None):
    """Build a minimal hourly DataFrame from a list of close prices."""
    if start_date is None:
        start_date = datetime(2025, 1, 6)  # A Monday
    trading_hours = [9, 10, 11, 12, 13, 14, 15]
    rows = []
    current_date = start_date
    idx = 0
    while idx < len(prices):
        while not is_trading_day(current_date):
            current_date += timedelta(days=1)
        for hour in trading_hours:
            if idx >= len(prices):
                break
            ts = current_date.replace(hour=hour, minute=30, second=0)
            p = prices[idx]
            rows.append({
                'datetime': ts,
                'date': current_date.date(),
                'hour': hour,
                'open': p,
                'high': p * 1.001,
                'low': p * 0.999,
                'close': p,
                'volume': 100000,
            })
            idx += 1
        current_date += timedelta(days=1)
    return pd.DataFrame(rows)


def _make_market_df(dates, uptrend=True, major_downtrend=False):
    """Build a minimal market_data DataFrame for a list of dates."""
    cond = (
        MarketCondition.MAJOR_DOWNTREND.value if major_downtrend else
        MarketCondition.UPTREND.value if uptrend else
        MarketCondition.DOWNTREND.value
    )
    rows = []
    for d in dates:
        rows.append({
            'date': d if isinstance(d, date) else d.date(),
            'market_open': 100.0,
            'market_close': 101.0,
            'week_start': 100.0,
            'is_uptrend': uptrend,
            'is_major_downtrend': major_downtrend,
            'market_condition': cond,
        })
    return pd.DataFrame(rows)


# ===================================================================
# Unit Tests – IntradayTradingStrategy
# ===================================================================

class TestCalculateIndicators:
    def test_columns_added(self):
        strategy = IntradayTradingStrategy(short_sma=3, long_sma=5)
        prices = list(range(100, 140))
        df = _make_hourly_df(prices)
        result = strategy.calculate_indicators(df)
        for col in ['sma_short', 'sma_long', 'sma_diff', 'sma_diff_prev',
                     'sma_short_downtrend', 'sma_short_take_profit', 'sma_long_take_profit']:
            assert col in result.columns, f"Missing column {col}"

    def test_sma_values(self):
        strategy = IntradayTradingStrategy(short_sma=3, long_sma=5)
        prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        df = _make_hourly_df(prices)
        result = strategy.calculate_indicators(df)
        # SMA(3) at idx 2 = mean(10,20,30) = 20
        assert result.iloc[2]['sma_short'] == pytest.approx(20.0)
        # SMA(5) at idx 4 = mean(10,20,30,40,50) = 30
        assert result.iloc[4]['sma_long'] == pytest.approx(30.0)

    def test_original_not_mutated(self):
        strategy = IntradayTradingStrategy(short_sma=3, long_sma=5)
        df = _make_hourly_df([100.0] * 10)
        original_cols = set(df.columns)
        strategy.calculate_indicators(df)
        assert set(df.columns) == original_cols


class TestCalculateSlope:
    def test_flat_prices(self):
        strategy = IntradayTradingStrategy()
        prices = [100.0] * 20
        df = _make_hourly_df(prices)
        slope = strategy.calculate_slope(df, 15, lookback_hours=7)
        assert slope == pytest.approx(0.0, abs=1e-10)

    def test_rising_prices(self):
        strategy = IntradayTradingStrategy()
        prices = [float(100 + i) for i in range(20)]
        df = _make_hourly_df(prices)
        slope = strategy.calculate_slope(df, 15, lookback_hours=7)
        assert slope > 0

    def test_falling_prices(self):
        strategy = IntradayTradingStrategy()
        prices = [float(200 - i) for i in range(20)]
        df = _make_hourly_df(prices)
        slope = strategy.calculate_slope(df, 15, lookback_hours=7)
        assert slope < 0

    def test_insufficient_data(self):
        strategy = IntradayTradingStrategy()
        df = _make_hourly_df([100.0] * 5)
        slope = strategy.calculate_slope(df, 0, lookback_hours=7)
        assert slope == 0.0


class TestIsEod:
    @pytest.mark.parametrize("hour,minute,expected", [
        (9, 30, False),
        (12, 0, False),
        (13, 0, False),
        (13, 29, False),
        (13, 30, True),
        (14, 0, True),
        (15, 0, True),
        (15, 59, True),
        (16, 0, False),
    ])
    def test_eod_times(self, hour, minute, expected):
        strategy = IntradayTradingStrategy()
        ts = datetime(2025, 1, 6, hour, minute)
        assert strategy.is_eod(ts) == expected


class TestCheckFiveYear:
    def test_rising_stock_passes(self):
        strategy = IntradayTradingStrategy()
        prices = [float(100 + i) for i in range(20)]
        df = _make_hourly_df(prices)
        assert strategy.check_five_year(df, 19) == True

    def test_declining_stock_fails(self):
        strategy = IntradayTradingStrategy()
        prices = [float(200 - i * 2) for i in range(20)]
        df = _make_hourly_df(prices)
        assert strategy.check_five_year(df, 19) == False

    def test_disabled(self):
        strategy = IntradayTradingStrategy(use_five_year_check=False)
        prices = [float(200 - i * 10) for i in range(20)]
        df = _make_hourly_df(prices)
        assert strategy.check_five_year(df, 19) is True


class TestStopLossAndTakeProfit:
    def test_stop_loss_triggered(self):
        strategy = IntradayTradingStrategy(stop_loss_pct=5.0)
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        # price = 94 → -6% → triggers 5% stop loss
        assert strategy.check_stop_loss(pos, 94.0) is True

    def test_stop_loss_not_triggered(self):
        strategy = IntradayTradingStrategy(stop_loss_pct=5.0)
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        assert strategy.check_stop_loss(pos, 96.0) is False

    def test_stop_loss_disabled(self):
        strategy = IntradayTradingStrategy(use_stop_loss=False)
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        assert strategy.check_stop_loss(pos, 50.0) is False

    def test_take_profit_triggered(self):
        strategy = IntradayTradingStrategy(take_profit_pct=0.7)
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        assert strategy.check_take_profit(pos, 100.71) is True

    def test_take_profit_not_triggered(self):
        strategy = IntradayTradingStrategy(take_profit_pct=0.7)
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        assert strategy.check_take_profit(pos, 100.50) is False


class TestSuddenDrop:
    def test_15pct_1hr_drop(self):
        strategy = IntradayTradingStrategy()
        prices = [100.0] * 10 + [84.0]
        df = _make_hourly_df(prices)
        assert strategy.check_sudden_drop(df, 10) is True

    def test_10pct_2hr_drop(self):
        strategy = IntradayTradingStrategy()
        prices = [100.0] * 10 + [95.0, 89.0]
        df = _make_hourly_df(prices)
        assert strategy.check_sudden_drop(df, 11) is True

    def test_no_drop(self):
        strategy = IntradayTradingStrategy()
        prices = [100.0] * 10 + [99.0]
        df = _make_hourly_df(prices)
        assert strategy.check_sudden_drop(df, 10) is False


class TestProfitBeforeEod:
    def test_profitable_during_eod(self):
        strategy = IntradayTradingStrategy()
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        ts = datetime(2025, 1, 6, 14, 0)  # 2pm = EOD
        assert strategy.check_profit_before_eod(pos, 101.0, ts) is True

    def test_loss_during_eod(self):
        strategy = IntradayTradingStrategy()
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        ts = datetime(2025, 1, 6, 14, 0)
        assert strategy.check_profit_before_eod(pos, 99.0, ts) is False

    def test_profitable_before_eod(self):
        strategy = IntradayTradingStrategy()
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        ts = datetime(2025, 1, 6, 10, 0)  # 10am = not EOD
        assert strategy.check_profit_before_eod(pos, 101.0, ts) is False

    def test_disabled(self):
        strategy = IntradayTradingStrategy(use_profit_before_eod=False)
        pos = Position("AAPL", 10, 100.0, datetime(2025, 1, 6))
        ts = datetime(2025, 1, 6, 14, 0)
        assert strategy.check_profit_before_eod(pos, 101.0, ts) is False


class TestDynamicSma:
    def test_default_periods(self):
        strategy = IntradayTradingStrategy(short_sma=20, long_sma=50)
        n1, n2 = strategy.get_dynamic_sma_periods(
            market_in_downtrend=False, took_profit_today=False, traded_today=False
        )
        assert (n1, n2) == (20, 50)

    def test_downtrend_periods(self):
        strategy = IntradayTradingStrategy(short_sma=20, long_sma=50, short_sma_downtrend=14)
        n1, n2 = strategy.get_dynamic_sma_periods(
            market_in_downtrend=True, took_profit_today=False, traded_today=False
        )
        assert (n1, n2) == (14, 50)

    def test_take_profit_with_pdt(self):
        strategy = IntradayTradingStrategy(
            short_sma=20, long_sma=50,
            short_sma_take_profit=5, long_sma_take_profit=7,
            account_balance=20000.0  # Under $25k PDT limit
        )
        n1, n2 = strategy.get_dynamic_sma_periods(
            market_in_downtrend=False, took_profit_today=True, traded_today=True
        )
        assert (n1, n2) == (5, 7)

    def test_take_profit_without_pdt(self):
        strategy = IntradayTradingStrategy(
            short_sma=20, long_sma=50,
            short_sma_take_profit=5, long_sma_take_profit=7,
            account_balance=30000.0  # Over $25k → PDT doesn't apply
        )
        n1, n2 = strategy.get_dynamic_sma_periods(
            market_in_downtrend=False, took_profit_today=True, traded_today=True
        )
        assert (n1, n2) == (20, 50)

    def test_disabled(self):
        strategy = IntradayTradingStrategy(use_dynamic_sma=False)
        n1, n2 = strategy.get_dynamic_sma_periods(
            market_in_downtrend=True, took_profit_today=True, traded_today=True
        )
        assert (n1, n2) == (strategy.short_sma, strategy.long_sma)


class TestCheckGoldenCross:
    def _build_cross_data(self, n1=3, n2=5):
        """
        Build data where SMA(3) crosses above SMA(5).
        Prices: 50 bars of declining then 10 bars of sharp rise.
        """
        prices = [100.0 - i * 0.5 for i in range(50)]
        prices += [prices[-1] + i * 2.0 for i in range(1, 11)]
        return prices

    def test_golden_cross_detected(self):
        strategy = IntradayTradingStrategy(
            short_sma=3, long_sma=5, golden_cross_buy_days=5,
            use_price_cross_check=False
        )
        prices = self._build_cross_data()
        df = _make_hourly_df(prices)
        df = strategy.calculate_indicators(df)
        is_cross, cross_price, price_5hr = strategy.check_golden_cross(
            df, len(df) - 1, n1=3, n2=5
        )
        assert is_cross is True
        assert cross_price is not None

    def test_no_cross_in_downtrend(self):
        strategy = IntradayTradingStrategy(short_sma=3, long_sma=5)
        prices = [float(200 - i) for i in range(30)]
        df = _make_hourly_df(prices)
        df = strategy.calculate_indicators(df)
        is_cross, _, _ = strategy.check_golden_cross(df, len(df) - 1, n1=3, n2=5)
        assert is_cross is False


class TestCheckDeathCross:
    def test_death_cross_detected(self):
        strategy = IntradayTradingStrategy(short_sma=3, long_sma=5)
        # Rise then sharp decline → SMA(3) falls below SMA(5)
        prices = [50.0 + i * 1.0 for i in range(30)]
        prices += [prices[-1] - i * 3.0 for i in range(1, 15)]
        df = _make_hourly_df(prices)
        df = strategy.calculate_indicators(df)
        is_death, price = strategy.check_death_cross(
            df, len(df) - 1, lookback_hours=20, n1=3, n2=5
        )
        assert is_death is True

    def test_no_death_cross_in_uptrend(self):
        strategy = IntradayTradingStrategy(short_sma=3, long_sma=5)
        prices = [float(50 + i) for i in range(30)]
        df = _make_hourly_df(prices)
        df = strategy.calculate_indicators(df)
        is_death, _ = strategy.check_death_cross(df, len(df) - 1, n1=3, n2=5)
        assert is_death is False


class TestPriceHigherThan5hrAgo:
    def test_higher(self):
        strategy = IntradayTradingStrategy()
        prices = [float(100 + i) for i in range(10)]
        df = _make_hourly_df(prices)
        assert strategy.check_price_higher_than_5hr_ago(df, 9) == True

    def test_lower(self):
        strategy = IntradayTradingStrategy()
        prices = [float(200 - i * 5) for i in range(10)]
        df = _make_hourly_df(prices)
        assert strategy.check_price_higher_than_5hr_ago(df, 9) == False

    def test_insufficient_data(self):
        strategy = IntradayTradingStrategy()
        df = _make_hourly_df([100.0] * 4)
        assert strategy.check_price_higher_than_5hr_ago(df, 3) is False


# ===================================================================
# Unit Tests – IntradayBacktester
# ===================================================================

class TestPositionSizing:
    def test_mode_2_per_stock(self):
        bt = IntradayBacktester(initial_capital=10000.0, max_position_pct=15.0,
                                use_total_investment_cap=2)
        # equity=10000, 15% = 1500, price=100 → 14 shares (0.95 * 10000 = 9500 cap)
        shares = bt.calculate_position_size(100.0)
        assert shares == 15  # min(10000*0.15, 10000*0.95) / 100 = 15

    def test_mode_1_total_cap(self):
        bt = IntradayBacktester(initial_capital=10000.0, max_position_pct=10.0,
                                use_total_investment_cap=1)
        # 10% of 10000 = 1000 total budget, price=50 → 19 shares (cash cap = 9500)
        shares = bt.calculate_position_size(50.0)
        assert shares == 20  # 1000/50=20

    def test_mode_0_legacy(self):
        bt = IntradayBacktester(initial_capital=10000.0, max_position_pct=5.0,
                                use_total_investment_cap=0)
        # equity / pct = 10000 / 5 = 2000, price=100 → 20 shares
        shares = bt.calculate_position_size(100.0)
        assert shares == 20


class TestExecuteBuySell:
    def test_buy_updates_state(self):
        bt = IntradayBacktester(initial_capital=10000.0)
        ts = datetime(2025, 1, 6, 10, 30)
        trade = bt.execute_buy("AAPL", 150.0, 10, ts, "golden_cross")
        assert trade is not None
        assert "AAPL" in bt.positions
        assert bt.cash == pytest.approx(10000.0 - 1500.0)
        assert bt.positions["AAPL"].shares == 10
        assert bt.day_trades_today == 1

    def test_sell_updates_state(self):
        bt = IntradayBacktester(initial_capital=10000.0)
        ts = datetime(2025, 1, 6, 10, 30)
        bt.execute_buy("AAPL", 100.0, 10, ts, "golden_cross")
        trade = bt.execute_sell("AAPL", 105.0, ts, SellReason.TAKE_PROFIT.value)
        assert trade is not None
        assert "AAPL" not in bt.positions
        assert bt.cash == pytest.approx(10000.0 - 1000.0 + 1050.0)
        assert bt.took_profit_today is True

    def test_sell_nonexistent_symbol(self):
        bt = IntradayBacktester(initial_capital=10000.0)
        ts = datetime(2025, 1, 6, 10, 30)
        trade = bt.execute_sell("AAPL", 100.0, ts, "death_cross")
        assert trade is None

    def test_buy_insufficient_funds(self):
        bt = IntradayBacktester(initial_capital=100.0)
        ts = datetime(2025, 1, 6, 10, 30)
        trade = bt.execute_buy("AAPL", 150.0, 10, ts, "golden_cross")
        assert trade is None


class TestPortfolioValue:
    def test_cash_only(self):
        bt = IntradayBacktester(initial_capital=10000.0)
        assert bt.get_portfolio_value({}) == pytest.approx(10000.0)

    def test_with_positions(self):
        bt = IntradayBacktester(initial_capital=10000.0)
        ts = datetime(2025, 1, 6, 10, 30)
        bt.execute_buy("AAPL", 100.0, 10, ts, "golden_cross")
        # cash = 9000, AAPL 10 shares @ 110 = 1100
        val = bt.get_portfolio_value({"AAPL": 110.0})
        assert val == pytest.approx(9000.0 + 1100.0)


class TestResetDailyState:
    def test_resets_properly(self):
        bt = IntradayBacktester(initial_capital=10000.0)
        bt.day_trades_today = 3
        bt.took_profit_today = True
        bt.symbols_traded_today = {"AAPL", "MSFT"}
        ts = datetime(2025, 1, 6, 10, 30)
        bt.execute_buy("AAPL", 100.0, 5, ts, "golden_cross")
        bt.positions["AAPL"].traded_today = True

        bt.reset_daily_state()
        assert bt.day_trades_today == 0
        assert bt.took_profit_today is False
        assert len(bt.symbols_traded_today) == 0
        assert bt.positions["AAPL"].traded_today is False


# ===================================================================
# Unit Tests – Data Generation helpers
# ===================================================================

class TestDataGeneration:
    def test_market_data_schema(self):
        md = generate_market_data(datetime(2025, 1, 6), 30, seed=42)
        required = {'date', 'market_open', 'market_close', 'week_start',
                    'is_uptrend', 'is_major_downtrend', 'market_condition'}
        assert required.issubset(set(md.columns))

    def test_intraday_data_schema(self):
        df = generate_intraday_data("TEST", datetime(2025, 1, 6), 10, seed=42)
        required = {'datetime', 'date', 'hour', 'open', 'high', 'low', 'close', 'volume'}
        assert required.issubset(set(df.columns))

    def test_golden_cross_data_schema(self):
        df = generate_golden_cross_intraday("TEST", datetime(2025, 1, 6), 10, seed=42)
        required = {'datetime', 'date', 'hour', 'open', 'high', 'low', 'close', 'volume'}
        assert required.issubset(set(df.columns))

    def test_synthetic_determinism(self):
        df1 = generate_golden_cross_intraday("AAPL", datetime(2025, 1, 6), 30, seed=123)
        df2 = generate_golden_cross_intraday("AAPL", datetime(2025, 1, 6), 30, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_trading_day_filter(self):
        # Saturday
        assert is_trading_day(date(2025, 1, 4)) is False
        # Sunday
        assert is_trading_day(date(2025, 1, 5)) is False
        # Monday
        assert is_trading_day(date(2025, 1, 6)) is True


# ===================================================================
# Integration Test – Full Backtest Determinism
# ===================================================================

class TestFullBacktestDeterminism:
    """
    Run a complete backtest with synthetic data and capture ALL key metrics.
    After optimizations, re-running this test ensures identical results.
    """

    SEED = 42
    SYMBOLS = ["SYM_A", "SYM_B", "SYM_C"]
    DAYS = 60
    CAPITAL = 10000.0

    def _run_full_backtest(self):
        """Run the reference backtest and return the result dict."""
        strategy = IntradayTradingStrategy(
            short_sma=20,
            long_sma=50,
            golden_cross_buy_days=2,
            short_sma_downtrend=14,
            short_sma_take_profit=5,
            long_sma_take_profit=7,
            stop_loss_pct=5.0,
            take_profit_pct=0.70,
            use_stop_loss=True,
            use_market_filter=True,
            use_eod_filter=True,
            use_profit_before_eod=True,
            use_price_5hr_check=True,
            use_price_cross_check=True,
            use_dynamic_sma=True,
            use_slope_ordering=True,
            slope_threshold=0.0008,
            account_balance=30000.0,
        )
        bt = IntradayBacktester(
            initial_capital=self.CAPITAL,
            max_positions=5,
            max_position_pct=15.0,
            use_total_investment_cap=2,
            strategy=strategy,
            day_trade_limit=3,
        )
        result = bt.run(
            symbols=self.SYMBOLS,
            days=self.DAYS,
            seed=self.SEED,
            verbose=False,
        )
        return result, bt

    def test_deterministic_results(self):
        """Two runs with same seed produce identical metrics."""
        r1, _ = self._run_full_backtest()
        r2, _ = self._run_full_backtest()
        for key in ['total_return_pct', 'win_rate', 'sharpe_ratio',
                     'max_drawdown_pct', 'total_trades', 'winning_trades',
                     'losing_trades', 'final_capital', 'trades_per_day']:
            assert r1[key] == r2[key], f"Mismatch on {key}: {r1[key]} vs {r2[key]}"

    def test_result_keys_present(self):
        """Result dict contains all expected keys."""
        r, _ = self._run_full_backtest()
        expected_keys = [
            'initial_capital', 'final_capital', 'total_return', 'total_return_pct',
            'total_trades', 'buy_trades', 'sell_trades',
            'winning_trades', 'losing_trades', 'win_rate',
            'avg_win', 'avg_loss', 'profit_factor',
            'max_drawdown', 'max_drawdown_pct', 'sharpe_ratio',
            'trading_days', 'trades_per_day',
            'exit_reasons', 'rejected_buys', 'features', 'strategy',
        ]
        for key in expected_keys:
            assert key in r, f"Missing key: {key}"

    def test_financial_accounting(self):
        """Final capital equals cash + positions (should be all cash at end)."""
        r, bt = self._run_full_backtest()
        # After backtest, all positions are closed
        assert len(bt.positions) == 0
        assert r['final_capital'] == pytest.approx(bt.cash, rel=1e-6)

    def test_trade_count_consistency(self):
        """buy_trades + sell_trades = total_trades; wins + losses = sell_trades."""
        r, _ = self._run_full_backtest()
        assert r['total_trades'] == r['buy_trades'] + r['sell_trades']
        assert r['winning_trades'] + r['losing_trades'] == r['sell_trades']

    def test_equity_curve_populated(self):
        """Equity curve should have entries for each simulated hour."""
        _, bt = self._run_full_backtest()
        assert len(bt.equity_curve) > 0
        # First entry should be close to initial capital
        assert bt.equity_curve[0][1] == pytest.approx(self.CAPITAL, rel=0.01)

    def test_exit_reasons_match_sells(self):
        """Sum of exit reason counts should equal sell_trades."""
        r, _ = self._run_full_backtest()
        total_exits = sum(v['count'] for v in r['exit_reasons'].values())
        assert total_exits == r['sell_trades']


class TestFullBacktestWithMoreSymbols:
    """
    A slightly larger integration test with more symbols to stress the
    simulation loop. Constrained to finish well under 2 minutes.
    """

    SEED = 99
    SYMBOLS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    DAYS = 90
    CAPITAL = 50000.0

    def _run_backtest(self):
        strategy = IntradayTradingStrategy(
            short_sma=20, long_sma=50,
            golden_cross_buy_days=3,
            stop_loss_pct=5.0,
            take_profit_pct=0.70,
            slope_threshold=0.0005,
            account_balance=50000.0,
        )
        bt = IntradayBacktester(
            initial_capital=self.CAPITAL,
            max_positions=10,
            max_position_pct=10.0,
            use_total_investment_cap=2,
            strategy=strategy,
        )
        return bt.run(symbols=self.SYMBOLS, days=self.DAYS, seed=self.SEED, verbose=False), bt

    def test_deterministic(self):
        r1, _ = self._run_backtest()
        r2, _ = self._run_backtest()
        assert r1['total_return_pct'] == r2['total_return_pct']
        assert r1['total_trades'] == r2['total_trades']
        assert r1['final_capital'] == r2['final_capital']

    def test_accounting(self):
        r, bt = self._run_backtest()
        assert len(bt.positions) == 0
        assert r['final_capital'] == pytest.approx(bt.cash, rel=1e-6)


class TestBacktestSnapshotValues:
    """
    Capture exact numerical results from the current (pre-optimization) code.
    After implementing speed improvements, these values MUST remain identical.

    The snapshot values are populated by running this test once first and then
    hard-coding the results. If they are None, the test just records them.
    """

    SEED = 42
    SYMBOLS = ["SYM_A", "SYM_B", "SYM_C"]
    DAYS = 60

    SNAPSHOT = {
        'total_return_pct': 0.76,
        'final_capital': 10075.52,
        'total_trades': 32,
        'winning_trades': 14,
        'losing_trades': 2,
        'win_rate': 87.5,
        'max_drawdown_pct': 0.47,
        'sharpe_ratio': 2.1037,
        'trades_per_day': 0.16,
    }

    def _run(self):
        strategy = IntradayTradingStrategy(
            short_sma=20, long_sma=50, golden_cross_buy_days=2,
            short_sma_downtrend=14, short_sma_take_profit=5, long_sma_take_profit=7,
            stop_loss_pct=5.0, take_profit_pct=0.70,
            use_stop_loss=True, use_market_filter=True, use_eod_filter=True,
            use_profit_before_eod=True, use_price_5hr_check=True,
            use_price_cross_check=True, use_dynamic_sma=True,
            use_slope_ordering=True, slope_threshold=0.0008,
            account_balance=30000.0,
        )
        bt = IntradayBacktester(
            initial_capital=10000.0, max_positions=5, max_position_pct=15.0,
            use_total_investment_cap=2, strategy=strategy, day_trade_limit=3,
        )
        return bt.run(symbols=self.SYMBOLS, days=self.DAYS, seed=self.SEED, verbose=False)

    def test_snapshot(self):
        """
        Verify exact results match captured snapshot.
        If SNAPSHOT is None, print values and skip (run once to capture).
        """
        r = self._run()

        snapshot_keys = [
            'total_return_pct', 'final_capital', 'total_trades',
            'winning_trades', 'losing_trades', 'win_rate',
            'max_drawdown_pct', 'sharpe_ratio', 'trades_per_day',
        ]

        for k in snapshot_keys:
            expected = self.SNAPSHOT[k]
            actual = r[k]
            if isinstance(expected, float):
                assert float(actual) == pytest.approx(expected, abs=1e-6), (
                    f"Snapshot mismatch on '{k}': expected {expected}, got {actual}"
                )
            else:
                assert actual == expected, (
                    f"Snapshot mismatch on '{k}': expected {expected}, got {actual}"
                )
