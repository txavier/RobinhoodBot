#!/usr/bin/env python3
"""
Intraday Genetic Algorithm Optimizer for RobinhoodBot

This module uses a genetic algorithm to evolve trading strategy parameters
for INTRADAY (day trading) using hourly data - matching actual main.py behavior.

Unlike genetic_optimizer.py (which uses daily data), this optimizer:
- Uses hourly data for realistic day trading simulation
- Includes all main.py filters (market trend, EOD, slope ordering, etc.)
- Can optimize filter parameters (slope threshold, price cap)
- Produces parameters that match real day trading performance

Usage:
    python genetic_optimizer_intraday.py --generations 20 --population 30
    python genetic_optimizer_intraday.py --symbols AAPL,MSFT --generations 15 --seed 42
"""

import argparse
import json
import os
import random
import copy
import shutil
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Ray support for distributed computing (optional)
# Falls back to multiprocessing.Pool if Ray is not available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from backtest_intraday import (
    IntradayBacktester, IntradayTradingStrategy,
    generate_market_data, generate_golden_cross_intraday,
    download_real_data, download_real_market_data,
    download_raw_market_index_data, compute_market_conditions,
    YFINANCE_AVAILABLE
)

# Load stock universe from external JSON file for --num-stocks mode.
_SYMBOLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'genetic_optimizer_test_symbols.json')
with open(_SYMBOLS_FILE, 'r') as _f:
    STOCK_UNIVERSE = json.load(_f)['symbols']

# Import config defaults
try:
    from config import (
        stop_loss_percent, take_profit_percent,
        golden_cross_buy_days
    )
except ImportError:
    stop_loss_percent = 5
    take_profit_percent = 0.70
    golden_cross_buy_days = 3


@dataclass
class IntradayTradingGene:
    """
    Represents a chromosome containing intraday trading strategy parameters.
    Each gene is a parameter that can be evolved.
    
    Includes main.py specific parameters like:
    - Hourly SMA periods (not daily)
    - Slope threshold for order_symbols_by_slope
    - Price cap
    - Filter toggle options
    """
    # SMA parameters (in HOURS, not days)
    short_sma: int = 20          # Range: 5-50 hours (~1-7 trading days)
    long_sma: int = 50           # Range: 20-100 hours (~3-14 trading days)
    golden_cross_buy_days: int = 2 # Range: 1-10 trading days (1 day = 7 hourly bars)
    
    # Dynamic SMA parameters (used when use_dynamic_sma=True)
    short_sma_downtrend: int = 14  # Range: 5-30 hours - used when market not in uptrend
    short_sma_take_profit: int = 5  # Range: 3-15 hours - used after take profit (only if balance < $25k PDT limit)
    long_sma_take_profit: int = 7   # Range: 5-20 hours - used after take profit (only if balance < $25k PDT limit)
    
    # Risk management
    use_stop_loss: bool = True   # Enable/disable stop loss selling
    stop_loss_pct: float = 5.0   # Range: 1-15%
    take_profit_pct: float = 0.7 # Range: 0.3-3.0%
    
    # Position sizing (per-stock % of equity, mode 2)
    position_size_pct: float = 5.0  # Range: 1-10%
    
    # Main.py specific parameters
    slope_threshold: float = 0.0008  # Range: 0.0001-0.002 (from order_symbols_by_slope)
    
    # Market trend detection parameters
    uptrend_threshold_pct: float = 0.1   # Range: 0.0-0.5 - min % above day open to count as uptrend
    major_downtrend_threshold_pct: float = 1.0  # Range: 0.3-3.0 - min % below week open for major downtrend
    use_momentum_check: bool = True      # Enable intraday momentum direction check
    momentum_lookback_bars: int = 12     # Range: 3-30 - number of 5-min bars for momentum (12 = 1hr)
    
    # Filter toggles (True = enabled, matching main.py behavior)
    use_market_filter: bool = True
    use_eod_filter: bool = True
    use_profit_before_eod: bool = True
    use_price_5hr_check: bool = True
    use_price_cross_check: bool = True
    use_dynamic_sma: bool = True
    use_slope_ordering: bool = True
    use_total_investment_cap: int = 2  # Fixed to mode 2 (per-stock pct)
    
    # Fitness score (set after evaluation)
    fitness: float = 0.0
    
    # Backtest results for reference
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    trades_per_day: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Exit reason tracking (Enhancement #10 - tracks which sell strategies are triggering)
    exit_reasons: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IntradayTradingGene':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def __str__(self) -> str:
        return (f"SMA({self.short_sma}/{self.long_sma}) GC:{self.golden_cross_buy_days}d "
                f"SL:{self.stop_loss_pct:.1f}% TP:{self.take_profit_pct:.2f}% "
                f"Slope:{self.slope_threshold:.4f} UT:{self.uptrend_threshold_pct:.2f}% "
                f"DT:{self.major_downtrend_threshold_pct:.1f}% Mom:{self.momentum_lookback_bars}b "
                f"| Fit:{self.fitness:.4f}")


@dataclass
class IntradayGeneticConfig:
    """Configuration for the intraday genetic algorithm"""
    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_size: int = 3  # Top performers kept unchanged
    tournament_size: int = 5
    num_workers: int = 0  # 0 = auto (cpu_count), 1 = no multiprocessing
    use_ray: bool = False  # True = use Ray for distributed computing (local or K8s cluster)
    disable_ray_mem_monitor: bool = False  # True = disable Ray memory monitor (fixes cgroup v2 issues)
    
    # Whether to optimize filter settings (True = can toggle filters on/off)
    optimize_filters: bool = False
    
    # Cross-validation settings (anti-overfitting)
    # validation_mode: 'kfold' = K-fold CV, 'walkforward' = walk-forward, 'both' = combined,
    #                  'simple' = legacy train/test split, 'none' = disabled
    validation_mode: str = 'both'  # Default: use both K-fold and walk-forward
    kfold_splits: int = 5          # Number of folds for K-fold CV
    walkforward_windows: int = 3   # Number of expanding windows for walk-forward validation
    
    # Parameter ranges for random generation and mutation
    param_ranges: Dict = field(default_factory=lambda: {
        'short_sma': (5, 80),         # Hours (upper expanded from 50 - best hit 46/50)
        'long_sma': (20, 100),        # Hours
        'golden_cross_buy_days': (1, 10), # Trading days (1 day = 7 hourly bars)
        'short_sma_downtrend': (5, 50),  # Hours - more conservative SMA in downtrend (upper expanded from 30 - best hit 27/30)
        'short_sma_take_profit': (3, 15),  # Hours - aggressive SMA after take profit
        'long_sma_take_profit': (2, 20),   # Hours - aggressive SMA after take profit (lower expanded from 5 - best hit 5/5)
        'stop_loss_pct': (0.5, 15.0),  # (lower expanded from 1.0 - best hit 2.1/1.0)
        'take_profit_pct': (0.3, 3.0), # Tighter range for day trading
        'position_size_pct': (1.0, 10.0),  # Per-stock % of equity (mode 2)
        'slope_threshold': (0.0001, 0.002),
        'uptrend_threshold_pct': (0.0, 0.5),
        'major_downtrend_threshold_pct': (0.3, 3.0),
        'momentum_lookback_bars': (3, 30),
    })
    
    # Fitness weights (how much each metric contributes to fitness)
    # Adjusted for day trading: emphasize consistency and risk-adjusted returns
    # UPDATED 2026-01-13: Recalibrated based on real trade log analysis:
    # - Real trading shows 75% win rate matters more than big returns
    # - Most wins are small (59% in 0-2% range), so consistency > magnitude
    # - profit_before_eod is most common successful exit (55% of exits)
    fitness_weights: Dict = field(default_factory=lambda: {
        'total_return_pct': 0.20,      # 20% weight on total return (down from 25% - big returns rare in real trading)
        'sharpe_ratio': 0.25,          # 25% weight on risk-adjusted return (down from 30%)
        'win_rate': 0.25,              # 25% weight on win rate (up from 15% - consistency matters most)
        'profit_factor': 0.15,         # 15% weight on profit factor
        'max_drawdown_penalty': 0.10,  # 10% penalty for drawdown
        'trades_per_day_bonus': 0.05,  # 5% bonus for active trading
    })


# Module-level globals for shared data across multiprocessing workers.
# On Linux (fork), child processes inherit these via copy-on-write —
# no pickling/piping needed. Set before Pool creation in _evaluate_with_multiprocessing().
_shared_real_data_cache = None
_shared_raw_market_index_data = None
# Cross-validation fold data (also copy-on-write via fork)
_shared_cv_folds = None    # List of (train_cache, train_market, test_cache, test_market) tuples
_shared_wf_windows = None  # List of (train_cache, train_market, test_cache, test_market) tuples


def _init_worker():
    """Initializer for Pool workers: ignore SIGTERM so only the parent handles it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _evaluate_gene_cv_impl(gene, symbols, days, initial_capital, fitness_weights, data_seed,
                            max_positions, cv_folds, wf_windows) -> IntradayTradingGene:
    """
    Module-level CV evaluation for multiprocessing compatibility.
    Evaluates a gene across all K-fold and/or walk-forward splits.
    Fitness = 70% average + 30% minimum across all splits (rewards consistency).
    """
    all_fitnesses = []

    # K-fold evaluations
    if cv_folds:
        for train_cache, train_market, test_cache, test_market in cv_folds:
            fold_gene = copy.deepcopy(gene)
            fold_gene = _evaluate_gene_impl(
                fold_gene, symbols, days, initial_capital,
                dict(fitness_weights), data_seed,
                max_positions, use_real_data=True,
                real_data_cache=test_cache,
                raw_market_index_data=test_market
            )
            all_fitnesses.append(fold_gene.fitness)

    # Walk-forward evaluations
    if wf_windows:
        for train_cache, train_market, test_cache, test_market in wf_windows:
            wf_gene = copy.deepcopy(gene)
            wf_gene = _evaluate_gene_impl(
                wf_gene, symbols, days, initial_capital,
                dict(fitness_weights), data_seed,
                max_positions, use_real_data=True,
                real_data_cache=test_cache,
                raw_market_index_data=test_market
            )
            all_fitnesses.append(wf_gene.fitness)

    if not all_fitnesses:
        # Fallback to standard evaluation
        return _evaluate_gene_impl(
            gene, symbols, days, initial_capital,
            dict(fitness_weights), data_seed,
            max_positions, False, None, None
        )

    # Fitness: 70% average + 30% minimum (rewards consistency across folds)
    avg_fitness = sum(all_fitnesses) / len(all_fitnesses)
    min_fitness = min(all_fitnesses)
    gene.fitness = 0.7 * avg_fitness + 0.3 * min_fitness

    # Get representative metrics from median fold for display
    median_idx = sorted(range(len(all_fitnesses)), key=lambda i: all_fitnesses[i])[len(all_fitnesses) // 2]
    n_cv = len(cv_folds) if cv_folds else 0
    if cv_folds and median_idx < n_cv:
        rep_cache = cv_folds[median_idx][2]  # test cache
        rep_market = cv_folds[median_idx][3]  # test market
    elif wf_windows:
        wf_idx = median_idx - n_cv
        rep_cache = wf_windows[wf_idx][2]
        rep_market = wf_windows[wf_idx][3]
    else:
        rep_cache = None
        rep_market = None

    if rep_cache:
        rep_gene = _evaluate_gene_impl(
            copy.deepcopy(gene), symbols, days, initial_capital,
            dict(fitness_weights), data_seed,
            max_positions, use_real_data=True,
            real_data_cache=rep_cache,
            raw_market_index_data=rep_market
        )
        gene.total_return_pct = rep_gene.total_return_pct
        gene.win_rate = rep_gene.win_rate
        gene.sharpe_ratio = rep_gene.sharpe_ratio
        gene.max_drawdown_pct = rep_gene.max_drawdown_pct
        gene.profit_factor = rep_gene.profit_factor
        gene.total_trades = rep_gene.total_trades
        gene.trades_per_day = rep_gene.trades_per_day
        gene.avg_win = rep_gene.avg_win
        gene.avg_loss = rep_gene.avg_loss
        gene.exit_reasons = rep_gene.exit_reasons

    return gene


def _evaluate_gene_worker(args: Tuple) -> IntradayTradingGene:
    """
    Module-level function for parallel gene evaluation.
    Must be at module level for multiprocessing to pickle it.
    
    When CV mode is active (_shared_cv_folds/_shared_wf_windows are set),
    evaluates across all folds/windows. Otherwise uses standard single evaluation.
    
    Large data is read from module-level globals set before Pool creation —
    avoids pickling ~4 MB per gene through pipes.
    
    Args is a tuple of (gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions,
                         use_real_data)
    """
    gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions, use_real_data = args

    # CV mode: evaluate across all folds/windows
    if _shared_cv_folds or _shared_wf_windows:
        return _evaluate_gene_cv_impl(
            gene, symbols, days, initial_capital, fitness_weights, data_seed,
            max_positions, _shared_cv_folds, _shared_wf_windows
        )

    # Standard single evaluation
    return _evaluate_gene_impl(
        gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions,
        use_real_data, _shared_real_data_cache, _shared_raw_market_index_data
    )


def _evaluate_gene_impl(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions,
                         use_real_data=False, real_data_cache=None, raw_market_index_data=None) -> IntradayTradingGene:
    """
    Core gene evaluation logic - used by both multiprocessing and Ray workers.
    """
    
    # Create strategy with gene parameters
    strategy = IntradayTradingStrategy(
        short_sma=gene.short_sma,
        long_sma=gene.long_sma,
        golden_cross_buy_days=gene.golden_cross_buy_days,
        short_sma_downtrend=gene.short_sma_downtrend,
        short_sma_take_profit=gene.short_sma_take_profit,
        long_sma_take_profit=gene.long_sma_take_profit,
        stop_loss_pct=gene.stop_loss_pct,
        take_profit_pct=gene.take_profit_pct,
        use_stop_loss=gene.use_stop_loss,
        use_market_filter=gene.use_market_filter,
        use_eod_filter=gene.use_eod_filter,
        use_profit_before_eod=gene.use_profit_before_eod,
        use_price_5hr_check=gene.use_price_5hr_check,
        use_price_cross_check=gene.use_price_cross_check,
        use_dynamic_sma=gene.use_dynamic_sma,
        use_slope_ordering=gene.use_slope_ordering,
        slope_threshold=gene.slope_threshold,
        # Market trend detection thresholds (evolved by genetic algorithm)
        uptrend_threshold_pct=gene.uptrend_threshold_pct,
        major_downtrend_threshold_pct=gene.major_downtrend_threshold_pct,
        use_momentum_check=gene.use_momentum_check,
    )
    
    # Compute market conditions using this gene's thresholds (no network call)
    real_market_cache = None
    if use_real_data and raw_market_index_data is not None:
        real_market_cache = compute_market_conditions(
            raw_market_index_data,
            uptrend_threshold_pct=gene.uptrend_threshold_pct,
            major_downtrend_threshold_pct=gene.major_downtrend_threshold_pct,
            use_momentum_check=gene.use_momentum_check,
        )
    
    # Run backtest
    backtester = IntradayBacktester(
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_position_pct=gene.position_size_pct,
        use_total_investment_cap=gene.use_total_investment_cap,
        strategy=strategy
    )
    
    result = backtester.run(
        symbols=symbols,
        days=days,
        seed=data_seed,
        verbose=False,
        use_real_data=use_real_data,
        real_data_cache=real_data_cache,
        real_market_cache=real_market_cache
    )
    
    # Store results in gene
    gene.total_return_pct = result.get('total_return_pct', 0)
    gene.win_rate = result.get('win_rate', 0)
    gene.sharpe_ratio = result.get('sharpe_ratio', 0)
    gene.max_drawdown_pct = result.get('max_drawdown_pct', 0)
    pf = result.get('profit_factor', 0)
    gene.profit_factor = min(pf, 10.0) if pf != 'inf' and pf != float('inf') else 10.0
    gene.total_trades = result.get('total_trades', 0)
    gene.trades_per_day = result.get('trades_per_day', 0)
    gene.avg_win = result.get('avg_win', 0)
    gene.avg_loss = result.get('avg_loss', 0)
    gene.exit_reasons = result.get('exit_reasons', {})
    
    # Calculate weighted fitness
    fitness = 0.0
    
    # Return contribution (can be negative)
    fitness += fitness_weights['total_return_pct'] * (gene.total_return_pct / 5.0)
    
    # Sharpe ratio contribution (typically -3 to +3)
    fitness += fitness_weights['sharpe_ratio'] * (gene.sharpe_ratio / 2.0)
    
    # Win rate contribution (0-100, normalize to 0-1)
    fitness += fitness_weights['win_rate'] * (gene.win_rate / 100.0)
    
    # Profit factor contribution (0-10, already capped)
    fitness += fitness_weights['profit_factor'] * (gene.profit_factor / 5.0)
    
    # Drawdown penalty (higher drawdown = lower fitness)
    fitness -= fitness_weights['max_drawdown_penalty'] * (gene.max_drawdown_pct / 5.0)
    
    # Trades per day bonus (reward active trading, but not too much)
    optimal_trades_per_day = 1.0
    trades_diff = abs(gene.trades_per_day - optimal_trades_per_day)
    trades_bonus = max(0, 1.0 - trades_diff * 0.5)
    fitness += fitness_weights['trades_per_day_bonus'] * trades_bonus
    
    # Penalties
    if gene.total_trades == 0:
        fitness -= 2.0
    elif gene.total_trades < 5:
        fitness -= 0.5
    
    # Bonus for good avg_win/avg_loss ratio
    if gene.avg_loss > 0:
        win_loss_ratio = gene.avg_win / gene.avg_loss
        if win_loss_ratio > 1.0:
            fitness += 0.1 * min(win_loss_ratio - 1.0, 1.0)
    
    gene.fitness = fitness
    return gene


class IntradayGeneticOptimizer:
    """
    Genetic algorithm optimizer for intraday trading strategy parameters.
    Uses hourly data and main.py matching filters.
    """
    
    def __init__(
        self,
        symbols: List[str],
        days: int = 60,
        initial_capital: float = 30000.0,
        config: Optional[IntradayGeneticConfig] = None,
        verbose: bool = True,
        seed: int = None,
        max_positions: int = 5,
        use_real_data: bool = False,
        log_file: Optional[str] = None,
        train_test_split: float = 0.0
    ):
        self.symbols = symbols
        self.days = days
        self.initial_capital = initial_capital
        self.config = config or IntradayGeneticConfig()
        self.verbose = verbose
        self.seed = seed
        self.max_positions = max_positions
        self.use_real_data = use_real_data
        self.log_file = log_file
        self._log_fh = None
        self.train_test_split = train_test_split  # 0.0 = disabled, 0.7 = 70% train / 30% test
        
        # Open log file if specified
        if self.log_file:
            self._log_fh = open(self.log_file, 'a')
        
        # Real data caches (populated in run() if use_real_data=True)
        self.real_data_cache: Optional[Dict] = None
        self.real_market_cache = None  # Legacy - kept for compatibility
        self.raw_market_index_data = None  # Raw SPY/DIA/QQQ OHLC for per-gene threshold computation
        
        # Train/test split caches (populated if train_test_split > 0)
        self._full_real_data_cache: Optional[Dict] = None   # All dates (for test evaluation)
        self._full_raw_market_index_data = None              # All dates (for test evaluation)
        self._test_real_data_cache: Optional[Dict] = None    # Test-only dates
        self._test_raw_market_index_data = None              # Test-only dates
        self._train_dates: Optional[List] = None
        self._test_dates: Optional[List] = None
        self.test_set_metrics: Optional[Dict] = None         # Best gene's test-set performance
        
        # Cross-validation data splits (populated by _create_cv_splits())
        self._cv_folds: Optional[List] = None                 # List of (train_cache, train_market, test_cache, test_market) tuples
        self._wf_windows: Optional[List] = None               # List of (train_cache, train_market, test_cache, test_market) tuples
        self._cv_fold_info: Optional[List] = None             # Metadata about each fold (dates, sizes)
        self._wf_window_info: Optional[List] = None           # Metadata about each window
        
        # Track evolution history
        self.generation_history: List[Dict] = []
        self.best_gene: Optional[IntradayTradingGene] = None
        self.best_fitness: float = float('-inf')
        
        # Checkpoint/resume support
        self.checkpoint_path: Optional[str] = None
        self._interrupted = False
    
    def _log(self, msg: str, **kwargs):
        """Print to stdout and optionally write to log file (unbuffered)."""
        print(msg, **kwargs)
        if self._log_fh:
            # Handle end='' or flush=True from kwargs
            end = kwargs.get('end', '\n')
            try:
                self._log_fh.write(msg + end)
                self._log_fh.flush()
            except Exception:
                pass  # Don't crash if log file write fails
        
    @staticmethod
    def _archive_file(filepath: str):
        """Archive an existing file by copying it to a timestamped version.
        
        Creates a copy like:
          genetic_optimization_intraday_result.json
          → genetic_optimization_intraday_result.2026-03-07_143022.json
        
        Only archives if the file exists and is non-empty.
        """
        if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
            return
        try:
            mtime = os.path.getmtime(filepath)
            ts = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d_%H%M%S')
            base, ext = os.path.splitext(filepath)
            archive_path = f"{base}.{ts}{ext}"
            # Don't overwrite an existing archive with the same timestamp
            if not os.path.exists(archive_path):
                shutil.copy2(filepath, archive_path)
        except OSError:
            pass  # Don't crash if archiving fails

    def save_checkpoint(self, filepath: str, generation: int, population: List[IntradayTradingGene]):
        """Save optimizer state to a checkpoint file after each generation.
        
        This allows resuming from the last completed generation if the process
        is interrupted (Ctrl+C, crash, etc.).
        """
        checkpoint = {
            'checkpoint_version': 1,
            'saved_at': str(datetime.now()),
            'start_time': str(self.start_time) if hasattr(self, 'start_time') else None,
            'completed_generation': generation,
            'total_generations': self.config.generations,
            # Optimizer parameters (to verify compatibility on resume)
            'symbols': self.symbols,
            'days': self.days,
            'initial_capital': self.initial_capital,
            'max_positions': self.max_positions,
            'seed': self.seed,
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size,
                'optimize_filters': self.config.optimize_filters,
                'num_workers': self.config.num_workers,
            },
            # State to restore
            'best_gene': self.best_gene.to_dict() if self.best_gene else None,
            'best_fitness': self.best_fitness,
            'generation_history': self.generation_history,
            'population': [g.to_dict() for g in population],
            # Random state for reproducibility
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
        }
        
        # Custom serializer for numpy types
        def numpy_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Write to temp file first, then rename (atomic on same filesystem)
        tmp_path = filepath + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=numpy_serializer)
        os.replace(tmp_path, filepath)
        
        if self.verbose:
            self._log(f"  💾 Checkpoint saved: Gen {generation + 1}/{self.config.generations} → {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Optional[dict]:
        """Load a checkpoint file and return the checkpoint data.
        
        Returns None if the file doesn't exist or is invalid.
        Validates that the checkpoint is compatible with current optimizer settings.
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️  Warning: Could not load checkpoint {filepath}: {e}")
            return None
        
        # Validate checkpoint version
        if checkpoint.get('checkpoint_version') != 1:
            print(f"  ⚠️  Warning: Unknown checkpoint version, ignoring")
            return None
        
        # Validate compatibility - warn but allow resume if symbols differ
        # (config params like population size must match)
        cp_config = checkpoint.get('config', {})
        if cp_config.get('population_size') != self.config.population_size:
            print(f"  ⚠️  Warning: Checkpoint population size ({cp_config.get('population_size')}) "
                  f"differs from current ({self.config.population_size}). Cannot resume.")
            return None
        if cp_config.get('generations') != self.config.generations:
            print(f"  ⚠️  Warning: Checkpoint generations ({cp_config.get('generations')}) "
                  f"differs from current ({self.config.generations}). Cannot resume.")
            return None
        
        # Validate symbols match
        if checkpoint.get('symbols') != self.symbols:
            print(f"  ⚠️  Warning: Checkpoint symbols differ from current run. Cannot resume.")
            return None
        
        return checkpoint
    
    def restore_from_checkpoint(self, checkpoint: dict) -> Tuple[int, List[IntradayTradingGene]]:
        """Restore optimizer state from a checkpoint.
        
        Returns (start_generation, population) to resume from.
        start_generation is the NEXT generation to run (0-indexed).
        """
        # Restore best gene and fitness
        if checkpoint['best_gene']:
            self.best_gene = IntradayTradingGene.from_dict(checkpoint['best_gene'])
        self.best_fitness = checkpoint['best_fitness']
        
        # Restore generation history
        self.generation_history = checkpoint['generation_history']
        
        # Restore population
        population = [IntradayTradingGene.from_dict(g) for g in checkpoint['population']]
        
        # Restore random state for reproducibility
        if 'random_state' in checkpoint:
            try:
                # random.getstate() returns a tuple with internal state
                state = checkpoint['random_state']
                # Convert the internal state list back to a tuple as required
                if isinstance(state, list):
                    state[1] = tuple(state[1])
                    state = tuple(state)
                random.setstate(state)
            except Exception:
                pass  # Non-critical - just won't be perfectly reproducible
        
        if 'numpy_random_state' in checkpoint:
            try:
                np_state = checkpoint['numpy_random_state']
                # numpy state: (string, ndarray, int, int, float)
                np.random.set_state((
                    np_state[0],
                    np.array(np_state[1], dtype=np.uint32),
                    np_state[2],
                    np_state[3],
                    np_state[4]
                ))
            except Exception:
                pass  # Non-critical
        
        # Restore start time
        if checkpoint.get('start_time') and checkpoint['start_time'] != 'None':
            try:
                self.start_time = datetime.fromisoformat(checkpoint['start_time'])
            except (ValueError, TypeError):
                self.start_time = datetime.now()
        
        # The next generation to run is completed_generation + 1 (0-indexed)
        # completed_generation is 0-indexed (gen 0 = "Generation 1/20")
        start_gen = checkpoint['completed_generation'] + 1
        
        return start_gen, population
    
    def _split_data_by_dates(self):
        """Split real data caches into train/test sets based on train_test_split ratio.
        
        Anti-overfitting: Evolution only sees train data. After optimization,
        the best gene is re-evaluated on unseen test data to verify generalization.
        """
        if self.train_test_split <= 0 or self.train_test_split >= 1.0:
            return
        if not self.real_data_cache:
            self._log("  ⚠️  Train/test split requires --real-data. Skipping split.")
            self.train_test_split = 0.0
            return
        
        # Collect all unique trading dates across all symbols
        all_dates = set()
        for df in self.real_data_cache.values():
            all_dates.update(df['date'].unique())
        all_dates = sorted(all_dates)
        
        if len(all_dates) < 10:
            self._log(f"  ⚠️  Only {len(all_dates)} trading days — too few for train/test split. Disabling.")
            self.train_test_split = 0.0
            return
        
        # Split dates
        split_idx = int(len(all_dates) * self.train_test_split)
        split_idx = max(5, min(split_idx, len(all_dates) - 3))  # Ensure at least 3 test days
        self._train_dates = all_dates[:split_idx]
        self._test_dates = all_dates[split_idx:]
        train_date_set = set(self._train_dates)
        test_date_set = set(self._test_dates)
        
        self._log(f"\n🔀 TRAIN/TEST SPLIT: {self.train_test_split:.0%}")
        self._log(f"   Train: {len(self._train_dates)} days ({self._train_dates[0]} → {self._train_dates[-1]})")
        self._log(f"   Test:  {len(self._test_dates)} days ({self._test_dates[0]} → {self._test_dates[-1]})")
        self._log(f"   Evolution fitness uses TRAIN data only. Test data is held out.")
        
        # Save full data for later test evaluation
        self._full_real_data_cache = self.real_data_cache
        self._full_raw_market_index_data = self.raw_market_index_data
        
        # Create train-only data cache (filter DataFrames to train dates)
        train_data_cache = {}
        for symbol, df in self.real_data_cache.items():
            train_df = df[df['date'].isin(train_date_set)].copy().reset_index(drop=True)
            if not train_df.empty:
                train_data_cache[symbol] = train_df
        
        # Create test-only data cache
        test_data_cache = {}
        for symbol, df in self.real_data_cache.items():
            test_df = df[df['date'].isin(test_date_set)].copy().reset_index(drop=True)
            if not test_df.empty:
                test_data_cache[symbol] = test_df
        
        # Create train-only raw market index data: {symbol: {date: {open, close}}}
        train_raw_market = {}
        for idx_sym, date_dict in self.raw_market_index_data.items():
            train_raw_market[idx_sym] = {d: v for d, v in date_dict.items() if d in train_date_set}
        
        # Create test-only raw market index data
        test_raw_market = {}
        for idx_sym, date_dict in self.raw_market_index_data.items():
            test_raw_market[idx_sym] = {d: v for d, v in date_dict.items() if d in test_date_set}
        
        # Store test caches
        self._test_real_data_cache = test_data_cache
        self._test_raw_market_index_data = test_raw_market
        
        # Replace active caches with train-only data (evolution will only see these)
        self.real_data_cache = train_data_cache
        self.raw_market_index_data = train_raw_market
        
        self._log(f"   Train symbols with data: {len(train_data_cache)}")
        self._log(f"   Test symbols with data:  {len(test_data_cache)}")
    
    def _filter_data_by_dates(self, date_set: set) -> Tuple[Dict, Dict]:
        """Filter data caches to a specific set of dates.
        
        Uses _full_real_data_cache if available (train/test split was done),
        otherwise falls back to self.real_data_cache (no split).
        
        Returns:
            (filtered_data_cache, filtered_raw_market_index_data)
        """
        source_cache = self._full_real_data_cache or self.real_data_cache
        source_market = self._full_raw_market_index_data or self.raw_market_index_data
        
        data_cache = {}
        for symbol, df in source_cache.items():
            filtered = df[df['date'].isin(date_set)].copy().reset_index(drop=True)
            if not filtered.empty:
                data_cache[symbol] = filtered
        
        raw_market = {}
        for idx_sym, date_dict in source_market.items():
            raw_market[idx_sym] = {d: v for d, v in date_dict.items() if d in date_set}
        
        return data_cache, raw_market
    
    def _create_cv_splits(self):
        """Create K-fold cross-validation and/or walk-forward validation splits.
        
        When train/test split is active, CV folds use only training data
        (test holdout remains completely unseen during evolution).
        When no split is active, CV folds use all available data.
        
        K-fold: Splits dates into K chronological folds. For each fold,
               that fold is the validation set and the rest are training.
               Fitness = average fitness across all K evaluations.
        
        Walk-forward: Creates expanding training windows with fixed-size test windows.
               Window 1: Train on first 40% of dates, test on next 20%
               Window 2: Train on first 60%, test on next 20%
               Window 3: Train on first 80%, test on final 20%
               Fitness = average fitness across all window evaluations.
        """
        # Use training data when split is active; otherwise all data
        source_cache = self.real_data_cache
        if not source_cache:
            self._log("  ⚠️  CV splits require real data. Skipping.")
            return
        
        # Collect unique trading dates from the source data
        all_dates = set()
        for df in source_cache.values():
            all_dates.update(df['date'].unique())
        all_dates = sorted(all_dates)
        n_dates = len(all_dates)
        
        mode = self.config.validation_mode
        
        # K-fold cross-validation splits
        if mode in ('kfold', 'both'):
            k = self.config.kfold_splits
            if n_dates < k * 5:
                self._log(f"  ⚠️  Only {n_dates} dates — too few for {k}-fold CV. Reducing to {max(2, n_dates // 5)} folds.")
                k = max(2, n_dates // 5)
                self.config.kfold_splits = k
            
            fold_size = n_dates // k
            self._cv_folds = []
            self._cv_fold_info = []
            
            self._log(f"\n🔀 K-FOLD CROSS-VALIDATION: {k} folds")
            
            for i in range(k):
                # Test fold is the i-th chunk
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < k - 1 else n_dates
                test_dates = set(all_dates[test_start:test_end])
                train_dates = set(all_dates) - test_dates
                
                train_cache, train_market = self._filter_data_by_dates(train_dates)
                test_cache, test_market = self._filter_data_by_dates(test_dates)
                
                self._cv_folds.append((train_cache, train_market, test_cache, test_market))
                
                sorted_test = sorted(test_dates)
                sorted_train = sorted(train_dates)
                fold_info = {
                    'fold': i + 1,
                    'train_days': len(train_dates),
                    'test_days': len(test_dates),
                    'test_period': f"{sorted_test[0]} → {sorted_test[-1]}",
                    'train_periods': f"{sorted_train[0]} → {sorted_train[-1]}",
                }
                self._cv_fold_info.append(fold_info)
                self._log(f"   Fold {i+1}: test={len(test_dates)}d ({sorted_test[0]}→{sorted_test[-1]}), "
                          f"train={len(train_dates)}d")
        
        # Walk-forward validation splits
        if mode in ('walkforward', 'both'):
            n_windows = self.config.walkforward_windows
            if n_dates < n_windows * 10:
                self._log(f"  ⚠️  Only {n_dates} dates — too few for {n_windows} walk-forward windows. Reducing to {max(2, n_dates // 10)}.")
                n_windows = max(2, n_dates // 10)
                self.config.walkforward_windows = n_windows
            
            # Each window's test period is ~20% of total dates
            test_size = max(5, n_dates // (n_windows + 1))
            
            self._wf_windows = []
            self._wf_window_info = []
            
            self._log(f"\n📈 WALK-FORWARD VALIDATION: {n_windows} expanding windows")
            
            for i in range(n_windows):
                # Training: all dates up to the test window start
                # Test: the next test_size dates
                train_end_idx = (i + 1) * test_size  # Expanding window
                if train_end_idx >= n_dates - test_size:
                    train_end_idx = n_dates - test_size  # Ensure there's a test window
                test_end_idx = min(train_end_idx + test_size, n_dates)
                
                # Skip if we'd have no test data
                if train_end_idx >= n_dates or test_end_idx <= train_end_idx:
                    continue
                
                train_dates = set(all_dates[:train_end_idx])
                test_dates_wf = set(all_dates[train_end_idx:test_end_idx])
                
                train_cache, train_market = self._filter_data_by_dates(train_dates)
                test_cache, test_market = self._filter_data_by_dates(test_dates_wf)
                
                self._wf_windows.append((train_cache, train_market, test_cache, test_market))
                
                sorted_train = sorted(train_dates)
                sorted_test = sorted(test_dates_wf)
                window_info = {
                    'window': i + 1,
                    'train_days': len(train_dates),
                    'test_days': len(test_dates_wf),
                    'train_period': f"{sorted_train[0]} → {sorted_train[-1]}",
                    'test_period': f"{sorted_test[0]} → {sorted_test[-1]}",
                }
                self._wf_window_info.append(window_info)
                self._log(f"   Window {i+1}: train={len(train_dates)}d ({sorted_train[0]}→{sorted_train[-1]}) | "
                          f"test={len(test_dates_wf)}d ({sorted_test[0]}→{sorted_test[-1]})")
        
        # Summary
        total_evals = 0
        if self._cv_folds:
            total_evals += len(self._cv_folds)
        if self._wf_windows:
            total_evals += len(self._wf_windows)
        if total_evals > 0:
            self._log(f"\n   Total evaluations per gene: {total_evals} "
                      f"(fitness = average across all splits)")
            self._log(f"   ⚠️  This will make each generation ~{total_evals}x slower but significantly reduces overfitting.\n")
    
    def _evaluate_gene_cv(self, gene: IntradayTradingGene, data_seed: Optional[int] = None) -> float:
        """Evaluate a gene across all cross-validation folds and walk-forward windows.
        
        Returns the average fitness across all splits. Also stores per-fold metrics
        in gene.cv_details for diagnostics.
        
        If no CV splits are configured, falls back to standard single evaluation.
        """
        has_cv = self._cv_folds is not None and len(self._cv_folds) > 0
        has_wf = self._wf_windows is not None and len(self._wf_windows) > 0
        
        if not has_cv and not has_wf:
            # No CV configured — use standard evaluation
            return self.evaluate_fitness(gene, data_seed)
        
        all_fitnesses = []
        all_metrics = []
        
        # K-fold evaluations
        if has_cv:
            for i, (train_cache, train_market, test_cache, test_market) in enumerate(self._cv_folds):
                fold_gene = copy.deepcopy(gene)
                # Evaluate on the TEST portion of this fold (training was done on the rest)
                fold_gene = _evaluate_gene_impl(
                    fold_gene, self.symbols, self.days, self.initial_capital,
                    dict(self.config.fitness_weights), data_seed,
                    self.max_positions, use_real_data=True,
                    real_data_cache=test_cache,
                    raw_market_index_data=test_market
                )
                all_fitnesses.append(fold_gene.fitness)
                all_metrics.append({
                    'type': f'kfold_{i+1}',
                    'fitness': fold_gene.fitness,
                    'return_pct': fold_gene.total_return_pct,
                    'win_rate': fold_gene.win_rate,
                    'sharpe': fold_gene.sharpe_ratio,
                    'trades': fold_gene.total_trades,
                })
        
        # Walk-forward evaluations
        if has_wf:
            for i, (train_cache, train_market, test_cache, test_market) in enumerate(self._wf_windows):
                wf_gene = copy.deepcopy(gene)
                # Evaluate on the test window (forward portion)
                wf_gene = _evaluate_gene_impl(
                    wf_gene, self.symbols, self.days, self.initial_capital,
                    dict(self.config.fitness_weights), data_seed,
                    self.max_positions, use_real_data=True,
                    real_data_cache=test_cache,
                    raw_market_index_data=test_market
                )
                all_fitnesses.append(wf_gene.fitness)
                all_metrics.append({
                    'type': f'walkforward_{i+1}',
                    'fitness': wf_gene.fitness,
                    'return_pct': wf_gene.total_return_pct,
                    'win_rate': wf_gene.win_rate,
                    'sharpe': wf_gene.sharpe_ratio,
                    'trades': wf_gene.total_trades,
                })
        
        # Average fitness across all splits
        avg_fitness = sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0.0
        
        # Also compute the minimum fitness (worst fold) — penalize inconsistency
        min_fitness = min(all_fitnesses) if all_fitnesses else 0.0
        fitness_std = np.std(all_fitnesses) if len(all_fitnesses) > 1 else 0.0
        
        # Final fitness: 70% avg + 30% min (rewards consistency across folds)
        gene.fitness = 0.7 * avg_fitness + 0.3 * min_fitness
        
        # Store metrics from the last evaluation for display purposes
        # (use the fold with median fitness for representative metrics)
        if all_fitnesses:
            median_idx = sorted(range(len(all_fitnesses)), key=lambda i: all_fitnesses[i])[len(all_fitnesses) // 2]
            # Re-evaluate on the median fold to get representative display metrics
            if has_cv and median_idx < len(self._cv_folds):
                rep_cache = self._cv_folds[median_idx][2]  # test cache
                rep_market = self._cv_folds[median_idx][3]  # test market
            elif has_wf:
                wf_idx = median_idx - (len(self._cv_folds) if has_cv else 0)
                rep_cache = self._wf_windows[wf_idx][2]
                rep_market = self._wf_windows[wf_idx][3]
            else:
                rep_cache = None
                rep_market = None
            
            if rep_cache:
                rep_gene = _evaluate_gene_impl(
                    copy.deepcopy(gene), self.symbols, self.days, self.initial_capital,
                    dict(self.config.fitness_weights), data_seed,
                    self.max_positions, use_real_data=True,
                    real_data_cache=rep_cache,
                    raw_market_index_data=rep_market
                )
                gene.total_return_pct = rep_gene.total_return_pct
                gene.win_rate = rep_gene.win_rate
                gene.sharpe_ratio = rep_gene.sharpe_ratio
                gene.max_drawdown_pct = rep_gene.max_drawdown_pct
                gene.profit_factor = rep_gene.profit_factor
                gene.total_trades = rep_gene.total_trades
                gene.trades_per_day = rep_gene.trades_per_day
                gene.avg_win = rep_gene.avg_win
                gene.avg_loss = rep_gene.avg_loss
                gene.exit_reasons = rep_gene.exit_reasons
        
        return gene.fitness
    
    def evaluate_on_test_set(self, gene: IntradayTradingGene) -> Dict:
        """Re-evaluate a gene on the held-out test data (unseen during evolution).
        
        Returns dict with test-set performance metrics.
        """
        if not self._test_real_data_cache or not self._test_raw_market_index_data:
            return {}
        
        # Evaluate using the same impl but with test data
        test_gene = copy.deepcopy(gene)
        test_gene = _evaluate_gene_impl(
            test_gene,
            self.symbols,
            self.days,  # days param is used for synthetic data sizing; real data uses actual dates
            self.initial_capital,
            dict(self.config.fitness_weights),
            data_seed=None,
            max_positions=self.max_positions,
            use_real_data=True,
            real_data_cache=self._test_real_data_cache,
            raw_market_index_data=self._test_raw_market_index_data
        )
        
        return {
            'total_return_pct': test_gene.total_return_pct,
            'win_rate': test_gene.win_rate,
            'sharpe_ratio': test_gene.sharpe_ratio,
            'max_drawdown_pct': test_gene.max_drawdown_pct,
            'profit_factor': test_gene.profit_factor,
            'total_trades': test_gene.total_trades,
            'trades_per_day': test_gene.trades_per_day,
            'avg_win': test_gene.avg_win,
            'avg_loss': test_gene.avg_loss,
            'fitness': test_gene.fitness,
            'exit_reasons': test_gene.exit_reasons,
        }

    def _setup_signal_handler(self):
        """Set up signal handlers to save checkpoint on interrupt."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def handler(signum, frame):
            sig_name = 'SIGINT (Ctrl+C)' if signum == signal.SIGINT else 'SIGTERM'
            print(f"\n\n  🛑 Received {sig_name}. Saving checkpoint before exit...")
            self._interrupted = True
            # The checkpoint will be saved at the next safe point in the run() loop
            # If we're in the middle of pool.map(), we need to let it finish or abort
            # Re-raise to stop the multiprocessing pool
            if self._original_sigint and signum == signal.SIGINT:
                signal.signal(signal.SIGINT, self._original_sigint)
                os.kill(os.getpid(), signal.SIGINT)
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        if hasattr(self, '_original_sigint') and self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if hasattr(self, '_original_sigterm') and self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def create_random_gene(self) -> IntradayTradingGene:
        """Create a random trading gene within parameter ranges"""
        ranges = self.config.param_ranges
        
        short_sma = random.randint(*ranges['short_sma'])
        # Ensure long_sma > short_sma
        long_sma_min = max(ranges['long_sma'][0], short_sma + 10)
        long_sma = random.randint(long_sma_min, ranges['long_sma'][1])
        
        gene = IntradayTradingGene(
            short_sma=short_sma,
            long_sma=long_sma,
            golden_cross_buy_days=random.randint(*ranges['golden_cross_buy_days']),
            short_sma_downtrend=random.randint(*ranges['short_sma_downtrend']),
            short_sma_take_profit=random.randint(*ranges['short_sma_take_profit']),
            long_sma_take_profit=random.randint(*ranges['long_sma_take_profit']),
            use_stop_loss=random.choice([True, True, True, False]) if self.config.optimize_filters else True,  # 75% chance True
            stop_loss_pct=round(random.uniform(*ranges['stop_loss_pct']), 1),
            take_profit_pct=round(random.uniform(*ranges['take_profit_pct']), 2),
            position_size_pct=round(random.uniform(*ranges['position_size_pct']), 1),
            slope_threshold=round(random.uniform(*ranges['slope_threshold']), 4),
            uptrend_threshold_pct=round(random.uniform(*ranges['uptrend_threshold_pct']), 2),
            major_downtrend_threshold_pct=round(random.uniform(*ranges['major_downtrend_threshold_pct']), 1),
            use_momentum_check=random.choice([True, True, True, False]),  # 75% chance True
            momentum_lookback_bars=random.randint(*ranges['momentum_lookback_bars']),
        )
        
        # Optionally randomize filter settings
        if self.config.optimize_filters:
            gene.use_market_filter = random.choice([True, False])
            gene.use_eod_filter = random.choice([True, False])
            gene.use_profit_before_eod = random.choice([True, False])
            gene.use_price_5hr_check = random.choice([True, False])
            gene.use_price_cross_check = random.choice([True, False])
            gene.use_dynamic_sma = random.choice([True, False])
            gene.use_slope_ordering = random.choice([True, False])
            gene.use_total_investment_cap = 2  # Fixed to mode 2
        
        return gene
    
    def initialize_population(self) -> List[IntradayTradingGene]:
        """Create initial random population"""
        population = []
        
        # Add some sensible defaults to seed the population
        defaults = [
            # Default main.py settings (matching config.py v0.9.21)
            IntradayTradingGene(
                short_sma=23, long_sma=100, golden_cross_buy_days=4,
                short_sma_downtrend=45, short_sma_take_profit=14, long_sma_take_profit=5,
                use_stop_loss=True, stop_loss_pct=7.9, take_profit_pct=1.18, position_size_pct=5.0,
                slope_threshold=0.0019,
                uptrend_threshold_pct=0.1, major_downtrend_threshold_pct=1.0,
                use_momentum_check=True, momentum_lookback_bars=12
            ),
            # More aggressive day trading
            IntradayTradingGene(
                short_sma=10, long_sma=30, golden_cross_buy_days=2,
                short_sma_downtrend=8, short_sma_take_profit=3, long_sma_take_profit=5,
                use_stop_loss=True, stop_loss_pct=3.0, take_profit_pct=1.0, position_size_pct=5.0,
                slope_threshold=0.001,
                uptrend_threshold_pct=0.05, major_downtrend_threshold_pct=0.5,
                use_momentum_check=True, momentum_lookback_bars=6
            ),
            # More conservative
            IntradayTradingGene(
                short_sma=30, long_sma=70, golden_cross_buy_days=5,
                short_sma_downtrend=20, short_sma_take_profit=8, long_sma_take_profit=12,
                use_stop_loss=True, stop_loss_pct=7.0, take_profit_pct=0.5, position_size_pct=8.0,
                slope_threshold=0.0005,
                uptrend_threshold_pct=0.2, major_downtrend_threshold_pct=1.5,
                use_momentum_check=True, momentum_lookback_bars=18
            ),
            # Very aggressive (quick scalping)
            IntradayTradingGene(
                short_sma=5, long_sma=15, golden_cross_buy_days=1,
                short_sma_downtrend=5, short_sma_take_profit=3, long_sma_take_profit=5,
                use_stop_loss=True, stop_loss_pct=2.0, take_profit_pct=0.5, position_size_pct=3.0,
                slope_threshold=0.0015,
                uptrend_threshold_pct=0.0, major_downtrend_threshold_pct=2.0,
                use_momentum_check=False, momentum_lookback_bars=6
            ),
        ]
        population.extend(defaults)
        
        # Fill rest with random genes
        while len(population) < self.config.population_size:
            population.append(self.create_random_gene())
        
        return population
    
    def evaluate_fitness(self, gene: IntradayTradingGene, data_seed: int = None) -> float:
        """
        Run intraday backtest with gene parameters and calculate fitness score.
        Fitness is a weighted combination of multiple metrics.
        """
        # Create strategy with gene parameters
        strategy = IntradayTradingStrategy(
            short_sma=gene.short_sma,
            long_sma=gene.long_sma,
            golden_cross_buy_days=gene.golden_cross_buy_days,
            short_sma_downtrend=gene.short_sma_downtrend,
            short_sma_take_profit=gene.short_sma_take_profit,
            long_sma_take_profit=gene.long_sma_take_profit,
            stop_loss_pct=gene.stop_loss_pct,
            take_profit_pct=gene.take_profit_pct,
            use_stop_loss=gene.use_stop_loss,
            # Main.py matching features
            use_market_filter=gene.use_market_filter,
            use_eod_filter=gene.use_eod_filter,
            use_profit_before_eod=gene.use_profit_before_eod,
            use_price_5hr_check=gene.use_price_5hr_check,
            use_price_cross_check=gene.use_price_cross_check,
            use_dynamic_sma=gene.use_dynamic_sma,
            use_slope_ordering=gene.use_slope_ordering,
            slope_threshold=gene.slope_threshold,
            # Market trend detection thresholds (evolved by genetic algorithm)
            uptrend_threshold_pct=gene.uptrend_threshold_pct,
            major_downtrend_threshold_pct=gene.major_downtrend_threshold_pct,
            use_momentum_check=gene.use_momentum_check,
        )
        
        # Compute market conditions using this gene's thresholds (no network call)
        real_market_cache = None
        if self.use_real_data and self.raw_market_index_data is not None:
            real_market_cache = compute_market_conditions(
                self.raw_market_index_data,
                uptrend_threshold_pct=gene.uptrend_threshold_pct,
                major_downtrend_threshold_pct=gene.major_downtrend_threshold_pct,
                use_momentum_check=gene.use_momentum_check,
            )
        
        # Run backtest
        backtester = IntradayBacktester(
            initial_capital=self.initial_capital,
            max_positions=5,
            max_position_pct=gene.position_size_pct,
            use_total_investment_cap=gene.use_total_investment_cap,
            strategy=strategy
        )
        
        result = backtester.run(
            symbols=self.symbols,
            days=self.days,
            seed=data_seed,
            verbose=False,
            use_real_data=self.use_real_data,
            real_data_cache=self.real_data_cache,
            real_market_cache=real_market_cache
        )
        
        # Store results in gene
        gene.total_return_pct = result.get('total_return_pct', 0)
        gene.win_rate = result.get('win_rate', 0)
        gene.sharpe_ratio = result.get('sharpe_ratio', 0)
        gene.max_drawdown_pct = result.get('max_drawdown_pct', 0)
        pf = result.get('profit_factor', 0)
        gene.profit_factor = min(pf, 10.0) if pf != 'inf' and pf != float('inf') else 10.0
        gene.total_trades = result.get('total_trades', 0)
        gene.trades_per_day = result.get('trades_per_day', 0)
        gene.avg_win = result.get('avg_win', 0)
        gene.avg_loss = result.get('avg_loss', 0)
        
        # Calculate weighted fitness
        weights = self.config.fitness_weights
        
        fitness = 0.0
        
        # Return contribution (can be negative)
        fitness += weights['total_return_pct'] * (gene.total_return_pct / 5.0)  # Normalize around 5%
        
        # Sharpe ratio contribution (typically -3 to +3)
        fitness += weights['sharpe_ratio'] * (gene.sharpe_ratio / 2.0)  # Normalize
        
        # Win rate contribution (0-100, normalize to 0-1)
        fitness += weights['win_rate'] * (gene.win_rate / 100.0)
        
        # Profit factor contribution (0-10, already capped)
        fitness += weights['profit_factor'] * (gene.profit_factor / 5.0)
        
        # Drawdown penalty (higher drawdown = lower fitness)
        fitness -= weights['max_drawdown_penalty'] * (gene.max_drawdown_pct / 5.0)
        
        # Trades per day bonus (reward active trading, but not too much)
        optimal_trades_per_day = 1.0  # Aim for ~1 trade per day
        trades_diff = abs(gene.trades_per_day - optimal_trades_per_day)
        trades_bonus = max(0, 1.0 - trades_diff * 0.5)  # Penalize deviation from optimal
        fitness += weights['trades_per_day_bonus'] * trades_bonus
        
        # Penalties
        if gene.total_trades == 0:
            fitness -= 2.0
        elif gene.total_trades < 5:
            fitness -= 0.5
        
        # Bonus for good avg_win/avg_loss ratio
        if gene.avg_loss > 0:
            win_loss_ratio = gene.avg_win / gene.avg_loss
            if win_loss_ratio > 1.0:
                fitness += 0.1 * min(win_loss_ratio - 1.0, 1.0)
        
        gene.fitness = fitness
        return fitness
    
    def evaluate_population(self, population: List[IntradayTradingGene], generation: int) -> List[IntradayTradingGene]:
        """Evaluate fitness for entire population (optionally in parallel)
        
        Supports three execution modes:
        1. Ray (distributed): Set use_ray=True - scales from local to Kubernetes cluster
        2. Multiprocessing (local): Default when Ray not available - uses CPU cores
        3. Sequential: Set num_workers=1 - useful for debugging
        """
        # Use consistent seed for data generation within a generation
        data_seed = (self.seed + generation * 1000) if self.seed else None
        
        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers == 0:
            num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
        
        if num_workers > 1 and len(population) > 1:
            # Prepare args for worker functions.
            # Large shared data (real_data_cache, raw_market_index_data) is passed
            # via module-level globals to avoid pickling ~4 MB per gene through pipes.
            eval_args = [
                (gene, self.symbols, self.days, self.initial_capital, 
                 dict(self.config.fitness_weights), data_seed, self.max_positions,
                 self.use_real_data)
                for gene in population
            ]
            
            # Try Ray first if enabled and available
            if self.config.use_ray and RAY_AVAILABLE:
                population = self._evaluate_with_ray(eval_args, num_workers)
            else:
                population = self._evaluate_with_multiprocessing(eval_args, num_workers)
            
            if self.verbose:
                for i, gene in enumerate(population):
                    self._log(f"    Gene {i+1}: {gene}")
        else:
            # Sequential evaluation (single worker or small population)
            use_cv = bool(self._cv_folds or self._wf_windows)
            for i, gene in enumerate(population):
                if use_cv:
                    self._evaluate_gene_cv(gene, data_seed)
                else:
                    self.evaluate_fitness(gene, data_seed)
                if self.verbose:
                    self._log(f"  Evaluating {i+1}/{len(population)}: {gene}")
        
        # Sort by fitness (descending)
        population.sort(key=lambda g: g.fitness, reverse=True)
        return population
    
    def _evaluate_with_ray(self, eval_args: List[Tuple], num_workers: int) -> List[IntradayTradingGene]:
        """Evaluate genes using Ray (works locally or distributed across Kubernetes)
        
        Large shared data is put into Ray's object store once via ray.put(),
        then passed by reference to each remote task (no redundant serialization).
        """
        if self.verbose:
            ray_info = ray.cluster_resources() if ray.is_initialized() else {}
            num_nodes = int(ray_info.get('CPU', num_workers))
            self._log(f"  Evaluating {len(eval_args)} genes using Ray ({num_nodes} CPUs available)...")
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            # Configure Ray init options
            init_kwargs = {"ignore_reinit_error": True}
            
            # Disable memory monitor if configured (fixes cgroup v2 issues on some Linux systems)
            if self.config.disable_ray_mem_monitor:
                init_kwargs["_system_config"] = {
                    "automatic_object_spilling_enabled": False,
                    "memory_monitor_refresh_ms": 0,
                }
                if self.verbose:
                    self._log("  Ray memory monitor disabled (--disable-ray-mem-monitor)")
            
            # ray.init() auto-detects: local CPUs or K8s cluster
            ray.init(**init_kwargs)
        
        # Put large shared data into Ray object store once (avoids per-task serialization)
        real_data_ref = ray.put(self.real_data_cache) if self.real_data_cache else None
        raw_market_ref = ray.put(self.raw_market_index_data) if self.raw_market_index_data else None
        
        # Check if CV mode is active
        use_cv = bool(self._cv_folds or self._wf_windows)
        
        if use_cv:
            cv_folds_ref = ray.put(self._cv_folds) if self._cv_folds else None
            wf_windows_ref = ray.put(self._wf_windows) if self._wf_windows else None
            
            @ray.remote
            def ray_evaluate_gene_cv(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions,
                                     use_real_data, cv_folds=None, wf_windows=None):
                return _evaluate_gene_cv_impl(gene, symbols, days, initial_capital, fitness_weights, data_seed,
                                              max_positions, cv_folds, wf_windows)
            
            futures = [
                ray_evaluate_gene_cv.remote(*args, cv_folds_ref, wf_windows_ref) for args in eval_args
            ]
        else:
            # Define remote function inside method to avoid module-level Ray dependency
            @ray.remote
            def ray_evaluate_gene(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions,
                                  use_real_data=False, real_data_cache=None, raw_market_index_data=None):
                return _evaluate_gene_impl(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions,
                                           use_real_data, real_data_cache, raw_market_index_data)
            
            # Submit all tasks — append shared data refs to each 8-element arg tuple
            futures = [
                ray_evaluate_gene.remote(*args, real_data_ref, raw_market_ref) for args in eval_args
            ]
        
        # Gather results
        evaluated_population = ray.get(futures)
        return evaluated_population
    
    def _evaluate_with_multiprocessing(self, eval_args: List[Tuple], num_workers: int) -> List[IntradayTradingGene]:
        """Evaluate genes using multiprocessing.Pool (local only)
        
        Large shared data is set in module-level globals before Pool creation.
        On Linux (fork), workers inherit these via copy-on-write — no pickling needed.
        Workers also ignore SIGTERM so only the parent process handles graceful shutdown.
        """
        global _shared_real_data_cache, _shared_raw_market_index_data, _shared_cv_folds, _shared_wf_windows
        _shared_real_data_cache = self.real_data_cache
        _shared_raw_market_index_data = self.raw_market_index_data
        _shared_cv_folds = self._cv_folds
        _shared_wf_windows = self._wf_windows
        
        if self.verbose:
            n_folds = (len(self._cv_folds) if self._cv_folds else 0) + (len(self._wf_windows) if self._wf_windows else 0)
            cv_info = f" (CV: {n_folds} splits each)" if n_folds > 0 else ""
            self._log(f"  Evaluating {len(eval_args)} genes using {num_workers} multiprocessing workers...{cv_info}")
        
        try:
            with Pool(processes=num_workers, initializer=_init_worker) as pool:
                evaluated_population = pool.map(_evaluate_gene_worker, eval_args)
        finally:
            # Clear globals after use (don't hold references longer than needed)
            _shared_real_data_cache = None
            _shared_raw_market_index_data = None
            _shared_cv_folds = None
            _shared_wf_windows = None
        
        return evaluated_population
    
    def tournament_select(self, population: List[IntradayTradingGene]) -> IntradayTradingGene:
        """Select a gene using tournament selection"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def crossover(self, parent1: IntradayTradingGene, parent2: IntradayTradingGene) -> Tuple[IntradayTradingGene, IntradayTradingGene]:
        """
        Perform crossover between two parent genes.
        Uses uniform crossover - each parameter randomly chosen from either parent.
        """
        if random.random() > self.config.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = IntradayTradingGene()
        child2 = IntradayTradingGene()
        
        # Numeric parameters
        params = ['short_sma', 'long_sma', 'golden_cross_buy_days',
                  'short_sma_downtrend', 'short_sma_take_profit', 'long_sma_take_profit',
                  'stop_loss_pct', 'take_profit_pct', 'position_size_pct',
                  'slope_threshold',
                  'uptrend_threshold_pct', 'major_downtrend_threshold_pct',
                  'momentum_lookback_bars']
        
        for param in params:
            if random.random() < 0.5:
                setattr(child1, param, getattr(parent1, param))
                setattr(child2, param, getattr(parent2, param))
            else:
                setattr(child1, param, getattr(parent2, param))
                setattr(child2, param, getattr(parent1, param))
        
        # Boolean parameters (use_stop_loss, use_momentum_check)
        for bool_param in ['use_stop_loss', 'use_momentum_check']:
            if random.random() < 0.5:
                setattr(child1, bool_param, getattr(parent1, bool_param))
                setattr(child2, bool_param, getattr(parent2, bool_param))
            else:
                setattr(child1, bool_param, getattr(parent2, bool_param))
                setattr(child2, bool_param, getattr(parent1, bool_param))
        
        # Filter toggles (if optimizing)
        if self.config.optimize_filters:
            filter_params = ['use_market_filter', 'use_eod_filter', 'use_profit_before_eod',
                            'use_price_5hr_check', 'use_price_cross_check', 'use_dynamic_sma', 'use_slope_ordering',
                            'use_total_investment_cap']
            for param in filter_params:
                if random.random() < 0.5:
                    setattr(child1, param, getattr(parent1, param))
                    setattr(child2, param, getattr(parent2, param))
                else:
                    setattr(child1, param, getattr(parent2, param))
                    setattr(child2, param, getattr(parent1, param))
        
        # Ensure long_sma > short_sma
        for child in [child1, child2]:
            if child.long_sma <= child.short_sma:
                child.long_sma = child.short_sma + 15
        
        return child1, child2
    
    def mutate(self, gene: IntradayTradingGene) -> IntradayTradingGene:
        """
        Apply random mutations to a gene.
        Each parameter has a chance to be mutated.
        """
        ranges = self.config.param_ranges
        mutated = copy.deepcopy(gene)
        
        # Mutate numeric parameters
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-5, 5)
            mutated.short_sma = max(ranges['short_sma'][0],
                                   min(ranges['short_sma'][1], mutated.short_sma + delta))
        
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-15, 15)
            mutated.long_sma = max(ranges['long_sma'][0],
                                  min(ranges['long_sma'][1], mutated.long_sma + delta))
        
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-2, 2)
            mutated.golden_cross_buy_days = max(ranges['golden_cross_buy_days'][0],
                                            min(ranges['golden_cross_buy_days'][1],
                                                mutated.golden_cross_buy_days + delta))
        
        # Mutate dynamic SMA parameters
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-5, 5)
            mutated.short_sma_downtrend = max(ranges['short_sma_downtrend'][0],
                                             min(ranges['short_sma_downtrend'][1],
                                                 mutated.short_sma_downtrend + delta))
        
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-3, 3)
            mutated.short_sma_take_profit = max(ranges['short_sma_take_profit'][0],
                                               min(ranges['short_sma_take_profit'][1],
                                                   mutated.short_sma_take_profit + delta))
        
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-3, 3)
            mutated.long_sma_take_profit = max(ranges['long_sma_take_profit'][0],
                                              min(ranges['long_sma_take_profit'][1],
                                                  mutated.long_sma_take_profit + delta))
        
        # Mutate use_stop_loss (if optimizing filters)
        if self.config.optimize_filters and random.random() < self.config.mutation_rate:
            mutated.use_stop_loss = not mutated.use_stop_loss
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-2.0, 2.0)
            mutated.stop_loss_pct = round(max(ranges['stop_loss_pct'][0],
                                             min(ranges['stop_loss_pct'][1],
                                                 mutated.stop_loss_pct + delta)), 1)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-0.3, 0.3)
            mutated.take_profit_pct = round(max(ranges['take_profit_pct'][0],
                                               min(ranges['take_profit_pct'][1],
                                                   mutated.take_profit_pct + delta)), 2)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-2.0, 2.0)
            mutated.position_size_pct = round(max(ranges['position_size_pct'][0],
                                                 min(ranges['position_size_pct'][1],
                                                     mutated.position_size_pct + delta)), 1)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-0.0003, 0.0003)
            mutated.slope_threshold = round(max(ranges['slope_threshold'][0],
                                               min(ranges['slope_threshold'][1],
                                                   mutated.slope_threshold + delta)), 4)
        
        # Mutate market trend / momentum parameters
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-0.1, 0.1)
            mutated.uptrend_threshold_pct = round(max(ranges['uptrend_threshold_pct'][0],
                                                     min(ranges['uptrend_threshold_pct'][1],
                                                         mutated.uptrend_threshold_pct + delta)), 2)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-0.5, 0.5)
            mutated.major_downtrend_threshold_pct = round(max(ranges['major_downtrend_threshold_pct'][0],
                                                             min(ranges['major_downtrend_threshold_pct'][1],
                                                                 mutated.major_downtrend_threshold_pct + delta)), 1)
        
        if random.random() < self.config.mutation_rate:
            mutated.use_momentum_check = not mutated.use_momentum_check
        
        if random.random() < self.config.mutation_rate:
            delta = random.randint(-5, 5)
            mutated.momentum_lookback_bars = max(ranges['momentum_lookback_bars'][0],
                                                min(ranges['momentum_lookback_bars'][1],
                                                    mutated.momentum_lookback_bars + delta))
        
        # Mutate filter toggles (if optimizing)
        if self.config.optimize_filters:
            if random.random() < self.config.mutation_rate:
                mutated.use_market_filter = not mutated.use_market_filter
            if random.random() < self.config.mutation_rate:
                mutated.use_eod_filter = not mutated.use_eod_filter
            if random.random() < self.config.mutation_rate:
                mutated.use_profit_before_eod = not mutated.use_profit_before_eod
            if random.random() < self.config.mutation_rate:
                mutated.use_price_5hr_check = not mutated.use_price_5hr_check
            if random.random() < self.config.mutation_rate:
                mutated.use_price_cross_check = not mutated.use_price_cross_check
            if random.random() < self.config.mutation_rate:
                mutated.use_dynamic_sma = not mutated.use_dynamic_sma
            if random.random() < self.config.mutation_rate:
                mutated.use_slope_ordering = not mutated.use_slope_ordering
            if random.random() < self.config.mutation_rate:
                mutated.use_total_investment_cap = 2  # Fixed to mode 2
        
        # Ensure long_sma > short_sma
        if mutated.long_sma <= mutated.short_sma:
            mutated.long_sma = mutated.short_sma + 15
        
        return mutated
    
    def create_next_generation(self, population: List[IntradayTradingGene]) -> List[IntradayTradingGene]:
        """Create the next generation through selection, crossover, and mutation"""
        next_gen = []
        
        # Elitism: keep top performers unchanged
        elite = population[:self.config.elite_size]
        next_gen.extend(copy.deepcopy(g) for g in elite)
        
        # Fill rest of population through breeding
        while len(next_gen) < self.config.population_size:
            # Tournament selection
            parent1 = self.tournament_select(population)
            parent2 = self.tournament_select(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            next_gen.append(child1)
            if len(next_gen) < self.config.population_size:
                next_gen.append(child2)
        
        return next_gen
    
    def run(self) -> IntradayTradingGene:
        """
        Run the genetic algorithm optimization.
        Returns the best gene found.
        
        Supports checkpoint/resume: if self.checkpoint_path is set and a checkpoint
        file exists, the optimizer will resume from the last completed generation.
        After each generation, progress is saved to the checkpoint file.
        On successful completion, the checkpoint file is removed.
        """
        start_gen = 0
        population = None
        resumed = False
        
        # Check for existing checkpoint to resume from
        if self.checkpoint_path:
            checkpoint = self.load_checkpoint(self.checkpoint_path)
            if checkpoint:
                completed = checkpoint['completed_generation'] + 1
                total = checkpoint['total_generations']
                self._log(f"\n🔄 RESUMING from checkpoint: Generation {completed}/{total} completed")
                self._log(f"   Checkpoint saved at: {checkpoint['saved_at']}")
                self._log(f"   Best fitness so far: {checkpoint['best_fitness']:.4f}")
                start_gen, population = self.restore_from_checkpoint(checkpoint)
                resumed = True
                # Create next generation from the restored population
                # (the checkpoint saved the population AFTER evaluation but BEFORE next_gen creation)
                if start_gen < self.config.generations:
                    population = self.create_next_generation(population)
                    self._log(f"   Resuming from Generation {start_gen + 1}...\n")
        
        if not resumed:
            self.start_time = datetime.now()
            # Archive previous result/checkpoint files before a fresh run overwrites them
            if self.checkpoint_path:
                self._archive_file(self.checkpoint_path)
        
        self._log(f"\n{'='*70}")
        self._log("INTRADAY GENETIC ALGORITHM OPTIMIZER")
        self._log(f"{'='*70}")
        self._log(f"Symbols: {', '.join(self.symbols)}")
        self._log(f"Trading Days: {self.days}")
        self._log(f"Initial Capital: ${self.initial_capital:,.2f}")
        self._log(f"Max Positions: {self.max_positions}")
        self._log(f"Population: {self.config.population_size}")
        self._log(f"Generations: {self.config.generations}")
        self._log(f"Mutation Rate: {self.config.mutation_rate}")
        self._log(f"Crossover Rate: {self.config.crossover_rate}")
        self._log(f"Optimize Filters: {self.config.optimize_filters}")
        num_workers = self.config.num_workers if self.config.num_workers > 0 else max(1, cpu_count() - 1)
        self._log(f"Workers: {num_workers} {'(auto)' if self.config.num_workers == 0 else ''}")
        data_source = "REAL (yfinance)" if self.use_real_data else "SYNTHETIC"
        self._log(f"Data Source: {data_source}")
        if self.train_test_split > 0:
            self._log(f"Train/Test Split: {self.train_test_split:.0%} train / {1 - self.train_test_split:.0%} test (anti-overfitting)")
        if self.config.validation_mode not in ('none', 'simple'):
            cv_parts = []
            if self.config.validation_mode in ('kfold', 'both'):
                cv_parts.append(f"{self.config.kfold_splits}-fold CV")
            if self.config.validation_mode in ('walkforward', 'both'):
                cv_parts.append(f"{self.config.walkforward_windows} walk-forward windows")
            self._log(f"Validation: {' + '.join(cv_parts)} (anti-overfitting)")
        if self.seed:
            self._log(f"Random Seed: {self.seed}")
        if self.checkpoint_path:
            self._log(f"Checkpoint: {self.checkpoint_path} {'(resumed)' if resumed else '(enabled)'}")
        self._log(f"{'='*70}\n")
        
        # Set random seed if provided (only on fresh start, not resume)
        if self.seed and not resumed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Initialize population (only on fresh start)
        if population is None:
            self._log("Initializing population...")
            population = self.initialize_population()
        
        # Pre-download real data once (shared across all generations and genes)
        if self.use_real_data:
            self._log(f"\n📥 Downloading real market data for {len(self.symbols)} symbols...")
            self._log("   (cached to disk - subsequent runs will be instant)")
            
            # Download raw market index data (threshold-independent, shared across all genes)
            self._log("   Downloading SPY/DIA/QQQ raw index data...")
            self.raw_market_index_data = download_raw_market_index_data(
                days=self.days,
                cache_max_age_hours=12
            )
            # Compute with default thresholds just for the summary count
            sample_market = compute_market_conditions(self.raw_market_index_data)
            self._log(f"   ✓ Market data: {len(sample_market)} trading days (each gene computes its own thresholds)")
            
            # Download each symbol
            self.real_data_cache = {}
            failed_symbols = []
            for i, symbol in enumerate(self.symbols):
                try:
                    self._log(f"   Downloading {symbol} ({i+1}/{len(self.symbols)})...", end="", flush=True)
                    df = download_real_data(symbol, self.days, cache_max_age_hours=12)
                    self.real_data_cache[symbol] = df
                    trading_days = df['date'].nunique()
                    self._log(f" ✓ {len(df)} bars ({trading_days} days)")
                except Exception as e:
                    self._log(f" ✗ FAILED: {e}")
                    failed_symbols.append(symbol)
            
            # Remove failed symbols
            if failed_symbols:
                self._log(f"\n   ⚠️  Removing {len(failed_symbols)} failed symbols: {', '.join(failed_symbols)}")
                self.symbols = [s for s in self.symbols if s not in failed_symbols]
                if not self.symbols:
                    raise ValueError("All symbols failed to download. Check internet connection and symbol names.")
            
            self._log(f"\n   ✓ Real data ready: {len(self.symbols)} symbols, {len(sample_market)} market days\n")
        
        # Apply train/test split if requested (must happen after data download)
        if self.train_test_split > 0:
            self._split_data_by_dates()
        
        # Create cross-validation splits (uses training data, or all data if no split)
        if self.config.validation_mode not in ('none', 'simple') and self.use_real_data:
            self._create_cv_splits()
        
        # Set up signal handler for graceful shutdown
        if self.checkpoint_path:
            self._setup_signal_handler()
        
        # Evolution loop
        try:
            for gen in range(start_gen, self.config.generations):
                # Check if SIGTERM was received — save checkpoint and exit cleanly
                if self._interrupted:
                    self._log(f"\n  🛑 SIGTERM detected before Generation {gen + 1}. Saving checkpoint and exiting cleanly...")
                    if self.checkpoint_path and self.generation_history:
                        # Checkpoint was already saved after last completed generation
                        self._log(f"     Checkpoint saved after Gen {len(self.generation_history)}.")
                        self._log(f"     Resume with: --resume to continue from Gen {len(self.generation_history) + 1}")
                    break
                
                self._log(f"\n--- Generation {gen + 1}/{self.config.generations} ---")
                
                # Evaluate fitness
                population = self.evaluate_population(population, gen)
                
                # Track best
                gen_best = population[0]
                gen_avg = sum(g.fitness for g in population) / len(population)
                gen_worst = population[-1].fitness
                
                if gen_best.fitness > self.best_fitness:
                    self.best_fitness = gen_best.fitness
                    self.best_gene = copy.deepcopy(gen_best)
                
                # Record generation stats
                self.generation_history.append({
                    'generation': gen + 1,
                    'best_fitness': gen_best.fitness,
                    'avg_fitness': gen_avg,
                    'worst_fitness': gen_worst,
                    'best_gene': gen_best.to_dict(),
                    'best_return_pct': gen_best.total_return_pct,
                    'best_sharpe': gen_best.sharpe_ratio,
                    'best_win_rate': gen_best.win_rate,
                    'best_trades_per_day': gen_best.trades_per_day,
                })
                
                self._log(f"\n  Best:  {gen_best}")
                self._log(f"  Avg Fitness: {gen_avg:.4f} | Worst: {gen_worst:.4f}")
                self._log(f"  Return: {gen_best.total_return_pct:+.2f}% | "
                      f"Sharpe: {gen_best.sharpe_ratio:.2f} | "
                      f"Win Rate: {gen_best.win_rate:.1f}% | "
                      f"Trades/Day: {gen_best.trades_per_day:.2f}")
                self._log(f"  ⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Save checkpoint after each completed generation
                if self.checkpoint_path:
                    self.save_checkpoint(self.checkpoint_path, gen, population)
                
                # Check if SIGTERM was received during evaluation — exit after saving checkpoint
                if self._interrupted:
                    self._log(f"\n  🛑 SIGTERM received during Generation {gen + 1}. Checkpoint saved. Exiting cleanly...")
                    self._log(f"     Resume with: --resume to continue from Gen {gen + 2}")
                    break
                
                # Create next generation (skip on last iteration)
                if gen < self.config.generations - 1:
                    population = self.create_next_generation(population)
        
        except KeyboardInterrupt:
            # Save emergency checkpoint if we have any progress
            if self.checkpoint_path and self.generation_history:
                last_completed_gen = len(self.generation_history) - 1 + start_gen
                # Only save if we haven't already (the per-generation save may have caught it)
                self._log(f"\n  🛑 Interrupted! Checkpoint was saved after Gen {len(self.generation_history)}.")
                self._log(f"     Resume with: --resume to continue from Gen {len(self.generation_history) + 1}")
            else:
                self._log(f"\n  🛑 Interrupted before any generation completed. No checkpoint to save.")
            
            # Restore signal handlers
            if self.checkpoint_path:
                self._restore_signal_handlers()
            
            # Still return best gene found so far (if any)
            if self.best_gene:
                self.end_time = datetime.now()
                return self.best_gene
            raise
        
        finally:
            # Restore signal handlers
            if self.checkpoint_path:
                self._restore_signal_handlers()
            # Close log file
            if self._log_fh:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
                self._log_fh = None
        
        self.end_time = datetime.now()
        
        # Clean up checkpoint file on successful completion
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            self._log(f"  🧹 Checkpoint file removed (optimization complete)")
        
        # Evaluate best gene on held-out test set (if train/test split was used)
        if self.train_test_split > 0 and self.best_gene and self._test_real_data_cache:
            self._log(f"\n{'='*70}")
            self._log("TEST SET EVALUATION (unseen data)")
            self._log(f"{'='*70}")
            self._log(f"Evaluating best gene on {len(self._test_dates)} held-out test days...")
            self._log(f"Test period: {self._test_dates[0]} → {self._test_dates[-1]}")
            
            self.test_set_metrics = self.evaluate_on_test_set(self.best_gene)
            
            self._log(f"\n  {'Metric':<22} {'Train':>10} {'Test':>10} {'Delta':>10}")
            self._log(f"  {'-'*52}")
            
            train_return = self.best_gene.total_return_pct
            test_return = self.test_set_metrics['total_return_pct']
            self._log(f"  {'Return %':<22} {train_return:>+9.2f}% {test_return:>+9.2f}% {test_return - train_return:>+9.2f}%")
            
            train_wr = self.best_gene.win_rate
            test_wr = self.test_set_metrics['win_rate']
            self._log(f"  {'Win Rate %':<22} {train_wr:>9.1f}% {test_wr:>9.1f}% {test_wr - train_wr:>+9.1f}%")
            
            train_sharpe = self.best_gene.sharpe_ratio
            test_sharpe = self.test_set_metrics['sharpe_ratio']
            self._log(f"  {'Sharpe Ratio':<22} {train_sharpe:>10.2f} {test_sharpe:>10.2f} {test_sharpe - train_sharpe:>+10.2f}")
            
            train_dd = self.best_gene.max_drawdown_pct
            test_dd = self.test_set_metrics['max_drawdown_pct']
            self._log(f"  {'Max Drawdown %':<22} {train_dd:>9.2f}% {test_dd:>9.2f}% {test_dd - train_dd:>+9.2f}%")
            
            train_pf = self.best_gene.profit_factor
            test_pf = self.test_set_metrics['profit_factor']
            self._log(f"  {'Profit Factor':<22} {train_pf:>10.2f} {test_pf:>10.2f} {test_pf - train_pf:>+10.2f}")
            
            train_trades = self.best_gene.total_trades
            test_trades = self.test_set_metrics['total_trades']
            self._log(f"  {'Total Trades':<22} {train_trades:>10} {test_trades:>10} {test_trades - train_trades:>+10}")
            
            train_fit = self.best_gene.fitness
            test_fit = self.test_set_metrics['fitness']
            self._log(f"  {'Fitness':<22} {train_fit:>10.4f} {test_fit:>10.4f} {test_fit - train_fit:>+10.4f}")
            
            # Overfitting assessment
            self._log(f"\n  📊 Overfitting Assessment:")
            if test_return <= 0 and train_return > 0:
                self._log(f"  ⚠️  OVERFIT: Train profitable but test negative. Strategy does not generalize.")
            elif train_return > 0 and test_return > 0 and test_return < train_return * 0.3:
                self._log(f"  ⚠️  LIKELY OVERFIT: Test return is <30% of train return ({test_return/train_return:.0%}).")
            elif train_return > 0 and test_return > 0 and test_return < train_return * 0.5:
                self._log(f"  ⚠️  MILD OVERFIT: Test return is {test_return/train_return:.0%} of train. Consider more generations or larger dataset.")
            elif test_return >= train_return * 0.5 or (train_return <= 0 and test_return >= train_return):
                self._log(f"  ✅ GOOD: Test performance is consistent with train ({test_return/train_return:.0%} retention)." if train_return > 0 else f"  ✅ Test performance consistent with train.")
            
            if abs(test_wr - train_wr) > 15:
                self._log(f"  ⚠️  Win rate dropped {abs(test_wr - train_wr):.1f}pp on test set — strategy may be curve-fit.")
            
            if test_sharpe < 0 and train_sharpe > 1:
                self._log(f"  ⚠️  Sharpe collapsed from {train_sharpe:.2f} to {test_sharpe:.2f} — strong overfitting signal.")
            
            self._log(f"{'='*70}")
        
        self._log(f"\n{'='*70}")
        self._log("OPTIMIZATION COMPLETE")
        self._log(f"{'='*70}")
        
        return self.best_gene
    
    def save_results(self, filepath: str = "genetic_optimization_intraday_result.json"):
        """Save optimization results to JSON file.
        
        Archives the previous result file (if any) before overwriting.
        """
        self._archive_file(filepath)
        # Calculate runtime
        start_time = getattr(self, 'start_time', None)
        end_time = getattr(self, 'end_time', None)
        runtime_seconds = (end_time - start_time).total_seconds() if start_time and end_time else None
        
        results = {
            'start_time': str(start_time) if start_time else None,
            'end_time': str(end_time) if end_time else None,
            'runtime_seconds': runtime_seconds,
            'runtime_minutes': round(runtime_seconds / 60, 2) if runtime_seconds else None,
            'optimizer_type': 'intraday',
            'num_symbols': len(self.symbols),
            'symbols': self.symbols,
            'days': self.days,
            'initial_capital': self.initial_capital,
            'train_test_split': self.train_test_split if self.train_test_split > 0 else None,
            'train_days': len(self._train_dates) if self._train_dates else None,
            'test_days': len(self._test_dates) if self._test_dates else None,
            'train_period': f"{self._train_dates[0]} → {self._train_dates[-1]}" if self._train_dates else None,
            'test_period': f"{self._test_dates[0]} → {self._test_dates[-1]}" if self._test_dates else None,
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size,
                'optimize_filters': self.config.optimize_filters,
                'validation_mode': self.config.validation_mode,
                'kfold_splits': self.config.kfold_splits if self.config.validation_mode in ('kfold', 'both') else None,
                'walkforward_windows': self.config.walkforward_windows if self.config.validation_mode in ('walkforward', 'both') else None,
            },
            'cross_validation': {
                'mode': self.config.validation_mode,
                'kfold_info': self._cv_fold_info if self._cv_fold_info else None,
                'walkforward_info': self._wf_window_info if self._wf_window_info else None,
                'total_eval_splits': (len(self._cv_folds) if self._cv_folds else 0) + (len(self._wf_windows) if self._wf_windows else 0),
                'fitness_formula': '0.7 * avg_fitness + 0.3 * min_fitness' if self.config.validation_mode not in ('none', 'simple') else 'standard',
            } if self.config.validation_mode not in ('none', 'simple') else None,
            'best_gene': self.best_gene.to_dict() if self.best_gene else None,
            'best_fitness': self.best_fitness,
            'test_set_metrics': self.test_set_metrics if self.test_set_metrics else None,
            'generation_history': self.generation_history,
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self._log(f"\nResults saved to {filepath}")
    
    def print_best_config(self):
        """Print the best configuration found in a copyable format"""
        if not self.best_gene:
            self._log("No optimization results available")
            return
        
        gene = self.best_gene
        self._log(f"\n{'='*70}")
        self._log("BEST INTRADAY CONFIGURATION FOUND")
        self._log(f"{'='*70}")
        self._log(f"\n# Add these to your config.py:")
        self._log(f"# Note: SMA values are in HOURS (for hourly data)")
        self._log(f"\n# SMA Settings (Hourly)")
        self._log(f"short_sma = {gene.short_sma}  # ~{gene.short_sma/7:.1f} trading days")
        self._log(f"long_sma = {gene.long_sma}   # ~{gene.long_sma/7:.1f} trading days")
        self._log(f"golden_cross_buy_days = {gene.golden_cross_buy_days}  # trading days (1 day = 7 hourly bars)")
        self._log(f"\n# Dynamic SMA Settings (used when use_dynamic_sma=True)")
        self._log(f"short_sma_downtrend = {gene.short_sma_downtrend}  # Used when market not in uptrend")
        self._log(f"short_sma_take_profit = {gene.short_sma_take_profit}  # Used after take profit (only if balance < $25k PDT limit)")
        self._log(f"long_sma_take_profit = {gene.long_sma_take_profit}   # Used after take profit (only if balance < $25k PDT limit)")
        self._log(f"\n# Risk Management")
        self._log(f"use_stop_loss = {gene.use_stop_loss}")
        self._log(f"stop_loss_percent = {gene.stop_loss_pct}")
        self._log(f"take_profit_percent = {gene.take_profit_pct}")
        self._log(f"\n# Position Sizing")
        self._log(f"purchase_limit_mode = {gene.use_total_investment_cap}  # 0=legacy, 1=total cap, 2=per-stock pct")
        self._log(f"purchase_limit_percentage = {gene.position_size_pct}")
        self._log(f"\n# Main.py Specific Settings")
        self._log(f"# slope_threshold = {gene.slope_threshold}  # For order_symbols_by_slope")
        self._log(f"\n# Market Trend Detection")
        self._log(f"uptrend_threshold_pct = {gene.uptrend_threshold_pct}")
        self._log(f"major_downtrend_threshold_pct = {gene.major_downtrend_threshold_pct}")
        self._log(f"use_momentum_check = {gene.use_momentum_check}")
        self._log(f"momentum_lookback_bars = {gene.momentum_lookback_bars}  # {gene.momentum_lookback_bars * 5} minutes")
        
        if self.config.optimize_filters:
            self._log(f"\n# Filter Settings (optimized)")
            self._log(f"# use_market_filter = {gene.use_market_filter}")
            self._log(f"# use_eod_filter = {gene.use_eod_filter}")
            self._log(f"# use_profit_before_eod = {gene.use_profit_before_eod}")
            self._log(f"# use_price_5hr_check = {gene.use_price_5hr_check}")
            self._log(f"# use_price_cross_check = {gene.use_price_cross_check}")
            self._log(f"# use_dynamic_sma = {gene.use_dynamic_sma}")
            self._log(f"# use_slope_ordering = {gene.use_slope_ordering}")
            self._log(f"# purchase_limit_mode = {gene.use_total_investment_cap}  # 0=legacy, 1=total cap, 2=per-stock pct")
        
        self._log(f"\n# Performance Metrics (TRAIN{' — evolution saw this data' if self.train_test_split > 0 else ''}):")
        self._log(f"# Total Return: {gene.total_return_pct:+.2f}%")
        self._log(f"# Win Rate: {gene.win_rate:.1f}%")
        self._log(f"# Sharpe Ratio: {gene.sharpe_ratio:.4f}")
        self._log(f"# Max Drawdown: {gene.max_drawdown_pct:.2f}%")
        self._log(f"# Profit Factor: {gene.profit_factor:.2f}")
        self._log(f"# Total Trades: {gene.total_trades}")
        self._log(f"# Trades/Day: {gene.trades_per_day:.2f}")
        self._log(f"# Avg Win: ${gene.avg_win:.2f}")
        self._log(f"# Avg Loss: ${gene.avg_loss:.2f}")
        self._log(f"# Fitness Score: {gene.fitness:.4f}")
        
        # Show test-set metrics if available
        if self.test_set_metrics:
            tm = self.test_set_metrics
            self._log(f"\n# Performance Metrics (TEST — unseen holdout data):")
            self._log(f"# Total Return: {tm['total_return_pct']:+.2f}%")
            self._log(f"# Win Rate: {tm['win_rate']:.1f}%")
            self._log(f"# Sharpe Ratio: {tm['sharpe_ratio']:.4f}")
            self._log(f"# Max Drawdown: {tm['max_drawdown_pct']:.2f}%")
            self._log(f"# Profit Factor: {tm['profit_factor']:.2f}")
            self._log(f"# Total Trades: {tm['total_trades']}")
            self._log(f"# Trades/Day: {tm['trades_per_day']:.2f}")
            self._log(f"# Fitness Score: {tm['fitness']:.4f}")
        
        # Enhancement #10: Display exit reason breakdown
        if gene.exit_reasons:
            self._log(f"\n# Exit Reason Breakdown (Enhancement #10):")
            total_exits = sum(r['count'] for r in gene.exit_reasons.values())
            for reason, stats in sorted(gene.exit_reasons.items(), key=lambda x: -x[1]['count']):
                pct = (stats['count'] / total_exits * 100) if total_exits > 0 else 0
                win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
                self._log(f"#   {reason:20}: {stats['count']:>3} ({pct:5.1f}%) | Win: {win_rate:5.1f}% | Avg: ${avg_pnl:>+7.2f}")
        
        self._log(f"{'='*70}\n")


def validate_against_real_trades(gene: IntradayTradingGene, tradehistory_path: str = "logs/tradehistory-real.json") -> Dict:
    """
    Enhancement #6: Validate optimized parameters against real trading history.
    
    Compares the genetic optimizer recommended configuration against actual
    trading performance from tradehistory-real.json to check alignment.
    
    Returns a dict with validation metrics and recommendations.
    """
    print(f"\n{'='*70}")
    print("REAL TRADE VALIDATION")
    print(f"{'='*70}")
    
    if not os.path.exists(tradehistory_path):
        print(f"⚠️  Warning: {tradehistory_path} not found. Skipping validation.")
        return {'status': 'skipped', 'reason': 'file_not_found'}
    
    with open(tradehistory_path, 'r') as f:
        trade_history = json.load(f)
    
    # Analyze real trading patterns
    real_profits = []
    real_losses = []
    exit_reasons_real = Counter()
    
    for timestamp, trades in trade_history.items():
        for symbol, trade in trades.items():
            pct_change = float(trade.get('percent_change', 0))
            if pct_change > 0:
                real_profits.append(pct_change)
            else:
                real_losses.append(abs(pct_change))
    
    # Load buy_reasons.json for exit reason analysis
    buy_reasons_path = "logs/buy_reasons.json"
    if os.path.exists(buy_reasons_path):
        with open(buy_reasons_path, 'r') as f:
            buy_reasons = json.load(f)
        for symbol, info in buy_reasons.items():
            reason = info.get('reason', 'unknown')
            # Handle compound reasons
            for r in reason.split(','):
                exit_reasons_real[r.strip()] += 1
    
    total_trades = len(real_profits) + len(real_losses)
    real_win_rate = len(real_profits) / total_trades * 100 if total_trades > 0 else 0
    real_avg_profit = sum(real_profits) / len(real_profits) if real_profits else 0
    real_avg_loss = sum(real_losses) / len(real_losses) if real_losses else 0
    
    print(f"\n📊 REAL TRADING STATISTICS (from {tradehistory_path})")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {real_win_rate:.1f}%")
    print(f"   Avg Profit: +{real_avg_profit:.2f}%")
    print(f"   Avg Loss: -{real_avg_loss:.2f}%")
    
    if exit_reasons_real:
        print(f"\n📤 REAL EXIT REASONS (from buy_reasons.json)")
        total_exits = sum(exit_reasons_real.values())
        for reason, count in exit_reasons_real.most_common():
            pct = count / total_exits * 100
            print(f"   {reason:20}: {count:>4} ({pct:5.1f}%)")
    
    # Compare with optimized gene parameters
    print(f"\n🔬 PARAMETER ALIGNMENT ANALYSIS")
    
    alignment_score = 0
    max_score = 5
    
    # 1. Take profit alignment
    # Real avg profit is ~1.87% for profitable trades, median is lower
    if 0.5 <= gene.take_profit_pct <= 2.5:
        alignment_score += 1
        tp_status = "✅ ALIGNED"
    else:
        tp_status = "⚠️  MISALIGNED"
    print(f"   Take Profit ({gene.take_profit_pct:.2f}%): {tp_status}")
    print(f"      Real avg profitable exit: +{real_avg_profit:.2f}%")
    
    # 2. Stop loss alignment  
    # Real avg loss around 4-5%
    if 3.0 <= gene.stop_loss_pct <= 7.0:
        alignment_score += 1
        sl_status = "✅ ALIGNED"
    else:
        sl_status = "⚠️  MISALIGNED"
    print(f"   Stop Loss ({gene.stop_loss_pct:.1f}%): {sl_status}")
    print(f"      Real avg loss: -{real_avg_loss:.2f}%")
    
    # 3. Win rate expectation
    # Real win rate ~75%
    if gene.win_rate >= 60:
        alignment_score += 1
        wr_status = "✅ ALIGNED"
    else:
        wr_status = "⚠️  LOWER THAN REAL"
    print(f"   Expected Win Rate ({gene.win_rate:.1f}%): {wr_status}")
    print(f"      Real win rate: {real_win_rate:.1f}%")
    
    # 4. Exit reason mix
    # Real: profit_before_eod dominates (55%), then take_profit (15%), stop_loss (13%)
    if gene.exit_reasons:
        gene_exit_total = sum(r['count'] for r in gene.exit_reasons.values())
        gene_pbe = gene.exit_reasons.get('profit_before_eod', {}).get('count', 0)
        gene_pbe_pct = gene_pbe / gene_exit_total * 100 if gene_exit_total > 0 else 0
        
        if gene_pbe_pct >= 30:  # profit_before_eod should be significant
            alignment_score += 1
            exit_status = "✅ ALIGNED"
        else:
            exit_status = "⚠️  DIFFERENT MIX"
        print(f"   Exit Reason Mix: {exit_status}")
        print(f"      Gene profit_before_eod: {gene_pbe_pct:.1f}%")
        print(f"      Real profit_before_eod: {exit_reasons_real.get('profit_before_eod', 0) / sum(exit_reasons_real.values()) * 100 if exit_reasons_real else 0:.1f}%")
    else:
        print(f"   Exit Reason Mix: ⚠️  No exit reason data in gene")
    
    # 5. Overall consistency
    if gene.sharpe_ratio > 1.0 and gene.max_drawdown_pct < 5.0:
        alignment_score += 1
        risk_status = "✅ GOOD RISK PROFILE"
    else:
        risk_status = "⚠️  REVIEW RISK"
    print(f"   Risk Profile: {risk_status}")
    print(f"      Sharpe: {gene.sharpe_ratio:.2f}, Max DD: {gene.max_drawdown_pct:.2f}%")
    
    # Overall score
    print(f"\n📈 ALIGNMENT SCORE: {alignment_score}/{max_score}")
    
    if alignment_score >= 4:
        recommendation = "✅ EXCELLENT - Parameters well-aligned with real trading patterns"
    elif alignment_score >= 3:
        recommendation = "✅ GOOD - Parameters reasonably aligned, minor tuning may help"
    elif alignment_score >= 2:
        recommendation = "⚠️  FAIR - Some misalignment, consider reviewing parameters"
    else:
        recommendation = "❌ POOR - Significant misalignment with real trading, reconsider strategy"
    
    print(f"\n{recommendation}")
    print(f"{'='*70}\n")
    
    return {
        'status': 'completed',
        'alignment_score': alignment_score,
        'max_score': max_score,
        'real_stats': {
            'total_trades': total_trades,
            'win_rate': real_win_rate,
            'avg_profit': real_avg_profit,
            'avg_loss': real_avg_loss,
            'exit_reasons': dict(exit_reasons_real)
        },
        'recommendation': recommendation
    }


def print_evolution_summary(optimizer: IntradayGeneticOptimizer):
    """Print a summary of how fitness evolved over generations"""
    if not optimizer.generation_history:
        return
    
    print(f"\n{'='*70}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Gen':>4} | {'Best Fit':>10} | {'Avg Fit':>10} | {'Return %':>10} | {'Sharpe':>8} | {'Win %':>7} | {'Tr/Day':>6}")
    print("-" * 78)
    
    for h in optimizer.generation_history:
        print(f"{h['generation']:>4} | {h['best_fitness']:>10.4f} | {h['avg_fitness']:>10.4f} | "
              f"{h['best_return_pct']:>+10.2f} | {h['best_sharpe']:>8.2f} | {h['best_win_rate']:>7.1f} | {h['best_trades_per_day']:>6.2f}")
    
    # Show improvement
    if len(optimizer.generation_history) > 1:
        first = optimizer.generation_history[0]
        last = optimizer.generation_history[-1]
        fitness_improvement = last['best_fitness'] - first['best_fitness']
        return_improvement = last['best_return_pct'] - first['best_return_pct']
        print("-" * 78)
        print(f"Improvement: Fitness {fitness_improvement:+.4f} | Return {return_improvement:+.2f}%")


def main():
    start_time = datetime.now()
    print(f"\n⏱️  Genetic optimizer started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(
        description='Intraday Genetic Algorithm Optimizer for RobinhoodBot Day Trading'
    )
    parser.add_argument(
        '--symbols', '-s', type=str, default=None,
        help='Comma-separated list of symbols (default: AAPL,MSFT,GOOGL). Ignored if --num-stocks is used.'
    )
    parser.add_argument(
        '--num-stocks', '-n', type=int, default=None,
        help='Randomly select N stocks from a built-in universe of ~500 liquid US equities. '
             'Overrides --symbols. Use with --seed for reproducibility. (e.g., --num-stocks 100)'
    )
    parser.add_argument(
        '--days', '-d', type=int, default=60,
        help='Trading days to simulate (default: 60)'
    )
    parser.add_argument(
        '--capital', '-c', type=float, default=30000,
        help='Initial capital (default: 30000)'
    )
    parser.add_argument(
        '--population', '-p', type=int, default=20,
        help='Population size (default: 20)'
    )
    parser.add_argument(
        '--generations', '-g', type=int, default=15,
        help='Number of generations (default: 15)'
    )
    parser.add_argument(
        '--mutation-rate', '-m', type=float, default=0.15,
        help='Mutation rate 0.0-1.0 (default: 0.15)'
    )
    parser.add_argument(
        '--crossover-rate', type=float, default=0.7,
        help='Crossover rate 0.0-1.0 (default: 0.7)'
    )
    parser.add_argument(
        '--optimize-filters', action='store_true',
        help='Also optimize filter on/off settings (advanced)'
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=0,
        help='Number of parallel workers (default: 0 = auto, uses cpu_count-1)'
    )
    parser.add_argument(
        '--use-ray', action='store_true',
        help='Use Ray for distributed computing (works locally or on Kubernetes cluster)'
    )
    parser.add_argument(
        '--disable-ray-mem-monitor', action='store_true',
        help='Disable Ray memory monitor (fixes cgroup v2 crashes on some Linux systems)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='genetic_optimization_intraday_result.json',
        help='Output JSON file (default: genetic_optimization_intraday_result.json)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--max-positions', type=int, default=5,
        help='Maximum concurrent positions (default: 5). More positions = more realistic trading volume.'
    )
    parser.add_argument(
        '--validate-real', action='store_true',
        help='After optimization, validate best gene against tradehistory-real.json patterns (Enhancement #6)'
    )
    parser.add_argument(
        '--real-data', action='store_true',
        help='Use real Yahoo Finance data (via yfinance) instead of synthetic data. '
             'Downloads hourly OHLCV for each symbol and SPY/DIA/QQQ for market conditions. '
             'Data is cached to disk (~12h freshness). Requires: pip install yfinance'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from checkpoint if one exists. Saves progress after each generation so interrupts are recoverable.'
    )
    parser.add_argument(
        '--checkpoint-file', type=str, default=None,
        help='Checkpoint file path (default: <output>.checkpoint.json). Used with --resume.'
    )
    parser.add_argument(
        '--log-file', type=str, default=None,
        help='Write all output to this log file in addition to stdout (unbuffered, survives broken pipes)'
    )
    parser.add_argument(
        '--train-test-split', type=float, default=0.0,
        help='Train/test split ratio for overfitting prevention (default: 0.0 = disabled). '
             'E.g., 0.7 = train on first 70%% of days, evaluate best gene on remaining 30%% as holdout. '
             'Requires --real-data. Recommended: 0.7'
    )
    parser.add_argument(
        '--no-kfold', action='store_true',
        help='Disable K-fold cross-validation during evolution. '
             'By default, K-fold CV is enabled (with --real-data) to reduce overfitting.'
    )
    parser.add_argument(
        '--no-walkforward', action='store_true',
        help='Disable walk-forward validation during evolution. '
             'By default, walk-forward is enabled (with --real-data) to reduce overfitting.'
    )
    parser.add_argument(
        '--kfold-splits', type=int, default=5,
        help='Number of K-fold cross-validation splits (default: 5). '
             'More folds = better generalization estimate but slower.'
    )
    parser.add_argument(
        '--walkforward-windows', type=int, default=3,
        help='Number of walk-forward validation windows (default: 3). '
             'More windows = better temporal robustness check but slower.'
    )
    
    args = parser.parse_args()
    
    # Resolve symbols: --num-stocks overrides --symbols
    if args.num_stocks is not None:
        universe = list(dict.fromkeys(STOCK_UNIVERSE))  # deduplicate preserving order
        if args.num_stocks > len(universe):
            print(f"⚠️  Requested {args.num_stocks} stocks but universe only has {len(universe)}. Using all.")
            args.num_stocks = len(universe)
        # Take the first N symbols (order in JSON matters — log-traded symbols come first)
        symbols = universe[:args.num_stocks]
        print(f"📊 Selected first {len(symbols)} stocks from universe of {len(universe)}")
    elif args.symbols is not None:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Determine validation mode from CLI flags
    # Default: 'both' (K-fold + walk-forward) when --real-data is used
    if args.real_data:
        if args.no_kfold and args.no_walkforward:
            validation_mode = 'simple'
        elif args.no_kfold:
            validation_mode = 'walkforward'
        elif args.no_walkforward:
            validation_mode = 'kfold'
        else:
            validation_mode = 'both'
    else:
        validation_mode = 'none'
    
    config = IntradayGeneticConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        optimize_filters=args.optimize_filters,
        num_workers=args.workers,
        use_ray=args.use_ray,
        disable_ray_mem_monitor=args.disable_ray_mem_monitor,
        validation_mode=validation_mode,
        kfold_splits=args.kfold_splits,
        walkforward_windows=args.walkforward_windows,
    )
    
    # Print Ray status if requested
    if args.use_ray:
        if RAY_AVAILABLE:
            print("🌟 Ray mode enabled - will use Ray for distributed computing")
        else:
            print("⚠️  Ray requested but not installed - falling back to multiprocessing")
            print("   Install with: pip install 'ray[default]'")
    
    # Check yfinance availability if --real-data requested
    if args.real_data and not YFINANCE_AVAILABLE:
        print("❌ --real-data requires yfinance. Install with: pip install yfinance")
        sys.exit(1)
    
    # Validate train-test-split
    if args.train_test_split > 0 and not args.real_data:
        print("⚠️  --train-test-split requires --real-data. Disabling split.")
        args.train_test_split = 0.0
    if args.train_test_split > 0 and (args.train_test_split < 0.5 or args.train_test_split > 0.95):
        print(f"⚠️  --train-test-split should be between 0.5 and 0.95 (got {args.train_test_split}). Using 0.7.")
        args.train_test_split = 0.7
    
    optimizer = IntradayGeneticOptimizer(
        symbols=symbols,
        days=args.days,
        initial_capital=args.capital,
        config=config,
        verbose=not args.quiet,
        seed=args.seed,
        max_positions=args.max_positions,
        use_real_data=args.real_data,
        log_file=args.log_file,
        train_test_split=args.train_test_split
    )
    
    # Set up checkpoint/resume
    if args.resume:
        checkpoint_file = args.checkpoint_file or (args.output.replace('.json', '') + '.checkpoint.json')
        optimizer.checkpoint_path = checkpoint_file
        print(f"📋 Checkpoint/resume enabled: {checkpoint_file}")
    
    # Run optimization
    best_gene = optimizer.run()
    
    # Print results
    print_evolution_summary(optimizer)
    optimizer.print_best_config()
    
    # Enhancement #6: Validate against real trades if requested
    if args.validate_real:
        validation_result = validate_against_real_trades(best_gene)
    
    # Save results
    optimizer.save_results(args.output)
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\n⏱️  Genetic optimizer completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total runtime: {elapsed.total_seconds():.2f} seconds ({elapsed.total_seconds()/60:.1f} minutes)")
    
    return best_gene


if __name__ == '__main__':
    main()
