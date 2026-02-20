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
    generate_market_data, generate_golden_cross_intraday
)

# Import config defaults
try:
    from config import (
        stop_loss_percent, take_profit_percent,
        golden_cross_buy_days, price_cap
    )
except ImportError:
    stop_loss_percent = 5
    take_profit_percent = 0.70
    golden_cross_buy_days = 3
    price_cap = 2100


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
    golden_cross_buy_days: int = 2 # Range: 1-10 trading days
    
    # Dynamic SMA parameters (used when use_dynamic_sma=True)
    short_sma_downtrend: int = 14  # Range: 5-30 hours - used when market not in uptrend
    short_sma_take_profit: int = 5  # Range: 3-15 hours - used after take profit (only if balance < $25k PDT limit)
    long_sma_take_profit: int = 7   # Range: 5-20 hours - used after take profit (only if balance < $25k PDT limit)
    
    # Risk management
    use_stop_loss: bool = True   # Enable/disable stop loss selling
    stop_loss_pct: float = 5.0   # Range: 1-15%
    take_profit_pct: float = 0.7 # Range: 0.3-3.0%
    
    # Position sizing (as percentage of portfolio)
    position_size_pct: float = 20.0  # Range: 5-30%
    
    # Main.py specific parameters
    slope_threshold: float = 0.0008  # Range: 0.0001-0.002 (from order_symbols_by_slope)
    price_cap_value: float = 2100.0  # Range: 500-5000
    
    # Filter toggles (True = enabled, matching main.py behavior)
    use_market_filter: bool = True
    use_eod_filter: bool = True
    use_profit_before_eod: bool = True
    use_price_5hr_check: bool = True
    use_dynamic_sma: bool = True
    use_slope_ordering: bool = True
    use_price_cap: bool = True
    
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
                f"Slope:{self.slope_threshold:.4f} | Fit:{self.fitness:.4f}")


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
    
    # Parameter ranges for random generation and mutation
    param_ranges: Dict = field(default_factory=lambda: {
        'short_sma': (5, 80),         # Hours (upper expanded from 50 - best hit 46/50)
        'long_sma': (20, 100),        # Hours
        'golden_cross_buy_days': (1, 10), # Trading days
        'short_sma_downtrend': (5, 50),  # Hours - more conservative SMA in downtrend (upper expanded from 30 - best hit 27/30)
        'short_sma_take_profit': (3, 15),  # Hours - aggressive SMA after take profit
        'long_sma_take_profit': (2, 20),   # Hours - aggressive SMA after take profit (lower expanded from 5 - best hit 5/5)
        'stop_loss_pct': (0.5, 15.0),  # (lower expanded from 1.0 - best hit 2.1/1.0)
        'take_profit_pct': (0.3, 3.0), # Tighter range for day trading
        'position_size_pct': (5.0, 30.0),
        'slope_threshold': (0.0001, 0.002),
        'price_cap_value': (500.0, 5000.0),
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


def _evaluate_gene_worker(args: Tuple) -> IntradayTradingGene:
    """
    Module-level function for parallel gene evaluation.
    Must be at module level for multiprocessing to pickle it.
    
    Args is a tuple of (gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions)
    """
    return _evaluate_gene_impl(*args)


def _evaluate_gene_impl(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions) -> IntradayTradingGene:
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
        use_dynamic_sma=gene.use_dynamic_sma,
        use_slope_ordering=gene.use_slope_ordering,
        use_price_cap=gene.use_price_cap,
        price_cap_value=gene.price_cap_value,
        slope_threshold=gene.slope_threshold,
    )
    
    # Run backtest
    backtester = IntradayBacktester(
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_position_pct=gene.position_size_pct,
        strategy=strategy
    )
    
    result = backtester.run(
        symbols=symbols,
        days=days,
        seed=data_seed,
        verbose=False
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
        max_positions: int = 5
    ):
        self.symbols = symbols
        self.days = days
        self.initial_capital = initial_capital
        self.config = config or IntradayGeneticConfig()
        self.verbose = verbose
        self.seed = seed
        self.max_positions = max_positions
        
        # Track evolution history
        self.generation_history: List[Dict] = []
        self.best_gene: Optional[IntradayTradingGene] = None
        self.best_fitness: float = float('-inf')
        
        # Checkpoint/resume support
        self.checkpoint_path: Optional[str] = None
        self._interrupted = False
        
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
            print(f"  💾 Checkpoint saved: Gen {generation + 1}/{self.config.generations} → {filepath}")
    
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
            price_cap_value=round(random.uniform(*ranges['price_cap_value']), 0),
        )
        
        # Optionally randomize filter settings
        if self.config.optimize_filters:
            gene.use_market_filter = random.choice([True, False])
            gene.use_eod_filter = random.choice([True, False])
            gene.use_profit_before_eod = random.choice([True, False])
            gene.use_price_5hr_check = random.choice([True, False])
            gene.use_dynamic_sma = random.choice([True, False])
            gene.use_slope_ordering = random.choice([True, False])
            gene.use_price_cap = random.choice([True, False])
        
        return gene
    
    def initialize_population(self) -> List[IntradayTradingGene]:
        """Create initial random population"""
        population = []
        
        # Add some sensible defaults to seed the population
        defaults = [
            # Default main.py settings (matching config.py)
            IntradayTradingGene(
                short_sma=20, long_sma=50, golden_cross_buy_days=2,
                short_sma_downtrend=14, short_sma_take_profit=5, long_sma_take_profit=7,
                use_stop_loss=True, stop_loss_pct=5.0, take_profit_pct=1.52, position_size_pct=15.0,
                slope_threshold=0.0008, price_cap_value=2100.0
            ),
            # More aggressive day trading
            IntradayTradingGene(
                short_sma=10, long_sma=30, golden_cross_buy_days=2,
                short_sma_downtrend=8, short_sma_take_profit=3, long_sma_take_profit=5,
                use_stop_loss=True, stop_loss_pct=3.0, take_profit_pct=1.0, position_size_pct=20.0,
                slope_threshold=0.001, price_cap_value=1500.0
            ),
            # More conservative
            IntradayTradingGene(
                short_sma=30, long_sma=70, golden_cross_buy_days=5,
                short_sma_downtrend=20, short_sma_take_profit=8, long_sma_take_profit=12,
                use_stop_loss=True, stop_loss_pct=7.0, take_profit_pct=0.5, position_size_pct=25.0,
                slope_threshold=0.0005, price_cap_value=3000.0
            ),
            # Very aggressive (quick scalping)
            IntradayTradingGene(
                short_sma=5, long_sma=15, golden_cross_buy_days=1,
                short_sma_downtrend=5, short_sma_take_profit=3, long_sma_take_profit=5,
                use_stop_loss=True, stop_loss_pct=2.0, take_profit_pct=0.5, position_size_pct=10.0,
                slope_threshold=0.0015, price_cap_value=1000.0
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
            use_dynamic_sma=gene.use_dynamic_sma,
            use_slope_ordering=gene.use_slope_ordering,
            use_price_cap=gene.use_price_cap,
            price_cap_value=gene.price_cap_value,
            slope_threshold=gene.slope_threshold,
        )
        
        # Run backtest
        backtester = IntradayBacktester(
            initial_capital=self.initial_capital,
            max_positions=5,
            max_position_pct=gene.position_size_pct,
            strategy=strategy
        )
        
        result = backtester.run(
            symbols=self.symbols,
            days=self.days,
            seed=data_seed,
            verbose=False
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
            # Prepare args for worker functions
            eval_args = [
                (gene, self.symbols, self.days, self.initial_capital, 
                 dict(self.config.fitness_weights), data_seed, self.max_positions)
                for gene in population
            ]
            
            # Try Ray first if enabled and available
            if self.config.use_ray and RAY_AVAILABLE:
                population = self._evaluate_with_ray(eval_args, num_workers)
            else:
                population = self._evaluate_with_multiprocessing(eval_args, num_workers)
            
            if self.verbose:
                for i, gene in enumerate(population):
                    print(f"    Gene {i+1}: {gene}")
        else:
            # Sequential evaluation (single worker or small population)
            for i, gene in enumerate(population):
                self.evaluate_fitness(gene, data_seed)
                if self.verbose:
                    print(f"  Evaluating {i+1}/{len(population)}: {gene}")
        
        # Sort by fitness (descending)
        population.sort(key=lambda g: g.fitness, reverse=True)
        return population
    
    def _evaluate_with_ray(self, eval_args: List[Tuple], num_workers: int) -> List[IntradayTradingGene]:
        """Evaluate genes using Ray (works locally or distributed across Kubernetes)"""
        if self.verbose:
            ray_info = ray.cluster_resources() if ray.is_initialized() else {}
            num_nodes = int(ray_info.get('CPU', num_workers))
            print(f"  Evaluating {len(eval_args)} genes using Ray ({num_nodes} CPUs available)...")
        
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
                    print("  Ray memory monitor disabled (--disable-ray-mem-monitor)")
            
            # ray.init() auto-detects: local CPUs or K8s cluster
            ray.init(**init_kwargs)
        
        # Define remote function inside method to avoid module-level Ray dependency
        @ray.remote
        def ray_evaluate_gene(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions):
            return _evaluate_gene_impl(gene, symbols, days, initial_capital, fitness_weights, data_seed, max_positions)
        
        # Submit all tasks
        futures = [
            ray_evaluate_gene.remote(*args) for args in eval_args
        ]
        
        # Gather results
        evaluated_population = ray.get(futures)
        return evaluated_population
    
    def _evaluate_with_multiprocessing(self, eval_args: List[Tuple], num_workers: int) -> List[IntradayTradingGene]:
        """Evaluate genes using multiprocessing.Pool (local only)"""
        if self.verbose:
            print(f"  Evaluating {len(eval_args)} genes using {num_workers} multiprocessing workers...")
        
        with Pool(processes=num_workers) as pool:
            evaluated_population = pool.map(_evaluate_gene_worker, eval_args)
        
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
                  'slope_threshold', 'price_cap_value']
        
        for param in params:
            if random.random() < 0.5:
                setattr(child1, param, getattr(parent1, param))
                setattr(child2, param, getattr(parent2, param))
            else:
                setattr(child1, param, getattr(parent2, param))
                setattr(child2, param, getattr(parent1, param))
        
        # Boolean parameters (use_stop_loss)
        if random.random() < 0.5:
            child1.use_stop_loss = parent1.use_stop_loss
            child2.use_stop_loss = parent2.use_stop_loss
        else:
            child1.use_stop_loss = parent2.use_stop_loss
            child2.use_stop_loss = parent1.use_stop_loss
        
        # Filter toggles (if optimizing)
        if self.config.optimize_filters:
            filter_params = ['use_market_filter', 'use_eod_filter', 'use_profit_before_eod',
                            'use_price_5hr_check', 'use_dynamic_sma', 'use_slope_ordering', 'use_price_cap']
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
            delta = random.uniform(-5.0, 5.0)
            mutated.position_size_pct = round(max(ranges['position_size_pct'][0],
                                                 min(ranges['position_size_pct'][1],
                                                     mutated.position_size_pct + delta)), 1)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-0.0003, 0.0003)
            mutated.slope_threshold = round(max(ranges['slope_threshold'][0],
                                               min(ranges['slope_threshold'][1],
                                                   mutated.slope_threshold + delta)), 4)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-500, 500)
            mutated.price_cap_value = round(max(ranges['price_cap_value'][0],
                                               min(ranges['price_cap_value'][1],
                                                   mutated.price_cap_value + delta)), 0)
        
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
                mutated.use_dynamic_sma = not mutated.use_dynamic_sma
            if random.random() < self.config.mutation_rate:
                mutated.use_slope_ordering = not mutated.use_slope_ordering
            if random.random() < self.config.mutation_rate:
                mutated.use_price_cap = not mutated.use_price_cap
        
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
                print(f"\n🔄 RESUMING from checkpoint: Generation {completed}/{total} completed")
                print(f"   Checkpoint saved at: {checkpoint['saved_at']}")
                print(f"   Best fitness so far: {checkpoint['best_fitness']:.4f}")
                start_gen, population = self.restore_from_checkpoint(checkpoint)
                resumed = True
                # Create next generation from the restored population
                # (the checkpoint saved the population AFTER evaluation but BEFORE next_gen creation)
                if start_gen < self.config.generations:
                    population = self.create_next_generation(population)
                    print(f"   Resuming from Generation {start_gen + 1}...\n")
        
        if not resumed:
            self.start_time = datetime.now()
        
        print(f"\n{'='*70}")
        print("INTRADAY GENETIC ALGORITHM OPTIMIZER")
        print(f"{'='*70}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Trading Days: {self.days}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Population: {self.config.population_size}")
        print(f"Generations: {self.config.generations}")
        print(f"Mutation Rate: {self.config.mutation_rate}")
        print(f"Crossover Rate: {self.config.crossover_rate}")
        print(f"Optimize Filters: {self.config.optimize_filters}")
        num_workers = self.config.num_workers if self.config.num_workers > 0 else max(1, cpu_count() - 1)
        print(f"Workers: {num_workers} {'(auto)' if self.config.num_workers == 0 else ''}")
        if self.seed:
            print(f"Random Seed: {self.seed}")
        if self.checkpoint_path:
            print(f"Checkpoint: {self.checkpoint_path} {'(resumed)' if resumed else '(enabled)'}")
        print(f"{'='*70}\n")
        
        # Set random seed if provided (only on fresh start, not resume)
        if self.seed and not resumed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Initialize population (only on fresh start)
        if population is None:
            print("Initializing population...")
            population = self.initialize_population()
        
        # Set up signal handler for graceful shutdown
        if self.checkpoint_path:
            self._setup_signal_handler()
        
        # Evolution loop
        try:
            for gen in range(start_gen, self.config.generations):
                print(f"\n--- Generation {gen + 1}/{self.config.generations} ---")
                
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
                
                print(f"\n  Best:  {gen_best}")
                print(f"  Avg Fitness: {gen_avg:.4f} | Worst: {gen_worst:.4f}")
                print(f"  Return: {gen_best.total_return_pct:+.2f}% | "
                      f"Sharpe: {gen_best.sharpe_ratio:.2f} | "
                      f"Win Rate: {gen_best.win_rate:.1f}% | "
                      f"Trades/Day: {gen_best.trades_per_day:.2f}")
                
                # Save checkpoint after each completed generation
                if self.checkpoint_path:
                    self.save_checkpoint(self.checkpoint_path, gen, population)
                
                # Create next generation (skip on last iteration)
                if gen < self.config.generations - 1:
                    population = self.create_next_generation(population)
        
        except KeyboardInterrupt:
            # Save emergency checkpoint if we have any progress
            if self.checkpoint_path and self.generation_history:
                last_completed_gen = len(self.generation_history) - 1 + start_gen
                # Only save if we haven't already (the per-generation save may have caught it)
                print(f"\n  🛑 Interrupted! Checkpoint was saved after Gen {len(self.generation_history)}.")
                print(f"     Resume with: --resume to continue from Gen {len(self.generation_history) + 1}")
            else:
                print(f"\n  🛑 Interrupted before any generation completed. No checkpoint to save.")
            
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
        
        self.end_time = datetime.now()
        
        # Clean up checkpoint file on successful completion
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print(f"  🧹 Checkpoint file removed (optimization complete)")
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        
        return self.best_gene
    
    def save_results(self, filepath: str = "genetic_optimization_intraday_result.json"):
        """Save optimization results to JSON file"""
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
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size,
                'optimize_filters': self.config.optimize_filters,
            },
            'best_gene': self.best_gene.to_dict() if self.best_gene else None,
            'best_fitness': self.best_fitness,
            'generation_history': self.generation_history,
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def print_best_config(self):
        """Print the best configuration found in a copyable format"""
        if not self.best_gene:
            print("No optimization results available")
            return
        
        gene = self.best_gene
        print(f"\n{'='*70}")
        print("BEST INTRADAY CONFIGURATION FOUND")
        print(f"{'='*70}")
        print(f"\n# Add these to your config.py:")
        print(f"# Note: SMA values are in HOURS (for hourly data)")
        print(f"\n# SMA Settings (Hourly)")
        print(f"short_sma = {gene.short_sma}  # ~{gene.short_sma/7:.1f} trading days")
        print(f"long_sma = {gene.long_sma}   # ~{gene.long_sma/7:.1f} trading days")
        print(f"golden_cross_buy_days = {gene.golden_cross_buy_days}  # trading days")
        print(f"\n# Dynamic SMA Settings (used when use_dynamic_sma=True)")
        print(f"short_sma_downtrend = {gene.short_sma_downtrend}  # Used when market not in uptrend")
        print(f"short_sma_take_profit = {gene.short_sma_take_profit}  # Used after take profit (only if balance < $25k PDT limit)")
        print(f"long_sma_take_profit = {gene.long_sma_take_profit}   # Used after take profit (only if balance < $25k PDT limit)")
        print(f"\n# Risk Management")
        print(f"use_stop_loss = {gene.use_stop_loss}")
        print(f"stop_loss_percent = {gene.stop_loss_pct}")
        print(f"take_profit_percent = {gene.take_profit_pct}")
        print(f"\n# Position Sizing")
        print(f"purchase_limit_percentage = {gene.position_size_pct}")
        print(f"\n# Main.py Specific Settings")
        print(f"# slope_threshold = {gene.slope_threshold}  # For order_symbols_by_slope")
        print(f"price_cap = {int(gene.price_cap_value)}")
        
        if self.config.optimize_filters:
            print(f"\n# Filter Settings (optimized)")
            print(f"# use_market_filter = {gene.use_market_filter}")
            print(f"# use_eod_filter = {gene.use_eod_filter}")
            print(f"# use_profit_before_eod = {gene.use_profit_before_eod}")
            print(f"# use_price_5hr_check = {gene.use_price_5hr_check}")
            print(f"# use_dynamic_sma = {gene.use_dynamic_sma}")
            print(f"# use_slope_ordering = {gene.use_slope_ordering}")
            print(f"# use_price_cap = {gene.use_price_cap}")
        
        print(f"\n# Performance Metrics:")
        print(f"# Total Return: {gene.total_return_pct:+.2f}%")
        print(f"# Win Rate: {gene.win_rate:.1f}%")
        print(f"# Sharpe Ratio: {gene.sharpe_ratio:.4f}")
        print(f"# Max Drawdown: {gene.max_drawdown_pct:.2f}%")
        print(f"# Profit Factor: {gene.profit_factor:.2f}")
        print(f"# Total Trades: {gene.total_trades}")
        print(f"# Trades/Day: {gene.trades_per_day:.2f}")
        print(f"# Avg Win: ${gene.avg_win:.2f}")
        print(f"# Avg Loss: ${gene.avg_loss:.2f}")
        print(f"# Fitness Score: {gene.fitness:.4f}")
        
        # Enhancement #10: Display exit reason breakdown
        if gene.exit_reasons:
            print(f"\n# Exit Reason Breakdown (Enhancement #10):")
            total_exits = sum(r['count'] for r in gene.exit_reasons.values())
            for reason, stats in sorted(gene.exit_reasons.items(), key=lambda x: -x[1]['count']):
                pct = (stats['count'] / total_exits * 100) if total_exits > 0 else 0
                win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
                print(f"#   {reason:20}: {stats['count']:>3} ({pct:5.1f}%) | Win: {win_rate:5.1f}% | Avg: ${avg_pnl:>+7.2f}")
        
        print(f"{'='*70}\n")


def validate_against_real_trades(gene: IntradayTradingGene, tradehistory_path: str = "tradehistory-real.json") -> Dict:
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
    buy_reasons_path = "buy_reasons.json"
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
        '--symbols', '-s', type=str, default='AAPL,MSFT,GOOGL',
        help='Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)'
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
        '--resume', action='store_true',
        help='Resume from checkpoint if one exists. Saves progress after each generation so interrupts are recoverable.'
    )
    parser.add_argument(
        '--checkpoint-file', type=str, default=None,
        help='Checkpoint file path (default: <output>.checkpoint.json). Used with --resume.'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    config = IntradayGeneticConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        optimize_filters=args.optimize_filters,
        num_workers=args.workers,
        use_ray=args.use_ray,
        disable_ray_mem_monitor=args.disable_ray_mem_monitor,
    )
    
    # Print Ray status if requested
    if args.use_ray:
        if RAY_AVAILABLE:
            print("🌟 Ray mode enabled - will use Ray for distributed computing")
        else:
            print("⚠️  Ray requested but not installed - falling back to multiprocessing")
            print("   Install with: pip install 'ray[default]'")
    
    optimizer = IntradayGeneticOptimizer(
        symbols=symbols,
        days=args.days,
        initial_capital=args.capital,
        config=config,
        verbose=not args.quiet,
        seed=args.seed,
        max_positions=args.max_positions
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
