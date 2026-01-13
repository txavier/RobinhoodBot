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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

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
    golden_cross_hours: int = 24 # Range: 7-72 hours (1-10 trading days equivalent)
    
    # Risk management
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
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IntradayTradingGene':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def __str__(self) -> str:
        return (f"SMA({self.short_sma}/{self.long_sma}) GC:{self.golden_cross_hours}h "
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
    
    # Whether to optimize filter settings (True = can toggle filters on/off)
    optimize_filters: bool = False
    
    # Parameter ranges for random generation and mutation
    param_ranges: Dict = field(default_factory=lambda: {
        'short_sma': (5, 50),         # Hours
        'long_sma': (20, 100),        # Hours
        'golden_cross_hours': (7, 72), # Hours (1-10 trading days)
        'stop_loss_pct': (1.0, 15.0),
        'take_profit_pct': (0.3, 3.0), # Tighter range for day trading
        'position_size_pct': (5.0, 30.0),
        'slope_threshold': (0.0001, 0.002),
        'price_cap_value': (500.0, 5000.0),
    })
    
    # Fitness weights (how much each metric contributes to fitness)
    # Adjusted for day trading: emphasize consistency and risk-adjusted returns
    fitness_weights: Dict = field(default_factory=lambda: {
        'total_return_pct': 0.25,      # 25% weight on total return
        'sharpe_ratio': 0.30,          # 30% weight on risk-adjusted return (higher for day trading)
        'win_rate': 0.15,              # 15% weight on win rate
        'profit_factor': 0.15,         # 15% weight on profit factor
        'max_drawdown_penalty': 0.10,  # 10% penalty for drawdown
        'trades_per_day_bonus': 0.05,  # 5% bonus for active trading
    })


class IntradayGeneticOptimizer:
    """
    Genetic algorithm optimizer for intraday trading strategy parameters.
    Uses hourly data and main.py matching filters.
    """
    
    def __init__(
        self,
        symbols: List[str],
        days: int = 60,
        initial_capital: float = 10000.0,
        config: Optional[IntradayGeneticConfig] = None,
        verbose: bool = True,
        seed: int = None
    ):
        self.symbols = symbols
        self.days = days
        self.initial_capital = initial_capital
        self.config = config or IntradayGeneticConfig()
        self.verbose = verbose
        self.seed = seed
        
        # Track evolution history
        self.generation_history: List[Dict] = []
        self.best_gene: Optional[IntradayTradingGene] = None
        self.best_fitness: float = float('-inf')
        
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
            golden_cross_hours=random.randint(*ranges['golden_cross_hours']),
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
            # Default main.py settings
            IntradayTradingGene(
                short_sma=20, long_sma=50, golden_cross_hours=24,
                stop_loss_pct=5.0, take_profit_pct=0.70, position_size_pct=20.0,
                slope_threshold=0.0008, price_cap_value=2100.0
            ),
            # More aggressive day trading
            IntradayTradingGene(
                short_sma=10, long_sma=30, golden_cross_hours=14,
                stop_loss_pct=3.0, take_profit_pct=1.0, position_size_pct=15.0,
                slope_threshold=0.001, price_cap_value=1500.0
            ),
            # More conservative
            IntradayTradingGene(
                short_sma=30, long_sma=70, golden_cross_hours=35,
                stop_loss_pct=7.0, take_profit_pct=0.5, position_size_pct=25.0,
                slope_threshold=0.0005, price_cap_value=3000.0
            ),
            # Very aggressive (quick scalping)
            IntradayTradingGene(
                short_sma=5, long_sma=15, golden_cross_hours=7,
                stop_loss_pct=2.0, take_profit_pct=0.5, position_size_pct=10.0,
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
            golden_cross_hours=gene.golden_cross_hours,
            stop_loss_pct=gene.stop_loss_pct,
            take_profit_pct=gene.take_profit_pct,
            use_stop_loss=True,
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
        """Evaluate fitness for entire population"""
        # Use consistent seed for data generation within a generation
        data_seed = (self.seed + generation * 1000) if self.seed else None
        
        for i, gene in enumerate(population):
            self.evaluate_fitness(gene, data_seed)
            if self.verbose:
                print(f"  Evaluating {i+1}/{len(population)}: {gene}")
        
        # Sort by fitness (descending)
        population.sort(key=lambda g: g.fitness, reverse=True)
        return population
    
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
        params = ['short_sma', 'long_sma', 'golden_cross_hours',
                  'stop_loss_pct', 'take_profit_pct', 'position_size_pct',
                  'slope_threshold', 'price_cap_value']
        
        for param in params:
            if random.random() < 0.5:
                setattr(child1, param, getattr(parent1, param))
                setattr(child2, param, getattr(parent2, param))
            else:
                setattr(child1, param, getattr(parent2, param))
                setattr(child2, param, getattr(parent1, param))
        
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
            delta = random.randint(-7, 7)
            mutated.golden_cross_hours = max(ranges['golden_cross_hours'][0],
                                            min(ranges['golden_cross_hours'][1],
                                                mutated.golden_cross_hours + delta))
        
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
        """
        print(f"\n{'='*70}")
        print("INTRADAY GENETIC ALGORITHM OPTIMIZER")
        print(f"{'='*70}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Trading Days: {self.days}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Population: {self.config.population_size}")
        print(f"Generations: {self.config.generations}")
        print(f"Mutation Rate: {self.config.mutation_rate}")
        print(f"Crossover Rate: {self.config.crossover_rate}")
        print(f"Optimize Filters: {self.config.optimize_filters}")
        if self.seed:
            print(f"Random Seed: {self.seed}")
        print(f"{'='*70}\n")
        
        # Set random seed if provided
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Initialize population
        print("Initializing population...")
        population = self.initialize_population()
        
        # Evolution loop
        for gen in range(self.config.generations):
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
            
            # Create next generation (skip on last iteration)
            if gen < self.config.generations - 1:
                population = self.create_next_generation(population)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        
        return self.best_gene
    
    def save_results(self, filepath: str = "genetic_optimization_intraday_result.json"):
        """Save optimization results to JSON file"""
        results = {
            'optimization_date': str(datetime.now()),
            'optimizer_type': 'intraday',
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
        print(f"# short_sma_hours = {gene.short_sma}  # ~{gene.short_sma/7:.1f} trading days")
        print(f"# long_sma_hours = {gene.long_sma}   # ~{gene.long_sma/7:.1f} trading days")
        print(f"golden_cross_buy_days = {max(1, gene.golden_cross_hours // 7)}  # {gene.golden_cross_hours} hours")
        print(f"\n# Risk Management")
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
        print(f"{'='*70}\n")


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
        '--capital', '-c', type=float, default=10000,
        help='Initial capital (default: 10000)'
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
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    config = IntradayGeneticConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        optimize_filters=args.optimize_filters,
    )
    
    optimizer = IntradayGeneticOptimizer(
        symbols=symbols,
        days=args.days,
        initial_capital=args.capital,
        config=config,
        verbose=not args.quiet,
        seed=args.seed
    )
    
    # Run optimization
    best_gene = optimizer.run()
    
    # Print results
    print_evolution_summary(optimizer)
    optimizer.print_best_config()
    
    # Save results
    optimizer.save_results(args.output)
    
    return best_gene


if __name__ == '__main__':
    main()
