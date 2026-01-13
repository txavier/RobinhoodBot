#!/usr/bin/env python3
"""
Genetic Algorithm Optimizer for RobinhoodBot

This module uses a genetic algorithm to evolve trading strategy parameters
by running backtests and selecting the fittest parameter combinations.

Usage:
    python genetic_optimizer.py --generations 20 --population 30
    python genetic_optimizer.py --symbols AAPL,MSFT --generations 50 --sample-data sample_data/
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

from backtest import (
    Backtester, TradingStrategy, HistoricalDataProvider,
    BacktestResult, print_result
)


@dataclass
class TradingGene:
    """
    Represents a chromosome containing trading strategy parameters.
    Each gene is a parameter that can be evolved.
    """
    # SMA parameters
    short_sma: int = 20          # Range: 5-50
    long_sma: int = 50           # Range: 20-200
    golden_cross_days: int = 3   # Range: 1-10
    
    # Risk management
    stop_loss_pct: float = 5.0   # Range: 1-15
    take_profit_pct: float = 0.7 # Range: 0.3-5.0
    
    # Position sizing (as percentage of portfolio)
    position_size_pct: float = 15.0  # Range: 5-30
    
    # Fitness score (set after evaluation)
    fitness: float = 0.0
    
    # Backtest results for reference
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingGene':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def __str__(self) -> str:
        return (f"SMA({self.short_sma}/{self.long_sma}) GC:{self.golden_cross_days}d "
                f"SL:{self.stop_loss_pct:.1f}% TP:{self.take_profit_pct:.1f}% "
                f"Pos:{self.position_size_pct:.0f}% | Fit:{self.fitness:.4f}")


@dataclass
class GeneticConfig:
    """Configuration for the genetic algorithm"""
    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_size: int = 3  # Top performers kept unchanged
    tournament_size: int = 5
    
    # Parameter ranges for random generation and mutation
    param_ranges: Dict = field(default_factory=lambda: {
        'short_sma': (5, 50),
        'long_sma': (20, 200),
        'golden_cross_days': (1, 10),
        'stop_loss_pct': (1.0, 15.0),
        'take_profit_pct': (0.3, 5.0),
        'position_size_pct': (5.0, 30.0),
    })
    
    # Fitness weights (how much each metric contributes to fitness)
    fitness_weights: Dict = field(default_factory=lambda: {
        'total_return_pct': 0.30,      # 30% weight on total return
        'sharpe_ratio': 0.25,          # 25% weight on risk-adjusted return
        'win_rate': 0.15,              # 15% weight on win rate
        'profit_factor': 0.15,         # 15% weight on profit factor
        'max_drawdown_penalty': 0.15,  # 15% penalty for drawdown
    })


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for trading strategy parameters.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        config: Optional[GeneticConfig] = None,
        data_provider: Optional[HistoricalDataProvider] = None,
        verbose: bool = True
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.config = config or GeneticConfig()
        self.data_provider = data_provider or HistoricalDataProvider()
        self.verbose = verbose
        
        # Track evolution history
        self.generation_history: List[Dict] = []
        self.best_gene: Optional[TradingGene] = None
        self.best_fitness: float = float('-inf')
        
    def create_random_gene(self) -> TradingGene:
        """Create a random trading gene within parameter ranges"""
        ranges = self.config.param_ranges
        
        short_sma = random.randint(*ranges['short_sma'])
        # Ensure long_sma > short_sma
        long_sma_min = max(ranges['long_sma'][0], short_sma + 10)
        long_sma = random.randint(long_sma_min, ranges['long_sma'][1])
        
        return TradingGene(
            short_sma=short_sma,
            long_sma=long_sma,
            golden_cross_days=random.randint(*ranges['golden_cross_days']),
            stop_loss_pct=round(random.uniform(*ranges['stop_loss_pct']), 1),
            take_profit_pct=round(random.uniform(*ranges['take_profit_pct']), 2),
            position_size_pct=round(random.uniform(*ranges['position_size_pct']), 1),
        )
    
    def initialize_population(self) -> List[TradingGene]:
        """Create initial random population"""
        population = []
        
        # Add some sensible defaults to seed the population
        defaults = [
            TradingGene(short_sma=20, long_sma=50, golden_cross_days=3, 
                       stop_loss_pct=5.0, take_profit_pct=0.7, position_size_pct=15.0),
            TradingGene(short_sma=10, long_sma=30, golden_cross_days=5,
                       stop_loss_pct=3.0, take_profit_pct=1.0, position_size_pct=10.0),
            TradingGene(short_sma=15, long_sma=40, golden_cross_days=3,
                       stop_loss_pct=7.0, take_profit_pct=1.5, position_size_pct=20.0),
        ]
        population.extend(defaults)
        
        # Fill rest with random genes
        while len(population) < self.config.population_size:
            population.append(self.create_random_gene())
        
        return population
    
    def evaluate_fitness(self, gene: TradingGene) -> float:
        """
        Run backtest with gene parameters and calculate fitness score.
        Fitness is a weighted combination of multiple metrics.
        """
        # Create strategy with gene parameters
        strategy = TradingStrategy(
            short_sma=gene.short_sma,
            long_sma=gene.long_sma,
            golden_cross_days=gene.golden_cross_days,
            stop_loss_pct=gene.stop_loss_pct,
            take_profit_pct=gene.take_profit_pct,
            use_stop_loss=True
        )
        
        # Run backtest
        backtester = Backtester(
            initial_capital=self.initial_capital,
            strategy=strategy,
            data_provider=self.data_provider
        )
        
        result = backtester.run(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            verbose=False
        )
        
        # Store results in gene
        gene.total_return_pct = result.total_return_pct
        gene.win_rate = result.win_rate
        gene.sharpe_ratio = result.sharpe_ratio
        gene.max_drawdown_pct = result.max_drawdown_pct
        gene.profit_factor = min(result.profit_factor, 10.0)  # Cap at 10 to avoid inf
        gene.total_trades = result.total_trades
        
        # Calculate weighted fitness
        weights = self.config.fitness_weights
        
        # Normalize metrics to similar scales
        fitness = 0.0
        
        # Return contribution (can be negative)
        fitness += weights['total_return_pct'] * (result.total_return_pct / 10.0)
        
        # Sharpe ratio contribution (typically -3 to +3)
        fitness += weights['sharpe_ratio'] * result.sharpe_ratio
        
        # Win rate contribution (0-100, normalize to 0-1)
        fitness += weights['win_rate'] * (result.win_rate / 100.0)
        
        # Profit factor contribution (0-10, already capped)
        pf = min(result.profit_factor, 10.0) if result.profit_factor != float('inf') else 10.0
        fitness += weights['profit_factor'] * (pf / 5.0)
        
        # Drawdown penalty (higher drawdown = lower fitness)
        fitness -= weights['max_drawdown_penalty'] * (result.max_drawdown_pct / 10.0)
        
        # Bonus for having trades (penalize strategies that never trade)
        if result.total_trades == 0:
            fitness -= 1.0
        elif result.total_trades < 3:
            fitness -= 0.3
        
        gene.fitness = fitness
        return fitness
    
    def evaluate_population(self, population: List[TradingGene]) -> List[TradingGene]:
        """Evaluate fitness for entire population"""
        for i, gene in enumerate(population):
            self.evaluate_fitness(gene)
            if self.verbose:
                print(f"  Evaluating {i+1}/{len(population)}: {gene}")
        
        # Sort by fitness (descending)
        population.sort(key=lambda g: g.fitness, reverse=True)
        return population
    
    def tournament_select(self, population: List[TradingGene]) -> TradingGene:
        """Select a gene using tournament selection"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def crossover(self, parent1: TradingGene, parent2: TradingGene) -> Tuple[TradingGene, TradingGene]:
        """
        Perform crossover between two parent genes.
        Uses uniform crossover - each parameter randomly chosen from either parent.
        """
        if random.random() > self.config.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = TradingGene()
        child2 = TradingGene()
        
        # For each parameter, randomly choose from either parent
        params = ['short_sma', 'long_sma', 'golden_cross_days', 
                  'stop_loss_pct', 'take_profit_pct', 'position_size_pct']
        
        for param in params:
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
    
    def mutate(self, gene: TradingGene) -> TradingGene:
        """
        Apply random mutations to a gene.
        Each parameter has a chance to be mutated.
        """
        ranges = self.config.param_ranges
        mutated = copy.deepcopy(gene)
        
        # Mutate each parameter with probability mutation_rate
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
            mutated.golden_cross_days = max(ranges['golden_cross_days'][0],
                                           min(ranges['golden_cross_days'][1], 
                                               mutated.golden_cross_days + delta))
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-2.0, 2.0)
            mutated.stop_loss_pct = round(max(ranges['stop_loss_pct'][0],
                                             min(ranges['stop_loss_pct'][1],
                                                 mutated.stop_loss_pct + delta)), 1)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-0.5, 0.5)
            mutated.take_profit_pct = round(max(ranges['take_profit_pct'][0],
                                               min(ranges['take_profit_pct'][1],
                                                   mutated.take_profit_pct + delta)), 2)
        
        if random.random() < self.config.mutation_rate:
            delta = random.uniform(-5.0, 5.0)
            mutated.position_size_pct = round(max(ranges['position_size_pct'][0],
                                                 min(ranges['position_size_pct'][1],
                                                     mutated.position_size_pct + delta)), 1)
        
        # Ensure long_sma > short_sma
        if mutated.long_sma <= mutated.short_sma:
            mutated.long_sma = mutated.short_sma + 15
        
        return mutated
    
    def create_next_generation(self, population: List[TradingGene]) -> List[TradingGene]:
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
    
    def run(self) -> TradingGene:
        """
        Run the genetic algorithm optimization.
        Returns the best gene found.
        """
        print(f"\n{'='*70}")
        print("GENETIC ALGORITHM OPTIMIZER")
        print(f"{'='*70}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Population: {self.config.population_size}")
        print(f"Generations: {self.config.generations}")
        print(f"Mutation Rate: {self.config.mutation_rate}")
        print(f"Crossover Rate: {self.config.crossover_rate}")
        print(f"{'='*70}\n")
        
        # Initialize population
        print("Initializing population...")
        population = self.initialize_population()
        
        # Evolution loop
        for gen in range(self.config.generations):
            print(f"\n--- Generation {gen + 1}/{self.config.generations} ---")
            
            # Evaluate fitness
            population = self.evaluate_population(population)
            
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
            })
            
            print(f"\n  Best:  {gen_best}")
            print(f"  Avg Fitness: {gen_avg:.4f} | Worst: {gen_worst:.4f}")
            print(f"  Return: {gen_best.total_return_pct:+.2f}% | "
                  f"Sharpe: {gen_best.sharpe_ratio:.2f} | "
                  f"Win Rate: {gen_best.win_rate:.1f}% | "
                  f"Trades: {gen_best.total_trades}")
            
            # Create next generation (skip on last iteration)
            if gen < self.config.generations - 1:
                population = self.create_next_generation(population)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        
        return self.best_gene
    
    def save_results(self, filepath: str = "genetic_optimization_result.json"):
        """Save optimization results to JSON file"""
        results = {
            'optimization_date': str(datetime.now()),
            'symbols': self.symbols,
            'start_date': str(self.start_date.date()),
            'end_date': str(self.end_date.date()),
            'initial_capital': self.initial_capital,
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size,
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
        print("BEST CONFIGURATION FOUND")
        print(f"{'='*70}")
        print(f"\n# Add these to your config.py:")
        print(f"# SMA Settings")
        print(f"short_sma = {gene.short_sma}")
        print(f"long_sma = {gene.long_sma}")
        print(f"golden_cross_buy_days = {gene.golden_cross_days}")
        print(f"\n# Risk Management")
        print(f"stop_loss_percent = {gene.stop_loss_pct}")
        print(f"take_profit_percent = {gene.take_profit_pct}")
        print(f"\n# Position Sizing")
        print(f"purchase_limit_percentage = {gene.position_size_pct}")
        print(f"\n# Performance Metrics:")
        print(f"# Total Return: {gene.total_return_pct:+.2f}%")
        print(f"# Win Rate: {gene.win_rate:.1f}%")
        print(f"# Sharpe Ratio: {gene.sharpe_ratio:.4f}")
        print(f"# Max Drawdown: {gene.max_drawdown_pct:.2f}%")
        print(f"# Profit Factor: {gene.profit_factor:.2f}")
        print(f"# Total Trades: {gene.total_trades}")
        print(f"# Fitness Score: {gene.fitness:.4f}")
        print(f"{'='*70}\n")


def print_evolution_summary(optimizer: GeneticOptimizer):
    """Print a summary of how fitness evolved over generations"""
    if not optimizer.generation_history:
        return
    
    print(f"\n{'='*70}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Gen':>4} | {'Best Fit':>10} | {'Avg Fit':>10} | {'Return %':>10} | {'Sharpe':>8} | {'Win %':>7}")
    print("-" * 70)
    
    for h in optimizer.generation_history:
        print(f"{h['generation']:>4} | {h['best_fitness']:>10.4f} | {h['avg_fitness']:>10.4f} | "
              f"{h['best_return_pct']:>+10.2f} | {h['best_sharpe']:>8.2f} | {h['best_win_rate']:>7.1f}")
    
    # Show improvement
    if len(optimizer.generation_history) > 1:
        first = optimizer.generation_history[0]
        last = optimizer.generation_history[-1]
        fitness_improvement = last['best_fitness'] - first['best_fitness']
        return_improvement = last['best_return_pct'] - first['best_return_pct']
        print("-" * 70)
        print(f"Improvement: Fitness {fitness_improvement:+.4f} | Return {return_improvement:+.2f}%")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm Optimizer for Trading Strategy'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        default='AAPL,MSFT,GOOGL',
        help='Comma-separated stock symbols (default: AAPL,MSFT,GOOGL)'
    )
    
    parser.add_argument(
        '--start', '-st',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end', '-e',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=200,
        help='Number of days to backtest (default: 200)'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000)'
    )
    
    parser.add_argument(
        '--population', '-p',
        type=int,
        default=20,
        help='Population size (default: 20)'
    )
    
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=15,
        help='Number of generations (default: 15)'
    )
    
    parser.add_argument(
        '--mutation-rate', '-m',
        type=float,
        default=0.15,
        help='Mutation rate (default: 0.15)'
    )
    
    parser.add_argument(
        '--crossover-rate',
        type=float,
        default=0.7,
        help='Crossover rate (default: 0.7)'
    )
    
    parser.add_argument(
        '--sample-data',
        type=str,
        help='Directory containing sample data files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='genetic_optimization_result.json',
        help='Output file for results (default: genetic_optimization_result.json)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Determine date range
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days)
    
    # Create config
    config = GeneticConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
    )
    
    # Create data provider
    data_provider = HistoricalDataProvider(
        sample_data_dir=args.sample_data
    )
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        config=config,
        data_provider=data_provider,
        verbose=not args.quiet
    )
    
    # Run optimization
    best_gene = optimizer.run()
    
    # Print results
    print_evolution_summary(optimizer)
    optimizer.print_best_config()
    
    # Save results
    optimizer.save_results(args.output)
    
    # Run final backtest with best parameters and show detailed results
    print("\n" + "="*70)
    print("FINAL BACKTEST WITH OPTIMIZED PARAMETERS")
    print("="*70)
    
    best_strategy = TradingStrategy(
        short_sma=best_gene.short_sma,
        long_sma=best_gene.long_sma,
        golden_cross_days=best_gene.golden_cross_days,
        stop_loss_pct=best_gene.stop_loss_pct,
        take_profit_pct=best_gene.take_profit_pct,
        use_stop_loss=True
    )
    
    final_backtester = Backtester(
        initial_capital=args.capital,
        strategy=best_strategy,
        data_provider=data_provider
    )
    
    final_result = final_backtester.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        verbose=True
    )
    
    print_result(final_result)


if __name__ == '__main__':
    main()
