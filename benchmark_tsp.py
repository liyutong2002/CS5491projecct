"""
Benchmark script for evaluating FunSearch-evolved TSP heuristics.

Compares:
  1. Nearest Neighbor (NN)
  2. NN + 2-opt
  3. FunSearch-evolved priority (from logs)
  4. FunSearch-evolved priority + 2-opt
  5. Random tour baseline

Usage:
    python benchmark_tsp.py [--tsplib_dir tsp_datasets/] [--log_dir logs/funsearch_tsp/]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple

import tsp_utils


def load_best_evolved_function(log_dir: str) -> str:
    """Load the best evolved priority function from FunSearch logs.

    Args:
        log_dir: path to FunSearch log directory

    Returns:
        function_body: string of the best evolved function body
    """
    samples_dir = os.path.join(log_dir, 'samples')
    if not os.path.exists(samples_dir):
        return None

    best_score = -float('inf')
    best_function = None

    for fname in os.listdir(samples_dir):
        if fname.endswith('.json'):
            with open(os.path.join(samples_dir, fname), 'r') as f:
                data = json.load(f)
                if data.get('score') is not None and data['score'] > best_score:
                    best_score = data['score']
                    best_function = data['function']

    if best_function:
        print(f"Loaded best evolved function (score: {best_score:.2f}):")
        print("-" * 40)
        print(best_function)
        print("-" * 40)

    return best_function


def nearest_neighbor_baseline(dist_matrix: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """Nearest neighbor heuristic."""
    tour = tsp_utils.nearest_neighbor_tour(dist_matrix, start)
    length = tsp_utils.compute_tour_length(dist_matrix, tour)
    return tour, length


def nn_multistart(dist_matrix: np.ndarray, num_starts: int = 5) -> Tuple[List[int], float]:
    """Nearest neighbor with multiple starting cities."""
    n = len(dist_matrix)
    best_tour = None
    best_length = float('inf')
    starts = list(range(min(n, num_starts)))
    for start in starts:
        tour = tsp_utils.nearest_neighbor_tour(dist_matrix, start)
        length = tsp_utils.compute_tour_length(dist_matrix, tour)
        if length < best_length:
            best_length = length
            best_tour = tour
    return best_tour, best_length


def nn_2opt(dist_matrix: np.ndarray, num_starts: int = 5, max_2opt_iter: int = 200) -> Tuple[List[int], float]:
    """Nearest neighbor + 2-opt."""
    n = len(dist_matrix)
    best_tour = None
    best_length = float('inf')
    starts = list(range(min(n, num_starts)))
    for start in starts:
        tour = tsp_utils.nearest_neighbor_tour(dist_matrix, start)
        tour = tsp_utils.two_opt_improve(dist_matrix, tour, max_iter=max_2opt_iter)
        length = tsp_utils.compute_tour_length(dist_matrix, tour)
        if length < best_length:
            best_length = length
            best_tour = tour
    return best_tour, best_length


def random_tour_baseline(dist_matrix: np.ndarray, num_trials: int = 10) -> Tuple[List[int], float]:
    """Random tour baseline."""
    n = len(dist_matrix)
    best_tour = None
    best_length = float('inf')
    for _ in range(num_trials):
        tour = list(np.random.permutation(n))
        length = tsp_utils.compute_tour_length(dist_matrix, tour)
        if length < best_length:
            best_length = length
            best_tour = tour
    return best_tour, best_length


def run_benchmark(instances: Dict[str, Dict], log_dir: str = None):
    """Run benchmark comparison across all instances."""
    print("=" * 100)
    print(f"{'Instance':<20} {'n':>5} {'Random':>12} {'NN':>12} {'NN+2opt':>12} "
          f"{'Optimal':>12} {'NN Gap%':>8} {'2opt Gap%':>8}")
    print("=" * 100)

    total_nn_gap = 0
    total_2opt_gap = 0
    num_with_optimal = 0

    for name, inst in sorted(instances.items(), key=lambda x: x[1]['num_cities']):
        n = inst['num_cities']
        dm = inst['dist_matrix']
        optimal = inst.get('optimal_tour_length')

        # Baselines
        t0 = time.time()
        _, random_len = random_tour_baseline(dm, num_trials=10)
        t_random = time.time() - t0

        t0 = time.time()
        _, nn_len = nn_multistart(dm, num_starts=min(n, 10))
        t_nn = time.time() - t0

        t0 = time.time()
        _, nn2opt_len = nn_2opt(dm, num_starts=min(n, 5), max_2opt_iter=200)
        t_2opt = time.time() - t0

        opt_str = f"{optimal:.0f}" if optimal else "N/A"

        nn_gap_str = ""
        opt2_gap_str = ""
        if optimal:
            nn_gap = (nn_len - optimal) / optimal * 100
            opt2_gap = (nn2opt_len - optimal) / optimal * 100
            nn_gap_str = f"{nn_gap:.1f}%"
            opt2_gap_str = f"{opt2_gap:.1f}%"
            total_nn_gap += nn_gap
            total_2opt_gap += opt2_gap
            num_with_optimal += 1

        print(f"{name:<20} {n:>5} {random_len:>12.0f} {nn_len:>12.0f} {nn2opt_len:>12.0f} "
              f"{opt_str:>12} {nn_gap_str:>8} {opt2_gap_str:>8}")

    if num_with_optimal > 0:
        print("-" * 100)
        print(f"{'Average gap':>59} {total_nn_gap/num_with_optimal:>8.1f}% "
              f"{total_2opt_gap/num_with_optimal:>8.1f}%")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Benchmark TSP heuristics')
    parser.add_argument('--tsplib_dir', type=str, default='tsp_datasets/',
                        help='Directory containing .tsp files')
    parser.add_argument('--log_dir', type=str, default='logs/funsearch_tsp/',
                        help='FunSearch log directory')
    parser.add_argument('--random_only', action='store_true',
                        help='Use only random instances (no TSPLIB files needed)')
    args = parser.parse_args()

    # Load instances
    if args.random_only or not os.path.exists(args.tsplib_dir):
        print("Using random TSP instances...")
        instances = tsp_utils.create_default_datasets()
    else:
        print(f"Loading instances from {args.tsplib_dir}...")
        instances = tsp_utils.load_tsplib_dir(args.tsplib_dir)
        if not instances:
            print("No TSPLIB files found, using random instances...")
            instances = tsp_utils.create_default_datasets()

    # Run benchmark
    run_benchmark(instances, args.log_dir)

    # Load and display best evolved function if available
    if os.path.exists(args.log_dir):
        print("\nBest evolved function from FunSearch:")
        load_best_evolved_function(args.log_dir)


if __name__ == '__main__':
    main()
