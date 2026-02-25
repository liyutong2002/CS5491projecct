"""
Benchmark script for evaluating FunSearch-evolved TSP heuristics.

Compares ALL baseline methods from the project requirements:
  1. Constructive Heuristics: Nearest Neighbor, Cheapest Insertion, Farthest Insertion
     (ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_baseline.py)
  2. 2-opt local search improvement
  3. Google OR-Tools
     (ref: https://developers.google.com/optimization/routing/tsp)
  4. LKH-3 (SOTA heuristic)
     (ref: http://webhotel4.ruc.dk/~keld/research/LKH-3/)
  5. Gurobi exact solver
     (ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_gurobi.py)
  6. FunSearch evolved priority (from logs)

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
    """Load the best evolved priority function from FunSearch logs."""
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
        print(f"Loaded best evolved function (score: {best_score:.2f})")
    return best_function


def run_with_timing(func, *args, **kwargs):
    """Run a function and return (result, elapsed_time)."""
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    return result, elapsed


def run_benchmark(instances: Dict[str, Dict], log_dir: str = None):
    """Run full benchmark comparison across all instances."""

    # Check which solvers are available
    has_ortools = False
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
        has_ortools = True
    except ImportError:
        pass

    has_lkh = False
    try:
        import lkh
        has_lkh = True
    except ImportError:
        pass

    has_gurobi = False
    try:
        import gurobipy
        has_gurobi = True
    except ImportError:
        pass

    print("\n" + "=" * 130)
    print("SOLVER AVAILABILITY")
    print("=" * 130)
    print(f"  Constructive (NN, Cheapest, Farthest): YES (built-in)")
    print(f"  2-opt local search:                    YES (built-in)")
    print(f"  Google OR-Tools:                       {'YES' if has_ortools else 'NO  (pip install ortools)'}")
    print(f"  LKH-3:                                 {'YES' if has_lkh else 'NO  (pip install lkh + LKH binary)'}")
    print(f"  Gurobi:                                {'YES' if has_gurobi else 'NO  (pip install gurobipy + license)'}")
    print(f"  Neural (AM/POMO):                      MANUAL (see README)")

    # Table header
    print("\n" + "=" * 130)
    header = f"{'Instance':<18} {'n':>4}"
    header += f" {'NN':>10} {'Cheap':>10} {'Far':>10}"
    header += f" {'NN+2opt':>10} {'Far+2opt':>10}"
    if has_ortools:
        header += f" {'OR-Tools':>10}"
    if has_lkh:
        header += f" {'LKH-3':>10}"
    if has_gurobi:
        header += f" {'Gurobi':>10}"
    header += f" {'Optimal':>10}"
    print(header)
    print("=" * 130)

    results_all = []

    for name, inst in sorted(instances.items(), key=lambda x: x[1]['num_cities']):
        n = inst['num_cities']
        dm = inst['dist_matrix']
        coords = inst['coords']
        optimal = inst.get('optimal_tour_length')

        row = {'name': name, 'n': n, 'optimal': optimal}

        # --- Constructive Heuristics ---
        # Nearest Neighbor (multi-start)
        best_nn_len = float('inf')
        for start in range(min(n, 10)):
            tour = tsp_utils.nearest_neighbor_tour(dm, start)
            length = tsp_utils.compute_tour_length(dm, tour)
            best_nn_len = min(best_nn_len, length)
        row['nn'] = best_nn_len

        # Cheapest Insertion
        ci_tour = tsp_utils.cheapest_insertion_tour(dm)
        row['cheapest'] = tsp_utils.compute_tour_length(dm, ci_tour)

        # Farthest Insertion
        fi_tour = tsp_utils.farthest_insertion_tour(dm)
        row['farthest'] = tsp_utils.compute_tour_length(dm, fi_tour)

        # --- 2-opt improvements ---
        # NN + 2-opt (multi-start)
        best_nn2opt_len = float('inf')
        for start in range(min(n, 5)):
            tour = tsp_utils.nearest_neighbor_tour(dm, start)
            tour = tsp_utils.two_opt_improve(dm, tour, max_iter=200)
            length = tsp_utils.compute_tour_length(dm, tour)
            best_nn2opt_len = min(best_nn2opt_len, length)
        row['nn_2opt'] = best_nn2opt_len

        # Farthest Insertion + 2-opt
        fi_2opt = tsp_utils.two_opt_improve(dm, fi_tour, max_iter=200)
        row['far_2opt'] = tsp_utils.compute_tour_length(dm, fi_2opt)

        # --- OR-Tools ---
        if has_ortools:
            ort_tour = tsp_utils.ortools_solve(dm, time_limit_seconds=min(30, max(5, n // 10)))
            if ort_tour:
                row['ortools'] = tsp_utils.compute_tour_length(dm, ort_tour)
            else:
                row['ortools'] = None

        # --- LKH-3 ---
        if has_lkh:
            lkh_tour = tsp_utils.lkh_solve(dm, coords=coords, runs=3)
            if lkh_tour:
                row['lkh'] = tsp_utils.compute_tour_length(dm, lkh_tour)
            else:
                row['lkh'] = None

        # --- Gurobi ---
        if has_gurobi and n <= 200:  # Gurobi is slow for large instances
            gurobi_tour = tsp_utils.gurobi_solve(dm, time_limit=60)
            if gurobi_tour:
                row['gurobi'] = tsp_utils.compute_tour_length(dm, gurobi_tour)
            else:
                row['gurobi'] = None

        # --- Print row ---
        line = f"{name:<18} {n:>4}"
        line += f" {row['nn']:>10.0f} {row['cheapest']:>10.0f} {row['farthest']:>10.0f}"
        line += f" {row['nn_2opt']:>10.0f} {row['far_2opt']:>10.0f}"
        if has_ortools:
            v = row.get('ortools')
            line += f" {v:>10.0f}" if v else f" {'N/A':>10}"
        if has_lkh:
            v = row.get('lkh')
            line += f" {v:>10.0f}" if v else f" {'N/A':>10}"
        if has_gurobi:
            v = row.get('gurobi')
            line += f" {v:>10.0f}" if v else f" {'N/A':>10}"
        line += f" {optimal:>10.0f}" if optimal else f" {'N/A':>10}"
        print(line)

        results_all.append(row)

    # --- Gap summary ---
    print("\n" + "=" * 130)
    print("OPTIMALITY GAP (%) = (heuristic - optimal) / optimal * 100")
    print("=" * 130)
    header2 = f"{'Instance':<18} {'n':>4}"
    header2 += f" {'NN':>8} {'Cheap':>8} {'Far':>8}"
    header2 += f" {'NN+2opt':>8} {'Far+2opt':>9}"
    if has_ortools:
        header2 += f" {'OR-Tools':>9}"
    if has_lkh:
        header2 += f" {'LKH-3':>8}"
    if has_gurobi:
        header2 += f" {'Gurobi':>8}"
    print(header2)
    print("-" * 130)

    gap_sums = {'nn': 0, 'cheapest': 0, 'farthest': 0, 'nn_2opt': 0, 'far_2opt': 0,
                'ortools': 0, 'lkh': 0, 'gurobi': 0}
    gap_counts = {k: 0 for k in gap_sums}

    for row in results_all:
        opt = row.get('optimal')
        if not opt:
            continue
        line = f"{row['name']:<18} {row['n']:>4}"
        for key in ['nn', 'cheapest', 'farthest', 'nn_2opt', 'far_2opt']:
            gap = (row[key] - opt) / opt * 100
            line += f" {gap:>8.1f}%"
            gap_sums[key] += gap
            gap_counts[key] += 1
        for key in ['ortools', 'lkh', 'gurobi']:
            if key == 'ortools' and not has_ortools:
                continue
            if key == 'lkh' and not has_lkh:
                continue
            if key == 'gurobi' and not has_gurobi:
                continue
            v = row.get(key)
            if v:
                gap = (v - opt) / opt * 100
                line += f" {gap:>8.1f}%"
                gap_sums[key] += gap
                gap_counts[key] += 1
            else:
                line += f" {'N/A':>9}"
        print(line)

    # Average gaps
    print("-" * 130)
    avg_line = f"{'AVERAGE':<18} {'':>4}"
    for key in ['nn', 'cheapest', 'farthest', 'nn_2opt', 'far_2opt']:
        if gap_counts[key] > 0:
            avg_line += f" {gap_sums[key]/gap_counts[key]:>8.1f}%"
        else:
            avg_line += f" {'N/A':>9}"
    for key in ['ortools', 'lkh', 'gurobi']:
        if key == 'ortools' and not has_ortools:
            continue
        if key == 'lkh' and not has_lkh:
            continue
        if key == 'gurobi' and not has_gurobi:
            continue
        if gap_counts[key] > 0:
            avg_line += f" {gap_sums[key]/gap_counts[key]:>8.1f}%"
        else:
            avg_line += f" {'N/A':>9}"
    print(avg_line)
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description='Benchmark TSP heuristics')
    parser.add_argument('--tsplib_dir', type=str, default='tsp_datasets/',
                        help='Directory containing .tsp files')
    parser.add_argument('--log_dir', type=str, default='logs/funsearch_tsp/',
                        help='FunSearch log directory')
    parser.add_argument('--random_only', action='store_true',
                        help='Use only random instances')
    args = parser.parse_args()

    if args.random_only or not os.path.exists(args.tsplib_dir):
        print("Using random TSP instances...")
        instances = tsp_utils.create_default_datasets()
    else:
        print(f"Loading instances from {args.tsplib_dir}...")
        instances = tsp_utils.load_tsplib_dir(args.tsplib_dir)
        if not instances:
            print("No TSPLIB files found, using random instances...")
            instances = tsp_utils.create_default_datasets()

    run_benchmark(instances, args.log_dir)

    if os.path.exists(args.log_dir):
        print("\nBest evolved function from FunSearch:")
        load_best_evolved_function(args.log_dir)


if __name__ == '__main__':
    main()
