"""
Benchmark script for evaluating FunSearch-evolved TSP heuristics.

Compares ALL baseline methods with TRAIN/TEST split:
  Training set: small instances (n <= 50) — FunSearch evolves on these
  Test set: larger instances (n > 50) — evaluate generalization

Baselines:
  1. Constructive Heuristics: Nearest Neighbor, Cheapest Insertion, Farthest Insertion
     (ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_baseline.py)
  2. Google OR-Tools
     (ref: https://developers.google.com/optimization/routing/tsp)
  3. LKH-3 (SOTA heuristic)
     (ref: http://webhotel4.ruc.dk/~keld/research/LKH-3/)
  4. Gurobi exact solver
     (ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_gurobi.py)
  5. FunSearch evolved priority (from logs)

Usage:
    python benchmark_tsp.py --tsplib_dir tsp_datasets/ --log_dir logs/funsearch_tsp/
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple

import tsp_utils


def load_best_evolved_function(log_dir: str):
    """Load the best evolved priority function from FunSearch logs."""
    samples_dir = os.path.join(log_dir, 'samples')
    if not os.path.exists(samples_dir):
        return None, None
    best_score = -float('inf')
    best_function = None
    for fname in os.listdir(samples_dir):
        if fname.endswith('.json'):
            with open(os.path.join(samples_dir, fname), 'r') as f:
                data = json.load(f)
                if data.get('score') is not None and data['score'] > best_score:
                    best_score = data['score']
                    best_function = data.get('function')
    return best_function, best_score


def funsearch_solve(dist_matrix, coords, evolved_function_body):
    """Run the evolved priority function on an instance."""
    if evolved_function_body is None:
        return None

    n = len(dist_matrix)
    # Build the full program
    program = f"""
import numpy as np
import math

def priority(current_city, unvisited_cities, dist_matrix, coords, visited, step, total_cities):
{evolved_function_body}

def construct_tour(dist_matrix, coords):
    n = len(dist_matrix)
    visited = np.zeros(n, dtype=bool)
    current_city = 0
    tour = [current_city]
    visited[current_city] = True
    for step in range(n - 1):
        unvisited = np.where(~visited)[0]
        scores = priority(current_city, unvisited, dist_matrix, coords, visited, step, n)
        best_idx = np.argmax(scores)
        next_city = unvisited[best_idx]
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city
    return tour
"""
    try:
        namespace = {}
        exec(program, namespace)
        # Try multiple starts
        best_tour = None
        best_length = float('inf')
        for start in range(min(n, 5)):
            perm = list(range(n))
            perm[0], perm[start] = perm[start], perm[0]
            remapped_dm = dist_matrix[np.ix_(perm, perm)]
            remapped_coords = coords[perm]
            tour = namespace['construct_tour'](remapped_dm, remapped_coords)
            length = tsp_utils.compute_tour_length(remapped_dm, tour)
            if length < best_length:
                best_length = length
                best_tour = tour
        return best_length
    except Exception as e:
        return None


def run_benchmark_on_split(instances: Dict, split_name: str, evolved_func=None):
    """Run benchmark on a set of instances."""

    # Check solver availability
    has_ortools = False
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
        has_ortools = True
    except ImportError:
        pass

    has_lkh = os.path.exists(r"C:\Users\liyutong\Desktop\LKH-3.0.13\LKH.exe") or \
              os.path.exists("LKH") or os.path.exists("/usr/local/bin/LKH")

    has_gurobi = False
    try:
        import gurobipy
        has_gurobi = True
    except ImportError:
        pass

    if not instances:
        print(f"  No instances in {split_name}.")
        return

    # Header
    print(f"\n{'='*130}")
    print(f"  {split_name}")
    print(f"{'='*130}")
    header = f"{'Instance':<15} {'n':>4}"
    header += f" {'NN':>9} {'Cheap':>9} {'Far':>9}"
    if has_ortools:
        header += f" {'OR-Tools':>9}"
    if has_lkh:
        header += f" {'LKH-3':>9}"
    if has_gurobi:
        header += f" {'Gurobi':>9}"
    if evolved_func:
        header += f" {'FunSearch':>9}"
    header += f" {'Best*':>9}"
    print(header)
    print("-" * 130)

    results = []

    for name, inst in sorted(instances.items(), key=lambda x: x[1]['num_cities']):
        n = inst['num_cities']
        dm = inst['dist_matrix']
        coords = inst['coords']
        optimal = inst.get('optimal_tour_length')

        row = {'name': name, 'n': n, 'optimal': optimal}

        # --- Constructive Heuristics ---
        best_nn = float('inf')
        for start in range(min(n, 10)):
            tour = tsp_utils.nearest_neighbor_tour(dm, start)
            best_nn = min(best_nn, tsp_utils.compute_tour_length(dm, tour))
        row['nn'] = best_nn

        ci_tour = tsp_utils.cheapest_insertion_tour(dm)
        row['cheapest'] = tsp_utils.compute_tour_length(dm, ci_tour)

        fi_tour = tsp_utils.farthest_insertion_tour(dm)
        row['farthest'] = tsp_utils.compute_tour_length(dm, fi_tour)

        # --- OR-Tools ---
        if has_ortools:
            ort_tour = tsp_utils.ortools_solve(dm, time_limit_seconds=min(30, max(5, n // 10)))
            row['ortools'] = tsp_utils.compute_tour_length(dm, ort_tour) if ort_tour else None

        # --- LKH-3 ---
        if has_lkh:
            lkh_tour = tsp_utils.lkh_solve(dm, coords=coords, runs=3)
            row['lkh'] = tsp_utils.compute_tour_length(dm, lkh_tour) if lkh_tour else None

        # --- Gurobi ---
        if has_gurobi and n <= 200:
            gurobi_tour = tsp_utils.gurobi_solve(dm, time_limit=60)
            row['gurobi'] = tsp_utils.compute_tour_length(dm, gurobi_tour) if gurobi_tour else None

        # --- FunSearch ---
        if evolved_func:
            fs_len = funsearch_solve(dm, coords, evolved_func)
            row['funsearch'] = fs_len

        # --- Best known (min of LKH, Gurobi, or TSPLIB optimal) ---
        best_known = optimal
        for key in ['lkh', 'gurobi']:
            v = row.get(key)
            if v is not None:
                if best_known is None or v < best_known:
                    best_known = v
        row['best_known'] = best_known

        # --- Print row ---
        line = f"{name:<15} {n:>4}"
        line += f" {row['nn']:>9.0f} {row['cheapest']:>9.0f} {row['farthest']:>9.0f}"
        if has_ortools:
            v = row.get('ortools')
            line += f" {v:>9.0f}" if v else f" {'N/A':>9}"
        if has_lkh:
            v = row.get('lkh')
            line += f" {v:>9.0f}" if v else f" {'N/A':>9}"
        if has_gurobi:
            v = row.get('gurobi')
            line += f" {v:>9.0f}" if v else f" {'N/A':>9}"
        if evolved_func:
            v = row.get('funsearch')
            line += f" {v:>9.0f}" if v else f" {'N/A':>9}"
        line += f" {best_known:>9.0f}" if best_known else f" {'N/A':>9}"
        print(line)
        results.append(row)

    # --- Gap table ---
    has_any_best = any(r.get('best_known') for r in results)
    if has_any_best:
        print(f"\n  Optimality Gap (%) = (method - best_known) / best_known * 100")
        print(f"  " + "-" * 120)
        header2 = f"  {'Instance':<15} {'n':>4}"
        header2 += f" {'NN':>8} {'Cheap':>8} {'Far':>8}"
        if has_ortools:
            header2 += f" {'OR-Tools':>9}"
        if evolved_func:
            header2 += f" {'FunSearch':>9}"
        print(header2)
        print(f"  " + "-" * 120)

        gap_sums = {}
        gap_counts = {}

        for row in results:
            bk = row.get('best_known')
            if not bk:
                continue
            line = f"  {row['name']:<15} {row['n']:>4}"
            for key in ['nn', 'cheapest', 'farthest']:
                gap = (row[key] - bk) / bk * 100
                line += f" {gap:>7.1f}%"
                gap_sums[key] = gap_sums.get(key, 0) + gap
                gap_counts[key] = gap_counts.get(key, 0) + 1
            if has_ortools:
                v = row.get('ortools')
                if v:
                    gap = (v - bk) / bk * 100
                    line += f" {gap:>8.1f}%"
                    gap_sums['ortools'] = gap_sums.get('ortools', 0) + gap
                    gap_counts['ortools'] = gap_counts.get('ortools', 0) + 1
                else:
                    line += f" {'N/A':>9}"
            if evolved_func:
                v = row.get('funsearch')
                if v:
                    gap = (v - bk) / bk * 100
                    line += f" {gap:>8.1f}%"
                    gap_sums['funsearch'] = gap_sums.get('funsearch', 0) + gap
                    gap_counts['funsearch'] = gap_counts.get('funsearch', 0) + 1
                else:
                    line += f" {'N/A':>9}"
            print(line)

        # Average
        print(f"  " + "-" * 120)
        avg_line = f"  {'AVERAGE':<15} {'':>4}"
        for key in ['nn', 'cheapest', 'farthest']:
            if gap_counts.get(key, 0) > 0:
                avg_line += f" {gap_sums[key]/gap_counts[key]:>7.1f}%"
            else:
                avg_line += f" {'N/A':>8}"
        if has_ortools and gap_counts.get('ortools', 0) > 0:
            avg_line += f" {gap_sums['ortools']/gap_counts['ortools']:>8.1f}%"
        if evolved_func and gap_counts.get('funsearch', 0) > 0:
            avg_line += f" {gap_sums['funsearch']/gap_counts['funsearch']:>8.1f}%"
        print(avg_line)

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark TSP heuristics')
    parser.add_argument('--tsplib_dir', type=str, default='tsp_datasets/')
    parser.add_argument('--log_dir', type=str, default='logs/funsearch_tsp/')
    parser.add_argument('--random_only', action='store_true')
    parser.add_argument('--train_max_cities', type=int, default=50,
                        help='Max cities for training set (default: 50)')
    args = parser.parse_args()

    # Load instances
    if args.random_only or not os.path.exists(args.tsplib_dir):
        all_instances = tsp_utils.create_default_datasets()
    else:
        all_instances = tsp_utils.load_tsplib_dir(args.tsplib_dir)
        if not all_instances:
            all_instances = tsp_utils.create_default_datasets()

    # Split into train/test
    train_instances = {
        k: v for k, v in all_instances.items()
        if v['num_cities'] <= args.train_max_cities
    }
    test_instances = {
        k: v for k, v in all_instances.items()
        if v['num_cities'] > args.train_max_cities
    }

    # Check for evolved function
    evolved_func = None
    evolved_score = None
    if os.path.exists(args.log_dir):
        evolved_func, evolved_score = load_best_evolved_function(args.log_dir)
        if evolved_func:
            print(f"Loaded best evolved function (score: {evolved_score:.2f})")

    # Solver availability
    print("\n" + "=" * 130)
    print("SOLVER AVAILABILITY")
    print("=" * 130)
    print(f"  Constructive (NN, Cheapest, Farthest):  YES")
    try:
        from ortools.constraint_solver import pywrapcp
        print(f"  Google OR-Tools:                        YES")
    except ImportError:
        print(f"  Google OR-Tools:                        NO")
    print(f"  LKH-3:                                  {'YES' if os.path.exists(r'C:\\Users\\liyutong\\Desktop\\LKH-3.0.13\\LKH.exe') else 'CHECK PATH'}")
    try:
        import gurobipy
        print(f"  Gurobi:                                 YES")
    except ImportError:
        print(f"  Gurobi:                                 NO")
    print(f"  FunSearch evolved:                      {'YES' if evolved_func else 'NO (run funsearch_tsp_llm_api.py first)'}")
    print(f"  Neural (AM/POMO):                       Reference from papers")

    # Run benchmarks
    print(f"\n{'#'*130}")
    print(f"  TRAINING SET (n <= {args.train_max_cities}) — FunSearch evolved on these")
    print(f"{'#'*130}")
    run_benchmark_on_split(train_instances, "TRAINING SET", evolved_func)

    print(f"\n{'#'*130}")
    print(f"  TEST SET (n > {args.train_max_cities}) — Generalization evaluation")
    print(f"{'#'*130}")
    run_benchmark_on_split(test_instances, "TEST SET", evolved_func)

    # Reference: AM/POMO results from papers
    print(f"\n{'='*130}")
    print("REFERENCE: Neural Solver Results (from original papers)")
    print("=" * 130)
    print("  Attention Model (AM):  ~3.5% gap on TSP100 (Kool et al., 2019)")
    print("  POMO:                  ~1.5% gap on TSP100 (Kwon et al., 2020)")
    print("  Source: https://github.com/wouterkool/attention-learn-to-route")
    print("  Source: https://github.com/yd-kwon/POMO")
    print("=" * 130)


if __name__ == '__main__':
    main()
