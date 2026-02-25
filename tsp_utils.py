"""
TSP utilities for FunSearch.
Provides:
  - TSPLIB instance parsing (.tsp files)
  - Random instance generation
  - Tour validation and cost computation
  - Dataset loading
  - Baseline algorithms:
    * Constructive Heuristics: nearest neighbor, cheapest insertion, farthest insertion
      (ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_baseline.py)
    * 2-opt local search
    * Google OR-Tools solver
      (ref: https://developers.google.com/optimization/routing/tsp)
    * LKH-3 via lkh Python wrapper
      (ref: http://webhotel4.ruc.dk/~keld/research/LKH-3/, https://github.com/ben-hudson/pylkh)
    * Gurobi exact solver (optional, requires license)
      (ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_gurobi.py)
"""

import os
import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


# =============================================================================
# TSPLIB Parser
# =============================================================================

def parse_tsplib(filepath: str) -> Dict[str, Any]:
    name = ""
    dimension = 0
    edge_weight_type = "EUC_2D"
    coords = []
    reading_coords = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NAME"):
                name = line.split(":")[1].strip() if ":" in line else line.split()[-1]
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip() if ":" in line else line.split()[-1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip() if ":" in line else line.split()[-1]
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line in ("EOF", "DISPLAY_DATA_SECTION", "EDGE_WEIGHT_SECTION"):
                reading_coords = False
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    coords = np.array(coords[:dimension])
    dist_matrix = compute_distance_matrix(coords, edge_weight_type)
    return {
        'name': name, 'num_cities': dimension, 'coords': coords,
        'dist_matrix': dist_matrix, 'optimal_tour_length': None,
    }


def compute_distance_matrix(coords: np.ndarray, edge_weight_type: str = "EUC_2D") -> np.ndarray:
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if edge_weight_type == "EUC_2D":
                d = math.sqrt(dx * dx + dy * dy)
                d = int(d + 0.5)
            elif edge_weight_type == "CEIL_2D":
                d = math.ceil(math.sqrt(dx * dx + dy * dy))
            elif edge_weight_type == "ATT":
                r = math.sqrt((dx * dx + dy * dy) / 10.0)
                t = int(r + 0.5)
                d = t + 1 if t < r else t
            else:
                d = math.sqrt(dx * dx + dy * dy)
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def compute_tour_length(dist_matrix: np.ndarray, tour: List[int]) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def validate_tour(tour: List[int], num_cities: int) -> bool:
    if len(tour) != num_cities:
        return False
    if set(tour) != set(range(num_cities)):
        return False
    return True


# =============================================================================
# Random TSP Instance Generation
# =============================================================================

def generate_random_instance(num_cities, seed=42, coord_range=1000.0, name="random"):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, coord_range, size=(num_cities, 2))
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return {
        'name': f'{name}_{num_cities}_seed{seed}', 'num_cities': num_cities,
        'coords': coords, 'dist_matrix': dist_matrix, 'optimal_tour_length': None,
    }


# =============================================================================
# Constructive Heuristics
# Ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/
#      problems/tsp/tsp_baseline.py
# =============================================================================

def nearest_neighbor_tour(dist_matrix: np.ndarray, start: int = 0) -> List[int]:
    """Nearest Neighbor constructive heuristic. O(n^2)."""
    n = len(dist_matrix)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        current = tour[-1]
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist_matrix[current][j] < best_dist:
                best_dist = dist_matrix[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True
    return tour


def cheapest_insertion_tour(dist_matrix: np.ndarray) -> List[int]:
    """Cheapest Insertion heuristic. O(n^3).
    Ref: attention-learn-to-route/problems/tsp/tsp_baseline.py"""
    n = len(dist_matrix)
    if n <= 3:
        return list(range(n))

    in_tour = [False] * n
    tour = [0]
    in_tour[0] = True

    farthest = max(range(n), key=lambda j: dist_matrix[0][j] if j != 0 else -1)
    tour.append(farthest)
    in_tour[farthest] = True

    best_third = -1
    best_val = -1
    for j in range(n):
        if in_tour[j]:
            continue
        val = min(dist_matrix[tour[0]][j], dist_matrix[tour[1]][j])
        if val > best_val:
            best_val = val
            best_third = j
    tour.append(best_third)
    in_tour[best_third] = True

    for _ in range(n - 3):
        best_cost_increase = float('inf')
        best_city = -1
        best_position = -1
        for city in range(n):
            if in_tour[city]:
                continue
            for pos in range(len(tour)):
                i = tour[pos]
                j = tour[(pos + 1) % len(tour)]
                cost_increase = (dist_matrix[i][city] + dist_matrix[city][j]
                                 - dist_matrix[i][j])
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_city = city
                    best_position = pos + 1
        tour.insert(best_position, best_city)
        in_tour[best_city] = True
    return tour


def farthest_insertion_tour(dist_matrix: np.ndarray) -> List[int]:
    """Farthest Insertion heuristic. O(n^2)~O(n^3).
    Ref: attention-learn-to-route/problems/tsp/tsp_baseline.py"""
    n = len(dist_matrix)
    if n <= 3:
        return list(range(n))

    in_tour = [False] * n
    best_pair = (0, 1)
    best_dist = dist_matrix[0][1]
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] > best_dist:
                best_dist = dist_matrix[i][j]
                best_pair = (i, j)

    tour = [best_pair[0], best_pair[1]]
    in_tour[best_pair[0]] = True
    in_tour[best_pair[1]] = True

    min_dist_to_tour = np.full(n, float('inf'))
    for j in range(n):
        if not in_tour[j]:
            min_dist_to_tour[j] = min(dist_matrix[tour[0]][j], dist_matrix[tour[1]][j])

    for _ in range(n - 2):
        farthest_city = -1
        farthest_dist = -1
        for j in range(n):
            if not in_tour[j] and min_dist_to_tour[j] > farthest_dist:
                farthest_dist = min_dist_to_tour[j]
                farthest_city = j
        if farthest_city == -1:
            break

        best_pos = 0
        best_cost = float('inf')
        for pos in range(len(tour)):
            i = tour[pos]
            j = tour[(pos + 1) % len(tour)]
            cost = dist_matrix[i][farthest_city] + dist_matrix[farthest_city][j] - dist_matrix[i][j]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos + 1

        tour.insert(best_pos, farthest_city)
        in_tour[farthest_city] = True

        for j in range(n):
            if not in_tour[j]:
                min_dist_to_tour[j] = min(min_dist_to_tour[j], dist_matrix[farthest_city][j])
    return tour


# =============================================================================
# Local Search: 2-opt
# =============================================================================

def two_opt_improve(dist_matrix: np.ndarray, tour: List[int], max_iter: int = 1000) -> List[int]:
    n = len(tour)
    improved = True
    iteration = 0
    tour = list(tour)
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                old_cost = (dist_matrix[tour[i - 1]][tour[i]] +
                            dist_matrix[tour[j]][tour[(j + 1) % n]])
                new_cost = (dist_matrix[tour[i - 1]][tour[j]] +
                            dist_matrix[tour[i]][tour[(j + 1) % n]])
                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


# =============================================================================
# Google OR-Tools Solver
# Ref: https://developers.google.com/optimization/routing/tsp
# =============================================================================

def ortools_solve(dist_matrix: np.ndarray, time_limit_seconds: int = 30) -> Optional[List[int]]:
    """Solve TSP using Google OR-Tools routing solver."""
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        print("  [OR-Tools] Not installed. Run: pip install ortools")
        return None

    n = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    int_dist = (dist_matrix * 1000).astype(int)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_dist[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return tour
    return None


# =============================================================================
# LKH-3 Solver
# Ref: http://webhotel4.ruc.dk/~keld/research/LKH-3/
#      https://github.com/ben-hudson/pylkh
# Requires: pip install lkh AND LKH-3 binary on PATH
# =============================================================================

# def lkh_solve(dist_matrix: np.ndarray, coords: np.ndarray = None,
#               lkh_path: str = "LKH", runs: int = 5) -> Optional[List[int]]:
# def lkh_solve(dist_matrix: np.ndarray, coords: np.ndarray = None,
#               lkh_path: str = r"C:\Users\liyutong\Desktop\LKH-3.0.13\LKH.exe", runs: int = 5) -> Optional[List[int]]:
#     """Solve TSP using LKH-3 (SOTA heuristic, <1% gap)."""
#     try:
#         import lkh
#     except ImportError:
#         print("  [LKH] Not installed. Run: pip install lkh")
#         return None

#     n = len(dist_matrix)
#     try:
#         problem_str = f"NAME: temp\nTYPE: TSP\nDIMENSION: {n}\n"
#         if coords is not None:
#             problem_str += "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
#             for i in range(n):
#                 problem_str += f"{i+1} {coords[i][0]:.6f} {coords[i][1]:.6f}\n"
#         else:
#             problem_str += "EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
#             problem_str += "EDGE_WEIGHT_SECTION\n"
#             for i in range(n):
#                 row = " ".join(str(int(dist_matrix[i][j])) for j in range(n))
#                 problem_str += row + "\n"
#         problem_str += "EOF\n"

#         tour = lkh.solve(lkh_path, problem=problem_str, runs=runs)
#         if tour and len(tour) > 0:
#             return [x - 1 for x in tour[0]]
#     except FileNotFoundError:
#         print(f"  [LKH] Binary not found. Download: http://webhotel4.ruc.dk/~keld/research/LKH-3/")
#     except Exception as e:
#         print(f"  [LKH] Error: {e}")
#     return None
def lkh_solve(dist_matrix: np.ndarray, coords: np.ndarray = None,
              lkh_path: str = r"C:\Users\liyutong\Desktop\LKH-3.0.13\LKH.exe",
              runs: int = 5) -> Optional[List[int]]:
    """Solve TSP using LKH-3 (SOTA heuristic, <1% gap). Direct call without lkh package."""
    import subprocess
    import tempfile

    n = len(dist_matrix)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write TSPLIB problem file
            prob_file = os.path.join(tmpdir, "problem.tsp")
            with open(prob_file, 'w') as f:
                f.write(f"NAME: temp\n")
                f.write(f"TYPE: TSP\n")
                f.write(f"DIMENSION: {n}\n")
                if coords is not None:
                    f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
                    f.write(f"NODE_COORD_SECTION\n")
                    for i in range(n):
                        f.write(f"{i+1} {coords[i][0]:.6f} {coords[i][1]:.6f}\n")
                else:
                    f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
                    f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
                    f.write(f"EDGE_WEIGHT_SECTION\n")
                    for i in range(n):
                        row = " ".join(str(int(dist_matrix[i][j])) for j in range(n))
                        f.write(row + "\n")
                f.write("EOF\n")

            # Write LKH parameter file
            par_file = os.path.join(tmpdir, "params.par")
            tour_file = os.path.join(tmpdir, "output.tour")
            with open(par_file, 'w') as f:
                f.write(f"PROBLEM_FILE = {prob_file}\n")
                f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
                f.write(f"RUNS = {runs}\n")
                f.write(f"SEED = 42\n")

            # Run LKH
            result = subprocess.run(
                [lkh_path, par_file],
                capture_output=True, text=True, timeout=120
            )

            # Parse output tour
            if os.path.exists(tour_file):
                tour = []
                reading_tour = False
                with open(tour_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line == "TOUR_SECTION":
                            reading_tour = True
                            continue
                        if line == "-1" or line == "EOF":
                            reading_tour = False
                            continue
                        if reading_tour:
                            city = int(line) - 1  # Convert to 0-indexed
                            if city >= 0:
                                tour.append(city)
                if len(tour) == n:
                    return tour

    except subprocess.TimeoutExpired:
        print(f"  [LKH] Timeout after 120 seconds")
    except FileNotFoundError:
        print(f"  [LKH] Binary not found at '{lkh_path}'")
    except Exception as e:
        print(f"  [LKH] Error: {e}")
    return None

# =============================================================================
# Gurobi Exact Solver (optional)
# Ref: https://github.com/wouterkool/attention-learn-to-route/blob/master/
#      problems/tsp/tsp_gurobi.py
# =============================================================================

def gurobi_solve(dist_matrix: np.ndarray, time_limit: float = 300) -> Optional[List[int]]:
    """Solve TSP exactly using Gurobi (requires license)."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  [Gurobi] Not installed. Run: pip install gurobipy (+ license)")
        return None

    n = len(dist_matrix)
    try:
        model = gp.Model("tsp")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY)

        u = {}
        for i in range(1, n):
            u[i] = model.addVar(lb=1, ub=n - 1, vtype=GRB.CONTINUOUS)

        model.update()
        model.setObjective(
            gp.quicksum(dist_matrix[i][j] * x[i, j]
                        for i in range(n) for j in range(n) if i != j),
            GRB.MINIMIZE)

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if i != j) == 1)
        for j in range(n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

        model.optimize()
        if model.SolCount > 0:
            tour = [0]
            current = 0
            for _ in range(n - 1):
                for j in range(n):
                    if j != current and x[current, j].X > 0.5:
                        tour.append(j)
                        current = j
                        break
            return tour
    except Exception as e:
        print(f"  [Gurobi] Error: {e}")
    return None


# =============================================================================
# Dataset Loading
# =============================================================================

TSPLIB_OPTIMAL = {
    'eil51': 426, 'berlin52': 7542, 'st70': 675, 'eil76': 538,
    'pr76': 108159, 'kroA100': 21282, 'kroB100': 22141, 'kroC100': 20749,
    'kroD100': 21294, 'kroE100': 22068, 'rd100': 7910, 'eil101': 629,
    'lin105': 14379, 'pr107': 44303, 'pr124': 59030, 'bier127': 118282,
    'ch130': 6110, 'pr136': 96772, 'pr144': 58537, 'ch150': 6528,
    'kroA150': 26524, 'kroB150': 26130, 'pr152': 73682, 'u159': 42080,
    'rat195': 2323, 'd198': 15780, 'kroA200': 29368, 'kroB200': 29437,
    'a280': 2579, 'pr299': 48191, 'lin318': 42029, 'rd400': 15281,
    'pcb442': 50778, 'att532': 27686, 'u574': 36905, 'rat575': 6773,
}


def load_tsplib_instance(filepath):
    instance = parse_tsplib(filepath)
    name_lower = instance['name'].lower()
    if name_lower in TSPLIB_OPTIMAL:
        instance['optimal_tour_length'] = TSPLIB_OPTIMAL[name_lower]
    return instance


def load_tsplib_dir(dirpath):
    datasets = {}
    for fname in sorted(os.listdir(dirpath)):
        if fname.endswith('.tsp'):
            filepath = os.path.join(dirpath, fname)
            try:
                instance = load_tsplib_instance(filepath)
                datasets[instance['name']] = instance
            except Exception as e:
                print(f"Warning: Failed to load {fname}: {e}")
    return datasets


def create_default_datasets():
    datasets = {}
    for n, seed in [(20, 42), (20, 123), (50, 42), (50, 123), (100, 42)]:
        instance = generate_random_instance(n, seed=seed)
        datasets[instance['name']] = instance
    return datasets


datasets = create_default_datasets()


if __name__ == '__main__':
    print("=" * 70)
    print("TSP Utils Demo - All Baselines")
    print("=" * 70)
    inst = generate_random_instance(50, seed=42)
    n = inst['num_cities']
    dm = inst['dist_matrix']
    coords = inst['coords']
    print(f"\nInstance: {inst['name']} ({n} cities)\n")

    nn_tour = nearest_neighbor_tour(dm, start=0)
    print(f"  Nearest Neighbor:       {compute_tour_length(dm, nn_tour):.2f}")

    ci_tour = cheapest_insertion_tour(dm)
    print(f"  Cheapest Insertion:     {compute_tour_length(dm, ci_tour):.2f}")

    fi_tour = farthest_insertion_tour(dm)
    print(f"  Farthest Insertion:     {compute_tour_length(dm, fi_tour):.2f}")

    nn_2opt = two_opt_improve(dm, nn_tour, max_iter=200)
    print(f"  NN + 2-opt:             {compute_tour_length(dm, nn_2opt):.2f}")

    fi_2opt = two_opt_improve(dm, fi_tour, max_iter=200)
    print(f"  Farthest Ins. + 2-opt:  {compute_tour_length(dm, fi_2opt):.2f}")

    ort_tour = ortools_solve(dm, time_limit_seconds=10)
    if ort_tour:
        print(f"  Google OR-Tools:        {compute_tour_length(dm, ort_tour):.2f}")
    else:
        print(f"  Google OR-Tools:        N/A")

    lkh_tour = lkh_solve(dm, coords=coords, runs=3)
    if lkh_tour:
        print(f"  LKH-3:                  {compute_tour_length(dm, lkh_tour):.2f}")
    else:
        print(f"  LKH-3:                  N/A (need binary)")

    gurobi_tour = gurobi_solve(dm, time_limit=30)
    if gurobi_tour:
        print(f"  Gurobi (exact):         {compute_tour_length(dm, gurobi_tour):.2f}")
    else:
        print(f"  Gurobi:                 N/A")
