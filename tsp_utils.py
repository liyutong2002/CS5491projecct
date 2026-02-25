"""
TSP utilities for FunSearch.
Provides:
  - TSPLIB instance parsing (.tsp files)
  - Random instance generation
  - Tour validation and cost computation
  - Dataset loading
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
    """Parse a TSPLIB format .tsp file and return a dict with instance data.

    Returns:
        dict with keys:
            'name': str, instance name
            'num_cities': int, number of cities
            'coords': np.ndarray of shape (num_cities, 2), city coordinates
            'dist_matrix': np.ndarray of shape (num_cities, num_cities), distance matrix
            'optimal_tour_length': float or None, known optimal if available
    """
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
        'name': name,
        'num_cities': dimension,
        'coords': coords,
        'dist_matrix': dist_matrix,
        'optimal_tour_length': None,  # can be filled from .opt.tour files
    }


def compute_distance_matrix(coords: np.ndarray, edge_weight_type: str = "EUC_2D") -> np.ndarray:
    """Compute distance matrix from coordinates.

    Args:
        coords: np.ndarray of shape (n, 2)
        edge_weight_type: TSPLIB edge weight type

    Returns:
        dist_matrix: np.ndarray of shape (n, n)
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if edge_weight_type == "EUC_2D":
                d = math.sqrt(dx * dx + dy * dy)
                d = int(d + 0.5)  # TSPLIB nint rounding
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
    """Compute the total tour length for a given tour.

    Args:
        dist_matrix: distance matrix (n x n)
        tour: list of city indices forming the tour

    Returns:
        total tour length (float)
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def validate_tour(tour: List[int], num_cities: int) -> bool:
    """Validate that a tour visits all cities exactly once."""
    if len(tour) != num_cities:
        return False
    if set(tour) != set(range(num_cities)):
        return False
    return True


# =============================================================================
# Random TSP Instance Generation
# =============================================================================

def generate_random_instance(
    num_cities: int,
    seed: int = 42,
    coord_range: float = 1000.0,
    name: str = "random"
) -> Dict[str, Any]:
    """Generate a random TSP instance with uniform coordinates.

    Args:
        num_cities: number of cities
        seed: random seed
        coord_range: coordinates in [0, coord_range]
        name: instance name

    Returns:
        dict with instance data (same format as parse_tsplib output)
    """
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, coord_range, size=(num_cities, 2))
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    return {
        'name': f'{name}_{num_cities}_seed{seed}',
        'num_cities': num_cities,
        'coords': coords,
        'dist_matrix': dist_matrix,
        'optimal_tour_length': None,
    }


# =============================================================================
# Baseline Heuristics (for comparison)
# =============================================================================

def nearest_neighbor_tour(dist_matrix: np.ndarray, start: int = 0) -> List[int]:
    """Classic nearest neighbor heuristic."""
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


def two_opt_improve(dist_matrix: np.ndarray, tour: List[int], max_iter: int = 1000) -> List[int]:
    """Apply 2-opt local search improvement to a tour."""
    n = len(tour)
    improved = True
    iteration = 0
    tour = list(tour)
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Compute improvement from reversing tour[i:j+1]
                old_cost = (dist_matrix[tour[i - 1]][tour[i]] +
                            dist_matrix[tour[j]][tour[(j + 1) % n]])
                new_cost = (dist_matrix[tour[i - 1]][tour[j]] +
                            dist_matrix[tour[i]][tour[(j + 1) % n]])
                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


# =============================================================================
# Dataset Loading
# =============================================================================

# Known optimal values for common TSPLIB instances
TSPLIB_OPTIMAL = {
    'eil51': 426,
    'berlin52': 7542,
    'st70': 675,
    'eil76': 538,
    'pr76': 108159,
    'kroA100': 21282,
    'kroB100': 22141,
    'kroC100': 20749,
    'kroD100': 21294,
    'kroE100': 22068,
    'rd100': 7910,
    'eil101': 629,
    'lin105': 14379,
    'pr107': 44303,
    'pr124': 59030,
    'bier127': 118282,
    'ch130': 6110,
    'pr136': 96772,
    'pr144': 58537,
    'ch150': 6528,
    'kroA150': 26524,
    'kroB150': 26130,
    'pr152': 73682,
    'u159': 42080,
    'rat195': 2323,
    'd198': 15780,
    'kroA200': 29368,
    'kroB200': 29437,
    'ts225': 126643,
    'tsp225': 3916,
    'pr226': 80369,
    'gil262': 2378,
    'pr264': 49135,
    'a280': 2579,
    'pr299': 48191,
    'lin318': 42029,
    'rd400': 15281,
    'fl417': 11861,
    'pr439': 107217,
    'pcb442': 50778,
    'd493': 35002,
    'att532': 27686,
    'ali535': 202339,
    'u574': 36905,
    'rat575': 6773,
    'p654': 34643,
    'd657': 48912,
    'u724': 41910,
    'rat783': 8806,
    'pr1002': 259045,
    'u1060': 224094,
    'vm1084': 239297,
    'd1291': 50801,
    'u1432': 152970,
    'fl1577': 22249,
    'd1655': 62128,
    'vm1748': 336556,
    'u1817': 57201,
    'rl1889': 316536,
    'u2152': 64253,
    'u2319': 234256,
    'pr2392': 378032,
    'pcb3038': 137694,
    'fl3795': 28772,
    'fnl4461': 182566,
    'rl5915': 565530,
    'rl5934': 556045,
}


def load_tsplib_instance(filepath: str) -> Dict[str, Any]:
    """Load a TSPLIB instance and attach known optimal value if available."""
    instance = parse_tsplib(filepath)
    name_lower = instance['name'].lower()
    if name_lower in TSPLIB_OPTIMAL:
        instance['optimal_tour_length'] = TSPLIB_OPTIMAL[name_lower]
    return instance


def load_tsplib_dir(dirpath: str) -> Dict[str, Dict[str, Any]]:
    """Load all .tsp files from a directory.

    Returns:
        dict mapping instance name -> instance data
    """
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


def create_default_datasets() -> Dict[str, Dict[str, Any]]:
    """Create a set of random TSP instances for testing when no TSPLIB files available.

    Returns:
        dict mapping instance name -> instance data
    """
    datasets = {}

    # Small instances for quick evaluation during search
    for n, seed in [(20, 42), (20, 123), (50, 42), (50, 123), (100, 42)]:
        instance = generate_random_instance(n, seed=seed)
        datasets[instance['name']] = instance

    return datasets


# Pre-built datasets (random instances for quick start)
datasets = create_default_datasets()


if __name__ == '__main__':
    # Demo: generate instances and compute baseline
    print("=" * 60)
    print("TSP Utils Demo")
    print("=" * 60)

    for name, inst in datasets.items():
        n = inst['num_cities']
        dm = inst['dist_matrix']

        nn_tour = nearest_neighbor_tour(dm, start=0)
        nn_length = compute_tour_length(dm, nn_tour)

        opt_tour = two_opt_improve(dm, nn_tour)
        opt_length = compute_tour_length(dm, opt_tour)

        print(f"\n{name}: {n} cities")
        print(f"  Nearest Neighbor tour length: {nn_length:.2f}")
        print(f"  After 2-opt improvement:      {opt_length:.2f}")
        print(f"  Improvement: {(nn_length - opt_length) / nn_length * 100:.1f}%")
