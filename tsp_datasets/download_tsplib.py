"""
Download TSPLIB instances and prepare datasets for FunSearch TSP.

Usage:
    python tsp_datasets/download_tsplib.py

This script downloads symmetric TSPLIB instances from the official source
and prepares them for use with FunSearch.
"""

import os
import sys
import urllib.request
import gzip
import shutil

# TSPLIB instances to download (name -> URL suffix)
# These are commonly used benchmark instances
TSPLIB_INSTANCES = {
    # Small (< 100 cities) - good for quick FunSearch iterations
    'eil51': 'eil51.tsp.gz',
    'berlin52': 'berlin52.tsp.gz',
    'st70': 'st70.tsp.gz',
    'eil76': 'eil76.tsp.gz',
    'pr76': 'pr76.tsp.gz',

    # Medium (100-200 cities)
    'kroA100': 'kroA100.tsp.gz',
    'kroB100': 'kroB100.tsp.gz',
    'rd100': 'rd100.tsp.gz',
    'eil101': 'eil101.tsp.gz',
    'lin105': 'lin105.tsp.gz',
    'ch130': 'ch130.tsp.gz',
    'ch150': 'ch150.tsp.gz',
    'kroA150': 'kroA150.tsp.gz',
    'rat195': 'rat195.tsp.gz',
    'kroA200': 'kroA200.tsp.gz',

    # Large (200+ cities) - for final evaluation
    'a280': 'a280.tsp.gz',
    'pr299': 'pr299.tsp.gz',
    'lin318': 'lin318.tsp.gz',
    'rd400': 'rd400.tsp.gz',
    'pcb442': 'pcb442.tsp.gz',
    'att532': 'att532.tsp.gz',
}

BASE_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"

# Optimal tour files
TSPLIB_OPT_TOURS = {
    'eil51': 'eil51.opt.tour.gz',
    'berlin52': 'berlin52.opt.tour.gz',
    'st70': 'st70.opt.tour.gz',
    'eil76': 'eil76.opt.tour.gz',
    'pr76': 'pr76.opt.tour.gz',
    'kroA100': 'kroA100.opt.tour.gz',
    'rd100': 'rd100.opt.tour.gz',
    'eil101': 'eil101.opt.tour.gz',
    'lin105': 'lin105.opt.tour.gz',
    'ch130': 'ch130.opt.tour.gz',
    'ch150': 'ch150.opt.tour.gz',
    'a280': 'a280.opt.tour.gz',
    'lin318': 'lin318.opt.tour.gz',
    'pcb442': 'pcb442.opt.tour.gz',
    'att532': 'att532.opt.tour.gz',
}


def download_file(url: str, filepath: str):
    """Download a file from URL to filepath."""
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def decompress_gz(gz_path: str, out_path: str):
    """Decompress a .gz file."""
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)


def download_tsplib(output_dir: str, instances: dict = None):
    """Download TSPLIB instances.

    Args:
        output_dir: directory to save .tsp files
        instances: dict of instance_name -> filename to download
                   If None, downloads all instances in TSPLIB_INSTANCES
    """
    os.makedirs(output_dir, exist_ok=True)

    if instances is None:
        instances = TSPLIB_INSTANCES

    print(f"Downloading {len(instances)} TSPLIB instances to {output_dir}/")
    print("=" * 60)

    success = 0
    failed = 0

    for name, filename in instances.items():
        tsp_path = os.path.join(output_dir, f"{name}.tsp")
        if os.path.exists(tsp_path):
            print(f"  {name}.tsp already exists, skipping.")
            success += 1
            continue

        url = BASE_URL + filename
        gz_path = os.path.join(output_dir, filename)

        if download_file(url, gz_path):
            try:
                decompress_gz(gz_path, tsp_path)
                print(f"  {name}.tsp downloaded successfully.")
                success += 1
            except Exception as e:
                print(f"  Failed to decompress {name}: {e}")
                failed += 1
        else:
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed.")
    return success, failed


def generate_synthetic_instances(output_dir: str):
    """Generate synthetic TSP instances in TSPLIB format for testing."""
    import random
    os.makedirs(output_dir, exist_ok=True)

    sizes = [20, 30, 50, 75, 100, 150, 200]

    print(f"\nGenerating {len(sizes)} synthetic instances...")
    for n in sizes:
        random.seed(42 + n)
        name = f"rand{n}"
        filepath = os.path.join(output_dir, f"{name}.tsp")

        with open(filepath, 'w') as f:
            f.write(f"NAME: {name}\n")
            f.write(f"COMMENT: Random {n}-city instance (seed={42+n})\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(1, n + 1):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                f.write(f"{i} {x:.4f} {y:.4f}\n")
            f.write("EOF\n")

        print(f"  Generated {name}.tsp ({n} cities)")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate synthetic instances (always works, no internet needed)
    generate_synthetic_instances(script_dir)

    # Try to download TSPLIB instances
    print("\n" + "=" * 60)
    print("Attempting to download TSPLIB instances...")
    print("(Requires internet connection)")
    print("=" * 60)

    # Download small instances first (most useful for FunSearch)
    small = {k: v for k, v in TSPLIB_INSTANCES.items()
             if k in ['eil51', 'berlin52', 'st70', 'eil76', 'pr76',
                       'kroA100', 'rd100', 'eil101']}

    try:
        download_tsplib(script_dir, small)
    except Exception as e:
        print(f"\nCould not download TSPLIB instances: {e}")
        print("You can manually download from: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/")
        print("Synthetic instances are available for testing.")
