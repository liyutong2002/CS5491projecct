"""
FunSearch for Traveling Salesman Problem (TSP).

This script adapts the FunSearch framework to evolve heuristic priority functions
for constructive TSP solving. The LLM-evolved function is a 'priority' function
that scores unvisited cities at each step of a greedy constructive heuristic.

Usage:
    python funsearch_tsp_llm_api.py

Configuration:
    - Set your LLM API endpoint and key below
    - Adjust dataset sizes and evaluation timeout as needed
    - Set global_max_sample_num to control total LLM calls
"""

import json
import multiprocessing
from typing import Collection, Any
import http.client
import os

from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
import tsp_utils


# =============================================================================
# Trim preface utility (for instruct-based LLMs)
# =============================================================================

def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions/symbols/'def' declaration before the function body.

    Since the LLM used in this file is not a pure code completion LLM, this trim function is required.
    It removes description above the function's signature and the signature itself.
    The indent of the code is preserved.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        if line.strip().startswith('def '):
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


# =============================================================================
# LLM API class
# =============================================================================

class LLMAPI(sampler.LLM):
    """Language model that predicts continuation of provided source code.

    Modify the _draw_sample method to use your preferred LLM API.
    Supported: OpenAI API, Anthropic API, local vLLM server, etc.
    """

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)
        additional_prompt = (
            'Complete a different and more complex Python function for TSP city selection priority. '
            'Be creative and consider factors like: distance, angle, position relative to centroid, '
            'nearest neighbor density, convex hull, and combinations of these features. '
            'You can use numpy operations. The function should return a numpy array of scores. '
            'Only output the Python code, no descriptions.'
        )
        self._additional_prompt = additional_prompt
        self._trim = trim

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        """Call the LLM API.

        IMPORTANT: Replace the API endpoint, model, and authorization key with your own.
        Below are example configurations for different providers.
        """
        prompt = '\n'.join([content, self._additional_prompt])

        while True:
            try:
                conn = http.client.HTTPSConnection("api.deepseek.com")
                payload = json.dumps({
                    "max_tokens": 1024,
                    "model": "deepseek-coder",
                    "temperature": 0.8,
                    "messages": [
                        {"role": "system", "content": "You are an expert algorithm designer. Write creative Python functions using numpy."},
                        {"role": "user", "content": prompt}
                    ]
                })
                headers = {
                    'Authorization': 'Bearer sk-085c14782c7241c3a4f677973393a4f0',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = json.loads(res.read().decode("utf-8"))
                response = data['choices'][0]['message']['content']

                # =============================================================
                # Option 2: Anthropic API (uncomment to use)
                # =============================================================
                # conn = http.client.HTTPSConnection("api.anthropic.com")
                # payload = json.dumps({
                #     "max_tokens": 1024,
                #     "model": "claude-sonnet-4-20250514",
                #     "messages": [
                #         {"role": "user", "content": prompt}
                #     ]
                # })
                # headers = {
                #     'x-api-key': 'sk-ant-YOUR_API_KEY_HERE',
                #     'anthropic-version': '2023-06-01',
                #     'Content-Type': 'application/json'
                # }
                # conn.request("POST", "/v1/messages", payload, headers)
                # res = conn.getresponse()
                # data = res.read().decode("utf-8")
                # data = json.loads(data)
                # response = data['content'][0]['text']

                # =============================================================
                # Option 3: Local vLLM / Ollama server (uncomment to use)
                # =============================================================
                # conn = http.client.HTTPConnection("localhost", 8000)
                # payload = json.dumps({
                #     "max_tokens": 1024,
                #     "model": "deepseek-coder-6.7b-instruct",
                #     "messages": [{"role": "user", "content": prompt}]
                # })
                # headers = {'Content-Type': 'application/json'}
                # conn.request("POST", "/v1/chat/completions", payload, headers)
                # res = conn.getresponse()
                # data = res.read().decode("utf-8")
                # data = json.loads(data)
                # response = data['choices'][0]['message']['content']

                # Trim function
                if self._trim:
                    response = _trim_preface_of_body(response)
                return response

            except Exception as e:
                print(f"LLM API error: {e}, retrying...")
                import time
                time.sleep(1)
                continue


# =============================================================================
# Sandbox for TSP evaluation
# =============================================================================

class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated TSP heuristic code.

    Executes the generated priority function within a constructive TSP solver
    and returns the negative tour length as the score (FunSearch maximizes).
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            test_input: str,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded."""
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve,
                  dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            results = None, False
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()
            else:
                results = None, False

        if self._verbose:
            print(f'================= Evaluated Program =================')
            program_: code_manipulation.Program = code_manipulation.text_to_program(text=program)
            function_: code_manipulation.Function = program_.get_function(function_to_evolve)
            function_str: str = str(function_).strip('\n')
            print(f'{function_str}')
            print(f'-----------------------------------------------------')
            print(f'Score: {str(results)}')
            print(f'=====================================================')
            print(f'\n\n')

        return results

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve,
                                  dataset, numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except Exception as e:
            result_queue.put((None, False))


# =============================================================================
# TSP Specification (Boilerplate Template)
# =============================================================================

specification = r'''
import numpy as np
import math


def construct_tsp_tour(dist_matrix: np.ndarray, coords: np.ndarray) -> list:
    """Construct a TSP tour using a priority-based greedy heuristic.

    At each step, the priority function scores all unvisited cities,
    and the city with the highest priority is added to the tour.

    Args:
        dist_matrix: Distance matrix of shape (n, n).
        coords: City coordinates of shape (n, 2).

    Returns:
        tour: A list of city indices forming a complete tour.
    """
    n = len(dist_matrix)
    visited = np.zeros(n, dtype=bool)
    # Start from city 0
    current_city = 0
    tour = [current_city]
    visited[current_city] = True

    for step in range(n - 1):
        # Get indices of unvisited cities
        unvisited = np.where(~visited)[0]
        # Compute priority scores for each unvisited city
        scores = priority(
            current_city=current_city,
            unvisited_cities=unvisited,
            dist_matrix=dist_matrix,
            coords=coords,
            visited=visited,
            step=step,
            total_cities=n
        )
        # Select the unvisited city with the highest priority
        best_idx = np.argmax(scores)
        next_city = unvisited[best_idx]
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city

    return tour


def apply_2opt(tour: list, dist_matrix: np.ndarray, max_iterations: int = 100) -> list:
    """Apply 2-opt local search to improve the tour."""
    n = len(tour)
    improved = True
    iteration = 0
    tour = list(tour)
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                old_cost = dist_matrix[a][b] + dist_matrix[c][d]
                new_cost = dist_matrix[a][c] + dist_matrix[b][d]
                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def compute_tour_length(dist_matrix: np.ndarray, tour: list) -> float:
    """Compute total tour length."""
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]
    return total


@funsearch.run
def evaluate(instance: dict) -> float:
    """Evaluate the evolved priority heuristic on a TSP instance.

    Returns the negative tour length (since FunSearch maximizes the score,
    and we want to minimize tour length).
    """
    dist_matrix = instance['dist_matrix']
    coords = instance['coords']
    n = instance['num_cities']

    # Run the constructive heuristic with the evolved priority function
    # Try multiple starting cities and pick the best
    best_length = float('inf')
    num_starts = min(n, 5)  # try up to 5 starting points for robustness

    for start in range(num_starts):
        # Rotate coordinate order to effectively change starting city
        perm = list(range(n))
        perm[0], perm[start] = perm[start], perm[0]
        remapped_dm = dist_matrix[np.ix_(perm, perm)]
        remapped_coords = coords[perm]

        tour_remapped = construct_tsp_tour(remapped_dm, remapped_coords)
        # Apply 2-opt improvement
        tour_remapped = apply_2opt(tour_remapped, remapped_dm, max_iterations=50)

        length = compute_tour_length(remapped_dm, tour_remapped)
        if length < best_length:
            best_length = length

    # Return negative tour length (FunSearch maximizes, we want to minimize)
    return -best_length


@funsearch.evolve
def priority(
    current_city: int,
    unvisited_cities: np.ndarray,
    dist_matrix: np.ndarray,
    coords: np.ndarray,
    visited: np.ndarray,
    step: int,
    total_cities: int
) -> np.ndarray:
    """Returns priority scores for selecting the next city to visit.

    Higher priority means the city is more likely to be chosen next.

    Args:
        current_city: Index of the current city in the tour.
        unvisited_cities: Array of indices of cities not yet visited.
        dist_matrix: Full distance matrix of shape (n, n).
        coords: City coordinates of shape (n, 2).
        visited: Boolean array indicating which cities are visited.
        step: Current step number (0-indexed).
        total_cities: Total number of cities.

    Returns:
        scores: Array of priority scores, one per unvisited city.
    """
    # Baseline: negative distance (nearest neighbor heuristic)
    distances = dist_matrix[current_city][unvisited_cities]
    scores = -distances
    return scores
'''


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == '__main__':
    # =========================================================================
    # Configuration
    # =========================================================================

    # Select dataset
    # Option 1: Use random instances (default, no external files needed)
    #tsp_datasets = tsp_utils.datasets
    tsp_datasets = tsp_utils.load_tsplib_dir('tsp_datasets/')
    # Option 2: Load TSPLIB instances from a directory
    # tsp_datasets = tsp_utils.load_tsplib_dir('tsp_datasets/')

    # Option 3: Load a single TSPLIB instance
    # tsp_datasets = {'eil51': tsp_utils.load_tsplib_instance('tsp_datasets/eil51.tsp')}

    # For quick testing, use only small instances
    small_instances = {
        k: v for k, v in tsp_datasets.items()
        if v['num_cities'] <= 50
    }
    if not small_instances:
        small_instances = tsp_datasets

    print(f"Using {len(small_instances)} TSP instances:")
    for name, inst in small_instances.items():
        print(f"  {name}: {inst['num_cities']} cities")

    # =========================================================================
    # FunSearch configuration
    # =========================================================================
    class_config = config.ClassConfig(
        llm_class=LLMAPI,
        sandbox_class=Sandbox
    )

    funsearch_config = config.Config(
        samples_per_prompt=4,       # Number of samples per LLM prompt
        evaluate_timeout_seconds=60, # Timeout for evaluating each program
    )

    # Maximum number of LLM samples before stopping
    # Set to None for endless loop
    global_max_sample_num = 20

    # =========================================================================
    # Launch FunSearch
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting FunSearch for TSP")
    print("=" * 60)

    funsearch.main(
        specification=specification,
        inputs=small_instances,
        config=funsearch_config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/funsearch_tsp',
    )
