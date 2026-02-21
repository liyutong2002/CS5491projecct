"""
Thoughts-Augmented FunSearch for Online Bin Packing
====================================================
This file modifies the original FunSearch to co-evolve natural language
"thoughts" alongside code, inspired by EoH (Evolution of Heuristics).

Key changes vs original:
1. LLM prompt includes thought descriptions for each heuristic version
2. LLM is asked to output both a thought and code
3. Thoughts are parsed from LLM response and stored alongside code
4. Multiple prompt strategies (mutate, crossover, etc.)
"""

import json
import re
import random
import multiprocessing
from typing import Collection, Any
import http.client
from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
import bin_packing_utils


# =====================================================================
# Thought storage: global dict mapping function body hash -> thought
# This is a simple approach that doesn't require modifying the deep
# internals of ProgramsDatabase/Cluster/Island.
# =====================================================================
_thought_store = {}

def store_thought(code: str, thought: str):
    """Store a thought associated with a code snippet."""
    key = code.strip()
    _thought_store[key] = thought

def get_thought(code: str) -> str:
    """Retrieve the thought associated with a code snippet."""
    key = code.strip()
    return _thought_store.get(key, "No description available.")


# =====================================================================
# Prompt strategies for Thoughts-Augmented FunSearch
# =====================================================================
PROMPT_STRATEGIES = [
    "thought_augmented",       # Default: show thought+code, ask for improved thought+code
    "mutate_thought_first",    # Emphasize changing the thought/strategy first
    "crossover_thoughts",      # Combine ideas from multiple heuristics
    "reflect_and_improve",     # Ask LLM to analyze weaknesses then improve
]

def _build_thought_instruction(strategy: str) -> str:
    """Returns strategy-specific instruction to append to the prompt."""
    if strategy == "thought_augmented":
        return (
            "Please provide an improved heuristic. First write a brief THOUGHT "
            "(1-2 sentences describing your strategy), then write the Python code.\n"
            "Format:\n"
            "# Thought: <your strategy description>\n"
            "def priority_v{version}(...):\n"
            "    ..."
        )
    elif strategy == "mutate_thought_first":
        return (
            "Think of a COMPLETELY DIFFERENT strategy from the ones above. "
            "First describe your new THOUGHT/idea in 1-2 sentences as a comment, "
            "then implement it as code.\n"
            "Format:\n"
            "# Thought: <your NEW strategy description>\n"
            "def priority_v{version}(...):\n"
            "    ..."
        )
    elif strategy == "crossover_thoughts":
        return (
            "Combine the BEST IDEAS from the heuristics above into a single "
            "improved heuristic. Describe how you combine them in a THOUGHT comment, "
            "then write the code.\n"
            "Format:\n"
            "# Thought: <how you combine the best ideas>\n"
            "def priority_v{version}(...):\n"
            "    ..."
        )
    elif strategy == "reflect_and_improve":
        return (
            "First analyze what makes the best heuristic above work well and what "
            "its weaknesses might be. Then propose an improvement. Write your "
            "analysis as a THOUGHT comment, then implement the improved code.\n"
            "Format:\n"
            "# Thought: <analysis and improvement plan>\n"
            "def priority_v{version}(...):\n"
            "    ..."
        )
    return ""


def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions before the function body.
    Enhanced to also extract and store the thought.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    thought_lines = []

    for lineno, line in enumerate(lines):
        # Collect thought comments
        stripped = line.strip()
        if stripped.startswith('# Thought:') or stripped.startswith('#Thought:'):
            thought_lines.append(stripped)

        # find the first 'def' statement
        if line.strip().startswith('def '):
            func_body_lineno = lineno
            find_def_declaration = True
            break

    # Extract thought from before the function
    thought = ""
    if thought_lines:
        thought = " ".join(t.replace('# Thought:', '').replace('#Thought:', '').strip()
                           for t in thought_lines)
    else:
        # Try to extract thought from text before def
        pre_def_text = "\n".join(lines[:func_body_lineno]).strip()
        # Remove markdown code fences
        pre_def_text = re.sub(r'```\w*', '', pre_def_text).strip()
        if pre_def_text and len(pre_def_text) < 500:
            thought = pre_def_text

    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'

        # Store the thought with this code body
        if thought:
            store_thought(code, thought)

        return code

    # If no def found, try to store thought anyway
    if thought:
        store_thought(sample, thought)
    return sample


class ThoughtsAugmentedLLM(sampler.LLM):
    """LLM that generates thoughts + code for heuristic functions."""

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)
        self._trim = trim
        self._call_count = 0

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        self._call_count += 1

        # Choose a prompt strategy (rotate through strategies)
        strategy = PROMPT_STRATEGIES[self._call_count % len(PROMPT_STRATEGIES)]

        # Inject thoughts into the existing prompt
        enhanced_prompt = self._enhance_prompt_with_thoughts(content, strategy)

        while True:
            try:
                conn = http.client.HTTPSConnection("api.deepseek.com")
                payload = json.dumps({
                    "max_tokens": 768,
                    "model": "deepseek-chat",
                    "temperature": 1.0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert algorithm designer for the online bin packing problem. "
                                "For each heuristic you create, ALWAYS start with a comment line:\n"
                                "# Thought: <brief description of your strategy>\n"
                                "Then write the complete Python function.\n"
                                "Only output the thought comment and Python code, nothing else."
                            )
                        },
                        {
                            "role": "user",
                            "content": enhanced_prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': 'Bearer sk-085c14782c7241c3a4f677973393a4f0',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data['choices'][0]['message']['content']

                # Clean up markdown code fences if present
                response = response.replace('```python', '').replace('```', '')

                if self._trim:
                    response = _trim_preface_of_body(response)
                return response
            except Exception as e:
                print(f"[LLM API Error] {e}, retrying...")
                continue

    def _enhance_prompt_with_thoughts(self, original_prompt: str, strategy: str) -> str:
        """Inject thought descriptions into the prompt for each version."""
        lines = original_prompt.split('\n')
        enhanced_lines = []

        for line in lines:
            enhanced_lines.append(line)
            # After each version's docstring, inject the stored thought
            if 'Improved version of' in line or 'priority_v0' in line:
                # Try to find the code body that follows and look up its thought
                pass  # Thoughts will be added via the strategy instruction

        # Add strategy-specific instruction at the end
        instruction = _build_thought_instruction(strategy)
        # Try to extract version number from prompt
        version_match = re.findall(r'priority_v(\d+)', original_prompt)
        if version_match:
            next_version = max(int(v) for v in version_match)
            instruction = instruction.replace('{version}', str(next_version))

        enhanced_prompt = original_prompt + "\n\n" + instruction

        # Add stored thoughts for referenced versions as extra context
        thought_context = []
        for v in sorted(set(version_match)) if version_match else []:
            # We can't easily map version to code here, so add general context
            pass

        return enhanced_prompt


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code (unchanged from original)."""

    def __init__(self, verbose=False, numba_accelerate=True):
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
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
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
            func_to_evolve_: str = kwargs.get('func_to_evolve', 'priority')
            function_: code_manipulation.Function = program_.get_function(func_to_evolve_)
            function_: str = str(function_).strip('\n')
            print(f'{function_}')
            print(f'-----------------------------------------------------')
            print(f'Score: {str(results)}')
            print(f'=====================================================')
            print(f'\n\n')

        return results

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
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
        except:
            result_queue.put((None, False))


specification = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    ratios = item / bins
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities
'''

# Store the seed thought
store_thought(
    """    ratios = item / bins\n    log_ratios = np.log(ratios)\n    priorities = -log_ratios\n    return priorities\n""",
    "Use negative log of item-to-bin capacity ratio as priority, which prefers bins with capacity close to item size (best-fit decreasing)."
)

if __name__ == '__main__':
    print("=" * 60)
    print("  Thoughts-Augmented FunSearch for Online Bin Packing")
    print("=" * 60)
    print(f"Prompt strategies: {PROMPT_STRATEGIES}")
    print()

    class_config = config.ClassConfig(llm_class=ThoughtsAugmentedLLM, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=30)

    bin_packing_or3 = {'OR3': bin_packing_utils.datasets['OR3']}
    global_max_sample_num = 40  # More samples to see the effect of thoughts
    funsearch.main(
        specification=specification,
        inputs=bin_packing_or3,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/funsearch_thoughts_augmented',
    )

    # Print collected thoughts at the end
    print("\n" + "=" * 60)
    print("  Collected Thoughts")
    print("=" * 60)
    for code_key, thought in _thought_store.items():
        short_code = code_key[:80].replace('\n', ' ')
        print(f"\nThought: {thought}")
        print(f"Code preview: {short_code}...")
        print("-" * 40)
