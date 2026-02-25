"""
FunSearch for TSP using a local LLM server (e.g., vLLM, text-generation-inference, Ollama).

Usage:
    1. Start your local LLM server (e.g., vLLM with code completion model)
    2. Configure the server URL and model name below
    3. Run: python funsearch_tsp_local_llm.py
"""

import json
import multiprocessing
from typing import Collection, Any
import http.client

from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
import tsp_utils


def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions before the function body."""
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


class LocalLLM(sampler.LLM):
    """Language model served locally via vLLM, Ollama, or text-generation-inference.

    For code completion models (e.g., CodeLlama, DeepSeek-Coder), set trim=False
    since they directly output code continuations.

    For instruct models, set trim=True to remove the 'def' header from output.
    """

    def __init__(self, samples_per_prompt: int, trim=False):
        super().__init__(samples_per_prompt)
        self._trim = trim
        # Configure your local LLM server here
        self._host = "localhost"
        self._port = 8000  # default vLLM port
        self._model = "deepseek-coder-6.7b-base"  # or any code model
        self._endpoint = "/v1/completions"  # use /v1/chat/completions for instruct models
        self._use_chat_api = False  # set True for instruct/chat models

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        while True:
            try:
                conn = http.client.HTTPConnection(self._host, self._port)

                if self._use_chat_api:
                    payload = json.dumps({
                        "model": self._model,
                        "max_tokens": 1024,
                        "temperature": 0.8,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert algorithm designer. "
                                    "Complete the given Python function for TSP optimization. "
                                    "Only output Python code."
                                )
                            },
                            {"role": "user", "content": content}
                        ]
                    })
                    endpoint = "/v1/chat/completions"
                else:
                    # Code completion API
                    payload = json.dumps({
                        "model": self._model,
                        "prompt": content,
                        "max_tokens": 1024,
                        "temperature": 0.8,
                        "stop": ["\ndef ", "\nclass ", "\n#"]
                    })
                    endpoint = "/v1/completions"

                headers = {'Content-Type': 'application/json'}
                conn.request("POST", endpoint, payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)

                if self._use_chat_api:
                    response = data['choices'][0]['message']['content']
                else:
                    response = data['choices'][0]['text']

                if self._trim:
                    response = _trim_preface_of_body(response)
                return response

            except Exception as e:
                print(f"Local LLM error: {e}, retrying...")
                import time
                time.sleep(1)
                continue


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated TSP heuristic code."""

    def __init__(self, verbose=False, numba_accelerate=False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(self, program, function_to_run, function_to_evolve, inputs,
            test_input, timeout_seconds, **kwargs):
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
        return results

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve,
                                  dataset, numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program, function_to_evolve=function_to_evolve)
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except Exception:
            result_queue.put((None, False))


# Use the same specification from the API version
from funsearch_tsp_llm_api import specification


if __name__ == '__main__':
    tsp_datasets = tsp_utils.datasets

    small_instances = {
        k: v for k, v in tsp_datasets.items()
        if v['num_cities'] <= 50
    }
    if not small_instances:
        small_instances = tsp_datasets

    print(f"Using {len(small_instances)} TSP instances:")
    for name, inst in small_instances.items():
        print(f"  {name}: {inst['num_cities']} cities")

    class_config = config.ClassConfig(
        llm_class=LocalLLM,
        sandbox_class=Sandbox
    )
    funsearch_config = config.Config(
        samples_per_prompt=4,
        evaluate_timeout_seconds=60,
    )
    global_max_sample_num = 20

    print("\nStarting FunSearch for TSP (Local LLM)")
    funsearch.main(
        specification=specification,
        inputs=small_instances,
        config=funsearch_config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/funsearch_tsp_local',
    )
