"""
Comparison Script: Original FunSearch vs Thoughts-Augmented FunSearch
=====================================================================
Run this after running both versions to compare results.

Usage:
  1. Run original:  python funsearch_bin_packing_llm_api.py
     (logs saved to logs/funsearch_llm_api/)
  2. Run augmented: python funsearch_bin_packing_thoughts.py
     (logs saved to logs/funsearch_thoughts_augmented/)
  3. Compare:       python compare_results.py

You can also view results with TensorBoard:
  tensorboard --logdir logs/
"""

import os
import glob


def read_tensorboard_logs(log_dir):
    """Try to read basic info from log directory."""
    if not os.path.exists(log_dir):
        return None
    files = glob.glob(os.path.join(log_dir, '**', '*'), recursive=True)
    return len(files) > 0


def main():
    print("=" * 60)
    print("  FunSearch Comparison: Original vs Thoughts-Augmented")
    print("=" * 60)

    original_dir = 'logs/funsearch_llm_api'
    augmented_dir = 'logs/funsearch_thoughts_augmented'

    print(f"\nOriginal FunSearch logs:  {original_dir}")
    if read_tensorboard_logs(original_dir):
        print("  Status: Logs found ✓")
    else:
        print("  Status: No logs found. Run funsearch_bin_packing_llm_api.py first.")

    print(f"\nThoughts-Augmented logs: {augmented_dir}")
    if read_tensorboard_logs(augmented_dir):
        print("  Status: Logs found ✓")
    else:
        print("  Status: No logs found. Run funsearch_bin_packing_thoughts.py first.")

    print("\n" + "-" * 60)
    print("To visualize and compare results:")
    print("  tensorboard --logdir logs/")
    print("  Then open http://localhost:6006 in your browser.")
    print("-" * 60)

    print("\n📋 Key metrics to compare in your report:")
    print("  1. Best score achieved (fewer bins = better)")
    print("  2. Convergence speed (iterations to reach threshold)")
    print("  3. Valid code rate (% of LLM outputs that execute)")
    print("  4. Diversity of generated heuristics")


if __name__ == '__main__':
    main()
