"""
Validation script to check the integrity of preprocessed pairs.

Usage:
    python validate_pairs.py --input pairs_output
"""

import argparse
import pickle
from pathlib import Path
from collections import Counter


def validate_pairs(pairs_dir):
    """
    Validate the preprocessed pairs for uniqueness and integrity.
    
    Args:
        pairs_dir: directory containing the preprocessed pairs
    """
    pairs_dir = Path(pairs_dir)
    
    print("="*80)
    print("VALIDATING PREPROCESSED PAIRS")
    print("="*80)
    
    # Load pairs
    pairs_file = pairs_dir / "pairs.pkl"
    print(f"\nLoading pairs from: {pairs_file}")
    
    with open(pairs_file, 'rb') as f:
        pairs = pickle.load(f)
    
    print(f"Total pairs loaded: {len(pairs)}")
    
    # Load config for context
    config_file = pairs_dir / "config.pkl"
    if config_file.exists():
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
        print(f"Dataset size: {config['dataset_size']}")
        print(f"Batch size used: {config['batch_size']}")
    
    print("\n" + "-"*80)
    print("VALIDATION CHECKS")
    print("-"*80)
    
    # Extract all p1 and p2 indices
    p1_indices = [p[0] for p in pairs]
    p2_indices = [p[1] for p in pairs]
    
    # Check 1: All p1 indices are unique
    print("\n1. Checking uniqueness of p1 (first element) indices...")
    p1_counter = Counter(p1_indices)
    p1_duplicates = {idx: count for idx, count in p1_counter.items() if count > 1}
    
    if p1_duplicates:
        print(f"   ❌ FAILED: Found {len(p1_duplicates)} duplicate p1 indices")
        print(f"   Example duplicates (showing first 5):")
        for idx, count in list(p1_duplicates.items())[:5]:
            print(f"      Index {idx} appears {count} times")
    else:
        print(f"   ✓ PASSED: All {len(p1_indices)} p1 indices are unique")
    
    # Check 2: All p2 indices are unique
    print("\n2. Checking uniqueness of p2 (second element) indices...")
    p2_counter = Counter(p2_indices)
    p2_duplicates = {idx: count for idx, count in p2_counter.items() if count > 1}
    
    if p2_duplicates:
        print(f"   ❌ FAILED: Found {len(p2_duplicates)} duplicate p2 indices")
        print(f"   Example duplicates (showing first 5):")
        for idx, count in list(p2_duplicates.items())[:5]:
            print(f"      Index {idx} appears {count} times")
    else:
        print(f"   ✓ PASSED: All {len(p2_indices)} p2 indices are unique")
    
    # Check 3: No index appears in both p1 and p2
    print("\n3. Checking for overlap between p1 and p2 indices...")
    p1_set = set(p1_indices)
    p2_set = set(p2_indices)
    overlap = p1_set & p2_set
    
    if overlap:
        print(f"   ⚠ WARNING: Found {len(overlap)} indices that appear in both p1 and p2")
        print(f"   Example overlapping indices (showing first 5): {list(overlap)[:5]}")
    else:
        print(f"   ✓ PASSED: No overlap between p1 and p2 indices")
    
    # Check 4: No self-pairs (p1 == p2)
    print("\n4. Checking for self-pairs (where p1 == p2)...")
    self_pairs = [(i, p) for i, p in enumerate(pairs) if p[0] == p[1]]
    
    if self_pairs:
        print(f"   ❌ FAILED: Found {len(self_pairs)} self-pairs")
        print(f"   Example self-pairs (showing first 5):")
        for pair_idx, (p1, p2) in self_pairs[:5]:
            print(f"      Pair {pair_idx}: ({p1}, {p2})")
    else:
        print(f"   ✓ PASSED: No self-pairs found")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    print(f"Total pairs: {len(pairs)}")
    print(f"Unique p1 indices: {len(p1_set)}")
    print(f"Unique p2 indices: {len(p2_set)}")
    print(f"Total unique indices across both: {len(p1_set | p2_set)}")
    print(f"Min p1 index: {min(p1_indices)}")
    print(f"Max p1 index: {max(p1_indices)}")
    print(f"Min p2 index: {min(p2_indices)}")
    print(f"Max p2 index: {max(p2_indices)}")
    
    # Overall result
    print("\n" + "="*80)
    all_passed = (
        not p1_duplicates and 
        not p2_duplicates and 
        not self_pairs
    )
    
    if all_passed:
        print("✓ ALL CHECKS PASSED - Pairs are valid!")
    else:
        print("❌ SOME CHECKS FAILED - Please review the results above")
    print("="*80)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate preprocessed pairs for contrastive learning"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to directory containing preprocessed pairs (e.g., pairs_output)'
    )
    
    args = parser.parse_args()
    validate_pairs(args.input)


if __name__ == "__main__":
    main()