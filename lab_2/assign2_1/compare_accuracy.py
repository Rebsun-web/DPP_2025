#!/usr/bin/env python3
"""Compare CPU and GPU results for accuracy"""

import sys
import numpy as np

def compare_results(cpu_file, gpu_file):
    try:
        cpu = np.loadtxt(cpu_file)
        gpu = np.loadtxt(gpu_file)
        
        if cpu.shape != gpu.shape:
            print(f"Shape mismatch: CPU {cpu.shape} vs GPU {gpu.shape}")
            return False
        
        abs_diff = np.abs(cpu - gpu)
        rel_diff = abs_diff / (np.abs(cpu) + 1e-10)
        
        print("="*60)
        print("ACCURACY COMPARISON: CPU vs GPU")
        print("="*60)
        print(f"Array size: {cpu.shape}")
        print(f"Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.2e}")
        print(f"Max relative difference: {np.max(rel_diff)*100:.4f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)
    
    success = compare_results(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)

