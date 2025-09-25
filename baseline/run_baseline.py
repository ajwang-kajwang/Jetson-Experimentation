#!/usr/bin/env python3
# baseline/run_baseline.py
"""
Main entry point for baseline inference
Compatible with original instructor's script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.baseline_infer_pytorch import main

if __name__ == '__main__':
    main()