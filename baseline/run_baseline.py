# baseline/run_baseline.py
"""
Alternative entry point for running baseline tests
Cleaner interface for development
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.baseline_infer_pytorch import main

if __name__ == '__main__':
    # Simply delegate to main script
    main()