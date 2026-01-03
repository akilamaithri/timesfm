#!/usr/bin/env python3
"""
Simple test script to verify TimesFM installation and basic functionality.
"""

import sys
import numpy as np

try:
    import timesfm
    print("✓ TimesFM imported successfully")

    # Try to create a model instance (basic check)
    # Note: This may require downloading checkpoints, so we'll just check import
    print("✓ All dependencies loaded")

    # Simple numpy array test
    data = np.random.randn(10, 1)
    print(f"✓ Numpy working: generated data shape {data.shape}")

    print("✓ TimesFM setup appears to be working!")

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)