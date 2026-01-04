#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

print("Starting simple test...")

try:
    from drift_detector import DriftDetector
    print("✓ DriftDetector imported")

    detector = DriftDetector()
    print("✓ Detector created")

    import pandas as pd
    test_df = pd.read_pickle('baseline_assets/val_test_data.pkl')
    print("✓ Test data loaded")

    import numpy as np
    test_X = np.stack(test_df['embedding_pca'].values).astype(np.float32)
    print("✓ Test embeddings created")

    # Test with minimal data
    X_small = detector.baseline_X[:10]
    Y_small = test_X[:10]

    print(f"Testing with shapes: {X_small.shape} vs {Y_small.shape}")

    gamma = detector._estimate_gamma(X_small, Y_small)
    print(f"✓ Gamma estimated: {gamma}")

    mmd = detector.calculate_mmd(X_small, Y_small, gamma)
    print(f"✓ MMD calculated: {mmd}")

    # Test permutation with 1 iteration
    result = detector.run_permutation_test(X_small, Y_small, iterations=1)
    print(f"✓ Permutation test successful: {result}")

    print("All tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
