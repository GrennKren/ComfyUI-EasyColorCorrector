#!/usr/bin/env python3
"""Test script to verify input validation fixes for EasyColorCorrection node."""

import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from src import EasyColorCorrection

def test_invalid_inputs():
    """Test that the node handles invalid inputs gracefully."""
    print("=" * 60)
    print("Testing EasyColorCorrection with invalid inputs")
    print("=" * 60)
    
    # Create a simple test image (1 batch, 64x64, RGB)
    test_image = torch.rand(1, 64, 64, 3)
    
    node = EasyColorCorrection()
    
    # Test case 1: None values for float parameters
    print("\n1. Testing None values for float parameters...")
    try:
        result = node.run(
            image=test_image,
            mode="Auto",
            noise=None,  # Should use default 0.0
            tint=None,   # Should use default 0.0
            skin_tone_adjustment=None,  # Should use default 0.0
        )
        print("✅ Test 1 passed: None values handled correctly")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test case 2: Empty list for float parameter
    print("\n2. Testing empty list for float parameter...")
    try:
        result = node.run(
            image=test_image,
            mode="Auto",
            skin_tone_adjustment=[],  # Should use default 0.0
        )
        print("✅ Test 2 passed: Empty list handled correctly")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test case 3: String preset name in float parameter
    print("\n3. Testing string preset name in float parameter...")
    try:
        result = node.run(
            image=test_image,
            mode="Auto",
            tint="Anime Bright",  # Should use default 0.0
        )
        print("✅ Test 3 passed: String preset name handled correctly")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test case 4: Invalid preset value
    print("\n4. Testing invalid preset value...")
    try:
        result = node.run(
            image=test_image,
            mode="Preset",
            preset='0',  # Should use first valid preset
        )
        print("✅ Test 4 passed: Invalid preset handled correctly")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    # Test case 5: Valid inputs (regression test)
    print("\n5. Testing valid inputs (regression)...")
    try:
        result = node.run(
            image=test_image,
            mode="Manual",
            warmth=0.1,
            contrast=0.2,
            brightness=0.05,
            tint=0.0,
            preset="Anime Bright",
            noise=0.0,
            skin_tone_adjustment=0.0,
        )
        print("✅ Test 5 passed: Valid inputs work correctly")
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_invalid_inputs()
    sys.exit(0 if success else 1)
