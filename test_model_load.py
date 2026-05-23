#!/usr/bin/env python
import os
import sys

print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

# Test TensorFlow import
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
except Exception as e:
    print(f"✗ Failed to import TensorFlow: {e}")
    sys.exit(1)

# Test Keras import
try:
    from tensorflow.keras.models import load_model
    print("✓ Keras load_model imported successfully")
except Exception as e:
    print(f"✗ Failed to import load_model: {e}")
    sys.exit(1)

# Test model loading
model_path = os.path.join(os.path.dirname(__file__), "digit_model.h5")
print(f"\nAttempting to load model from: {model_path}")

if not os.path.exists(model_path):
    print(f"✗ Model file does not exist at {model_path}")
    sys.exit(1)

file_size = os.path.getsize(model_path)
print(f"  File size: {file_size / (1024*1024):.2f} MB")

try:
    print("\n  Loading model...")
    model = load_model(model_path)
    print(f"✓ Model loaded successfully!")
    print(f"  Model type: {type(model)}")
    print(f"  Model summary:")
    model.summary()
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Model is ready for use.")
print("=" * 60)
