#!/usr/bin/env python3
"""
Quick test to verify JSON serialization fix for numpy types
"""

import json
import numpy as np
import pandas as pd

# Import the custom encoder and conversion function
import sys
sys.path.append('./src')
from socialMediaInsightsBot import NumpyJSONEncoder, convert_numpy_types

# Test data with various numpy types that would cause the original error
test_data = {
    "int32_value": np.int32(42),
    "int64_value": np.int64(100),
    "float32_value": np.float32(3.14),
    "float64_value": np.float64(2.718),
    "numpy_array": np.array([1, 2, 3]),
    "bool_value": np.bool_(True),
    "nested_dict": {
        "another_int32": np.int32(999),
        "another_float": np.float64(1.234)
    },
    "list_with_numpy": [np.int32(1), np.int32(2), np.int32(3)],
    "regular_values": {
        "normal_int": 42,
        "normal_float": 3.14,
        "normal_string": "hello",
        "normal_bool": True
    }
}

print("Testing JSON serialization fix...")

# Test 1: Direct JSON serialization with custom encoder
print("\n1. Testing NumpyJSONEncoder:")
try:
    json_str = json.dumps(test_data, cls=NumpyJSONEncoder)
    print("✅ SUCCESS: NumpyJSONEncoder works!")
    print(f"   Serialized data length: {len(json_str)} characters")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Using conversion function + JSON encoder
print("\n2. Testing convert_numpy_types + NumpyJSONEncoder:")
try:
    clean_data = convert_numpy_types(test_data)
    json_str = json.dumps(clean_data, cls=NumpyJSONEncoder)
    print("✅ SUCCESS: convert_numpy_types + NumpyJSONEncoder works!")
    print(f"   Serialized data length: {len(json_str)} characters")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Verify the converted data types
print("\n3. Checking converted data types:")
clean_data = convert_numpy_types(test_data)
print(f"   int32_value type: {type(clean_data['int32_value'])}")
print(f"   float64_value type: {type(clean_data['float64_value'])}")
print(f"   numpy_array type: {type(clean_data['numpy_array'])}")
print(f"   nested int32 type: {type(clean_data['nested_dict']['another_int32'])}")

# Test 4: Test with pandas-like data (simulating the actual issue)
print("\n4. Testing pandas-like data:")
try:
    # Simulate pandas operations that return numpy types
    df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    pandas_data = {
        "max_value": df['values'].max(),  # This would be numpy.int64
        "mean_value": df['values'].mean(),  # This would be numpy.float64
        "count": df['values'].count()  # This would be numpy.int64
    }
    
    # Convert and serialize
    clean_pandas_data = convert_numpy_types(pandas_data)
    json_str = json.dumps(clean_pandas_data, cls=NumpyJSONEncoder)
    print("✅ SUCCESS: Pandas data serialization works!")
    print(f"   Original max type: {type(pandas_data['max_value'])}")
    print(f"   Converted max type: {type(clean_pandas_data['max_value'])}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n✨ All tests completed!")