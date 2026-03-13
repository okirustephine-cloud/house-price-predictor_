# check_model.py - Diagnostic tool
import pickle
import os

print("=" * 60)
print(" CHECKING MODEL FILE")
print("=" * 60)

# Check if file exists
model_path = 'california_knn_pipeline.pkl'

if not os.path.exists(model_path):
    print(f" File not found: {model_path}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files here: {os.listdir('.')}")
else:
    print(f" File found: {model_path}")
    print(f"   File size: {os.path.getsize(model_path)} bytes")

    # Try to load it
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n Loaded object type: {type(data)}")
        print(f"   Object value: {repr(data)[:200]}...")  # First 200 chars

        if isinstance(data, str):
            print(f"\n PROBLEM CONFIRMED: Your pickle file contains a STRING!")
            print(f"   Content: '{data}'")
            print(f"\n This usually happens if you accidentally pickled a filename")
            print(f"   instead of the actual model object.")

        elif hasattr(data, 'predict'):
            print(f"\n GOOD: Object has predict method!")
            print(f"   Model type: {type(data).__name__}")

            # Test prediction
            import pandas as pd
            test_df = pd.DataFrame({
                'MedInc': [3.0], 'HouseAge': [20.0], 'AveRooms': [5.0],
                'AveBedrms': [1.0], 'Population': [1000.0], 'AveOccup': [3.0],
                'Latitude': [37.0], 'Longitude': [-122.0]
            })
            pred = data.predict(test_df)
            print(f"   Test prediction: {pred[0]:.4f}")
        else:
            print(f"\n UNKNOWN: Object type is {type(data)}")

    except Exception as e:
        print(f"\n Error loading pickle: {e}")

print("=" * 60)
