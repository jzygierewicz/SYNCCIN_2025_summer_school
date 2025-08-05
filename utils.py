import numpy as np  # type: ignore
def describe_dict(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: ndarray, shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: list, length={len(value)}")
        elif isinstance(value, dict):
            print(f"{key}: dict, keys={list(value.keys())}")
        else:
            print(f"{key}: {type(value).__name__}, value={value}")

