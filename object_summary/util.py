import pandas as pd
import numpy as np

def split_df(df, num_splits):
    if num_splits <= 0:
        raise ValueError('Number of splits cannot be less than or equal to zero.')
        
    N = df.shape[0]
    split_ends = np.linspace(0, N, num_splits + 1, dtype=np.int32)
    parts = []
    for i in range(1, len(split_ends)):
        start = split_ends[i - 1]
        end = split_ends[i]
        parts.append(df.iloc[start:end])
        
    return parts
