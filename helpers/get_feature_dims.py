import pandas as pd

def get_feature_dims(df: pd.DataFrame, features: list, output_dim = 8):
    feature_dims = {}
    for feature in features:
        sample = df.iloc[0][feature]
        if isinstance(sample, list):
            num_unique_values = df[feature].explode().nunique()
            feature_dims[feature] = (num_unique_values, output_dim)
        else:
            feature_dims[feature] = (1, output_dim)
    return feature_dims