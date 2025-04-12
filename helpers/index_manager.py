import pandas as pd
import pickle
import os

def _check_missing_cols_df(df, cols):
    missing_cols = set(cols) - set(df.columns)
    if len(missing_cols) > 0:
        raise ValueError(f"Columns {missing_cols} not found in dataframe")

class IndexManager:

    def __init__(self):
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = {}

    def fit(self, df: pd.DataFrame, cols):
        _check_missing_cols_df(df, cols)
        for col in cols:
            unique_values = df[col].unique()
            id_to_idx = self.id_to_idx.get(col, {})
            idx_to_id = self.idx_to_id.get(col, {})
            next_idx = self.next_idx.get(col, 0)
            for value in unique_values:
                if value not in id_to_idx:
                    id_to_idx[value] = next_idx
                    idx_to_id[next_idx] = value
                    next_idx += 1
            self.id_to_idx[col] = id_to_idx
            self.idx_to_id[col] = idx_to_id
            self.next_idx[col] = next_idx
        return self

    def transform(self, df: pd.DataFrame, cols, inplace=False):
        self._check_missing_cols_index(cols)
        df = df if inplace else df.copy()
        for col in cols:
            df[col] = df[col].map(self.id_to_idx[col])
        return df

    def inverse_transform(self, df, cols, inplace=False):
        self._check_missing_cols_index(cols)
        df = df if inplace else df.copy()
        for col in cols:
            df[col] = df[col].map(self.idx_to_id[col])
        return df

    def get_id(self, col, idx):
        self._check_missing_cols_index([col])
        return self.idx_to_id[col].get(idx, None)

    def get_indexed_values(self, col):
        self._check_missing_cols_index([col])
        return list(self.idx_to_id[col].keys())

    def _check_missing_cols_index(self, cols):
        missing_cols = set(cols) - set(self.id_to_idx.keys())
        if len(missing_cols) > 0:
            raise ValueError(f"Columns {missing_cols} not found in index")

    def save(self, path='../data/index.pkl'):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self, path='../data/index.pkl'):
        if not os.path.exists(path):
            raise ValueError(f"Index file {path} not found")
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.__dict__.update(data.__dict__)