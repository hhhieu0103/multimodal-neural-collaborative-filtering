import pickle
import os
import pandas as pd

class CategoryIndexManager:
    def __init__(self):
        self.cat_to_idx = {}
        self.idx_to_cat = {}

    def fit(self, df_features: pd.DataFrame, features):
        for feature in features:
            self.cat_to_idx[feature] = {}
            self.idx_to_cat[feature] = {}
            categories = df_features[feature].explode().unique()
            for idx, category in enumerate(categories):
                self.cat_to_idx[feature][category] = idx
                self.idx_to_cat[feature][idx] = category

    def transform(self, df_features: pd.DataFrame, features, inplace=False):
        df = df_features if inplace else df_features.copy()
        for feature in features:
            df[feature] = df[feature].apply(lambda cats: [self.cat_to_idx[feature][cat] for cat in cats])
        return df

    def inverse_transform(self, df_features: pd.DataFrame, features, inplace=False):
        df = df_features if inplace else df_features.copy()
        for feature in features:
            df[feature] = df[feature].apply(lambda indices: [self.idx_to_cat[feature][idx] for idx in indices])
        return df

    def save(self, path='../data/category-index.pkl'):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self, path='../data/category-index.pkl'):
        if not os.path.exists(path):
            raise ValueError(f"Index file {path} not found")
        with open(path, 'rb') as file:
            self.__dict__.update(pickle.load(file))