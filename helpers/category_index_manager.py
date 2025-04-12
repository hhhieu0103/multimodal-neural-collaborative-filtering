
class CategoryIndexManager:
    def __init__(self):
        self.cat_to_idx = {}
        self.idx_to_cat = {}

    def fit(self, df_features, features):
        for feature in features:
            self.cat_to_idx[feature] = {}
            self.idx_to_cat[feature] = {}
            categories = df_features[feature].explode().unique()
            for idx, category in enumerate(categories):
                self.cat_to_idx[feature][category] = idx
                self.idx_to_cat[feature][idx] = category

    def transform(self, df_features, features, inplace=False):
        df = df_features if inplace else df_features.copy()
        for feature in features:
            df[feature] = df[feature].apply(lambda cats: [self.cat_to_idx[feature][cat] for cat in cats])
        return df

    def inverse_transform(self, df_features, features, inplace=False):
        df = df_features if inplace else df_features.copy()
        for feature in features:
            df[feature] = df[feature].apply(lambda indices: [self.idx_to_cat[feature][idx] for idx in indices])
        return df