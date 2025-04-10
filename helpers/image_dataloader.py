from helpers.image_cacher import DoubleCache, SingleCache, UnlimitedCache, CacheType
import lmdb
import numpy as np
import torch

class ImageDataLoader:
    def __init__(self, file_dir, index_manager, cache_size = 1000, cache_type: CacheType = CacheType.DOUBLE):
        self.missing_idx = set()
        self.index_manager = index_manager
        self.file_dir = file_dir

        if cache_type == CacheType.DOUBLE:
            self.cache = DoubleCache(cache_size)
        elif cache_type == CacheType.SINGLE:
            self.cache = SingleCache(cache_size)
        else:
            self.cache = UnlimitedCache()

        self.env = None
        self.txn = None

    def get_tensor(self, item_idx, device='cpu'):

        if item_idx in self.missing_idx:
            return torch.zeros(512, device=device)

        # Find item in most used cache
        tensor = self.cache.get_image_tensor(item_idx)

        if tensor is not None:
            return tensor

        # Find item in all files
        item_id = self.index_manager.item_id(item_idx)
        tensor = self._get_tensor_from_file(item_id, device)

        if tensor is not None:
            self.cache.insert_image_tensor(item_idx, tensor)
            return tensor

        self.missing_idx.add(item_idx)
        return torch.zeros(512, device=device)

    def _get_tensor_from_file(self, item_id, device):
        tensor_data = self.txn.get(str(item_id).encode())
        if tensor_data is None:
            return None
        tensor_data = np.frombuffer(tensor_data, dtype=np.float32)
        return torch.tensor(tensor_data, device=device)

    def get_batch_tensors(self, item_indices, device='cuda'):
        batch_features = np.zeros((len(item_indices), 512), dtype=np.float32)

        batch_missing_idx = []
        batch_missing_pos = []

        for i, item_idx in enumerate(item_indices):
            if item_idx in self.missing_idx:
                continue

            tensor = self.cache.get_image_tensor(item_idx)
            if tensor is not None:
                batch_features[i] = tensor
            else:
                batch_missing_idx.append(item_idx)
                batch_missing_pos.append(i)

        if len(batch_missing_idx) == 0:
            return torch.tensor(batch_features, device=device)

        missing_keys = []
        for item_idx in batch_missing_idx:
            item_id = self.index_manager.item_id(item_idx)
            missing_keys.append(str(item_id).encode())

        env = self._init_env()
        with env.begin(write=False) as txn:
            for key, pos in zip(missing_keys, batch_missing_pos):
                data = txn.get(key)
                if data is not None:
                    feature = np.frombuffer(data, dtype=np.float32)
                    batch_features[pos] = feature
                    item_idx = item_indices[pos]
                    self.cache.insert_image_tensor(item_idx, feature)
                else:
                    self.missing_idx.add(item_idx)

        return torch.tensor(batch_features, device=device)

    def open_lmdb(self):
        self.env = self._init_env()
        self.txn = self.env.begin(write=False)

    def _init_env(self):
        return lmdb.open(
            self.file_dir,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )