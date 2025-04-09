import h5py
import torch
from helpers.image_cacher import DoubleCache
import os
import math
import lmdb
import pickle

class H5DataLoader:
    def __init__(self, file_dir, index_manager, cache_size = 1000):
        self.missing_idx = set()
        self.index_manager = index_manager
        self.file_dir = file_dir
        self.cache = DoubleCache(cache_size)

        self.envs = {}
        self.txns = {}

    def get_tensor(self, item_idx):
        if item_idx in self.missing_idx:
            return torch.zeros(3, 224, 224)

        # Find item in most used cache
        tensor = self.cache.get_image_tensor(item_idx)

        if tensor is not None:
            return tensor

        # Find item in all files
        item_id = self.index_manager.item_id(item_idx)
        shard_idx = math.floor(item_id / 500000)
        tensor = self._get_file_tensor(shard_idx, item_id)

        if tensor is not None:
            self.cache.insert_image_tensor(item_idx, tensor)
            return tensor

        self.missing_idx.add(item_idx)
        return torch.zeros(3, 224, 224)

    def _get_file_tensor(self, shard_idx, item_id):
        tensor_data = self.txns[shard_idx].get(str(item_id).encode())
        if tensor_data is None:
            return None
        tensor = pickle.loads(tensor_data)
        return tensor

    def _open_lmdb(self):
        shard_paths = os.listdir(self.file_dir)
        for i, shard_path in enumerate(shard_paths):
            self.envs[i] = lmdb.open(
                os.path.join(self.file_dir, shard_path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )
            self.txns[i] = self.envs[i].begin(write=False)