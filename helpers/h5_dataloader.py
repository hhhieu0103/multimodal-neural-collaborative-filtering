import h5py
import torch
from helpers.image_cacher import DoubleCache

class H5DataLoader:
    def __init__(self, h5_files, index_manager, cache_size = 1000):
        self.missing_idx = set()
        self.index_manager = index_manager
        self.h5_files = h5_files
        self.file_to_ids = self._get_file_to_ids()
        self.cache = DoubleCache(cache_size)

    def get_tensor(self, item_idx):
        if item_idx in self.missing_idx:
            return torch.zeros(3, 224, 224)

        # Find item in most used cache
        tensor = self.cache.get_image_tensor(item_idx)

        if tensor is not None:
            return tensor

        # Find item in all files
        item_id = str(self.index_manager.item_id(item_idx))
        file_path = None
        for file, ids in self.file_to_ids.items():
            if item_id in ids:
                file_path = file
                break

        if file_path is None:
            self.missing_idx.add(item_idx)
            return torch.zeros(3, 224, 224)
        else:
            tensor = self._get_file_tensor(file_path, item_id)
            self.cache.insert_image_tensor(item_idx, tensor)

        return tensor

    def _get_file_tensor(self, file_path, item_id):
        with h5py.File(file_path, 'r') as f:
            tensor = f[item_id][:]
        return torch.tensor(tensor, dtype=torch.float32)

    def _get_file_to_ids(self):
        file_to_ids = {}
        for file_path in self.h5_files:
            with h5py.File(file_path, 'r') as f:
                file_to_ids[file_path] = list(f.keys())
        return file_to_ids