import h5py
import torch

class H5DataLoader:
    def __init__(self, h5_files, index_manager, cache_size = 10000):
        self.missing_idx = set()
        self.idx_tensors = {}
        self.most_used = {}
        self.index_manager = index_manager
        self.h5_files = h5_files
        self.cache_size = cache_size

    def get_tensor(self, item_idx):
        if item_idx in self.missing_idx:
            return torch.zeros(3, 224, 224)

        # Find item in most used cache
        tensor = self.most_used.get(item_idx, None)

        if tensor is not None:
            self.most_used[item_idx]['count'] += 1
            return self.most_used[item_idx]['tensor']

        # Find item in all files
        tensor = self._find_tensor_in_files(item_idx)

        if tensor is None:
            self.missing_idx.add(item_idx)
            return torch.zeros(3, 224, 224)
        else:
            self._insert_to_most_used(item_idx, tensor)

        return tensor

    def _find_tensor_in_files(self, item_idx):
        tensor = None

        for i, file_path in enumerate(self.h5_files):
            item_id = self.index_manager.item_id(item_idx)
            tensor = self._get_file_tensor(file_path, item_id)

            if tensor is not None:
                break

        return tensor

    def _get_file_tensor(self, file_path, item_id):
        item_id = str(item_id)
        with h5py.File(file_path, 'r') as f:
            if item_id in f:
                tensor = f[item_id][:]
                return torch.tensor(tensor, dtype=torch.float32)
            return None

    def _insert_to_most_used(self, item_idx, tensor):
        if len(self.most_used) >= self.cache_size:
            min_idx = min(self.most_used, key=lambda x: self.most_used[x]['count'])
            self.most_used.pop(min_idx)

        self.most_used[item_idx] = {'count': 1, 'tensor': tensor}
