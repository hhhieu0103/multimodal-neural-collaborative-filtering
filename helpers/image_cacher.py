import enum

class CacheType(enum.Enum):
    DOUBLE = 1
    SINGLE = 2
    UNLIMITED = 3

class DoubleCache:
    def __init__(self, main_size=1000, queue_size=200):
        self.main_size = main_size
        self.queue_size = queue_size
        self.main = {}
        self.queue = {}
        self.main_min = None
        self.queue_max = None
        self.queue_min = None
        self.hit = 0
        self.miss = 0

    def get_image_tensor(self, item_idx):
        image = self.main.get(item_idx, None)
        tensor = None
        if image is not None:
            self.main[item_idx]['count'] += 1
            if self.main_min == item_idx:
                self.main_min = min(self.main, key=lambda x: self.main[x]['count'])
            self.hit += 1
            tensor = self.main[item_idx]['tensor']

        image = self.queue.get(item_idx, None)
        if image is not None:
            self.queue[item_idx]['count'] += 1
            if self.queue_min == item_idx:
                self.queue_min = min(self.queue, key=lambda x: self.queue[x]['count'])

            if self.queue_max is None:
                self.queue_max = item_idx
            else:
                queue_max_count = self.queue[self.queue_max]['count']
                item_count = self.queue[item_idx]['count']

                if item_count > queue_max_count:
                    self.queue_max = item_idx

            tensor = self.queue[item_idx]['tensor']
            self._update_main_cache()
            self.hit += 1

        if tensor is None:
            self.miss += 1
        return tensor

    def insert_image_tensor(self, item_idx, tensor):
        if len(self.main) < self.main_size:
            self.main[item_idx] = {'count': 1, 'tensor': tensor}
            self.main_min = item_idx
            return

        if len(self.queue) < self.queue_size:
            self.queue[item_idx] = {'count': 1, 'tensor': tensor}
            self.queue_min = item_idx
            return

        if self.queue_min is None:
            self.queue_min = min(self.queue, key=lambda x: self.main[x]['count'])
        if self.queue_min == self.queue_max:
            self.queue_max = None
        self.queue.pop(self.queue_min)
        self.queue[item_idx] = {'count': 1, 'tensor': tensor}
        self.queue_min = item_idx

    def _update_main_cache(self):
        if self.main_min is None:
            self.main_min = min(self.main, key=lambda x: self.main[x]['count'])
        if self.queue_max is None:
            self.queue_max = max(self.queue, key=lambda x: self.queue[x]['count'])

        main_min_count = self.main[self.main_min]['count']
        queue_max_count = self.queue[self.queue_max]['count']

        if queue_max_count > main_min_count:

            main_min_item = self.main.pop(self.main_min)
            queue_max_item = self.queue.pop(self.queue_max)

            self.main[self.queue_max] = queue_max_item
            self.queue[self.main_min] = main_min_item

            self.main_min = min(self.main, key=lambda x: self.main[x]['count'])
            self.queue_max = max(self.queue, key=lambda x: self.queue[x]['count'])

    def hit_rate(self):
        return self.hit / (self.hit + self.miss)

class SingleCache:
    def __init__(self, main_size=1000):
        self.main_size = main_size
        self.main = {}
        self.min_idx = None
        self.hit = 0
        self.miss = 0

    def get_image_tensor(self, item_idx):
        image = self.main.get(item_idx, None)
        if image is not None:
            self.hit += 1
            self.main[item_idx]['count'] += 1
            if self.min_idx == item_idx:
                self.min_idx = min(self.main, key=lambda x: self.main[x]['count'])
            return self.main[item_idx]['tensor']

        self.miss += 1
        return None

    def insert_image_tensor(self, item_idx, tensor):
        if len(self.main) >= self.main_size:
            if self.min_idx is None:
                self.min_idx = min(self.main, key=lambda x: self.main[x]['count'])
            self.main.pop(self.min_idx)

        self.main[item_idx] = {'count': 1, 'tensor': tensor}
        self.min_idx = item_idx

    def hit_rate(self):
        return self.hit / (self.hit + self.miss)

class UnlimitedCache:
    def __init__(self):
        self.cache = {}

    def get_image_tensor(self, item_idx):
        return self.cache.get(item_idx, None)

    def insert_image_tensor(self, item_idx, tensor):
        self.cache[item_idx] = tensor
