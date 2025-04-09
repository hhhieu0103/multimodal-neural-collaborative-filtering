class DoubleCache:
    def __init__(self, main_size=1000, queue_size=400):
        self.main_size = main_size
        self.queue_size = queue_size
        self.main = {}
        self.queue = {}

    def get_image_tensor(self, item_idx):
        image = self.main.get(item_idx, None)
        if image is not None:
            self.main[item_idx]['count'] += 1
            return self.main[item_idx]['tensor']

        image = self.queue.get(item_idx, None)
        if image is not None:
            self.queue[item_idx]['count'] += 1
            self._update_main_cache()
            return self.queue[item_idx]['tensor']

        return None

    def insert_image_tensor(self, item_idx, tensor):
        if len(self.main) < self.main_size:
            self.main[item_idx] = {'count': 1, 'tensor': tensor}
            return

        if len(self.queue) < self.queue_size:
            self.queue[item_idx] = {'count': 1, 'tensor': tensor}
            return

        min_idx = min(self.queue, key=lambda x: self.queue[x]['count'])
        self.queue.pop(min_idx)
        self.queue[item_idx] = {'count': 1, 'tensor': tensor}

    def _update_main_cache(self):
        main_min_idx = min(self.main, key=lambda x: self.main[x]['count'])
        queue_max_idx = max(self.queue, key=lambda x: self.queue[x]['count'])

        main_min = self.main[main_min_idx]['count']
        queue_max = self.queue[queue_max_idx]['count']

        if queue_max > main_min:
            temp = self.main[main_min_idx]
            self.main[main_min_idx] = self.queue[queue_max_idx]
            self.queue[queue_max_idx] = temp

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