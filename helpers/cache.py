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

    def get(self, key, default=None):
        value = self.main.get(key, None)
        if value is not None:
            self.main[key]['count'] += 1
            if self.main_min == key:
                self.main_min = min(self.main, key=lambda x: self.main[x]['count'])
            self.hit += 1

        value = self.queue.get(key, None)
        if value is not None:
            self.queue[key]['count'] += 1
            if self.queue_min == key:
                self.queue_min = min(self.queue, key=lambda x: self.queue[x]['count'])

            if self.queue_max is None:
                self.queue_max = key
            else:
                queue_max_count = self.queue[self.queue_max]['count']
                value_count = self.queue[key]['count']

                if value_count > queue_max_count:
                    self.queue_max = key

            self._update_main_cache()
            self.hit += 1

        if value is None:
            self.miss += 1
            return default

        return value

    def insert(self, key, value):
        if len(self.main) < self.main_size:
            self.main[key] = {'count': 1, 'value': value}
            self.main_min = key
            return

        if len(self.queue) < self.queue_size:
            self.queue[key] = {'count': 1, 'value': value}
            self.queue_min = key
            return

        if self.queue_min is None:
            self.queue_min = min(self.queue, key=lambda x: self.main[x]['count'])
        if self.queue_min == self.queue_max:
            self.queue_max = None
        self.queue.pop(self.queue_min)
        self.queue[key] = {'count': 1, 'value': value}
        self.queue_min = key

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
    def __init__(self, size=1000):
        self.size = size
        self.cache = {}
        self.min_key = None
        self.hit = 0
        self.miss = 0

    def get(self, key, default=None):
        value = self.cache.get(key, default)
        if value is not None:
            self.hit += 1
            self.cache[key]['count'] += 1
            if self.min_key == key:
                self.min_key = min(self.cache, key=lambda x: self.cache[x]['count'])
            return self.cache[key]['value']

        self.miss += 1
        return None

    def insert(self, key, value):
        if len(self.cache) >= self.size:
            if self.min_key is None:
                self.min_key = min(self.cache, key=lambda x: self.cache[x]['count'])
            self.cache.pop(self.min_key)

        self.cache[key] = {'count': 1, 'value': value}
        self.min_key = key

    def hit_rate(self):
        return self.hit / (self.hit + self.miss)

class UnlimitedCache:
    def __init__(self):
        self.cache = {}

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def insert(self, key, value):
        self.cache[key] = value
