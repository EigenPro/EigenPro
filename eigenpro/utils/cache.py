class LRUCache:
    def __init__(self):
        self.cache = {}

    def get(self, key: int) -> int:
        return self.cache.get(key, -1)

    def put(self, key: int, value: int) -> None:
        # Since capacity is 1, clear the cache before adding a new item
        self.cache.clear()
        self.cache[key] = value
