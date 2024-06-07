import time


class Timer:
    def __init__(self):
        self.elapsed_time = 0
        self.count = 0

    def __enter__(self):
        self._started_at = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ended_at = time.time()
        self.elapsed_time += self._ended_at - self._started_at
        self.count += 1

    def frequency(self) -> float:
        if self.elapsed_time == 0:
            return 0
        return self.count / self.elapsed_time
