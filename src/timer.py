import time
from functools import wraps
from typing import Dict, Optional


class Timer:
    _instance: Optional["Timer"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.function_times: Dict[str, float] = {}
        self.total_time: float = 0.0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            execution_time = end_time - start_time
            func_name = func.__name__

            if func_name in self.function_times:
                self.function_times[func_name] += execution_time
            else:
                self.function_times[func_name] = execution_time

            self.total_time += execution_time

            return result

        return wrapper

    def get_stats(self) -> Dict[str, float]:
        """Returns dictionary with execution times for all tracked functions"""
        stats = self.function_times.copy()
        stats["total"] = self.total_time
        return stats

    def reset(self):
        """Resets all timing statistics"""
        self.function_times.clear()
        self.total_time = 0.0


timer = Timer()
