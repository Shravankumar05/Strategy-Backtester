import pandas as pd
from typing import Dict, Optional
import os
import pickle
from datetime import datetime, timedelta


class CacheManager: 
    def __init__(self, cache_dir: Optional[str] = None, expiry_hours: int = 24):
        self._memory_cache: Dict[str, tuple] = {}
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if datetime.now() - timestamp < timedelta(hours=self.expiry_hours):
                return data
            else:
                del self._memory_cache[key]
        
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        data, timestamp = pickle.load(f)
                    
                    if datetime.now() - timestamp < timedelta(hours=self.expiry_hours):
                        self._memory_cache[key] = (data, timestamp)
                        return data
                    else:
                        os.remove(cache_file)
                except Exception:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
        
        return None
    
    def cache_data(self, key: str, data: pd.DataFrame) -> None:
        timestamp = datetime.now()
        self._memory_cache[key] = (data, timestamp)
        
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((data, timestamp), f)
            except Exception:
                pass
    
    def clear_cache(self) -> None:
        self._memory_cache.clear()
        if self.cache_dir and os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))