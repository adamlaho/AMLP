"""
Cache module for storing and retrieving data.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class Cache:
    """
    Simple file-based cache for storing API responses and other data.
    """
    
    def __init__(self, cache_dir, ttl=86400):
        """
        Initialize the cache.
        
        Args:
            cache_dir (str): Directory to store cache files.
            ttl (int): Time to live in seconds. Default is 24 hours.
        """
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.ttl = ttl
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache initialized at {self.cache_dir} with TTL of {ttl} seconds")
    
    def _get_cache_file_path(self, key):
        """
        Get the file path for a cache key.
        
        Args:
            key (str): Cache key.
        
        Returns:
            Path: Path to the cache file.
        """
        # Create a hash of the key to use as the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key):
        """
        Get a value from the cache.
        
        Args:
            key (str): Cache key.
        
        Returns:
            Any: Cached value or None if not found or expired.
        """
        cache_file = self._get_cache_file_path(key)
        
        if not cache_file.exists():
            logger.debug(f"Cache miss: {key}")
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            if time.time() - cache_data['timestamp'] > self.ttl:
                logger.debug(f"Cache expired: {key}")
                os.remove(cache_file)
                return None
            
            logger.debug(f"Cache hit: {key}")
            return cache_data['value']
        
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, key, value):
        """
        Set a value in the cache.
        
        Args:
            key (str): Cache key.
            value: Value to cache.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        cache_file = self._get_cache_file_path(key)
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'value': value
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"Cache set: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            return False
    
    def clear(self, key=None):
        """
        Clear cache entries.
        
        Args:
            key (str, optional): Specific key to clear. If None, clear all cache.
        
        Returns:
            int: Number of entries cleared.
        """
        if key:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                os.remove(cache_file)
                logger.debug(f"Cleared cache entry: {key}")
                return 1
            return 0
        
        # Clear all cache entries
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            os.remove(cache_file)
            count += 1
        
        logger.debug(f"Cleared {count} cache entries")
        return count
    
    def cleanup_expired(self):
        """
        Remove all expired cache entries.
        
        Returns:
            int: Number of expired entries removed.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                if time.time() - cache_data['timestamp'] > self.ttl:
                    os.remove(cache_file)
                    count += 1
            
            except Exception as e:
                logger.error(f"Error cleaning up cache: {e}")
                # Remove corrupted cache files
                os.remove(cache_file)
                count += 1
        
        logger.debug(f"Removed {count} expired cache entries")
        return count