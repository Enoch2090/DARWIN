import uuid
import os
import pickle
import logging
from config import CACHE_CONFIG
from streamlit import experimental_singleton

@experimental_singleton
class PipelineCache:
    def __init__(self, name=None):
        self.name = name if name else uuid.uuid1()
        self.logger = logging.getLogger("darwin")
        self.cache_dir = CACHE_CONFIG.cache_dir
        self.cache_file = os.path.join(self.cache_dir, f"{self.name}.pkl")
        self.init_cache()
        self.log(f"PipelineCache <{self.name}> - created")
        
    def init_cache(self):
        self.cache = {}
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
        self.cache_counter = len(self.cache)
                
    def cache_to_file(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f) # TODO: may lead to performance deterioration after self.cache grows large. consider other databases.
        self.log(f"PipelineCache <{self.name}> - cache saved to {self.cache_file}")
                
    def log(self, log, level="info"):
        if self.logger:
            if level=="info":
                self.logger.info(log)
            elif level=="warning":
                self.logger.warning(log)
            elif level=="error":
                self.logger.error(log)
            elif level=="debug":
                self.logger.debug(log)
            elif level=="critical":
                self.logger.critical(log)
                
    def get(self, query):
        if query in self.cache.keys():
            value = self.cache[query]
            value_str = str(value)
            if len(value_str) > CACHE_CONFIG.log_character_limit:
                value_str = value_str[0:CACHE_CONFIG.log_character_limit] + "...(output exceeds limit, see interface/config.py.CACHE_CONFIG.log_character_limit)"
            self.log(f"PipelineCache <{self.name}> - got query {query}, return cached value {type(value)} {value_str}")
        else:
            value = None
            self.log(f"PipelineCache <{self.name}> - got query {query}, return None")
        return value
    
    def update(self, query, value):
        self.cache[query] = value
        self.cache_counter += 1
        value_str = str(value)
        if len(value_str) > CACHE_CONFIG.log_character_limit:
                value_str = value_str[0:CACHE_CONFIG.log_character_limit] + "...(output exceeds limit, see interface/config.py.CACHE_CONFIG.log_character_limit)"
        self.log(f"PipelineCache <{self.name}> - update query {query} to {type(value)} {value_str}")
        if self.cache_counter % CACHE_CONFIG["interval"] == 0:
            self.cache_to_file()