from easydict import EasyDict as edict

INTERFACE_CONFIG = edict(
    {
        "dev_mode": True,                  # allow the dev mode menu
        "query_dir": "../queries",         # directory with query files (in .tsv format)
        "doc": "../info.json",             # path to the original document retrieved by get_data.ipynb
        "max_results": 5,                  # max results to save for each query
        "max_queries": 20,                 # max queries in each query file
        "max_displayed_results": 20,       # max results to display for each query
        "default_placeholder_delay": 0.01, # dev usage, ignore this
        "users": ["test_user"],            # names for multiple users in dev mode
        "cache_edit": False                # allow to edit cache for FeatureExtractor in dev mode. 
    }
)

CACHE_CONFIG = edict(
    {
        "cache_dir": "./.cache",           # directory to caches
        "interval": 5,                     # after how many new entries should a cache be saved
        "log_character_limit": 20          # character upper limit of a variable to be printed in log
    }
)

PIPELINE_CONFIG = edict(
    {
        "index_fname": "../model/index",   # pyserini index location
        "doc_fname": "../model/data_dropped/data_dropped.json"  # dropped data
    }
)