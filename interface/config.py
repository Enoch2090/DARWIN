from easydict import EasyDict as edict

INTERFACE_CONFIG = edict(
    {
        "dev_mode": True,
        "query_dir": "../queries",
        "doc": "../info.json",
        "max_results": 5,
        "max_queries": 20,
        "max_displayed_results": 20,
        "default_placeholder_delay": 0.01,
        "users": ["test_user"],
        "cache_edit": False
    }
)

CACHE_CONFIG = edict(
    {
        "cache_dir": "./.cache",
        "interval": 5,
        "log_character_limit": 20
    }
)

PIPELINE_CONFIG = edict(
    {
        "index_fname": "../model/index",
        "doc_fname": "../model/data_dropped/data_dropped.json"
    }
)