import logging
import settings
import json

CONFIG_PATH = './config.json'

class DictObject:
    def __init__(self, dict: dict = {}):
        self._dict = dict
        
    def __getattr__(self, __name: str):
        if __name in self._dict:
            return self._dict[__name]
        raise Exception(f'settings does not contain attribute "{__name}"')

def read_config(config_path):
    """ Read config from specified 'config_path' """
    # open path
    with open(config_path, 'r') as config_file:   
        config = dict(json.load(config_file))
    return DictObject(config)

def configure_logger():
    # Map log level names to their corresponding constants
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    # Fetch the log level from environment variables
    # log_level = read_config(CONFIG_PATH).LOG_LEVEL
    # log_level = settings.LOG_LEVEL

    # Set the log level to the corresponding constant from the mapping, or use INFO as the default
    log_level = read_config(CONFIG_PATH).LOG_LEVEL
    print(f'Logging level set as: ' + str(log_level))
    log_level = log_level_mapping.get(log_level, logging.INFO)
    print(f'Logging level set as: ' + str(log_level))

    # Configure the logger with the specified log level and other settings
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )