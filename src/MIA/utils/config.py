from .singleton import Singleton
class ConfigClass(metaclass=Singleton):
    def __init__(self, config=None):
        assert config is not None, 'config should be provided as a singleton dictionary of config values for use in the code'
        self.config = config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value