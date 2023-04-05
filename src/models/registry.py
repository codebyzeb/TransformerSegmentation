MODEL_REGISTRY = {}
CONFIG_REGISTRY = {}


def register_model(name, config_cls):
    def _register(cls):
        MODEL_REGISTRY[name] = cls
        CONFIG_REGISTRY[name] = config_cls
        return cls

    return _register
