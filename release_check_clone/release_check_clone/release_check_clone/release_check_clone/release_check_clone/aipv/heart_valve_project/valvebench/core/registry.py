MODEL_REGISTRY = {}


def register_model(name):
    """Register a model class by name for benchmark pipelines."""
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def get_model(name):
    return MODEL_REGISTRY.get(name)
