from importlib import import_module

_SUBMODULES = {
    "bsq",
    "bsq_no_upsample",
    "causal_cnn",
    "firefly",
    "firefly_encoder",
    "fsq",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
