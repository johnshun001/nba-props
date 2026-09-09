"""Feature engineering for production pregame forecasts."""

__all__ = ["FEATURE_COLUMNS", "build_pregame_features", "persist_feature_store"]


def __getattr__(name):
    if name in __all__:
        from . import pregame_features
        return getattr(pregame_features, name)
    raise AttributeError(name)
