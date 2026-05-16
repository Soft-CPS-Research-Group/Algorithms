import importlib
import types


def test_module_importable():
    mod = importlib.import_module("scripts.explore_features")
    assert isinstance(mod, types.ModuleType)


def test_has_main():
    from scripts import explore_features
    assert callable(explore_features.main)
