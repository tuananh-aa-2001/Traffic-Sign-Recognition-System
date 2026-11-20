"""Data module initialization

Die Klassen werden erst beim ersten Zugriff importiert, damit leichte Module wie
src.data.zusatz_klassen (vom Prüfwerkzeug genutzt) nicht PyTorch mitladen.
"""
import importlib

_EXPORTS = {
    'GTSRBDataset': '.data_loader',
    'get_data_loaders': '.data_loader',
    'preprocess_image': '.preprocessing',
    'preprocess_for_inference': '.preprocessing',
    'CLAHEPreprocessor': '.preprocessing',
    'GTSRBAugmentation': '.augmentation',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
