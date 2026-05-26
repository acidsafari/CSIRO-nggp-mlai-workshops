"""Workshop package for the NextGen 2026 ML/AI workshops."""

__all__ = ["__version__", "show_polynomial_recap"]

__version__ = "0.1.0"


def show_polynomial_recap(*args, **kwargs):
    """Lazy package-level access to the Notebook 00 recap plot."""
    from .plots import show_polynomial_recap as _show_polynomial_recap

    return _show_polynomial_recap(*args, **kwargs)
