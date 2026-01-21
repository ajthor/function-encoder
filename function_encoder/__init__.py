from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("function_encoder")
except PackageNotFoundError:  # pragma: no cover - local/edited source
    __version__ = "0.0.0"

__all__ = ["__version__"]
