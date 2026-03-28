"""mlx-turboquant: Apple-Silicon KV-cache compression for MLX/MLX-LM."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("mlx-turboquant")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
