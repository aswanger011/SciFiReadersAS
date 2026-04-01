"""
Model Context Protocol server for SciFiReaders.

The server intentionally exposes a single tool:
``read_file``. It automatically selects the best reader from the package's
registered readers, executes that reader, and returns the extracted data,
metadata, and dimension information as JSON-friendly dictionaries.
"""

from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from warnings import warn

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional runtime dependency
    FastMCP = None


def _as_builtin(value: Any) -> Any:
    """Convert nested numpy/sidpy-heavy structures into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(key): _as_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        for encoding in ("utf-8", "latin-1"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.name
    return value


def _load_reader_classes() -> list[type]:
    """Import the package reader registry lazily."""
    readers_module = importlib.import_module("SciFiReaders.readers")
    return list(getattr(readers_module, "all_readers", []))


def _reader_summary(reader_cls: type) -> Dict[str, Any]:
    """Return a short, serializable description of a reader class."""
    doc = (reader_cls.__doc__ or "").strip()
    return {
        "name": reader_cls.__name__,
        "module": reader_cls.__module__,
        "doc": doc.splitlines()[0] if doc else "",
    }


def _matches_extension(reader_cls: type, file_suffix: str) -> bool:
    """Best-effort extension routing for legacy readers whose can_read() is not compatible."""
    suffix = file_suffix.lower()
    module_name = reader_cls.__module__.lower()
    class_name = reader_cls.__name__.lower()

    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
        return class_name == "imagereader" or module_name.endswith("generic.image")
    if suffix == ".ibw":
        return class_name == "igoribwreader"
    if suffix == ".spe":
        return class_name == "ramanspereader" or "spe" in module_name
    return False


def available_readers() -> list[Dict[str, Any]]:
    """Advertise the reader classes currently registered by the package."""
    try:
        reader_classes = _load_reader_classes()
    except Exception as exc:  # pragma: no cover - runtime import problems
        return [{"name": "<unavailable>", "module": "", "doc": f"Could not load readers: {exc}"}]
    return [_reader_summary(reader_cls) for reader_cls in reader_classes]


def _is_dataset_like(value: Any) -> bool:
    """Best-effort check for sidpy.Dataset without importing sidpy eagerly."""
    if (
        hasattr(value, "compute")
        and hasattr(value, "shape")
        and hasattr(value, "metadata")
        and hasattr(value, "get_dimension_by_number")
    ):
        return True
    return False


def _dimension_payload(dataset: Any, index: int) -> Dict[str, Any]:
    """Serialize a single sidpy dimension."""
    try:
        dimension = dataset.get_dimension_by_number(index)[0]
    except Exception:
        return {"index": index}

    return {
        "index": index,
        "name": getattr(dimension, "name", f"dim_{index}"),
        "quantity": getattr(dimension, "quantity", ""),
        "units": getattr(dimension, "units", ""),
        "dimension_type": getattr(getattr(dimension, "dimension_type", None), "name", str(getattr(dimension, "dimension_type", ""))),
        "values": _as_builtin(getattr(dimension, "values", [])),
    }


def _dataset_payload(dataset: Any) -> Dict[str, Any]:
    """Convert a sidpy.Dataset into a serializable dictionary."""
    try:
        data = dataset.compute()
    except Exception:
        data = np.asarray(dataset)

    shape = list(getattr(dataset, "shape", np.asarray(data).shape))
    dimensions = [_dimension_payload(dataset, index) for index in range(len(shape))]
    data_type = getattr(dataset, "data_type", None)
    data_type_name = getattr(data_type, "name", str(data_type)) if data_type is not None else "UNKNOWN"

    return {
        "title": getattr(dataset, "title", ""),
        "quantity": getattr(dataset, "quantity", ""),
        "units": getattr(dataset, "units", ""),
        "data_type": data_type_name,
        "shape": shape,
        "dtype": str(getattr(np.asarray(data), "dtype", "")),
        "labels": _as_builtin(getattr(dataset, "labels", [])),
        "data": _as_builtin(data),
        "metadata": _as_builtin(getattr(dataset, "metadata", {})),
        "original_metadata": _as_builtin(getattr(dataset, "original_metadata", {})),
        "dimensions": dimensions,
    }


def _serialize_result(result: Any) -> Any:
    """Recursively serialize reader output."""
    if _is_dataset_like(result):
        return _dataset_payload(result)
    if isinstance(result, Mapping):
        return {str(key): _serialize_result(val) for key, val in result.items()}
    if isinstance(result, (list, tuple)):
        return [_serialize_result(item) for item in result]
    return _as_builtin(result)


def _looks_like_dataset_payload(value: Any) -> bool:
    """Detect a serialized single-dataset payload."""
    if not isinstance(value, dict):
        return False
    return {"title", "shape", "data_type", "data"}.issubset(value.keys())


def _select_reader(file_path: str) -> tuple[type, list[Dict[str, Any]]]:
    """Select the best matching reader using the package's own registry."""
    reader_classes = _load_reader_classes()
    matches: list[type] = []
    file_suffix = str(Path(file_path).suffix).lower()

    for reader_cls in reader_classes:
        if _matches_extension(reader_cls, file_suffix):
            matches.append(reader_cls)
            continue
        try:
            candidate = reader_cls(file_path)
            readable = candidate.can_read()
        except TypeError:
            # Older SciFiReaders readers call sidpy.Reader.can_read(extension=...)
            # which no longer exists in newer sidpy releases. Fall back to
            # extension-based routing for those readers.
            continue
        except Exception:
            continue
        if readable:
            matches.append(reader_cls)

    if not matches:
        raise TypeError("The automatic search for a suitable reader was unsuccessful.")

    if len(matches) > 1:
        warn(
            "Multiple readers may be able to read this file. "
            f"Using {matches[-1].__name__}."
        )

    return matches[-1], [_reader_summary(reader_cls) for reader_cls in matches]


def read_file(file_path: str) -> Dict[str, Any]:
    """
    Read a file with the appropriate SciFiReaders reader and return all data.

    The payload includes:
    - ``available_readers``: all registered readers in the package
    - ``matched_readers``: readers that reported they could read the file
    - ``selected_reader``: the reader that was actually used
    - ``datasets``: one entry per returned sidpy.Dataset or reader output item
    """
    reader_cls, matched_readers = _select_reader(file_path)
    extracted = reader_cls(file_path).read()
    serialized = _serialize_result(extracted)

    if isinstance(serialized, list):
        datasets = {f"Channel_{index:03d}": item for index, item in enumerate(serialized)}
    elif _looks_like_dataset_payload(serialized):
        datasets = {"Channel_000": serialized}
    elif isinstance(serialized, dict):
        datasets = serialized
    else:
        datasets = {"Channel_000": serialized}

    return {
        "file_path": file_path,
        "available_readers": available_readers(),
        "matched_readers": matched_readers,
        "selected_reader": _reader_summary(reader_cls),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def create_mcp_server(server_name: str = "scifireaders") -> Any:
    """Create the MCP server exposing the single file-reading tool."""
    if FastMCP is None:  # pragma: no cover - optional runtime dependency
        raise ImportError("The 'mcp' package is required to create the SciFiReaders MCP server.")

    server = FastMCP(server_name)

    @server.tool()
    def read_file_tool(file_path: str) -> Dict[str, Any]:
        """Read a file with the automatically selected SciFiReaders reader."""
        return read_file(file_path)

    return server


if FastMCP is not None:  # pragma: no cover - optional runtime dependency
    mcp = create_mcp_server()
else:  # pragma: no cover - optional runtime dependency
    mcp = None


def main() -> None:
    """Run the MCP server over stdio."""
    if mcp is None:  # pragma: no cover - optional runtime dependency
        raise ImportError("Install the 'mcp' package or the optional 'SciFiReaders[mcp]' extra to run this server.")
    try:
        mcp.run(transport="stdio")
    except TypeError:  # pragma: no cover - compatibility with older MCP releases
        mcp.run()


__all__ = [
    "AVAILABLE_READERS",
    "available_readers",
    "create_mcp_server",
    "main",
    "read_file",
]


AVAILABLE_READERS = available_readers()
