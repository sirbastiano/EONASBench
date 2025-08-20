"""
Layer registry for modular benchmarking framework.
"""

LAYERS = {}

def register(name: str):
    """Decorator to register a layer class by name.
    Args:
        name (str): Layer ID.
    Returns:
        Callable: Decorator.
    """
    def _wrap(cls):
        LAYERS[name] = cls
        return cls
    return _wrap

def build_cell(cell_id: str, C: int, **kw):
    """Build a cell from the registry.
    Args:
        cell_id (str): Layer ID.
        C (int): Number of channels.
        **kw: Additional arguments.
    Returns:
        nn.Module: Instantiated cell.
    """
    assert cell_id in LAYERS, f"Unknown cell_id: {cell_id}"
    return LAYERS[cell_id](C, **kw)
