from dataclasses import dataclass
from typing import Callable

@dataclass
class ModelSpec:
    name: str
    bind_fn: Callable          # () -> serve.Application (deployment.bind())
    route_prefix: str          # used by api router to mount handle

_REGISTRY: dict[str, ModelSpec] = {}

def register_model(spec: ModelSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Model '{spec.name}' already registered")
    _REGISTRY[spec.name] = spec

def all_models() -> dict[str, ModelSpec]:
    return dict(_REGISTRY)