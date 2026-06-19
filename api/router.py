"""
Aggregates per-model routers into the gateway's FastAPI app.

Keeps IngressDeployment itself dumb — it just calls register_routes()
with whatever handles it was bound with. Adding a model means adding
one line here, nothing in ingress.py changes.
"""

from importlib import import_module
from fastapi import FastAPI
from ray.serve.handle import DeploymentHandle
from loguru import logger


def register_routes(app: FastAPI, handles: dict[str, DeploymentHandle]) -> None:
    """
    For each registered model, import its api/routes/<name>.py module
    and mount the router it builds. Route modules are expected to live
    at api.routes.<model_name> and expose build_router(handle).
    """
    for name, handle in handles.items():
        try:
            route_module = import_module(f"api.routes.{name}")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Model '{name}' is registered but api/routes/{name}.py "
                f"is missing or not importable."
            ) from e

        if not hasattr(route_module, "build_router"):
            raise RuntimeError(
                f"api/routes/{name}.py must define build_router(handle) -> APIRouter"
            )

        router = route_module.build_router(handle)
        app.include_router(router)
        logger.info(f"[router] mounted /{name} routes")