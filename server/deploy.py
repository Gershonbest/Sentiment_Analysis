from config.settings import settings

import ray
from ray import serve
from loguru import logger

import models.sentiment.deployment  # noqa: F401
import models.summarization.deployment  # noqa: F401
from core.registry import all_models
from server.ingress import IngressDeployment


def deploy():
    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": settings.host, "port": settings.http_port})

    specs = all_models()
    if not specs:
        raise RuntimeError(
            "No models registered. Check models/__init__.py imports all "
            "model packages, and that each deployment.py calls register_model() "
            "at module level."
        )
    handles = {name: spec.bind_fn() for name, spec in specs.items()}

    ingress = IngressDeployment.bind(handles)
    serve.run(ingress, route_prefix="/")

    logger.info(f"Live at http://{settings.host}:{settings.http_port}")
    for name in specs:
        logger.info(f"  POST /{name}/predict")
    logger.info("  GET  /docs")