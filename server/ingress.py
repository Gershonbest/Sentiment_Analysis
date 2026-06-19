import asyncio
from ray import serve
from ray.serve.handle import DeploymentHandle
from loguru import logger

from api.app import app
from api.router import register_routes


@serve.deployment(name="api_gateway", num_replicas=1)
@serve.ingress(app)
class IngressDeployment:
    """
    Single HTTP front door. Holds one handle per registered model;
    route wiring is delegated to api/router.py so this class never
    needs to change when a model is added.
    """

    def __init__(self, handles: dict[str, DeploymentHandle]):
        self.handles = handles
        self._warmup_task: asyncio.Task | None = None
        self._warmup_status: dict[str, object] = {
            "state": "idle",
            "warmed": [],
            "failed": {},
        }
        register_routes(app, handles)

    @app.get("/health")
    async def health(self):
        return {"status": "ok"}

    async def _run_warmup(self) -> None:
        warmed: list[str] = []
        failed: dict[str, str] = {}
        self._warmup_status = {"state": "running", "warmed": warmed, "failed": failed}

        for name, handle in self.handles.items():
            try:
                if name == "sentiment":
                    await handle.predict.remote(["warmup"])
                elif name == "summarization":
                    await handle.predict.remote(
                        ["Ray Serve warmup request for model initialization."],
                        max_length=20,
                        min_length=5,
                    )
                else:
                    logger.warning(f"[warmup] no warmup payload configured for '{name}', skipping")
                    continue
                warmed.append(name)
            except Exception as exc:
                failed[name] = str(exc)
                logger.exception(f"[warmup] failed for '{name}'")

        self._warmup_status = {
            "state": "completed" if not failed else "partial",
            "warmed": warmed,
            "failed": failed,
        }

    @app.post("/warmup")
    async def warmup(self):
        """Start model warmup in the background and return immediately."""
        if self._warmup_task is not None and not self._warmup_task.done():
            return {"status": "running", **self._warmup_status}

        self._warmup_task = asyncio.create_task(self._run_warmup())
        return {"status": "started", **self._warmup_status}

    @app.get("/warmup/status")
    async def warmup_status(self):
        return self._warmup_status