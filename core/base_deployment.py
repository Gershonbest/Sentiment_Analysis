from abc import ABC, abstractmethod
from typing import Any


class BaseModelDeployment(ABC):
    """Contract every model deployment must satisfy."""

    @abstractmethod
    async def predict(self, payload: Any) -> Any:
        """Run inference. Called via DeploymentHandle.predict.remote(...)."""
        raise NotImplementedError

    async def health(self) -> dict:
        return {"status": "ok"}