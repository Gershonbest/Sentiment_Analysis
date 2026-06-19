from ray import serve
from transformers import pipeline
from loguru import logger
from typing import Any, Optional

from core.base_deployment import BaseModelDeployment
from core.registry import ModelSpec, register_model
from config.settings import settings
from .schema import SentimentResult


@serve.deployment(
    name="sentiment_classifier",
    num_replicas=settings.sentiment.num_replicas,
    ray_actor_options={"num_cpus": settings.sentiment.num_cpus},
)
class SentimentClassifier(BaseModelDeployment):
    def __init__(self):
        self.classifier: Optional[Any] = None
        logger.info("[sentiment] model load deferred until first request")

    def _ensure_loaded(self) -> None:
        if self.classifier is not None:
            return

        logger.info("[sentiment] loading model…")
        self.classifier = pipeline(
            task="sentiment-analysis",
            model=settings.sentiment.model_name,
            device=settings.sentiment.device,   # -1 = CPU, 0 = first GPU
        )
        logger.info("[sentiment] ready ✓")

    async def predict(self, texts: list[str]) -> list[SentimentResult]:
        self._ensure_loaded()
        raw = self.classifier(texts, truncation=True, max_length=512)
        return [
            SentimentResult(text=t, label=r["label"], score=round(r["score"], 4))
            for t, r in zip(texts, raw)
        ]


register_model(
    ModelSpec(
        name="sentiment",
        bind_fn=lambda: SentimentClassifier.bind(),
        route_prefix="/sentiment",
    )
)