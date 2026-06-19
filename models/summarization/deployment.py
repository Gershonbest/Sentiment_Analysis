from ray import serve
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from loguru import logger
from typing import Any, Optional

from core.base_deployment import BaseModelDeployment
from core.registry import ModelSpec, register_model
from config.settings import settings
from .schema import SummarizationResult


@serve.deployment(
    name="summarizer",
    num_replicas=settings.summarization.num_replicas,
    ray_actor_options={"num_cpus": settings.summarization.num_cpus},
)
class Summarizer(BaseModelDeployment):
    def __init__(self):
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = torch.device("cpu")
        logger.info("[summarization] model load deferred until first request")

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info("[summarization] loading model…")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.summarization.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.summarization.model_name)

        if settings.summarization.device >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{settings.summarization.device}")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info("[summarization] ready ✓")

    async def predict(
        self,
        texts: list[str],
        max_length: int = 130,
        min_length: int = 30,
    ) -> list[SummarizationResult]:
        self._ensure_loaded()
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )

        summaries = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [
            SummarizationResult(source_text=t, summary=s)
            for t, s in zip(texts, summaries)
        ]


register_model(
    ModelSpec(
        name="summarization",
        bind_fn=lambda: Summarizer.bind(),
        route_prefix="/summarization",
    )
)