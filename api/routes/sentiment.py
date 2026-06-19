from fastapi import APIRouter
from ray.serve.handle import DeploymentHandle

from models.sentiment.schema import SentimentRequest, SentimentResponse

def build_router(handle: DeploymentHandle) -> APIRouter:
    router = APIRouter(prefix="/sentiment", tags=["sentiment"])

    @router.post("/predict", response_model=SentimentResponse)
    async def predict(request: SentimentRequest) -> SentimentResponse:
        texts = request.text if isinstance(request.text, list) else [request.text]
        results = await handle.predict.remote(texts)
        return SentimentResponse(results=results)

    @router.get("/health")
    async def health():
        return await handle.health.remote()

    return router