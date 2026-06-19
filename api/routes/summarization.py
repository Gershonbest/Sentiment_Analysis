from fastapi import APIRouter
from ray.serve.handle import DeploymentHandle

from models.summarization.schema import SummarizationRequest, SummarizationResponse

def build_router(handle: DeploymentHandle) -> APIRouter:
    router = APIRouter(prefix="/summarization", tags=["summarization"])

    @router.post("/predict", response_model=SummarizationResponse)
    async def predict(request: SummarizationRequest) -> SummarizationResponse:
        texts = request.text if isinstance(request.text, list) else [request.text]
        results = await handle.predict.remote(
            texts,
            max_length=request.max_length,
            min_length=request.min_length,
        )
        return SummarizationResponse(results=results)

    @router.get("/health")
    async def health():
        return await handle.health.remote()

    return router