from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    text: str | list[str]
    max_length: int = 130
    min_length: int = 30

class SummarizationResult(BaseModel):
    source_text: str
    summary: str

class SummarizationResponse(BaseModel):
    results: list[SummarizationResult]