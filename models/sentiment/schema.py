from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str | list[str]

class SentimentResult(BaseModel):
    text: str
    label: str
    score: float

class SentimentResponse(BaseModel):
    results: list[SentimentResult]