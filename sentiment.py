from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Sentiment_Analysis.predict import SentimentAnalysis
import uvicorn


app = FastAPI()

class Sentiment(BaseModel):
    text: str = None

async def get_stream(text):
    for chunk in text.split(" "):
        yield  chunk + " "

@app.get("/")
async def home():
    return {"message": "Welcome to the Sentiment Analysis API!"}
@app.post('/api/sentiment')
def get_sentiment(sentiment: Sentiment):
    analysis = SentimentAnalysis()
    return analysis(sentiment.text)


@app.post("/api/data/stream")
async def get_data_stream(text: Sentiment):
    async for stream in get_stream(text.text):
        print(stream)
    return StreamingResponse(get_stream(text.text), media_type="text/plain")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
