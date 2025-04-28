from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import SentimentAnalyzer
import uvicorn

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading sentiment analysis model...")
sentiment_analyzer = SentimentAnalyzer()
print("Model loaded successfully. Server is ready to analyze text.")

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: dict
    confidence: float
    
@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = sentiment_analyzer.analyze(input_data.text)
    
    return {
        "sentiment": result["sentiment"],
        "probabilities": result["probabilities"],
        "confidence": result["confidence"]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting Sentiment Analysis API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)