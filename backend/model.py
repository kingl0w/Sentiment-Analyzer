from transformers import pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        #load pretrained model from Hugging Face
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
        
        #label mapping
        self.id2label = {
            "NEGATIVE": "Negative",
            "POSITIVE": "Positive"
        }
    
    def analyze(self, text):
        #get raw predictions from model
        raw_results = self.classifier(text)[0]
        
        #convert to readable format
        probabilities = {self.id2label[result["label"]]: result["score"] for result in raw_results}
        
        #add neutral probability (if not present in model)
        if "Neutral" not in probabilities:
            #for models that only have positive/negative, we infer neutral
            #when both scores are similar
            pos_score = probabilities.get("Positive", 0)
            neg_score = probabilities.get("Negative", 0)
            
            #if scores close, consider it neutral
            diff = abs(pos_score - neg_score)
            neutral_score = max(0, 0.5 - diff)
            
            #renormalize
            total = pos_score + neg_score + neutral_score
            probabilities = {
                "Positive": pos_score / total,
                "Negative": neg_score / total,
                "Neutral": neutral_score / total
            }
        
        #get highest probability sentiment
        sentiment = max(probabilities, key=probabilities.get)
        confidence = probabilities[sentiment]
        
        return {
            "sentiment": sentiment,
            "probabilities": probabilities,
            "confidence": confidence
        }