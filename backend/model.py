from transformers import pipeline
import torch
import re

class SentimentAnalyzer:
    def __init__(self):
        #load model from Hugging Face
        print("Device set to use", "cuda" if torch.cuda.is_available() else "cpu")
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
        
        #defines contrast words that indicate mixed sentiment
        self.contrast_words = [
            'but', 'however', 'although', 'though', 'yet', 'nonetheless',
            'nevertheless', 'despite', 'in spite of', 'while', 'whereas',
            'even though', 'on the other hand', 'on the contrary'
        ]
        
        #define intensifiers that impact confidence
        self.intensifiers = [
            'very', 'extremely', 'absolutely', 'completely', 'totally',
            'utterly', 'really', 'definitely', 'certainly', 'undoubtedly',
            'indeed', 'truly', 'highly'
        ]
        
        #define uncertainty words that reduce confidence
        self.uncertainty_words = [
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could be',
            'somewhat', 'sort of', 'kind of', 'a bit', 'a little', 'slightly',
            'fairly', 'rather', 'quite', 'seemingly', 'supposedly'
        ]
    
    def detect_mixed_sentiment(self, text):
        """Detect if text contains mixed sentiment indicators"""
        text_lower = text.lower()
        
        #checks for contrast words
        has_contrast = any(word in text_lower for word in self.contrast_words)
        
        if has_contrast:
            #splits the text around the first contrast word found
            contrast_word = next((word for word in self.contrast_words if word in text_lower), None)
            if contrast_word:
                parts = re.split(f'\\b{contrast_word}\\b', text_lower, 1)
                if len(parts) == 2:
                    #analyzes both parts separately
                    first_part_result = self.classifier(parts[0])[0]
                    second_part_result = self.classifier(parts[1])[0]
                    
                    first_sentiment = max(first_part_result, key=lambda x: x["score"])["label"]
                    second_sentiment = max(second_part_result, key=lambda x: x["score"])["label"]
                    
                    #if the two parts have different sentiments, it's mixed
                    if first_sentiment != second_sentiment:
                        return True, first_part_result, second_part_result
        
        return False, None, None
    
    def calculate_confidence_modifier(self, text):
        """Calculate a confidence modifier based on linguistic cues"""
        text_lower = text.lower()
        
        #checks for intensifiers (increase confidence)
        intensifier_count = sum(1 for word in self.intensifiers if f" {word} " in f" {text_lower} ")
        
        #checks for uncertainty words (decrease confidence)
        uncertainty_count = sum(1 for word in self.uncertainty_words if f" {word} " in f" {text_lower} ")
        
        #calculate modifier (positive for more confidence, negative for less)
        modifier = (intensifier_count * 0.05) - (uncertainty_count * 0.1)
        
        #limits the range of the modifier
        return max(min(modifier, 0.15), -0.3)
    
    def analyze(self, text):
        #check if potentially mixed sentiment
        is_mixed, first_part_results, second_part_results = self.detect_mixed_sentiment(text)
        
        #calculates confidence modifier based on language
        confidence_modifier = self.calculate_confidence_modifier(text)
        
        #if mixed sentiment detected, handle specially
        if is_mixed and first_part_results and second_part_results:
            #gets the dominant sentiment from each part
            first_sentiment = max(first_part_results, key=lambda x: x["score"])
            second_sentiment = max(second_part_results, key=lambda x: x["score"])
            
            #calculates blended probabilities
            pos_score = 0
            neg_score = 0
            
            #accumulates scores from both parts, weighing the second part more
            #(since it usually comes after a "but" and carries more weight)
            for result in first_part_results:
                if result["label"] == "POSITIVE":
                    pos_score += result["score"] * 0.4 
                else:
                    neg_score += result["score"] * 0.4
                    
            for result in second_part_results:
                if result["label"] == "POSITIVE":
                    pos_score += result["score"] * 0.6  
                else:
                    neg_score += result["score"] * 0.6
            
            #adjusts scores to reduce over-confidence for mixed sentiments
            pos_score = min(pos_score, 0.85)
            neg_score = min(neg_score, 0.85)
            
            #calculates neutral score
            diff = abs(pos_score - neg_score)
            if diff < 0.3:  
                neutral_score = 0.7 - diff  
            else:
                neutral_score = 0.2  
            
            #renormalize
            total = pos_score + neg_score + neutral_score
            probabilities = {
                "Positive": pos_score / total,
                "Negative": neg_score / total,
                "Neutral": neutral_score / total
            }
            
            #determine overall sentiment
            if diff < 0.15:
                sentiment = "Neutral" 
            elif pos_score > neg_score:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
                
            #adjusts confidence based on linguistic cues
            confidence = probabilities[sentiment] + confidence_modifier
            confidence = max(min(confidence, 0.95), 0.4)  
            
            return {
                "sentiment": sentiment,
                "probabilities": probabilities,
                "confidence": confidence
            }
        
        #for non-mixed sentiments, use the standard approach
        raw_results = self.classifier(text)[0]
        
        #convert to readable format
        probabilities = {self.id2label[result["label"]]: result["score"] for result in raw_results}
        
        #add neutral probability
        pos_score = probabilities.get("Positive", 0)
        neg_score = probabilities.get("Negative", 0)
        
        #calculates neutral score based on difference
        diff = abs(pos_score - neg_score)
        
        #more nuanced neutral score calculation
        if diff < 0.2:
            neutral_score = 0.6 - diff * 2 
        elif diff < 0.4:
            neutral_score = 0.3 - (diff - 0.2) 
        else:
            neutral_score = 0.05 
            
        #renormalize
        total = pos_score + neg_score + neutral_score
        probabilities = {
            "Positive": pos_score / total,
            "Negative": neg_score / total,
            "Neutral": neutral_score / total
        }
        
        #gets highest probability sentiment
        sentiment = max(probabilities, key=probabilities.get)
        
        #adjusts confidence based on linguistic cues
        confidence = probabilities[sentiment] + confidence_modifier
        
        #caps confidence to prevent unrealistic 99% values
        confidence = max(min(confidence, 0.95), 0.4)
        
        return {
            "sentiment": sentiment,
            "probabilities": probabilities,
            "confidence": confidence
        }