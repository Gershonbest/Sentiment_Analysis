from transformers import pipeline
import torch

# Load the sentiment analysis pipeline
class SentimentAnalysis:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda") # For GPU only
            print("Running on CUDA device")
        elif torch.mps.is_available():
            self.device = torch.device("mps") # For Apple M1 and M2 chips
            print("Running on MPS device") 
        else:
            self.device = torch.device("cpu") # For CPU 
            print("Running on CPU")
            print("No GPU or MPS available, running on CPU")

        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        self.sentiment_classifier = pipeline(
            model= self.model_name, 
            top_k = None,
            device=self.device,
        )


    def __call__(self,text): 
        # call the pipeline function with the text parameter 
        return self.sentiment_classifier(text)[0]
