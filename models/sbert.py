from sentence_transformers import SentenceTransformer

class SBERT:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, sentences):
        return self.model.encode(sentences)
    
    def get_model(self):
        return self.model
    
    def __call__(self, sentences):
        return self.encode(sentences)
