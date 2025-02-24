import openai
from typing import List, Dict

class EmbeddingGenerator:
    def __init__(self, api_key, model="text-embedding-ada-002"):
        openai.api_key = api_key
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a given text"""
        response = openai.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def batch_generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for a batch of text chunks"""
        result = []
        for chunk in chunks:
            embedding = self.generate_embedding(chunk["content"])
            chunk["embedding"] = embedding
            result.append(chunk)
        return result