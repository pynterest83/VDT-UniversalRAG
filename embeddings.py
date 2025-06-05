import json
from typing import List, Dict, Any, Protocol
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...


class GoogleEmbeddings:
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str = None):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key or os.getenv("GEMINI_API_KEY")
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)


class OpenAIEmbeddings:
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        from langchain_openai import OpenAIEmbeddings
        self.model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)


class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.model = HuggingFaceEmbeddings(model_name=model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)


class CustomEmbeddings:
    def __init__(self, model_path: str, tokenizer_path: str = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model.eval()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import torch
        
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding.tolist())
        
        return embeddings


class EmbeddingProcessor:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
    
    def embed_chunks(self, input_path: str, output_path: str, batch_size: int = 50):
        chunks = self._load_chunks(input_path)
        
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]
            embeddings = self.embedding_model.embed_documents(texts)
            
            for chunk, embedding in zip(batch, embeddings):
                chunk['embedding'] = embedding
        
        self._save_chunks(chunks, output_path)
        return chunks
    
    def _load_chunks(self, path: str) -> List[Dict[str, Any]]:
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
    
    def _save_chunks(self, chunks: List[Dict[str, Any]], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def create_embedding_model(provider: str, model_name: str = None, **kwargs) -> EmbeddingModel:
    if provider == "google":
        return GoogleEmbeddings(model_name or "models/text-embedding-004", **kwargs)
    elif provider == "openai":
        return OpenAIEmbeddings(model_name or "text-embedding-3-small", **kwargs)
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    elif provider == "custom":
        return CustomEmbeddings(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def embed_chunked_documents(input_path: str = 'datasets/chunked_medical_documents.jsonl',
                           output_path: str = 'datasets/embedded_medical_documents.jsonl',
                           provider: str = "google",
                           model_name: str = None,
                           batch_size: int = 50,
                           **kwargs):
    embedding_model = create_embedding_model(provider, model_name, **kwargs)
    processor = EmbeddingProcessor(embedding_model)
    return processor.embed_chunks(input_path, output_path, batch_size)


if __name__ == "__main__":
    # Google (default)
    embed_chunked_documents()
    
    # OpenAI
    # embed_chunked_documents(provider="openai", model_name="text-embedding-3-large")
    
    # Hugging Face
    # embed_chunked_documents(provider="huggingface", model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Custom model
    # embed_chunked_documents(provider="custom", model_name="/path/to/your/model")
