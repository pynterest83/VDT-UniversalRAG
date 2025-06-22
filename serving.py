import modal
import os
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

models_volume = modal.Volume.from_name("rag-models", create_if_missing=True)

embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "sentence-transformers>=2.2.2",
        "langchain-huggingface>=0.0.3",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "accelerate>=0.20.0",
        "fastapi[standard]>=0.100.0"
    ])
)

reranker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "accelerate>=0.20.0",
        "fastapi[standard]>=0.100.0"
    ])
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi[standard]>=0.100.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "httpx>=0.24.0"
    ])
)

app = modal.App("universal-rag-models")

class EmbeddingRequest(BaseModel):
    text: str

class RerankRequest(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    top_k: Optional[int] = 5

def find_model_path(base_dir: str, model_name: str) -> str:
    import os
    
    possible_paths = [
        os.path.join(base_dir, model_name),
        os.path.join(base_dir, "models", model_name),
        os.path.join(base_dir, "rag-models", "models", model_name)
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            print(f"Found model at: {path}")
            return path
    
    raise FileNotFoundError(f"Model {model_name} not found in any of: {possible_paths}")

@app.function(
    image=embedding_image,
    gpu="T4",
    volumes={"/models": models_volume},
    scaledown_window=300,
    timeout=120
)
@modal.fastapi_endpoint(method="POST")
def embed_context(request: EmbeddingRequest):
    """Endpoint for context embedding using local bge-m3-v3 model"""
    import torch  # Import torch here where it's needed
    from langchain_huggingface import HuggingFaceEmbeddings
    
    if not hasattr(embed_context, "model"):
        try:
            model_path = find_model_path("/models", "bge-m3-v3")
            print(f"Loading context embedding model from {model_path}")
            embed_context.model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'trust_remote_code': True
                }
            )
            print("Context embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading context model: {e}")
            return {"error": f"Model loading error: {str(e)}"}
    
    try:
        embedding = embed_context.model.embed_query(request.text)
        return {
            "embedding": embedding, 
            "dimension": len(embedding),
            "model": "bge-m3-v3"
        }
    except Exception as e:
        return {"error": f"Context embedding error: {str(e)}"}

@app.function(
    image=embedding_image,
    gpu="T4",
    volumes={"/models": models_volume},
    scaledown_window=300,
    timeout=120
)
@modal.fastapi_endpoint(method="POST")
def embed_image_caption(request: EmbeddingRequest):
    """Endpoint for image caption embedding using local bge-m3-image model"""
    import torch  # Import torch here where it's needed
    from langchain_huggingface import HuggingFaceEmbeddings
    
    if not hasattr(embed_image_caption, "model"):
        try:
            model_path = find_model_path("/models", "bge-m3-image")
            print(f"Loading image embedding model from {model_path}")
            embed_image_caption.model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'trust_remote_code': True
                }
            )
            print("Image embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading image model: {e}")
            return {"error": f"Model loading error: {str(e)}"}
    
    try:
        embedding = embed_image_caption.model.embed_query(request.text)
        return {
            "embedding": embedding, 
            "dimension": len(embedding),
            "model": "bge-m3-image"
        }
    except Exception as e:
        return {"error": f"Image embedding error: {str(e)}"}

@app.function(
    image=reranker_image,
    gpu="T4",
    volumes={"/models": models_volume},
    scaledown_window=300,
    timeout=240
)
@modal.fastapi_endpoint(method="POST")
def rerank_documents(request: RerankRequest):
    """Endpoint for document reranking using local bge_m3_reranker model"""
    import torch  # Import torch here where it's needed
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    if not hasattr(rerank_documents, "tokenizer") or not hasattr(rerank_documents, "model"):
        try:
            model_path = find_model_path("/models", "bge_m3_reranker")
            print(f"Loading reranker model from {model_path}")
            
            rerank_documents.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True
            )
            
            rerank_documents.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            rerank_documents.model.eval()
            print("Reranker model loaded successfully")
        except Exception as e:
            print(f"Error loading reranker model: {e}")
            # Return a proper error response that matches expected format
            return {
                "error": f"Model loading error: {str(e)}",
                "reranked_documents": []  # Include this field for compatibility
            }
    
    try:
        documents = request.documents
        query = request.query
        top_k = min(request.top_k, len(documents))
        MAX_LENGTH = 2304
        
        if len(documents) <= top_k:
            for doc in documents:
                doc['rerank_score'] = 1.0
            return {"reranked_documents": documents}
        
        doc_contents = [doc['content'] for doc in documents]
        pairs = [[query, doc_content] for doc_content in doc_contents]
        
        with torch.no_grad():
            inputs = rerank_documents.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=MAX_LENGTH
            )
            
            if torch.cuda.is_available() and rerank_documents.model.device.type == 'cuda':
                inputs = {k: v.to(rerank_documents.model.device) for k, v in inputs.items()}
            
            scores = rerank_documents.model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        document_scores = list(zip(documents, scores.cpu().numpy()))
        document_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = []
        for i, (doc, score) in enumerate(document_scores[:top_k]):
            doc['rerank_score'] = float(score)
            reranked_docs.append(doc)
        
        return {
            "reranked_documents": reranked_docs,
            "model": "bge_m3_reranker"
        }
        
    except Exception as e:
        print(f"Reranking processing error: {e}")
        # Return documents with default scores as fallback
        fallback_docs = []
        for i, doc in enumerate(documents[:top_k]):
            doc['rerank_score'] = 1.0 - (i * 0.1)  # Descending scores
            fallback_docs.append(doc)
        
        return {
            "error": f"Reranking error: {str(e)}",
            "reranked_documents": fallback_docs,
            "model": "bge_m3_reranker"
        }

# --- Health Check Endpoint ---
@app.function(image=api_image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Universal RAG Models",
        "models": ["bge-m3-v3", "bge-m3-image", "bge_m3_reranker"]
    }

# --- Model Info Endpoint ---
@app.function(
    image=api_image,
    volumes={"/models": models_volume}
)
@modal.fastapi_endpoint(method="GET")
def model_info():
    """Get information about available models"""
    import os
    
    def search_models(base_dir):
        """Recursively search for model directories"""
        models = {}
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                # This is likely a model directory
                rel_path = os.path.relpath(root, base_dir)
                model_name = os.path.basename(root)
                
                safetensors_exists = "model.safetensors" in files
                pytorch_exists = "pytorch_model.bin" in files
                
                models[model_name] = {
                    "path": root,
                    "relative_path": rel_path,
                    "config_exists": True,
                    "safetensors_exists": safetensors_exists,
                    "pytorch_exists": pytorch_exists,
                    "model_exists": safetensors_exists or pytorch_exists,
                    "ready": True and (safetensors_exists or pytorch_exists),
                    "files_count": len(files)
                }
        return models
    
    if os.path.exists("/models"):
        models = search_models("/models")
        return {"available_models": models}
    else:
        return {"error": "Models directory not found", "available_models": {}}

# --- Test All Models Endpoint ---
@app.function(image=api_image)
@modal.fastapi_endpoint(method="POST")
def test_all_models():
    """Test all three models with sample data"""
    import httpx
    
    test_text = "Paracetamol là thuốc giảm đau và hạ sốt"
    test_query = "tác dụng của paracetamol"
    test_documents = [
        {
            "content": "Paracetamol là thuốc giảm đau, hạ sốt an toàn và hiệu quả",
            "chunk_id": "test_chunk_1"
        },
        {
            "content": "Aspirin cũng có tác dụng giảm đau nhưng có thể gây kích ứng dạ dày",
            "chunk_id": "test_chunk_2"
        }
    ]
    
    results = {}
    
    base_url = "https://ise703--universal-rag-models"
    
    try:
        # Test context embedding
        context_response = httpx.post(
            f"{base_url}-embed-context.modal.run",
            json={"text": test_text},
            timeout=60
        )
        results["context_embedding"] = {
            "status": "success" if context_response.status_code == 200 else "failed",
            "response": context_response.json() if context_response.status_code == 200 else context_response.text
        }
    except Exception as e:
        results["context_embedding"] = {"status": "error", "error": str(e)}
    
    try:
        # Test image embedding
        image_response = httpx.post(
            f"{base_url}-embed-image-caption.modal.run",
            json={"text": test_text},
            timeout=60
        )
        results["image_embedding"] = {
            "status": "success" if image_response.status_code == 200 else "failed",
            "response": image_response.json() if image_response.status_code == 200 else image_response.text
        }
    except Exception as e:
        results["image_embedding"] = {"status": "error", "error": str(e)}
    
    try:
        # Test reranker
        rerank_response = httpx.post(
            f"{base_url}-rerank-documents.modal.run",
            json={
                "query": test_query,
                "documents": test_documents,
                "top_k": 2
            },
            timeout=60
        )
        results["reranker"] = {
            "status": "success" if rerank_response.status_code == 200 else "failed",
            "response": rerank_response.json() if rerank_response.status_code == 200 else rerank_response.text
        }
    except Exception as e:
        results["reranker"] = {"status": "error", "error": str(e)}
    
    return results
