import json
import asyncio
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import os
import re
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from langchain_core.documents import Document
import uuid
import modal
from pathlib import Path
from huggingface_hub import login

# Define the Modal image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "transformers",
    "sentence-transformers",
    "langchain-huggingface",
    "langchain-qdrant", 
    "qdrant-client",
    "tqdm",
    "python-dotenv",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "langchain-core",
    "numpy",
    "scipy",
    "tiktoken",  # Added this missing package
    "sentencepiece",  # Also needed for Vietnamese_Reranker
).add_local_dir("bge-m3-image", "/bge-m3-image").add_local_dir("datasets", "/datasets", ignore=["images/*", "corpus/*"])

app = modal.App(name="bge-m3-image-vietnamese-reranker-benchmark", image=image)

class VectorStore: 
    def __init__(self, collection_name: str = "image-captions-store", delete_existing: bool = False):
        # Login to Hugging Face first
        login(token=os.getenv("HF_TOKEN"))
        
        # Use the local uploaded image model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="/bge-m3-image"
        )

        self.collection_name = collection_name

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        if delete_existing:
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
            except Exception:
                pass

        self._setup()

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client, 
            collection_name=self.collection_name, 
            embedding=self.embedding_model
        )

    def _setup(self): 
        try: 
            all_collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in all_collections.collections]

            if self.collection_name not in collection_names:
                embedding_dim = 1024
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name, 
                    vectors_config=VectorParams(
                        size=embedding_dim, 
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            raise Exception(f"Error setting up Qdrant collection: {e}")

    def load_documents_from_jsonl(self, jsonl_path: str): 
        """Load documents from JSONL file with pre-computed embeddings"""
        points = []
        total_docs = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        for line in tqdm(lines, desc="Loading documents from JSONL"):
            item = json.loads(line.strip())
            
            chunk_id = item.get('chunk_id', str(uuid.uuid4()))
            content = item.get('content', 'No Content')
            metadata = item.get('metadata', {})
            embedding = item.get('embedding', None)
            
            # Use pre-computed embedding if available
            if embedding is None:
                print(f"Warning: No embedding found for chunk_id: {chunk_id}, skipping...")
                continue

            # Create point for Qdrant insertion
            point = PointStruct(
                id=chunk_id, 
                vector=embedding,
                payload={
                    'page_content': content,
                    'chunk_id': chunk_id,
                    **metadata
                }
            )

            points.append(point)
            total_docs += 1

            if len(points) >= 100: 
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, 
                    points=points
                )
                points = []
        
        if points: 
            self.qdrant_client.upsert(
                collection_name=self.collection_name, 
                points=points
            )
        
        print(f"Successfully loaded {total_docs} documents from {jsonl_path} into the vector store.")

    def add_documents(self, documents: list[Document], ids: list[str] = None):
        """Add documents with on-demand embedding computation (fallback for manual additions)"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        points = []
        for doc, doc_id in zip(documents, ids):
            embedding = self.embedding_model.embed_query(doc.page_content)
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    'page_content': doc.page_content,
                    'chunk_id': doc_id,
                    **doc.metadata
                }
            )
            points.append(point)
        
        return self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def delete_documents(self, ids: list[str]):
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )

    def similarity_search(self, query_embedding: list[float], k: int = 5, filter_dict: dict = None):
        """Search using a pre-computed query embedding"""
        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=filter_dict
            )
            
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload.get('page_content', ''),
                    metadata={k: v for k, v in result.payload.items() if k != 'page_content'}
                )
                documents.append(doc)
            
            return documents  
        except Exception as e:
            raise Exception(f"Error during similarity search: {e}")

    def similarity_search_with_score(self, query_embedding: list[float], k: int = 5, filter_dict: dict = None):
        """Search using a pre-computed query embedding, with scores"""
        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=filter_dict
            )
            
            results = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload.get('page_content', ''),
                    metadata={k: v for k, v in result.payload.items() if k != 'page_content'}
                )
                results.append((doc, result.score))
            
            return results  
        except Exception as e:
            raise Exception(f"Error during similarity search with score: {e}")

    def as_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)


class ImageTopKAccuracyBenchmarkWithReranker:
    def __init__(self, vector_store: VectorStore, reranker_model_path: str = "AITeamVN/Vietnamese_Reranker"):
        self.vector_store = vector_store
        
        # Login to Hugging Face
        login(token=os.getenv("HF_TOKEN"))
        
        # Create embedding model for query encoding using local uploaded image model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="/bge-m3-image"
        )
        
        # Load Vietnamese reranker model from HuggingFace with proper tokenizer settings
        print(f"Loading Vietnamese reranker model from {reranker_model_path}...")
        
        try:
            # Try loading with use_fast=False to avoid fast tokenizer conversion issues
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                reranker_model_path,
                use_fast=False  # Use slow tokenizer to avoid conversion issues
            )
        except Exception as e:
            print(f"Failed to load with slow tokenizer, trying default: {e}")
            # Fallback to default loading
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
        
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.reranker_model.eval()
        print("Vietnamese reranker model loaded successfully!")
        
        # Store model path for results
        self.reranker_model_path = reranker_model_path
        self.MAX_LENGTH = 2304  # As specified in the Vietnamese_Reranker docs
    
    def load_image_qa_dataset(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load image Q&A dataset from JSONL file"""
        qa_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa_data.append(json.loads(line))
        return qa_data
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def rerank_documents(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Rerank documents using the Vietnamese reranker model"""
        if len(documents) <= top_k:
            return documents
        
        # Prepare query-document pairs for reranking (following the exact pattern from the docs)
        pairs = [[query, doc] for doc in documents]  # Note: using list of lists, not tuples
        
        # Tokenize all pairs using the exact pattern from Vietnamese_Reranker docs
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.MAX_LENGTH
            )
            
            # Move to device if using GPU
            if torch.cuda.is_available() and self.reranker_model.device.type == 'cuda':
                inputs = {k: v.to(self.reranker_model.device) for k, v in inputs.items()}
            
            # Get reranking scores using the exact pattern from docs
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        # Sort documents by reranking scores (descending)
        document_scores = list(zip(documents, scores.cpu().numpy()))
        document_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k reranked documents
        reranked_documents = [doc for doc, score in document_scores[:top_k]]
        return reranked_documents
    
    def retrieve_captions_only(self, context: str, k: int = 5) -> List[str]:
        """Retrieve captions using context as query, without reranking"""
        query_embedding = self.embedding_model.embed_query(context)
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding, 
            k=k
        )
        return [doc.page_content for doc in results]
    
    def retrieve_and_rerank_captions(self, context: str, initial_k: int = 10, final_k: int = 5) -> List[str]:
        """Retrieve captions using context as query, then rerank them"""
        # First, retrieve top-k captions from vector store using context
        query_embedding = self.embedding_model.embed_query(context)
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding, 
            k=initial_k
        )
        initial_captions = [doc.page_content for doc in results]
        
        # Then rerank and get top final_k
        reranked_captions = self.rerank_documents(context, initial_captions, top_k=final_k)
        return reranked_captions
    
    def check_caption_in_retrieved(self, ground_truth_caption: str, retrieved_captions: List[str]) -> bool:
        """Check if ground truth caption appears in retrieved captions"""
        normalized_gt = self.normalize_text(ground_truth_caption)
        
        for retrieved_caption in retrieved_captions:
            normalized_retrieved = self.normalize_text(retrieved_caption)
            
            # Check for exact match
            if normalized_gt == normalized_retrieved:
                return True
            
            # Check for substantial overlap (>80% of ground truth content)
            if len(normalized_gt) > 0:
                # Count overlapping words
                gt_words = set(normalized_gt.split())
                retrieved_words = set(normalized_retrieved.split())
                overlap = len(gt_words.intersection(retrieved_words))
                overlap_ratio = overlap / len(gt_words)
                
                if overlap_ratio >= 0.8:
                    return True
        
        return False
    
    def evaluate_sample(self, image_qa_item: Dict[str, Any], sample_index: int = None, use_reranker: bool = True) -> Dict[str, Any]:
        """Evaluate a single image Q&A sample for top-k accuracy with or without reranking"""
        question = image_qa_item.get('question', '')
        answer = image_qa_item.get('answer', '')
        context = image_qa_item.get('context', '')  # Use context as search query
        caption = image_qa_item.get('caption', '')  # Use caption as ground truth
        image_filename = image_qa_item.get('image_filename', '')
        
        if not context or not caption:
            error_msg = 'Missing context or caption'
            print(f"⚠️ Error Sample #{sample_index}: {error_msg}")
            print(f"   Context: {'✓' if context else '✗'} ({len(context)} chars)")
            print(f"   Caption: {'✓' if caption else '✗'} ({len(caption)} chars)")
            print(f"   Image: {image_filename}")
            return {
                'found': False, 
                'error': error_msg,
                'sample_index': sample_index,
                'context': context,
                'caption': caption,
                'image_filename': image_filename,
                'error_details': {
                    'missing_context': not context,
                    'missing_caption': not caption
                }
            }
        
        try:
            if use_reranker:
                # Retrieve 10 captions using context and rerank to get top 5
                retrieved_captions = self.retrieve_and_rerank_captions(
                    context, 
                    initial_k=10, 
                    final_k=5
                )
            else:
                # Just retrieve top 5 captions using context without reranking
                retrieved_captions = self.retrieve_captions_only(context, k=5)
            
            found = self.check_caption_in_retrieved(caption, retrieved_captions)
            
            return {
                'found': found,
                'context': context,  # The search query
                'ground_truth_caption': caption,  # What we're looking for
                'question': question,
                'answer': answer,
                'image_filename': image_filename,
                'retrieved_count': len(retrieved_captions),
                'error': None,
                'sample_index': sample_index,
                'used_reranker': use_reranker
            }
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Error Sample #{sample_index}: {error_msg}")
            print(f"   Context: {context[:100]}...")
            print(f"   Image: {image_filename}")
            print(f"   Exception: {type(e).__name__}: {error_msg}")
            return {
                'found': False, 
                'error': error_msg,
                'sample_index': sample_index,
                'context': context,
                'caption': caption,
                'image_filename': image_filename,
                'error_details': {
                    'exception_type': type(e).__name__,
                    'exception_message': error_msg
                }
            }
    
    def calculate_topk_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate top-k accuracy metrics"""
        valid_results = [r for r in results if r.get('error') is None]
        error_results = [r for r in results if r.get('error') is not None]
        
        if not valid_results:
            return {
                'accuracy': 0.0,
                'total_evaluated': 0,
                'found_count': 0,
                'error_count': len(error_results),
                'error_samples': error_results
            }
        
        found_count = sum(1 for r in valid_results if r['found'])
        accuracy = found_count / len(valid_results)
        
        return {
            'accuracy': accuracy,
            'total_evaluated': len(valid_results),
            'found_count': found_count,
            'error_count': len(error_results),
            'error_samples': error_results
        }
    
    def benchmark_comparison(self, image_qa_jsonl_path: str) -> Dict[str, Any]:
        """Run benchmark comparison: Top-5 without reranker vs Top-5 with reranker (retrieve 10, rerank to 5)"""
        print(f"Loading image Q&A dataset from {image_qa_jsonl_path}...")
        image_qa_dataset = self.load_image_qa_dataset(image_qa_jsonl_path)
        print(f"Loaded {len(image_qa_dataset)} image Q&A pairs")
        
        print(f"\nSearch Method: Context → Related Captions")
        print(f"Evaluation: Check if ground truth caption is retrieved")
        
        # Test 1: Top-5 without reranker (baseline)
        print(f"\n" + "="*60)
        print("BASELINE: Top-5 Caption Retrieval WITHOUT Reranker")
        print("="*60)
        baseline_results = []
        
        for idx, image_qa_item in enumerate(tqdm(image_qa_dataset, desc="Evaluating baseline (context→captions, no reranker)")):
            result = self.evaluate_sample(image_qa_item, sample_index=idx, use_reranker=False)
            baseline_results.append(result)
        
        baseline_metrics = self.calculate_topk_accuracy(baseline_results)
        print(f"Baseline Top-5 Accuracy (context→captions, no reranker): {baseline_metrics['accuracy']:.4f} ({baseline_metrics['found_count']}/{baseline_metrics['total_evaluated']})")
        
        # Test 2: Top-5 with reranker (retrieve 10, rerank to 5)
        print(f"\n" + "="*60)
        print("WITH RERANKER: Context→Retrieve 10 Captions→Rerank to Top-5")
        print("="*60)
        reranker_results = []
        
        for idx, image_qa_item in enumerate(tqdm(image_qa_dataset, desc="Evaluating with Vietnamese reranker (context→captions)")):
            result = self.evaluate_sample(image_qa_item, sample_index=idx, use_reranker=True)
            reranker_results.append(result)
        
        reranker_metrics = self.calculate_topk_accuracy(reranker_results)
        print(f"Top-5 Accuracy (context→captions, with Vietnamese reranker): {reranker_metrics['accuracy']:.4f} ({reranker_metrics['found_count']}/{reranker_metrics['total_evaluated']})")
        
        # Calculate improvement
        improvement = reranker_metrics['accuracy'] - baseline_metrics['accuracy']
        improvement_pct = (improvement / baseline_metrics['accuracy'] * 100) if baseline_metrics['accuracy'] > 0 else 0
        
        print(f"\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Baseline (Context→Top-5 Captions):     {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
        print(f"With Reranker (Context→10→5 Captions): {reranker_metrics['accuracy']:.4f} ({reranker_metrics['accuracy']*100:.2f}%)")
        print(f"Improvement:                           {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        # Log errors for both approaches
        if baseline_metrics['error_count'] > 0:
            print(f"\n⚠️  Baseline errors: {baseline_metrics['error_count']}")
        if reranker_metrics['error_count'] > 0:
            print(f"⚠️  Reranker errors: {reranker_metrics['error_count']}")
        
        # Prepare final benchmark results
        benchmark_results = {
            'dataset_info': {
                'total_questions': len(image_qa_dataset),
                'dataset_path': image_qa_jsonl_path,
                'dataset_type': 'image_qa_with_captions'
            },
            'search_explanation': {
                'query_field': 'context',
                'target_field': 'caption',
                'description': 'Use context as query to find captions, check if ground truth caption is retrieved'
            },
            'baseline_setup': {
                'method': 'context_to_top_5_captions',
                'k': 5,
                'reranker_used': False
            },
            'reranking_setup': {
                'method': 'context_to_retrieve_10_rerank_to_5_captions',
                'initial_retrieval_k': 10,
                'final_reranked_k': 5,
                'reranker_model_path': self.reranker_model_path,
                'reranker_model_name': 'AITeamVN/Vietnamese_Reranker',
                'max_length': self.MAX_LENGTH
            },
            'baseline_metrics': baseline_metrics,
            'reranker_metrics': reranker_metrics,
            'improvement': {
                'absolute': improvement,
                'percentage': improvement_pct
            },
            'vector_store_info': {
                'collection_name': self.vector_store.collection_name,
                'embedding_model': '/bge-m3-image'
            }
        }
        
        return benchmark_results


@app.function(
    gpu="A100",
    timeout=43200,  # 12 hours
    memory=32768,   # 32GB RAM
)
def run_benchmark_comparison():
    """Main function to run the comparison benchmark on Modal"""
    
    # Login to Hugging Face at the start
    login(token="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")
    
    # Initialize vector store for image captions
    print("Initializing Image Vector Store...")
    vector_store = VectorStore(
        collection_name="image-captions-store", 
        delete_existing=False
    )
    
    # Initialize benchmark with Vietnamese reranker for image data
    print("Initializing Image Caption Retrieval Benchmark with Vietnamese Reranker...")
    benchmark = ImageTopKAccuracyBenchmarkWithReranker(vector_store)
    
    # Run comparison benchmark
    print("Running caption retrieval comparison benchmark (baseline vs reranker)...")
    results = benchmark.benchmark_comparison(
        image_qa_jsonl_path="/datasets/image_question_mappings_test_updated.jsonl"
    )
    
    return results


@app.function()
def save_results(results: Dict[str, Any]):
    """Save benchmark results to a file"""
    output_data = json.dumps(results, ensure_ascii=False, indent=2)
    
    print(f"Caption Retrieval Comparison Benchmark Results:")
    print(output_data)
    
    return output_data


@app.function(
    gpu="A100",
    timeout=43200,
    memory=32768,
)
def run_complete_benchmark():
    """Run the complete comparison benchmark workflow and save results"""
    
    print("🚀 Starting Image Caption Retrieval Benchmark Comparison (Baseline vs Reranker) on Modal...")
    
    # Run the comparison benchmark
    results = run_benchmark_comparison.remote()
    
    # Save and display results
    output = save_results.remote(results)
    
    print("\n✅ Caption retrieval comparison benchmark completed successfully!")
    return results
