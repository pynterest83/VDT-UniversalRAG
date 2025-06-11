import json
import asyncio
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from vectorstore import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import re
from collections import defaultdict

load_dotenv()

class TopKAccuracyBenchmark:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        # Create embedding model for query encoding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bge_m3_v2/checkpoint-780"
        )
    
    def load_qa_dataset(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load Q&A dataset from JSONL file"""
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
    
    def retrieve_documents_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents with similarity scores"""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.similarity_search_with_score(
            query_embedding=query_embedding, 
            k=k
        )
        return [(doc.page_content, score) for doc, score in results]
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-k documents"""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding, 
            k=k
        )
        return [doc.page_content for doc in results]
    
    def check_context_in_retrieved(self, ground_truth_context: str, retrieved_contexts: List[str]) -> bool:
        """Check if ground truth context appears in retrieved documents"""
        normalized_gt = self.normalize_text(ground_truth_context)
        
        for retrieved_context in retrieved_contexts:
            normalized_retrieved = self.normalize_text(retrieved_context)
            
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
    
    def evaluate_sample(self, qa_item: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
        """Evaluate a single Q&A sample for top-k accuracy"""
        question = qa_item.get('question', '')
        answer = qa_item.get('answer', '')
        ground_truth_context = qa_item.get('context', '')
        
        if not question or not ground_truth_context:
            return {'found': False, 'error': 'Missing question or context'}
        
        try:
            retrieved_contexts = self.retrieve_documents(question, k=k)
            found = self.check_context_in_retrieved(ground_truth_context, retrieved_contexts)
            
            return {
                'found': found,
                'question': question,
                'ground_truth_context': ground_truth_context,
                'retrieved_count': len(retrieved_contexts),
                'error': None
            }
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def calculate_topk_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate top-k accuracy metrics"""
        valid_results = [r for r in results if r.get('error') is None]
        
        if not valid_results:
            return {
                'accuracy': 0.0,
                'total_evaluated': 0,
                'found_count': 0,
                'error_count': len(results) - len(valid_results)
            }
        
        found_count = sum(1 for r in valid_results if r['found'])
        accuracy = found_count / len(valid_results)
        
        return {
            'accuracy': accuracy,
            'total_evaluated': len(valid_results),
            'found_count': found_count,
            'error_count': len(results) - len(valid_results)
        }
    
    def benchmark(self, qa_jsonl_path: str, k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, Any]:
        """Run top-k accuracy benchmark"""
        print(f"Loading Q&A dataset from {qa_jsonl_path}...")
        qa_dataset = self.load_qa_dataset(qa_jsonl_path)
        print(f"Loaded {len(qa_dataset)} Q&A pairs")
        
        results = {}
        
        for k in k_values:
            print(f"\nEvaluating Top-{k} Accuracy...")
            sample_results = []
            
            for qa_item in tqdm(qa_dataset, desc=f"Evaluating k={k}"):
                result = self.evaluate_sample(qa_item, k=k)
                sample_results.append(result)
            
            # Calculate metrics for this k value
            metrics = self.calculate_topk_accuracy(sample_results)
            results[f"top_{k}"] = metrics
            
            print(f"Top-{k} Accuracy: {metrics['accuracy']:.4f} ({metrics['found_count']}/{metrics['total_evaluated']})")
        
        # Prepare final benchmark results
        benchmark_results = {
            'dataset_info': {
                'total_questions': len(qa_dataset),
                'dataset_path': qa_jsonl_path
            },
            'topk_accuracy_metrics': results,
            'vector_store_info': {
                'collection_name': self.vector_store.collection_name,
                'embedding_model': str(type(self.vector_store.embedding_model).__name__)
            }
        }
        
        return benchmark_results
    
    def detailed_analysis(self, qa_jsonl_path: str, k: int = 10, sample_size: int = 10) -> None:
        """Perform detailed analysis on a sample of results"""
        qa_dataset = self.load_qa_dataset(qa_jsonl_path)
        sample_dataset = qa_dataset[:sample_size]
        
        print(f"\n=== Detailed Analysis (Top-{k}, {sample_size} samples) ===")
        
        for i, qa_item in enumerate(sample_dataset):
            print(f"\n--- Sample {i+1} ---")
            print(f"Question: {qa_item.get('question', '')[:100]}...")
            
            result = self.evaluate_sample(qa_item, k=k)
            
            if result.get('error'):
                print(f"Error: {result['error']}")
                continue
                
            print(f"Found in Top-{k}: {'✓' if result['found'] else '✗'}")
            print(f"Ground Truth Context: {qa_item.get('context', '')[:200]}...")
            
            # Show top-3 retrieved contexts
            retrieved_with_scores = self.retrieve_documents_with_scores(qa_item.get('question', ''), k=3)
            print("Top-3 Retrieved Contexts:")
            for j, (context, score) in enumerate(retrieved_with_scores):
                print(f"  {j+1}. (Score: {score:.4f}) {context[:150]}...")


def main():
    # Initialize vector store
    print("Initializing Vector Store...")
    vector_store = VectorStore(
        collection_name="universal-rag-precomputed", 
        delete_existing=False
    )
    
    # Initialize benchmark
    benchmark = TopKAccuracyBenchmark(vector_store)
    
    # Run benchmark
    results = benchmark.benchmark(
        qa_jsonl_path="datasets/q_a_test_filtered.jsonl",
        k_values=[5]
    )
    
    # Print results
    print("\n" + "="*50)
    print("TOP-K ACCURACY BENCHMARK RESULTS")
    print("="*50)
    
    for k_setting, metrics in results['topk_accuracy_metrics'].items():
        k_value = k_setting.replace('top_', '')
        print(f"\nTop-{k_value} Accuracy:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Found: {metrics['found_count']}/{metrics['total_evaluated']}")
        print(f"  Errors: {metrics['error_count']}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/topk_accuracy_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Run detailed analysis on a sample
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    benchmark.detailed_analysis(
        qa_jsonl_path="datasets/q_a_test_filtered.jsonl",
        k=10,
        sample_size=5
    )


if __name__ == "__main__":
    main()
