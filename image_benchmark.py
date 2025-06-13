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

class ImageTopKAccuracyBenchmark:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        # Create embedding model for query encoding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bge-m3-image"
        )
    
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
    
    def evaluate_sample(self, image_qa_item: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
        """Evaluate a single image Q&A sample for top-k accuracy"""
        question = image_qa_item.get('question', '')
        answer = image_qa_item.get('answer', '')
        context = image_qa_item.get('context', '')  # Use context as search query
        caption = image_qa_item.get('caption', '')  # Use caption as ground truth
        image_filename = image_qa_item.get('image_filename', '')
        
        if not context or not caption:
            return {
                'found': False, 
                'error': 'Missing context or caption',
                'context': context,
                'caption': caption,
                'image_filename': image_filename
            }
        
        try:
            # Use context as query to search for related captions
            retrieved_captions = self.retrieve_documents(context, k=k)
            found = self.check_caption_in_retrieved(caption, retrieved_captions)
            
            return {
                'found': found,
                'context': context,  # The search query
                'ground_truth_caption': caption,  # What we're looking for
                'question': question,
                'answer': answer,
                'image_filename': image_filename,
                'retrieved_count': len(retrieved_captions),
                'error': None
            }
        except Exception as e:
            return {
                'found': False, 
                'error': str(e),
                'context': context,
                'caption': caption,
                'image_filename': image_filename
            }
    
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
    
    def benchmark(self, image_qa_jsonl_path: str, k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, Any]:
        """Run top-k accuracy benchmark for image Q&A dataset"""
        print(f"Loading image Q&A dataset from {image_qa_jsonl_path}...")
        image_qa_dataset = self.load_image_qa_dataset(image_qa_jsonl_path)
        print(f"Loaded {len(image_qa_dataset)} image Q&A pairs")
        
        results = {}
        
        for k in k_values:
            print(f"\nEvaluating Top-{k} Accuracy...")
            print(f"Query: Context → Search for: Related Captions → Check: Ground Truth Caption")
            sample_results = []
            
            for image_qa_item in tqdm(image_qa_dataset, desc=f"Evaluating k={k}"):
                result = self.evaluate_sample(image_qa_item, k=k)
                sample_results.append(result)
            
            # Calculate metrics for this k value
            metrics = self.calculate_topk_accuracy(sample_results)
            results[f"top_{k}"] = metrics
            
            print(f"Top-{k} Accuracy: {metrics['accuracy']:.4f} ({metrics['found_count']}/{metrics['total_evaluated']})")
            
            # Show some error examples for debugging
            error_samples = [r for r in sample_results if r.get('error') is not None]
            if error_samples and k == k_values[0]:  # Only show for first k value
                print(f"\nError Examples (showing first 3 of {len(error_samples)}):")
                for i, error_sample in enumerate(error_samples[:3]):
                    print(f"  Error {i+1}: {error_sample['error']}")
                    print(f"    Context: {error_sample.get('context', 'N/A')[:100]}...")
                    print(f"    Caption: {error_sample.get('caption', 'N/A')[:100]}...")
                    print(f"    Image: {error_sample.get('image_filename', 'N/A')}")
        
        # Prepare final benchmark results
        benchmark_results = {
            'dataset_info': {
                'total_questions': len(image_qa_dataset),
                'dataset_path': image_qa_jsonl_path,
                'dataset_type': 'image_qa_with_captions',
                'search_method': 'context_to_caption_retrieval'
            },
            'search_explanation': {
                'query_field': 'context',
                'target_field': 'caption', 
                'description': 'Use context as query to find captions, check if ground truth caption is retrieved'
            },
            'topk_accuracy_metrics': results,
            'vector_store_info': {
                'collection_name': self.vector_store.collection_name,
                'embedding_model': str(type(self.vector_store.embedding_model).__name__)
            }
        }
        
        return benchmark_results


def main():
    # Initialize vector store
    print("Initializing Vector Store...")
    vector_store = VectorStore(
        collection_name="image-captions-store", 
        delete_existing=False
    )
    
    # Initialize benchmark
    benchmark = ImageTopKAccuracyBenchmark(vector_store)
    
    # Run benchmark on image dataset
    results = benchmark.benchmark(
        image_qa_jsonl_path="datasets/image_question_mappings_test_updated.jsonl",
        k_values=[5]
    )
    
    # Print results
    print("\n" + "="*50)
    print("IMAGE CAPTION RETRIEVAL BENCHMARK RESULTS")
    print("="*50)
    print("Search Method: Context → Related Captions")
    print("Evaluation: Check if ground truth caption is retrieved")
    
    for k_setting, metrics in results['topk_accuracy_metrics'].items():
        k_value = k_setting.replace('top_', '')
        print(f"\nTop-{k_value} Accuracy:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Found: {metrics['found_count']}/{metrics['total_evaluated']}")
        print(f"  Errors: {metrics['error_count']}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/image_caption_retrieval_results_bge_m3.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")
    

if __name__ == "__main__":
    main()
