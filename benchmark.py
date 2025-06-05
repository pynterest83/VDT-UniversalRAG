import json
import asyncio
from typing import List, Dict, Any
from tqdm import tqdm
from vectorstore import VectorStore
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

class RAGASBenchmark:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
        )
        
        self.context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
        self.context_recall = LLMContextRecall(llm=evaluator_llm)
    
    def load_qa_dataset(self, jsonl_path: str) -> List[Dict[str, Any]]:
        qa_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa_data.append(json.loads(line))
        return qa_data
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[str]:
        results = self.vector_store.similarity_search(query=query, k=k)
        return [doc.page_content for doc in results]
    
    async def evaluate_sample(self, qa_item: Dict[str, Any], k: int = 5) -> Dict[str, float]:
        question = qa_item.get('question', '')
        answer = qa_item.get('answer', '')
        context = qa_item.get('context', '')
        
        retrieved_contexts = self.retrieve_documents(question, k=k)
        
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=context,
            retrieved_contexts=retrieved_contexts
        )
        
        precision_score = await self.context_precision.single_turn_ascore(sample)
        recall_score = await self.context_recall.single_turn_ascore(sample)
        
        return {
            'context_precision': precision_score,
            'context_recall': recall_score
        }
    
    async def benchmark(self, qa_jsonl_path: str, k_values: List[int] = [5, 10]) -> Dict[str, Any]:
        qa_dataset = self.load_qa_dataset(qa_jsonl_path)
        
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            
            for qa_item in tqdm(qa_dataset, desc=f"Evaluating k={k}"):
                try:
                    scores = await self.evaluate_sample(qa_item, k=k)
                    precision_scores.append(scores['context_precision'])
                    recall_scores.append(scores['context_recall'])
                except Exception as e:
                    continue
            
            results[f"k_{k}"] = {
                'context_precision_mean': sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
                'context_recall_mean': sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
                'samples_evaluated': len(precision_scores)
            }
        
        benchmark_results = {
            'dataset_info': {
                'total_questions': len(qa_dataset),
                'dataset_path': qa_jsonl_path
            },
            'ragas_metrics': results,
            'vector_store_info': {
                'index_name': self.vector_store.index_name,
                'embedding_model': str(type(self.vector_store.embedding_model).__name__)
            }
        }
        
        return benchmark_results

async def main():
    vector_store = VectorStore(
        index_name="universal-rag-google-paragraphs",
        provider="google",
        delete_existing=False
    )
    
    benchmark = RAGASBenchmark(vector_store)
    
    results = await benchmark.benchmark(
        qa_jsonl_path="datasets/q_a_test_filtered.jsonl",
        k_values=[5, 20]
    )
    
    print("\n=== RAGAS Benchmark Results ===")
    for k_setting, metrics in results['ragas_metrics'].items():
        print(f"\n{k_setting}:")
        print(f"  Context Precision: {metrics['context_precision_mean']:.4f}")
        print(f"  Context Recall: {metrics['context_recall_mean']:.4f}")
        print(f"  Samples: {metrics['samples_evaluated']}")
    
    with open('ragas_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
