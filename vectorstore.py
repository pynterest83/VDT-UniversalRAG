import json 
from dotenv import load_dotenv 
from langchain_elasticsearch import ElasticsearchStore
import os 
from tqdm import tqdm
import uuid 
from langchain_core.documents import Document

load_dotenv(dotenv_path = ".env")

def create_embedding_model(provider: str = "google", model_name: str = None, **kwargs):
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model_name or "models/text-embedding-004",
            google_api_key=kwargs.get('api_key') or os.getenv("GEMINI_API_KEY")
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            openai_api_key=kwargs.get('api_key') or os.getenv("OPENAI_API_KEY")
        )
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

class VectorStore: 

    def __init__(self, index_name: str = "universal-rag", 
                 provider: str = "google", 
                 model_name: str = None,
                 delete_existing: bool = False,
                 **embedding_kwargs):
        
        self.embedding_model = create_embedding_model(
            provider=provider,
            model_name=model_name,
            **embedding_kwargs
        )

        self.index_name = index_name

        if delete_existing:
            from elasticsearch import Elasticsearch
            es_client = Elasticsearch([os.getenv("ELASTIC_URL", "http://localhost:9200")])
            if es_client.indices.exists(index=self.index_name):
                es_client.indices.delete(index=self.index_name)

        self.elasticsearch_client = ElasticsearchStore(
            es_url=os.getenv("ELASTIC_URL", "http://localhost:9200"),
            index_name=self.index_name,
            embedding=self.embedding_model,
            distance_strategy="COSINE"
        )

    def load_documents_from_jsonl(self, jsonl_path: str): 
        documents = []
        uuids = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        for line in tqdm(lines, desc="Loading documents from JSONL"):
            item = json.loads(line.strip())
            
            content = item.get('content', 'No Content')
            metadata = item.get('metadata', {})
            
            chunk_id = item.get('chunk_id')
            if chunk_id:
                doc_id = chunk_id
            else:
                doc_id = str(uuid.uuid4())

            doc = Document(
                page_content=content,
                metadata=metadata
            )

            documents.append(doc)
            uuids.append(doc_id)

            if len(documents) >= 100: 
                self.elasticsearch_client.add_documents(
                    documents=documents, 
                    ids=uuids
                )
                documents = []
                uuids = []
        
        if documents: 
            self.elasticsearch_client.add_documents(
                documents=documents, 
                ids=uuids
            )

    def add_documents(self, documents: list[Document], ids: list[str] = None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        return self.elasticsearch_client.add_documents(documents=documents, ids=ids)

    def delete_documents(self, ids: list[str]):
        self.elasticsearch_client.delete(ids=ids)

    def similarity_search(self, query: str, k: int = 5, filter_dict: dict = None):
        return self.elasticsearch_client.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )

    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: dict = None):
        return self.elasticsearch_client.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )

    def as_retriever(self, **kwargs):
        return self.elasticsearch_client.as_retriever(**kwargs)


if __name__ == "__main__": 
    jsonl_file = "datasets/chunked_medical_paragraphs.jsonl"
    vector_store = VectorStore(index_name="universal-rag-google-paragraphs", delete_existing=False)
    # vector_store.load_documents_from_jsonl(jsonl_file)
    
    test_queries = [
        "Có bắt buộc phải sử dụng thuốc Carbogast trong khi mang thai và cho con bú không?"
    ]
    
    for query in test_queries:
        results = vector_store.similarity_search_with_score(query=query, k=10)
        for doc in results:
            print(doc[0].page_content)
            print(doc[1])
            print("-"*100)

    # Google (default)
    # vector_store = VectorStore(index_name="test-index")
    
    # OpenAI
    # vector_store = VectorStore(
    #     index_name="test-index", 
    #     provider="openai", 
    #     model_name="text-embedding-3-large"
    # )
    
    # HuggingFace
    # vector_store = VectorStore(
    #     index_name="test-index", 
    #     provider="huggingface", 
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )
    
    # Custom model
    # vector_store = VectorStore(
    #     index_name="test-index", 
    #     provider="custom", 
    #     model_name="/path/to/your/model"
    # )
    
