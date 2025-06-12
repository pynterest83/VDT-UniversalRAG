import json 
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from langchain_core.documents import Document
import os 
from tqdm import tqdm
import uuid 

load_dotenv(dotenv_path=".env")

class VectorStore: 

    def __init__(self, collection_name: str = "universal-rag", delete_existing: bool = False):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bge-m3-v3"
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


if __name__ == "__main__": 
    # Example usage with JSONL file
    jsonl_file = "datasets/context_corpus_embedded_clean_2.jsonl"
    vector_store = VectorStore(collection_name="universal-rag-precomputed-clean-2", delete_existing=True)
    vector_store.load_documents_from_jsonl(jsonl_file)
    
    # Embedding model is now created outside the vector store for querying
    embedding_model = HuggingFaceEmbeddings(
        model_name="bge-m3-v3"
    )

    test_queries = [
        "Có bắt buộc phải sử dụng thuốc Carbogast trong khi mang thai và cho con bú không?"
    ]
    
    for query in test_queries:
        query_embedding = embedding_model.embed_query(query)
        results = vector_store.similarity_search_with_score(query_embedding=query_embedding, k=10)
        for doc, score in results:
            print(doc.page_content)
            print(score)
            print("-"*100)
