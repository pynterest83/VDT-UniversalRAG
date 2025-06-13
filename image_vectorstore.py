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

class ImageVectorStore: 

    def __init__(self, collection_name: str = "image-caption-rag", delete_existing: bool = False):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bge-m3-image"
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

    def load_image_captions_from_jsonl(self, jsonl_path: str): 
        """Load image caption documents from JSONL file with pre-computed embeddings"""
        points = []
        total_docs = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        for line in tqdm(lines, desc="Loading image captions from JSONL"):
            item = json.loads(line.strip())
            
            chunk_id = item.get('chunk_id', str(uuid.uuid4()))
            caption = item.get('content', 'No Caption')  # Content field contains the caption
            metadata = item.get('metadata', {})
            embedding = item.get('embedding', None)
            
            # Use pre-computed embedding if available
            if embedding is None:
                print(f"Warning: No embedding found for chunk_id: {chunk_id}, skipping...")
                continue

            # Create point for Qdrant insertion with image-specific metadata
            point = PointStruct(
                id=chunk_id, 
                vector=embedding,
                payload={
                    'page_content': caption,
                    'chunk_id': chunk_id,
                    'image_path': metadata.get('image_path', ''),
                    'filename': metadata.get('filename', ''),
                    'original_caption': metadata.get('original_caption', ''),
                    'original_alt_text': metadata.get('original_alt_text', ''),
                    'original_title': metadata.get('original_title', ''),
                    'source_url': metadata.get('source_url', ''),
                    'file_size': metadata.get('file_size', 0),
                    'image_hash': metadata.get('image_hash', ''),
                    **{k: v for k, v in metadata.items() if k not in [
                        'image_path', 'filename', 'original_caption', 'original_alt_text', 
                        'original_title', 'source_url', 'file_size', 'image_hash'
                    ]}
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
        
        print(f"Successfully loaded {total_docs} image caption documents from {jsonl_path} into the vector store.")

    def add_image_documents(self, captions: list[str], metadata_list: list[dict] = None, ids: list[str] = None):
        """Add image caption documents with on-demand embedding computation"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(captions))]
        
        if metadata_list is None:
            metadata_list = [{}] * len(captions)
        
        points = []
        for caption, metadata, doc_id in zip(captions, metadata_list, ids):
            embedding = self.embedding_model.embed_query(caption)
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    'page_content': caption,
                    'chunk_id': doc_id,
                    'image_path': metadata.get('image_path', ''),
                    'filename': metadata.get('filename', ''),
                    'original_caption': metadata.get('original_caption', ''),
                    'original_alt_text': metadata.get('original_alt_text', ''),
                    'original_title': metadata.get('original_title', ''),
                    'source_url': metadata.get('source_url', ''),
                    'file_size': metadata.get('file_size', 0),
                    'image_hash': metadata.get('image_hash', ''),
                    **{k: v for k, v in metadata.items() if k not in [
                        'image_path', 'filename', 'original_caption', 'original_alt_text', 
                        'original_title', 'source_url', 'file_size', 'image_hash'
                    ]}
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

    def search_by_image_metadata(self, filename: str = None, source_url: str = None, k: int = 5):
        """Search for images by metadata fields"""
        filter_conditions = []
        
        if filename:
            filter_conditions.append({"key": "filename", "match": {"value": filename}})
        
        if source_url:
            filter_conditions.append({"key": "source_url", "match": {"value": source_url}})
        
        filter_dict = None
        if filter_conditions:
            filter_dict = {"must": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
        
        # For metadata-only search, we can use a dummy query vector or search all
        try:
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=k,
                scroll_filter=filter_dict
            )
            
            documents = []
            for result in search_results[0]:  # scroll returns (points, next_page_offset)
                doc = Document(
                    page_content=result.payload.get('page_content', ''),
                    metadata={k: v for k, v in result.payload.items() if k != 'page_content'}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            raise Exception(f"Error during metadata search: {e}")

    def as_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)


if __name__ == "__main__": 
    # Example usage with image caption JSONL file
    jsonl_file = "datasets/caption_embeddings.jsonl"
    image_vector_store = ImageVectorStore(collection_name="image-captions-store", delete_existing=False)
    # image_vector_store.load_image_captions_from_jsonl(jsonl_file)
    
    # Embedding model is now created outside the vector store for querying
    embedding_model = HuggingFaceEmbeddings(
        model_name="bge-m3-image"
    )

    test_queries = [
        "Bàn long sâm với thân thảo mảnh mai",
        "Rễ cây có hình dạng dài màu nâu đất",
        "thuốc có nhiều công dụng"
    ]
    
    print("Testing image caption similarity search:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        query_embedding = embedding_model.embed_query(query)
        results = image_vector_store.similarity_search_with_score(query_embedding=query_embedding, k=3)
        for doc, score in results:
            print(f"Score: {score:.4f}")
            print(f"Caption: {doc.page_content}")
            print(f"Image: {doc.metadata.get('filename', 'N/A')}")
            print(f"Source: {doc.metadata.get('source_url', 'N/A')}")
            print("-" * 50)
