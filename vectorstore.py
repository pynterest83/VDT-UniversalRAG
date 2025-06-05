import json 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from langchain.schema import Document 
import os 
from tqdm import tqdm
import uuid 

load_dotenv(dotenv_path = ".env")

class VectorStore: 

    def __init__(self, index_name: str = "universal-rag"):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model = "models/text-embedding-004", 
            google_api_key = os.getenv("GEMINI_API_KEY")
        )

        self.index_name = index_name

        try:
            from elasticsearch import Elasticsearch
            es_client = Elasticsearch([os.getenv("ELASTIC_URL", "http://localhost:9200")])
            if es_client.indices.exists(index=self.index_name):
                es_client.indices.delete(index=self.index_name)
        except:
            pass

        self.elasticsearch_client = ElasticsearchStore(
            es_url=os.getenv("ELASTIC_URL", "http://localhost:9200"),
            index_name=self.index_name,
            embedding=self.embedding_model,
            distance_strategy="COSINE"
        )

    def load_documents_from_json(self, json_path: str): 
        with open(json_path, 'r') as file:
            data = json.load(file)

        documents = []
        uuids = []
        
        for item in tqdm(data, desc="Loading documents"):
            title = item.get('title', 'No Title')
            passage = item.get('passage', 'No Passage')

            doc = Document(
                page_content = f"{title}\n{passage}",
                metadata = {
                    'title': title,
                    'passage': passage,
                }
            )

            documents.append(doc)
            uuids.append(str(uuid.uuid4()))

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
    vector_store = VectorStore(index_name="test-index")
    
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="Tesla Model 3 has a battery capacity of 75 kWh for long range variant.",
            metadata={"source": "specs"}
        ),
        Document(
            page_content="The weather forecast shows sunny skies with temperature of 25°C.",
            metadata={"source": "weather"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "docs"}
        )
    ]

    doc_ids = vector_store.add_documents(documents)
    print(f"Added {len(doc_ids)} documents")

    results = vector_store.similarity_search(
        query="What is the battery capacity of Tesla Model 3?", 
        k=2
    )
    
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")

    scored_results = vector_store.similarity_search_with_score(
        query="LangChain framework",
        k=1
    )
    
    for doc, score in scored_results:
        print(f"Score: {score:.3f} - {doc.page_content}")
    
