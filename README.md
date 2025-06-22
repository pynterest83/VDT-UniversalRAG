# VDT Universal RAG - Vietnamese Medical Question Answering System

A comprehensive Retrieval-Augmented Generation (RAG) system specifically designed for Vietnamese medical content, featuring multimodal capabilities with text and image understanding, advanced reranking, and benchmarking tools.

## 🌟 Key Features

- **Vietnamese Medical RAG**: Specialized for Vietnamese healthcare content with medical terminology understanding
- **Multimodal Support**: Text and image retrieval with enhanced caption embeddings
- **Advanced Reranking**: BGE-M3 reranker for improved context relevance
- **Real-time Image Generation**: AI-powered medical image generation based on answers
- **Comprehensive Benchmarking**: Top-K accuracy evaluation for both text and image retrieval
- **Streamlit Web Interface**: User-friendly chat interface with Vietnamese support
- **Production-Ready Serving**: Modal-based microservices architecture for scalable deployment

## 🏗️ Architecture

### Core Components

1. **Pipeline Engine** (`pipeline.py`): LangGraph-based processing workflow
2. **Vector Stores** (`vectorstore.py`): Qdrant-based document and image storage
3. **Embedding Models**: 
   - BGE-M3-v3 for text content embedding
   - BGE-M3-image for image caption embedding
4. **Reranker** (`reranker.py`): BGE-M3 reranker for document ranking
5. **Serving Layer** (`serving.py`): Modal-based API endpoints
6. **Web Interface** (`app.py`): Streamlit chat application

### Data Flow

```
User Question → Context Embedding → Vector Search → Reranking → 
Context Selection → Answer Generation → Image Search → 
Image Generation (optional) → Final Response
```

## 📋 Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for embeddings and reranking)
- Qdrant vector database
- Azure OpenAI API access
- Modal account (for cloud deployment)

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd VDT-UniversalRAG
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration:**
Create a `.env` file with the following variables:
```env
# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Azure OpenAI Configuration
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_azure_api_key
AZURE_VERSION=your_api_version

# Azure Image Generation
AZURE_IMAGE_API_KEY=your_image_api_key
AZURE_IMAGE_VERSION=your_image_api_version
AZURE_IMAGE_ENDPOINT=your_image_endpoint

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_key
```

## 📊 Dataset Structure

The system works with several dataset formats:

### Text Corpus
- **Location**: `datasets/context_corpus_embedded_enhanced.jsonl`
- **Format**: Pre-computed embeddings with metadata
```json
{
  "chunk_id": "unique_id",
  "content": "Vietnamese medical content",
  "embedding": [float_vector],
  "metadata": {
    "title": "Document title",
    "keyword": "Medical topic",
    "source": "youmed.vn"
  }
}
```

### Q&A Datasets
- **Training**: `datasets/q_a_train_filtered.jsonl`
- **Testing**: `datasets/q_a_test_filtered.jsonl`
- **Validation**: `datasets/q_a_validation_filtered.jsonl`

### Image Datasets
- **Image-Question Mappings**: `datasets/image_question_mappings_*.jsonl`
- **Caption Embeddings**: `datasets/caption_embeddings.jsonl`

## 🔧 Usage

### Running the Web Interface

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501` for an interactive Vietnamese medical Q&A experience.

### Programmatic Usage

```python
from pipeline import create_app_rag_graph

# Initialize the RAG graph
graph = create_app_rag_graph()

# Process a question
state = {
    "question": "Thuốc Paracetamol có tác dụng gì?"
}

# Run the pipeline
result = graph.invoke(state)
print(result["final_answer"])
```

### Vector Store Setup

```python
from vectorstore import VectorStore

# Initialize vector store
vector_store = VectorStore(
    collection_name="universal-rag-precomputed-enhanced"
)

# Load pre-computed embeddings
vector_store.load_documents_from_jsonl(
    "datasets/context_corpus_embedded_enhanced.jsonl"
)
```

## 🧪 Benchmarking

### Text Retrieval Benchmark

```bash
python benchmark.py
```

Evaluates Top-K accuracy for text retrieval using various K values (1, 3, 5, 10, 20).

### Image Retrieval Benchmark

```bash
python image_benchmark.py
```

Evaluates multimodal retrieval performance for image-question pairs.

### Reranker Benchmark

```bash
python benchmark_reranker.py
```

Compares performance with and without reranking.

## 🏭 Production Deployment

### Modal Cloud Deployment

Deploy the serving layer to Modal:

```bash
modal deploy serving.py
```

This provides scalable API endpoints:
- `/embed-context`: Context embedding
- `/embed-image-caption`: Image caption embedding  
- `/rerank-documents`: Document reranking

### Local Serving

For local development:

```bash
python serving.py
```

## 🔬 Model Training & Fine-tuning

### Embedding Model Training

```bash
modal run embeddings.py::finetune_embeddings --num-train-epochs=4
```

### Reranker Training

```bash
modal run reranker.py::train_reranker_with_prebuilt_data --num-train-epochs=2
```

### Multimodal Model Training

```bash
modal run multimodal_embeddings.py::run_training --max-train-samples=5000
```

## 🔧 Configuration Options

### Chunking Strategy
- **Chunk Size**: 512 tokens (optimized for BGE-M3)
- **Overlap**: 128 tokens
- **Context Prefix**: Vietnamese medical context

### Retrieval Parameters
- **Initial Retrieval**: Top-20 documents
- **Reranking**: Top-5 documents
- **Final Selection**: Top-3 contexts

### Generation Settings
- **Model**: Azure OpenAI GPT-4
- **Temperature**: 0.1 (deterministic responses)
- **Max Tokens**: 1000

## 📈 Performance Metrics

Recent benchmark results:
- **Text Retrieval Top-5 Accuracy**: 85%+
- **Image Retrieval Top-5 Accuracy**: 78%+
- **Reranker Improvement**: +12% over baseline
- **Average Response Time**: <3 seconds

## 🛠️ Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
   - Check network connectivity

2. **Azure OpenAI Rate Limits**
   - Implement retry logic
   - Consider using multiple API keys

3. **CUDA Memory Issues**
   - Reduce batch sizes in training
   - Use gradient checkpointing

4. **Modal Deployment Issues**
   - Ensure Modal CLI is installed: `pip install modal`
   - Authenticate: `modal token set`

### Performance Optimization

- Use SSD storage for vector databases
- Enable GPU acceleration for embeddings
- Implement request batching for high throughput
- Use CDN for static assets (images)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Work

- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding): Multilingual embedding model
- [LangGraph](https://github.com/langchain-ai/langgraph): Workflow orchestration
- [Qdrant](https://qdrant.tech/): Vector database
- [Modal](https://modal.com/): Cloud compute platform

## 📞 Support

For questions and support:
- Create an issue in this repository
- Review the troubleshooting section
- Check the benchmark results in `results/` directory

---

**Note**: This system is designed specifically for Vietnamese medical content and may require adaptation for other languages or domains.
