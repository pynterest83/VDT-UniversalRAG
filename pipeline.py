import uuid
from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv
import requests
import json
import tiktoken
from openai import AzureOpenAI

load_dotenv()

from langgraph.graph import StateGraph, END, START
from vectorstore import VectorStore

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class GraphState(TypedDict):
    question: str
    initial_chunks: Optional[List[dict]]
    reranked_chunks: Optional[List[dict]]
    selected_contexts: Optional[List[dict]]
    answer: Optional[str]
    image_info: Optional[dict]
    final_answer: Optional[str]
    error: Optional[str]

# --- Modal API Configuration ---
MODAL_BASE_URL = "https://ise703--universal-rag-models"
MODAL_ENDPOINTS = {
    "context_embedding": f"{MODAL_BASE_URL}-embed-context.modal.run",
    "image_embedding": f"{MODAL_BASE_URL}-embed-image-caption.modal.run",
    "reranker": f"{MODAL_BASE_URL}-rerank-documents.modal.run"
}

# --- Initialize components (only local ones) ---
def initialize_components():
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_VERSION")
    )
    
    context_store = VectorStore(collection_name="universal-rag-precomputed-enhanced")
    image_store = VectorStore(collection_name="image-captions-store")
    
    return azure_client, context_store, image_store

azure_client, context_store, image_store = initialize_components()

# Initialize tiktoken encoder for GPT-4o
encoding = tiktoken.encoding_for_model("gpt-4o")

# --- Modal API Helper Functions ---
def get_context_embedding(text: str) -> List[float]:
    """Get context embedding from Modal endpoint"""
    try:
        response = requests.post(
            MODAL_ENDPOINTS["context_embedding"],
            json={"text": text},
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        raise Exception(f"Context embedding API error: {str(e)}")

def get_image_embedding(text: str) -> List[float]:
    """Get image embedding from Modal endpoint"""
    try:
        response = requests.post(
            MODAL_ENDPOINTS["image_embedding"],
            json={"text": text},
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        raise Exception(f"Image embedding API error: {str(e)}")

def rerank_documents_api(query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
    """Rerank documents using Modal endpoint"""
    try:
        response = requests.post(
            MODAL_ENDPOINTS["reranker"],
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k
            },
            timeout=240
        )
        response.raise_for_status()
        result = response.json()
        
        # Check if we got a successful reranking result
        if "reranked_documents" in result and result["reranked_documents"]:
            return result["reranked_documents"]
        elif "error" in result:
            print(f"Reranker returned error: {result['error']}")
            # Use fallback reranked_documents if available
            if "reranked_documents" in result:
                return result["reranked_documents"]
        
        # If no valid reranked documents, return original with default scores
        raise Exception("No valid reranked documents returned")
        
    except Exception as e:
        print(f"Reranker API error: {str(e)}")
        # Fallback: return original documents with default scores
        fallback_docs = []
        for i, doc in enumerate(documents[:top_k]):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = 1.0 - (i * 0.1)  # Descending scores
            fallback_docs.append(doc_copy)
        return fallback_docs

def llm_select_context(question: str, ranked_chunks: List[dict]) -> List[dict]:
    
    if not ranked_chunks:
        return []
    
    try:
        context_list = []
        total_context_length = 0
        
        for i, chunk in enumerate(ranked_chunks):
            content = chunk['content']
            total_context_length += len(content)
            context_list.append({
                "priority_order": i + 1,
                "rerank_score": chunk.get('rerank_score', 0),
                "content": content,
                "chunk_id": chunk.get('chunk_id', f"chunk_{i+1}"),
            })
        
        max_context_tokens = max(len(encoding.encode(chunk['content'])) for chunk in ranked_chunks) if ranked_chunks else 0
        max_tokens = max_context_tokens + 200
        
        prompt = f"""
# NHIỆM VỤ: PHÂN TÍCH VÀ LỰA CHỌN CONTEXT TỐI ƯU

## BỐI CẢNH
Bạn là một chuyên gia phân tích thông tin y tế với chuyên môn sâu về:
- Y học lâm sàng và chẩn đoán bệnh
- Dược học và tác dụng của thuốc
- Thảo dược và y học tự nhiên  
- Sinh lý bệnh và cơ chế bệnh tật
- Điều trị và phòng ngừa bệnh

**CÂU HỎI Y TẾ:**
{question}

**DANH SÁCH CONTEXT (theo thứ tự ưu tiên từ reranking model):**
{json.dumps(context_list, ensure_ascii=False, indent=2)}

## FORMAT TRẢ LỜI
Trả về ĐÚNG format JSON sau (không thêm text nào khác):

{{
    "selected_chunk_ids": ["chunk_X", "chunk_Y"],
    "reasoning": "Phân tích chi tiết",
    "confidence": 0.XX,
    "analysis": {{
        "primary_context": "chunk_X",
        "supplementary_contexts": ["chunk_Y"] hoặc [],
        "coverage_assessment": "Đánh giá mức độ đủ thông tin"
    }}
}}
"""
        
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia phân tích văn bản và lựa chọn context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(result_text)
            selected_chunk_ids = result.get("selected_chunk_ids", [])
            
            selected_contexts = []
            for chunk_id in selected_chunk_ids:
                for chunk in ranked_chunks:
                    if chunk.get('chunk_id') == chunk_id:
                        selected_contexts.append(chunk)
                        break
            
            return selected_contexts
            
        except json.JSONDecodeError:
            return [ranked_chunks[0]]
        
    except Exception as e:
        return [ranked_chunks[0]] if ranked_chunks else []

def llm_generate_answer(question: str, selected_contexts: List[dict]) -> str:
    if not selected_contexts:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này."
    
    try:
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx['content']}" for i, ctx in enumerate(selected_contexts)])
        
        prompt = f"""
# NHIỆM VỤ: TRẢ LỜI CÂU HỎI Y TẾ DỰA TRÊN CONTEXT

## VAI TRÒ VÀ CHUYÊN MÔN
Bạn là một chuyên gia y tế đa lĩnh vực với kiến thức sâu về y học lâm sàng, dược học, thảo dược, sinh lý bệnh.

## NGUYÊN TẮC TRẢ LỜI
1. **Chính xác khoa học**: Thông tin phải có cơ sở khoa học rõ ràng
2. **Dựa trên context**: Chỉ sử dụng thông tin có trong context được cung cấp
3. **NGẮN GỌN TỐI ĐA**: 20-80 từ, ưu tiên 30-50 từ
4. **Trực tiếp**: Trả lời thẳng vào vấn đề, không dài dòng

**CÂU HỎI CẦN TRẢ LỜI:**
{question}

**CONTEXT THAM KHẢO:**
{context_text}

## TRẢ LỜI (NGẮN GỌN):
"""
        
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y tế chuyên trả lời CỰC NGẮN GỌN và CHÍNH XÁC."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        context_preview = selected_contexts[0]['content'][:300] if selected_contexts else ""
        return f"Dựa trên thông tin tìm được: {context_preview}..."

def search_initial_context_node(state: GraphState) -> GraphState:
    question = state.get("question")
    
    try:
        # Use Modal API for context embedding
        query_embedding = get_context_embedding(question)
        results = context_store.similarity_search_with_score(query_embedding=query_embedding, k=10)
        
        if not results:
            return {**state, "error": "Không tìm thấy context phù hợp."}
        
        initial_chunks = []
        for doc, score in results:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "chunk_id": doc.metadata.get("chunk_id", str(uuid.uuid4()))
            }
            initial_chunks.append(chunk)
        
        return {**state, "initial_chunks": initial_chunks}
        
    except Exception as e:
        return {**state, "error": f"Lỗi khi tìm kiếm context: {str(e)}"}

def rerank_context_node(state: GraphState) -> GraphState:
    question = state.get("question")
    initial_chunks = state.get("initial_chunks", [])
    
    if state.get("error") or not initial_chunks:
        return state
    
    try:
        # Use Modal API for reranking
        reranked_chunks = rerank_documents_api(question, initial_chunks, top_k=5)
        return {**state, "reranked_chunks": reranked_chunks}
        
    except Exception as e:
        return {**state, "error": f"Lỗi khi rerank context: {str(e)}"}

def select_context_node(state: GraphState) -> GraphState:
    question = state.get("question")
    reranked_chunks = state.get("reranked_chunks", [])
    
    if state.get("error") or not reranked_chunks:
        return state
    
    try:
        selected_contexts = llm_select_context(question, reranked_chunks)
        return {**state, "selected_contexts": selected_contexts}
        
    except Exception as e:
        return {**state, "error": f"Lỗi khi lựa chọn context: {str(e)}"}

def generate_answer_node(state: GraphState) -> GraphState:
    question = state.get("question")
    selected_contexts = state.get("selected_contexts", [])
    
    if state.get("error"):
        return state
    
    answer = llm_generate_answer(question, selected_contexts)
    return {**state, "answer": answer}

def search_image_node(state: GraphState) -> GraphState:
    selected_contexts = state.get("selected_contexts", [])
    
    if state.get("error") or not selected_contexts:
        return state
    
    try:
        # Use Modal API for image embedding
        main_context = selected_contexts[0]['content']
        context_embedding = get_image_embedding(main_context)
        results = image_store.similarity_search_with_score(query_embedding=context_embedding, k=1)
        
        if results:
            doc, score = results[0]
            image_info = {
                "caption": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "image_name": doc.metadata.get("image_name"),
                "image_path": doc.metadata.get("image_path"), 
                "image_url": doc.metadata.get("image_url"),
                "source": doc.metadata.get("source"),
                "image_id": doc.metadata.get("image_id")
            }
        else:
            image_info = None
        
        return {**state, "image_info": image_info}
        
    except Exception as e:
        print(f"Warning: Image search failed: {str(e)}")
        return {**state, "image_info": None}

def finalize_answer_node(state: GraphState) -> GraphState:
    answer = state.get("answer")
    image_info = state.get("image_info")
    
    if state.get("error"):
        final_answer = f"Đã có lỗi xảy ra: {state.get('error')}"
    else:
        final_answer = answer
        
        if image_info:
            final_answer += f"\n\n🖼️ Ảnh minh họa: {image_info.get('caption', 'Ảnh liên quan')}"
            
            if image_info.get('image_name'):
                final_answer += f"\nTên ảnh: {image_info['image_name']}"
            if image_info.get('image_path'):
                final_answer += f"\nĐường dẫn: {image_info['image_path']}"
            if image_info.get('source'):
                final_answer += f"\nNguồn: {image_info['source']}"
            if image_info.get('image_url'):
                final_answer += f"\nURL: {image_info['image_url']}"
    
    return {**state, "final_answer": final_answer}

def create_enhanced_rag_graph() -> "CompiledGraph":
    workflow = StateGraph(GraphState)

    workflow.add_node("search_initial_context", search_initial_context_node)
    workflow.add_node("rerank_context", rerank_context_node)
    workflow.add_node("select_context", select_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("search_image", search_image_node)
    workflow.add_node("finalize_answer", finalize_answer_node)

    workflow.set_entry_point("search_initial_context")
    workflow.add_edge("search_initial_context", "rerank_context")
    workflow.add_edge("rerank_context", "select_context")
    workflow.add_edge("select_context", "generate_answer")
    workflow.add_edge("generate_answer", "search_image")
    workflow.add_edge("search_image", "finalize_answer")
    workflow.add_edge("finalize_answer", END)

    return workflow.compile()

# Test the updated pipeline
if __name__ == "__main__":
    graph = create_enhanced_rag_graph()
    test_question = "Paracetamol có tác dụng gì?"
    result = graph.invoke({"question": test_question})
    print(result["final_answer"])