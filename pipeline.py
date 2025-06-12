import uuid
from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

from langgraph.graph import StateGraph, END, START
from vectorstore import VectorStore

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class GraphState(TypedDict):
    """
    Định nghĩa cấu trúc dữ liệu cho toàn bộ pipeline.

    Attributes:
        question: Câu hỏi gốc từ người dùng.
        text_chunks: Danh sách các đoạn văn bản truy xuất được.
        answer: Câu trả lời được tạo từ context.
        caption_query: Query để tìm kiếm caption/ảnh.
        image_info: Thông tin ảnh tìm được.
        final_answer: Câu trả lời cuối cùng cho người dùng (text + ảnh).
        error: Thông báo lỗi nếu có.
    """
    question: str
    text_chunks: Optional[List[dict]]
    answer: Optional[str]
    caption_query: Optional[str]
    image_info: Optional[dict]
    final_answer: Optional[str]
    error: Optional[str]


# --- Khởi tạo các thành phần ---
def initialize_components():
    """Khởi tạo embedding model và vector stores"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="bge-m3-v3"
    )
    
    # Vector store cho context (văn bản)
    context_store = VectorStore(collection_name="universal-rag-precomputed-clean-2")
    
    # Vector store cho caption/ảnh (giả sử có collection khác cho ảnh)
    image_store = VectorStore(collection_name="universal-rag-precomputed-clean-2")
    
    return embedding_model, context_store, image_store

# Khởi tạo global components
embedding_model, context_store, image_store = initialize_components()

# --- Mock LLM functions ---
def mock_llm_generate_answer(question: str, context_chunks: List[dict]) -> str:
    """Giả lập LLM sinh câu trả lời từ context."""
    print("🤖 >> Giả lập LLM: Đang sinh câu trả lời từ context...")
    if not context_chunks:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này."
    
    context_text = "\n".join([chunk['content'] for chunk in context_chunks])
    answer = f"Dựa trên thông tin tìm được, để trả lời câu hỏi '{question}':\n\n{context_text}"
    return answer

def mock_llm_generate_caption_query(question: str, answer: str) -> str:
    """Giả lập LLM tạo query để tìm kiếm caption/ảnh."""
    print("🤖 >> Giả lập LLM: Đang tạo query tìm kiếm ảnh...")
    # Trong thực tế, LLM sẽ phân tích câu hỏi và câu trả lời để tạo query mô tả ảnh cần tìm
    return f"ảnh minh họa {question}"

def mock_llm_generate_final_answer(answer: str, image_info: Optional[dict]) -> str:
    """Giả lập LLM tạo câu trả lời cuối cùng kết hợp text và ảnh."""
    print("✨ >> Giả lập LLM: Đang tạo câu trả lời cuối cùng...")
    final_answer = answer
    
    if image_info:
        final_answer += f"\n\n🖼️ Ảnh minh họa: {image_info.get('caption', 'Ảnh liên quan')}"
        if image_info.get('image_url'):
            final_answer += f"\nURL: {image_info['image_url']}"
    
    return final_answer

# --- Định nghĩa các Node của Graph ---

def search_context_node(state: GraphState) -> GraphState:
    """Node 1: Tìm kiếm context từ vector store."""
    print("\n--- BƯỚC 1: TÌM KIẾM CONTEXT ---")
    question = state.get("question")
    
    try:
        # Embed query
        query_embedding = embedding_model.embed_query(question)
        
        # Tìm kiếm context với score
        results = context_store.similarity_search_with_score(
            query_embedding=query_embedding, 
            k=5
        )
        
        if not results:
            return {**state, "error": "Không tìm thấy context phù hợp."}
        
        # Chuyển đổi kết quả thành format phù hợp
        text_chunks = []
        for doc, score in results:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "chunk_id": doc.metadata.get("chunk_id", str(uuid.uuid4()))
            }
            text_chunks.append(chunk)
        
        print(f"✅ Tìm thấy {len(text_chunks)} đoạn context phù hợp")
        return {**state, "text_chunks": text_chunks}
        
    except Exception as e:
        return {**state, "error": f"Lỗi khi tìm kiếm context: {str(e)}"}

def generate_answer_node(state: GraphState) -> GraphState:
    """Node 2: Sinh câu trả lời từ context."""
    print("\n--- BƯỚC 2: SINH CÂU TRẢ LỜI ---")
    question = state.get("question")
    text_chunks = state.get("text_chunks", [])
    
    if state.get("error"):
        return state
    
    answer = mock_llm_generate_answer(question, text_chunks)
    return {**state, "answer": answer}

def generate_caption_query_node(state: GraphState) -> GraphState:
    """Node 3: Tạo query để tìm kiếm ảnh."""
    print("\n--- BƯỚC 3: TẠO QUERY TÌM KIẾM ẢNH ---")
    question = state.get("question")
    answer = state.get("answer")
    
    if state.get("error"):
        return state
    
    caption_query = mock_llm_generate_caption_query(question, answer)
    return {**state, "caption_query": caption_query}

def search_image_node(state: GraphState) -> GraphState:
    """Node 4: Tìm kiếm ảnh từ caption query."""
    print("\n--- BƯỚC 4: TÌM KIẾM ẢNH ---")
    caption_query = state.get("caption_query")
    
    if state.get("error") or not caption_query:
        return state
    
    try:
        # Embed caption query
        query_embedding = embedding_model.embed_query(caption_query)
        
        # Tìm kiếm trong image store
        results = image_store.similarity_search_with_score(
            query_embedding=query_embedding,
            k=1  # Chỉ lấy 1 ảnh đại diện
        )
        
        if results:
            doc, score = results[0]
            image_info = {
                "caption": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "image_url": doc.metadata.get("image_url"),  # Giả sử metadata chứa URL ảnh
                "image_id": doc.metadata.get("image_id")
            }
            print(f"✅ Tìm thấy ảnh phù hợp với score: {score}")
        else:
            print("⚠️ Không tìm thấy ảnh phù hợp")
            image_info = None
        
        return {**state, "image_info": image_info}
        
    except Exception as e:
        print(f"⚠️ Lỗi khi tìm kiếm ảnh: {str(e)}")
        return {**state, "image_info": None}

def finalize_answer_node(state: GraphState) -> GraphState:
    """Node 5: Tạo câu trả lời cuối cùng kết hợp text và ảnh."""
    print("\n--- BƯỚC 5: HOÀN THIỆN CÂU TRẢ LỜI ---")
    answer = state.get("answer")
    image_info = state.get("image_info")
    
    if state.get("error"):
        final_answer = f"Đã có lỗi xảy ra: {state.get('error')}"
    else:
        final_answer = mock_llm_generate_final_answer(answer, image_info)
    
    return {**state, "final_answer": final_answer}

# --- Xây dựng và Compile Graph ---

def create_enhanced_rag_graph() -> "CompiledGraph":
    """Tạo graph cho RAG pipeline mới."""
    workflow = StateGraph(GraphState)

    # Thêm các node vào graph
    workflow.add_node("search_context", search_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("generate_caption_query", generate_caption_query_node)
    workflow.add_node("search_image", search_image_node)
    workflow.add_node("finalize_answer", finalize_answer_node)

    # Kết nối các node theo flow tuần tự
    workflow.set_entry_point("search_context")
    workflow.add_edge("search_context", "generate_answer")
    workflow.add_edge("generate_answer", "generate_caption_query")
    workflow.add_edge("generate_caption_query", "search_image")
    workflow.add_edge("search_image", "finalize_answer")
    workflow.add_edge("finalize_answer", END)

    # Compile graph để sẵn sàng sử dụng
    return workflow.compile()

# --- Chạy Pipeline ---

if __name__ == "__main__":
    # Khởi tạo graph
    app = create_enhanced_rag_graph()

    # Định nghĩa câu hỏi đầu vào
    inputs = {"question": "Bách thảo sương dùng để chữa chảy máu kẽ răng như thế nào?"}
    
    # Chạy pipeline và xem các bước thực thi
    print("🚀 BẮT ĐẦU CHẠY ENHANCED RAG PIPELINE 🚀")
    print(f"📝 Câu hỏi: {inputs['question']}")
    
    for output in app.stream(inputs, stream_mode="values"):
        # `stream_mode="values"` sẽ trả về state sau mỗi bước
        print("\n" + "="*50)
        print("Trạng thái hiện tại của Graph:")
        
        # Hiển thị thông tin quan trọng
        if output.get("text_chunks"):
            print(f"📄 Context chunks: {len(output['text_chunks'])} đoạn")
        if output.get("answer"):
            print(f"💬 Answer: {output['answer'][:100]}...")
        if output.get("caption_query"):
            print(f"🔍 Caption query: {output['caption_query']}")
        if output.get("image_info"):
            print(f"🖼️ Image found: {output['image_info'].get('caption', 'N/A')}")
        if output.get("error"):
            print(f"❌ Error: {output['error']}")
            
        print("="*50)

    print("\n🏁 PIPELINE HOÀN TẤT! 🏁")
    print("\nCâu trả lời cuối cùng:")
    print("-" * 50)
    print(output.get("final_answer"))