import uuid
from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END, START

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class GraphState(TypedDict):
    """
    Định nghĩa cấu trúc dữ liệu cho toàn bộ pipeline.

    Attributes:
        question: Câu hỏi gốc từ người dùng.
        entities: Các thực thể (keyword, topic) được LLM trích xuất.
        text_chunks: Danh sách các đoạn văn bản truy xuất được.
        image_url: URL của hình ảnh được tìm thấy hoặc sinh ra.
        final_answer: Câu trả lời cuối cùng cho người dùng.
        error: Thông báo lỗi nếu có.
    """
    question: str
    entities: Optional[dict]
    text_chunks: Optional[List[dict]]
    image_url: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]


# --- 2. Các hàm giả lập (Mock Functions) ---
# Thay thế các hàm này bằng các lời gọi API/logic thực tế của bạn.

def mock_llm_extract_entities(question: str) -> dict:
    """Giả lập việc gọi LLM để trích xuất entities."""
    print("🤖 >> Giả lập LLM: Đang trích xuất entities...")
    if "bách thảo sương" in question.lower():
        return {"keyword": "Bách thảo sương", "topic": "chảy máu chân răng"}
    return {"keyword": "unknown", "topic": "general"}

def mock_text_hybrid_search(question: str, entities: dict, k: int = 3) -> List[dict]:
    """Giả lập việc tìm kiếm văn bản kết hợp (vector + filter)."""
    print(f"📄 >> Giả lập VectorStore: Đang tìm kiếm văn bản với filter '{entities}'...")
    # Trong thực tế, bạn sẽ dùng entities để tạo metadata filter
    return [
        {
            "chunk_id": "chunk_xyz_789",
            "content": "Để chữa chứng kẽ răng ra máu, dùng một ít Bách thảo sương làm ra bột, xát trực tiếp vào chân răng.",
            "metadata": {"keyword": "Bách thảo sương", "source": "Tập giản phương"}
        }
    ]

def mock_image_lookup_by_chunk_id(chunk_id: str) -> Optional[str]:
    """Giả lập việc tra cứu ảnh bằng chunk_id."""
    print(f"🖼️ >> Giả lập VectorStore: Đang tra cứu ảnh với chunk_id '{chunk_id}'...")
    if chunk_id == "chunk_xyz_789":
        # Giả sử tìm thấy ảnh tương ứng
        return "https://example.com/images/bach_thao_suong_on_gums.jpg"
    # Giả sử không tìm thấy
    return None

def mock_image_search_by_entities(entities: dict) -> Optional[str]:
    """Giả lập việc tìm kiếm ảnh bằng entities khi tra cứu thất bại."""
    print(f"🖼️ >> Giả lập VectorStore: Tra cứu thất bại, chuyển sang tìm kiếm ảnh bằng entities '{entities}'...")
    # Tạo query giàu mô tả từ entities và tìm kiếm vector
    return "https://example.com/images/generic_medicinal_powder.jpg"

def mock_llm_generate_final_answer(question: str, context: List[dict], image_url: Optional[str]) -> str:
    """Giả lập LLM đa phương thức sinh câu trả lời cuối cùng."""
    print("✨ >> Giả lập LLM: Đang tổng hợp và sinh câu trả lời cuối cùng...")
    text_context = "\n".join([chunk['content'] for chunk in context])
    answer = f"Để trả lời câu hỏi '{question}', bạn có thể tham khảo thông tin sau: {text_context}."
    if image_url:
        answer += f"\n\nẢnh minh họa: {image_url}"
    return answer

# --- 3. Định nghĩa các Node của Graph ---

def analyze_query_node(state: GraphState) -> GraphState:
    """Node 1: Phân tích câu hỏi và trích xuất entities."""
    print("\n--- BƯỚC 1: PHÂN TÍCH QUERY ---")
    question = state.get("question")
    entities = mock_llm_extract_entities(question)
    return {**state, "entities": entities}

def enhanced_text_retrieval_node(state: GraphState) -> GraphState:
    """Node 2: Truy xuất văn bản tăng cường."""
    print("\n--- BƯỚC 2: TRUY XUẤT VĂN BẢN ---")
    question = state.get("question")
    entities = state.get("entities")
    text_chunks = mock_text_hybrid_search(question, entities)
    if not text_chunks:
        return {**state, "error": "Không tìm thấy văn bản nào phù hợp."}
    return {**state, "text_chunks": text_chunks}

def image_lookup_node(state: GraphState) -> GraphState:
    """Node 3: Tra cứu ảnh bằng chunk_id (Ưu tiên 1)."""
    print("\n--- BƯỚC 3.1: TRUY XUẤT ẢNH (ƯU TIÊN 1 - DÙNG CHUNK_ID) ---")
    text_chunks = state.get("text_chunks")
    if not text_chunks:
        return {**state, "image_url": None} # Bỏ qua nếu không có text
    
    top_chunk_id = text_chunks[0].get("chunk_id")
    image_url = mock_image_lookup_by_chunk_id(top_chunk_id)
    return {**state, "image_url": image_url}

def image_search_node(state: GraphState) -> GraphState:
    """Node 4: Tìm kiếm ảnh bằng entities (Phương án 2)."""
    print("\n--- BƯỚC 3.2: TRUY XUẤT ẢNH (PHƯƠNG ÁN 2 - DÙNG ENTITIES) ---")
    entities = state.get("entities")
    image_url = mock_image_search_by_entities(entities)
    return {**state, "image_url": image_url}
    
def generate_answer_node(state: GraphState) -> GraphState:
    """Node 5: Sinh câu trả lời cuối cùng."""
    print("\n--- BƯỚC 4: SINH CÂU TRẢ LỜI ---")
    question = state.get("question")
    text_chunks = state.get("text_chunks")
    image_url = state.get("image_url")
    
    if state.get("error"):
         final_answer = f"Đã có lỗi xảy ra: {state.get('error')}"
    else:
        final_answer = mock_llm_generate_final_answer(question, text_chunks, image_url)
    
    return {**state, "final_answer": final_answer}

# --- 4. Định nghĩa các cạnh và điều kiện rẽ nhánh (Edges) ---

def should_fallback_to_image_search(state: GraphState) -> str:
    """
    Hàm quyết định: Nếu đã tìm thấy ảnh bằng chunk_id thì đi đến bước sinh câu trả lời.
    Nếu không, thực hiện tìm kiếm ảnh bằng entities.
    """
    print("--- KIỂM TRA ĐIỀU KIỆN RẼ NHÁNH ---")
    if state.get("image_url"):
        print("✅ Đã tìm thấy ảnh bằng chunk_id. Chuyển đến bước sinh câu trả lời.")
        return "generate_answer"
    else:
        print("⚠️ Không tìm thấy ảnh. Chuyển sang phương án 2: tìm kiếm bằng entities.")
        return "image_search"

# --- 5. Xây dựng và Compile Graph ---

def create_smart_rag_graph() -> "CompiledGraph":
    workflow = StateGraph(GraphState)

    # Thêm các node vào graph
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("enhanced_text_retrieval", enhanced_text_retrieval_node)
    workflow.add_node("image_lookup", image_lookup_node)
    workflow.add_node("image_search", image_search_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Kết nối các node
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "enhanced_text_retrieval")
    workflow.add_edge("enhanced_text_retrieval", "image_lookup")
    
    # Thêm cạnh điều kiện sau bước tra cứu ảnh
    workflow.add_conditional_edges(
        "image_lookup",
        should_fallback_to_image_search,
        {
            "image_search": "image_search",
            "generate_answer": "generate_answer",
        }
    )
    
    workflow.add_edge("image_search", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile graph để sẵn sàng sử dụng
    return workflow.compile()

# --- 6. Chạy Pipeline ---

if __name__ == "__main__":
    # Khởi tạo graph
    app = create_smart_rag_graph()

    # Định nghĩa câu hỏi đầu vào
    inputs = {"question": "Bách thảo sương dùng để chữa chảy máu kẽ răng như thế nào?"}
    
    # Chạy pipeline và xem các bước thực thi
    print("🚀 BẮT ĐẦU CHẠY PIPELINE RAG ĐA PHƯƠNG THỨC 🚀")
    for output in app.stream(inputs, stream_mode="values"):
        # `stream_mode="values"` sẽ trả về state sau mỗi bước
        print("\n" + "="*50)
        print("Trạng thái hiện tại của Graph:")
        print(output)
        print("="*50)

    print("\n🏁 PIPELINE HOÀN TẤT! 🏁")
    print("\nCâu trả lời cuối cùng là:")
    print(output.get("final_answer"))