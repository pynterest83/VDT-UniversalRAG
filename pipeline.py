import uuid
from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv
import requests
import json
import tiktoken
from openai import AzureOpenAI
import base64
from PIL import Image
from io import BytesIO

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
    available_images: Optional[List[dict]]
    selected_image: Optional[dict]
    generated_image: Optional[dict]
    final_answer: Optional[str]
    error: Optional[str]
    user_choice: Optional[str]

MODAL_BASE_URL = "https://ise703--universal-rag-models"
MODAL_ENDPOINTS = {
    "context_embedding": f"{MODAL_BASE_URL}-embed-context.modal.run",
    "image_embedding": f"{MODAL_BASE_URL}-embed-image-caption.modal.run",
    "reranker": f"{MODAL_BASE_URL}-rerank-documents.modal.run"
}

def initialize_components():
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_VERSION")
    )

    image_client = AzureOpenAI(
        api_key=os.getenv("AZURE_IMAGE_API_KEY"),
        api_version=os.getenv("AZURE_IMAGE_VERSION"),
        azure_endpoint=os.getenv("AZURE_IMAGE_ENDPOINT")
    )
    
    context_store = VectorStore(collection_name="universal-rag-precomputed-enhanced")
    image_store = VectorStore(collection_name="image-captions-store")
    
    return azure_client, image_client, context_store, image_store

azure_client, image_client, context_store, image_store = initialize_components()
encoding = tiktoken.encoding_for_model("gpt-4o")

def create_image_prompt(question: str, context: str, answer: str):
    return f"""
Generate a realistic, high-quality, professional medical image that visually represents the following information. The image should be suitable for a medical consultation or educational material.

**Main Answer to Illustrate:**
{answer}

**Full Context for Detail:**
- **User's Question:** {question}
- **Supporting Information:** {context}

**Guidelines:**
- **Style**: Realistic photography. Avoid illustrations, cartoons, or abstract art.
- **Focus**: The main goal is to visually represent the **Main Answer**. Use the question and supporting information to add accurate details.
- **Clarity**: The image must be clear, professional, and easy to understand.
- **Setting**: A clean, modern medical environment like a clinic, hospital, or pharmacy.
- **Important**: Do not include any visible text, logos, or branding in the image. The focus is on the visual representation of the medical information.
- **Tone**: Professional, trustworthy, and informative.

Create an image that effectively visualizes the provided medical answer, informed by the original question and context.
"""

def generate_medical_image(question: str, context: str, answer: str):
    prompt = create_image_prompt(question, context, answer)
    os.makedirs("imgs", exist_ok=True)
    
    result = image_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((512, 512), Image.LANCZOS)
    
    import time
    timestamp = int(time.time())
    filename = f"generated_{timestamp}.jpg"
    full_path = f"imgs/{filename}"
    
    image.save(full_path, format="JPEG", quality=95, optimize=True)
    
    return {
        "image_path": full_path,
        "image_name": filename,
        "caption": f"Ảnh được sinh tự động cho câu hỏi: {question}",
        "source": "AI Generated",
        "score": 1.0,
        "type": "generated"
    }

def get_context_embedding(text: str) -> List[float]:
    response = requests.post(
        MODAL_ENDPOINTS["context_embedding"],
        json={"text": text},
        timeout=120
    )
    response.raise_for_status()
    result = response.json()
    return result["embedding"]

def get_image_embedding(text: str) -> List[float]:
    response = requests.post(
        MODAL_ENDPOINTS["image_embedding"],
        json={"text": text},
        timeout=120
    )
    response.raise_for_status()
    result = response.json()
    return result["embedding"]

def rerank_documents_api(query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
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
        
        if "reranked_documents" in result and result["reranked_documents"]:
            return result["reranked_documents"]
        
        raise Exception("No valid reranked documents returned")
        
    except Exception as e:
        print(f"Reranker API error, falling back: {str(e)}")
        fallback_docs = []
        for i, doc in enumerate(documents[:top_k]):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = 1.0 - (i * 0.1)
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

## NHIỆM VỤ CỤ THỂ
Phân tích danh sách context đã được sắp xếp theo độ liên quan (từ mô hình reranking) và lựa chọn context tối ưu nhất để trả lời câu hỏi y tế.

## NGUYÊN TẮC LỰA CHỌN
1. **Độ chính xác**: Context phải chứa thông tin chính xác, khoa học về chủ đề được hỏi
2. **Độ đầy đủ**: Context phải cung cấp đủ thông tin để trả lời hoàn chỉnh câu hỏi
3. **Độ tin cậy**: Ưu tiên context từ nguồn uy tín, có cơ sở khoa học
4. **Tính cụ thể**: Context phải cụ thể về cơ chế, liều lượng, cách sử dụng
5. **Tối ưu số lượng**: Ưu tiên chọn 1 context đầy đủ, chỉ lấy thêm nếu thực sự cần thiết

## THÔNG TIN CONTEXT
- Tất cả context được hiển thị đầy đủ (không bị cắt bớt)
- Field "content_length" cho biết độ dài của từng context
- Có thể phân tích toàn bộ nội dung để đưa ra quyết định chính xác

## DỮ LIỆU ĐẦU VÀO

**CÂU HỎI Y TẾ:**
{question}

**DANH SÁCH CONTEXT (theo thứ tự ưu tiên từ reranking model):**
{json.dumps(context_list, ensure_ascii=False, indent=2)}

## YÊU CẦU PHÂN TÍCH

### BƯỚC 1: Đánh giá từng context
- Xác định mức độ liên quan trực tiếp đến câu hỏi (0-10)
- Đánh giá độ đầy đủ thông tin (có đủ để trả lời không?)
- Kiểm tra tính chính xác và cơ sở khoa học
- Xác định điểm mạnh và điểm yếu của mỗi context

### BƯỚC 2: Lựa chọn context tối ưu
- Chọn context có điểm tổng hợp cao nhất (không nhất thiết phải top 1)
- Quyết định có cần kết hợp thêm context khác không
- Ưu tiên giải pháp tối thiểu (1 context nếu đủ)

### BƯỚC 3: Đưa ra quyết định cuối cùng

## FORMAT TRẢ LỜI
Trả về ĐÚNG format JSON sau (không thêm text nào khác):

{{
    "selected_chunk_ids": ["chunk_X", "chunk_Y"],
    "reasoning": "Phân tích chi tiết: Context chunk_X được chọn vì [lý do cụ thể về độ chính xác, đầy đủ, tin cậy]. [Nếu chọn thêm context khác thì giải thích tại sao cần thiết]",
    "confidence": 0.XX,
    "analysis": {{
        "primary_context": "chunk_X",
        "supplementary_contexts": ["chunk_Y"] hoặc [],
        "coverage_assessment": "Đánh giá mức độ đủ thông tin để trả lời (đầy đủ/một phần/không đủ)"
    }}
}}

**LƯU Ý QUAN TRỌNG:**
- CHỈ trả về JSON, không có text giải thích thêm
- Confidence score phải phản ánh chính xác mức độ tin tưởng vào lựa chọn
- Reasoning phải cụ thể và có căn cứ khoa học
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
Bạn là một chuyên gia y tế đa lĩnh vực với kiến thức sâu về:
- **Y học lâm sàng**: Chẩn đoán, điều trị, theo dõi bệnh nhân
- **Dược học**: Cơ chế tác dụng, tương tác thuốc, liều lượng, tác dụng phụ
- **Thảo dược**: Thành phần hoạt tính, công dụng, cách chế biến và sử dụng
- **Sinh lý bệnh**: Cơ chế phát sinh và tiến triển bệnh
- **Y học phòng chống**: Biện pháp phòng ngừa và chăm sóc sức khỏe
- **Y học cổ truyền**: Phương pháp điều trị truyền thống có cơ sở khoa học

## NGUYÊN TẮC TRẢ LỜI
1. **Chính xác khoa học**: Thông tin phải có cơ sở khoa học rõ ràng
2. **Dựa trên context**: Chỉ sử dụng thông tin có trong context được cung cấp
3. **NGẮN GỌN TỐI ĐA**: 20-80 từ, ưu tiên 30-50 từ
4. **Trực tiếp**: Trả lời thẳng vào vấn đề, không dài dòng
5. **Thực tiễn**: Cung cấp thông tin cốt lõi nhất

## DỮ LIỆU ĐẦU VÀO

**CÂU HỎI CẦN TRẢ LỜI:**
{question}

**CONTEXT THAM KHẢO:**
{context_text}

## YÊU CẦU CỤ THỂ

### Cấu trúc câu trả lời (NGẮN GỌN):
1. **Trả lời trực tiếp** trong 1-2 câu chính
2. **Thông tin cốt lõi** (cơ chế/liều lượng/cách dùng) nếu có trong context
3. **Lưu ý quan trọng** (nếu cần thiết)

### Tiêu chuẩn chất lượng:
- **Độ dài**: Không quá dài (tối ưu 30-50 từ)
- **Ngôn ngữ**: Tiếng Việt súc tích, khoa học
- **Cấu trúc**: Trực tiếp, không giải thích dài dòng
- **Nội dung**: Chỉ thông tin thiết yếu nhất

### Lưu ý đặc biệt:
- KHÔNG bịa đặt thông tin không có trong context
- KHÔNG giải thích chi tiết nếu không cần thiết
- Ưu tiên thông tin thực tiễn, cụ thể
- Sử dụng "theo tài liệu" khi cần thiết

## TRẢ LỜI (NGẮN GỌN):
"""
        
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y tế chuyên trả lời CỰC NGẮN GỌN và CHÍNH XÁC. Chỉ nói những gì cần thiết nhất."},
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
        return {**state, "error": f"Lỗi khi tìm kiếm context: {str(e)}", "initial_chunks": []}

def rerank_context_node(state: GraphState) -> GraphState:
    if state.get("error") or not state.get("initial_chunks"):
        return state
    
    question = state.get("question")
    initial_chunks = state.get("initial_chunks", [])
    reranked_chunks = rerank_documents_api(question, initial_chunks, top_k=5)
    return {**state, "reranked_chunks": reranked_chunks}

def select_context_node(state: GraphState) -> GraphState:
    if state.get("error") or not state.get("reranked_chunks"):
        return state
    
    question = state.get("question")
    reranked_chunks = state.get("reranked_chunks", [])
    selected_contexts = llm_select_context(question, reranked_chunks)
    return {**state, "selected_contexts": selected_contexts}

def generate_answer_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    
    question = state.get("question")
    selected_contexts = state.get("selected_contexts", [])
    answer = llm_generate_answer(question, selected_contexts)
    return {**state, "answer": answer}

def search_images_node(state: GraphState) -> GraphState:
    if state.get("error") or not state.get("selected_contexts"):
        return state
    
    try:
        main_context = state["selected_contexts"][0]['content']
        context_embedding = get_image_embedding(main_context)
        results = image_store.similarity_search_with_score(query_embedding=context_embedding, k=5)
        
        available_images = []
        if results:
            for i, (doc, score) in enumerate(results):
                available_images.append({
                    "id": i,
                    "caption": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "image_name": doc.metadata.get("image_name"),
                    "image_path": doc.metadata.get("image_path"), 
                    "image_url": doc.metadata.get("image_url"),
                    "source": doc.metadata.get("source"),
                    "image_id": doc.metadata.get("image_id"),
                    "type": "existing"
                })
        
        return {**state, "available_images": available_images}
    except Exception as e:
        print(f"Image search failed, continuing without images: {str(e)}")
        return {**state, "available_images": []}

def generate_image_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    
    question = state.get("question")
    selected_contexts = state.get("selected_contexts", [])
    answer = state.get("answer")

    if not answer:
        return {**state, "error": "Không có câu trả lời để tạo ảnh."}

    main_context = selected_contexts[0]['content'] if selected_contexts else ""
    
    generated_image = generate_medical_image(question, main_context, answer)
    return {**state, "generated_image": generated_image}

def finalize_answer_node(state: GraphState) -> GraphState:
    answer = state.get("answer")
    selected_image = state.get("selected_image")
    generated_image = state.get("generated_image")
    user_choice = state.get("user_choice")
    
    if state.get("error"):
        final_answer = f"Đã có lỗi xảy ra: {state.get('error')}"
    else:
        final_answer = answer
        
        if user_choice == "select_existing" and selected_image:
            final_answer += f"\n\n🖼️ Ảnh minh họa: {selected_image.get('caption', 'Ảnh liên quan')}"
            if selected_image.get('source'):
                final_answer += f"\nNguồn: {selected_image['source']}"
                
        elif user_choice == "generate_new" and generated_image:
            final_answer += f"\n\n🖼️ Ảnh được sinh tự động: {generated_image.get('caption', 'Ảnh liên quan')}"
            final_answer += f"\nNguồn: {generated_image['source']}"
    
    return {**state, "final_answer": final_answer}

def create_app_rag_graph() -> "CompiledGraph":
    """Create RAG graph optimized for app usage - stops after generating answer and finding images"""
    workflow = StateGraph(GraphState)

    workflow.add_node("search_initial_context", search_initial_context_node)
    workflow.add_node("rerank_context", rerank_context_node)
    workflow.add_node("select_context", select_context_node)
    workflow.add_node("search_images", search_images_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("search_initial_context")
    workflow.add_edge("search_initial_context", "rerank_context")
    workflow.add_edge("rerank_context", "select_context")
    workflow.add_edge("select_context", "search_images")
    workflow.add_edge("search_images", "generate_answer")

    return workflow.compile()

def create_enhanced_rag_graph() -> "CompiledGraph":
    workflow = StateGraph(GraphState)

    workflow.add_node("search_initial_context", search_initial_context_node)
    workflow.add_node("rerank_context", rerank_context_node)
    workflow.add_node("select_context", select_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("search_images", search_images_node)
    workflow.add_node("generate_image", generate_image_node)
    workflow.add_node("finalize_answer", finalize_answer_node)

    workflow.set_entry_point("search_initial_context")
    workflow.add_edge("search_initial_context", "rerank_context")
    workflow.add_edge("rerank_context", "select_context")
    workflow.add_edge("select_context", "search_images")
    workflow.add_edge("search_images", "generate_answer")
    workflow.add_edge("generate_answer", "finalize_answer")

    return workflow.compile()

def run_image_generation(state: GraphState) -> GraphState:
    """Run only image generation node"""
    return generate_image_node(state)

if __name__ == "__main__":
    graph = create_enhanced_rag_graph()
    test_question = "Paracetamol có tác dụng gì?"
    result = graph.invoke({"question": test_question})
    print("Available images:", len(result.get("available_images", [])))
    print("Final answer:", result["final_answer"])