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

def create_medical_prompt(question_data):
    keyword = question_data.get('keyword', 'y tế')
    question = question_data.get('question', '')
    context = question_data.get('context', '')
    
    props_description = f"realistic medical items related to {keyword}, medicine packages, medical documents on desk"
    
    return f"""
Create a realistic, professional medical consultation photo about "{keyword}".

Medical context: {question}
Information: {context}

Scene requirements:
- Style: Realistic photography, not illustration or cartoon
- People: Asian doctor consulting with Asian patient
- Doctor: Professional attire (white coat), confident and caring expression
- Patient: Comfortable, attentive, trusting
- Setting: Modern medical clinic or pharmacy, clean and professional
- Props: {props_description}, medical documents, computer/tablet
- Lighting: Natural, warm, professional medical lighting
- Composition: Clear view of consultation, focused on interaction
- Quality: High resolution, sharp details

Important: NO TEXT visible in the image, no signs with words, focus on realistic medical consultation scene.

Photography style: Medical consultation photography, professional healthcare setting.
"""

def generate_medical_image(question_data):
    prompt = create_medical_prompt(question_data)
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
        "caption": f"Ảnh được sinh tự động cho câu hỏi: {question_data.get('question', '')}",
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
        for i, chunk in enumerate(ranked_chunks):
            context_list.append({
                "priority_order": i + 1,
                "rerank_score": chunk.get('rerank_score', 0),
                "content": chunk['content'],
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
        
        result = json.loads(response.choices[0].message.content.strip())
        selected_chunk_ids = result.get("selected_chunk_ids", [])
        
        selected_contexts = []
        for chunk_id in selected_chunk_ids:
            for chunk in ranked_chunks:
                if chunk.get('chunk_id') == chunk_id:
                    selected_contexts.append(chunk)
                    break
        
        return selected_contexts
            
    except Exception:
        return [ranked_chunks[0]]

def llm_generate_answer(question: str, selected_contexts: List[dict]) -> str:
    if not selected_contexts:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này."
    
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
    
    return response.choices[0].message.content.strip()

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
    
    main_context = selected_contexts[0]['content'] if selected_contexts else ""
    
    question_data = {
        "question": question,
        "context": main_context,
        "keyword": extract_keyword_from_question(question),
    }
    
    generated_image = generate_medical_image(question_data)
    return {**state, "generated_image": generated_image}

def extract_keyword_from_question(question: str) -> str:
    question_lower = question.lower()
    
    medical_keywords = [
        "paracetamol", "aspirin", "vitamin", "thuốc", "bệnh", "điều trị",
        "carbogast", "calcium", "dược", "y tế", "sức khỏe"
    ]
    
    for keyword in medical_keywords:
        if keyword in question_lower:
            return keyword
    
    words = question.split()
    for word in words:
        if len(word) > 3 and word.lower() not in ["trong", "của", "với", "như", "thế", "nào"]:
            return word
    
    return "y tế"

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
    workflow.add_edge("select_context", "generate_answer")
    workflow.add_edge("generate_answer", "search_images")
    workflow.add_edge("search_images", "finalize_answer")

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