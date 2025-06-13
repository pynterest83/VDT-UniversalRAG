import uuid
from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import AzureOpenAI
import json
import tiktoken

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


# --- Khởi tạo các thành phần ---
def initialize_components():
    context_embedding_model = HuggingFaceEmbeddings(model_name="./bge-m3-v3")
    
    reranker_model_path = "AITeamVN/Vietnamese_Reranker"
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path, use_fast=False)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        reranker_model_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    reranker_model.eval()
    
    image_embedding_model = HuggingFaceEmbeddings(model_name="./bge-m3-image")
    
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_VERSION")
    )
    
    context_store = VectorStore(collection_name="universal-rag-precomputed-enhanced")
    image_store = VectorStore(collection_name="image-captions-store")
    
    return context_embedding_model, reranker_tokenizer, reranker_model, image_embedding_model, azure_client, context_store, image_store

context_embedding_model, reranker_tokenizer, reranker_model, image_embedding_model, azure_client, context_store, image_store = initialize_components()
MAX_LENGTH = 2304

# Initialize tiktoken encoder for GPT-4o
encoding = tiktoken.encoding_for_model("gpt-4o")

def rerank_documents(query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
    if len(documents) <= top_k:
        return documents
    
    doc_contents = [doc['content'] for doc in documents]
    pairs = [[query, doc_content] for doc_content in doc_contents]
    
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=MAX_LENGTH)
        if torch.cuda.is_available() and reranker_model.device.type == 'cuda':
            inputs = {k: v.to(reranker_model.device) for k, v in inputs.items()}
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    document_scores = list(zip(documents, scores.cpu().numpy()))
    document_scores.sort(key=lambda x: x[1], reverse=True)
    
    reranked_docs = []
    for i, (doc, score) in enumerate(document_scores[:top_k]):
        doc['rerank_score'] = float(score)
        reranked_docs.append(doc)
    
    return reranked_docs

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
        query_embedding = context_embedding_model.embed_query(question)
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
        reranked_chunks = rerank_documents(question, initial_chunks, top_k=5)
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
        main_context = selected_contexts[0]['content']
        context_embedding = image_embedding_model.embed_query(main_context)
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