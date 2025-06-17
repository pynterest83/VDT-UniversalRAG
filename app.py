import gradio as gr
from pipeline import create_enhanced_rag_graph, run_image_generation
import os
from PIL import Image
import base64
from io import BytesIO

# Initialize pipeline globally
app = create_enhanced_rag_graph()

# Global state to store current pipeline result
current_state = None

def process_medical_question(message, progress=gr.Progress()):
    """Process medical question with detailed step-by-step output"""
    global current_state
    
    if not message.strip():
        return "Vui lòng nhập câu hỏi.", "", [], "", "", False
    
    inputs = {"question": message}
    
    step_details = []
    final_answer = ""
    available_images = []
    context_info = ""
    
    progress(0, desc="Khởi tạo pipeline...")
    
    step_count = 0
    total_steps = 6
    
    for output in app.stream(inputs, stream_mode="values"):
        step_count += 1
        progress(step_count / total_steps, desc=f"Bước {step_count}/{total_steps}")
        
        if output.get("initial_chunks") and not any("BƯỚC 1" in detail for detail in step_details):
            initial_chunks = output["initial_chunks"]
            step_details.append(f"✅ **BƯỚC 1: TÌM KIẾM CONTEXT BAN ĐẦU**")
            step_details.append(f"   📄 Tìm thấy {len(initial_chunks)} đoạn context từ vector store")
            step_details.append(f"   🔍 Model embedding: bge-m3-v3")
            step_details.append("")
        
        if output.get("reranked_chunks") and not any("BƯỚC 2" in detail for detail in step_details):
            reranked_chunks = output["reranked_chunks"]
            step_details.append(f"✅ **BƯỚC 2: RERANK CONTEXT VỚI VIETNAMESE RERANKER**")
            step_details.append(f"   🔄 Đã rerank và chọn top {len(reranked_chunks)} context")
            step_details.append(f"   🤖 Model: AITeamVN/Vietnamese_Reranker")
            
            for i, chunk in enumerate(reranked_chunks[:3]):
                score = chunk.get('rerank_score', 0)
                preview = chunk['content'][:80] + "..." if len(chunk['content']) > 80 else chunk['content']
                step_details.append(f"      {i+1}. Score: {score:.4f} - {preview}")
            step_details.append("")
        
        if output.get("selected_contexts") and not any("BƯỚC 3" in detail for detail in step_details):
            selected_contexts = output["selected_contexts"]
            step_details.append(f"✅ **BƯỚC 3: GPT-4O LỰA CHỌN CONTEXT TỐT NHẤT**")
            step_details.append(f"   🎯 Đã chọn {len(selected_contexts)} context từ {len(output.get('reranked_chunks', []))} context")
            step_details.append(f"   🧠 AI phân tích và chọn context phù hợp nhất")
            
            for i, ctx in enumerate(selected_contexts):
                preview = ctx['content'][:100] + "..." if len(ctx['content']) > 100 else ctx['content']
                rerank_score = ctx.get('rerank_score', 0)
                step_details.append(f"      Context {i+1} (Score: {rerank_score:.4f}): {preview}")
            step_details.append("")
        
        if output.get("answer") and not any("BƯỚC 4" in detail for detail in step_details):
            answer = output["answer"]
            word_count = len(answer.split())
            step_details.append(f"✅ **BƯỚC 4: GPT-4O SINH CÂU TRẢ LỜI**")
            step_details.append(f"   💬 Đã sinh câu trả lời ({word_count} từ)")
            step_details.append(f"   🎯 Dựa trên {len(output.get('selected_contexts', []))} context được chọn")
            step_details.append("")
        
        if output.get("available_images") and not any("BƯỚC 5" in detail for detail in step_details):
            available_images_data = output["available_images"]
            step_details.append(f"✅ **BƯỚC 5: TÌM KIẾM 5 ẢNH MINH HỌA**")
            step_details.append(f"   🖼️ Tìm thấy {len(available_images_data)} ảnh liên quan")
            step_details.append(f"   📸 Model embedding: bge-m3-image")
            
            for i, img in enumerate(available_images_data[:3]):
                step_details.append(f"      {i+1}. Score: {img.get('score', 0):.4f} - {img.get('caption', 'N/A')[:60]}...")
            step_details.append("")
    
    final_output = output
    current_state = final_output
    
    if final_output:
        final_answer = final_output.get("answer", "")
        available_images_data = final_output.get("available_images", [])
        available_images = process_available_images(available_images_data)
        context_info = create_context_summary(final_output)
    else:
        final_answer = "Xin lỗi, tôi không thể trả lời câu hỏi này."
    
    step_display = "\n".join(step_details)
    show_image_selection = len(available_images) > 0
    
    return final_answer, step_display, available_images, context_info, "", show_image_selection

def process_available_images(available_images_data):
    """Process image information to display in Gradio"""
    images = []
    
    for img_info in available_images_data:
        image_path = img_info.get('image_path')
        image_name = img_info.get('image_name')
        
        loaded_image = None
        actual_path = None
        
        if image_path and os.path.isfile(image_path):
            loaded_image = Image.open(image_path)
            actual_path = image_path
        else:
            possible_paths = [
                f"images/{image_name}" if image_name else None,
                f"datasets/images/{image_name}" if image_name else None,
                f"./images/{image_name}" if image_name else None,
            ]
            
            for path in possible_paths:
                if path and os.path.isfile(path):
                    try:
                        loaded_image = Image.open(path)
                        actual_path = path
                        break
                    except Exception:
                        continue
        
        if loaded_image and actual_path:
            caption = (
                f"Score: {img_info.get('score', 0):.4f}\n"
                f"{img_info.get('caption', 'Ảnh liên quan')}"
            )
            images.append((actual_path, caption))
    
    return images

def select_existing_image(evt: gr.SelectData):
    """Handle selection of existing image from gallery"""
    global current_state
    
    if not current_state or not current_state.get("available_images"):
        return "Không có ảnh để chọn.", None
    
    selected_index = evt.index
    available_images = current_state["available_images"]
    
    if 0 <= selected_index < len(available_images):
        selected_image = available_images[selected_index]
        
        current_state["selected_image"] = selected_image
        current_state["user_choice"] = "select_existing"
        
        final_result = finalize_with_selected_image(current_state)
        selected_img = load_single_image(selected_image)
        
        return final_result["final_answer"], selected_img
    
    return "Lỗi khi chọn ảnh.", None

def generate_new_image():
    """Generate new image using AI"""
    global current_state
    
    if not current_state:
        return "Vui lòng đặt câu hỏi trước.", None
    
    result = run_image_generation(current_state)
    generated_image = result.get("generated_image")
    
    if generated_image:
        current_state["generated_image"] = generated_image
        current_state["user_choice"] = "generate_new"
        
        final_result = finalize_with_selected_image(current_state)
        generated_img = load_single_image(generated_image)
        
        return final_result["final_answer"], generated_img
    else:
        return "Không thể tạo ảnh mới. Vui lòng thử lại.", None

def finalize_with_selected_image(state):
    """Create final answer with selected image"""
    from pipeline import finalize_answer_node
    return finalize_answer_node(state)

def load_single_image(image_info):
    """Load a single image for display"""
    if not image_info:
        return None
        
    image_path = image_info.get('image_path')
    image_name = image_info.get('image_name')
    
    if image_path and os.path.isfile(image_path):
        return Image.open(image_path)
    
    possible_paths = [
        f"images/{image_name}" if image_name else None,
        f"datasets/images/{image_name}" if image_name else None,
        f"./images/{image_name}" if image_name else None,
        f"imgs/{image_name}" if image_name else None,
    ]
    
    for path in possible_paths:
        if path and os.path.isfile(path):
            try:
                return Image.open(path)
            except Exception:
                continue
    
    return None

def create_context_summary(output):
    """Create summary of context and processing information"""
    summary_parts = []
    
    question = output.get("question", "N/A")
    summary_parts.append(f"**CÂU HỎI:** {question}")
    summary_parts.append("")
    
    initial_chunks = output.get("initial_chunks", [])
    reranked_chunks = output.get("reranked_chunks", [])
    selected_contexts = output.get("selected_contexts", [])
    
    summary_parts.append("**THỐNG KÊ CONTEXT:**")
    summary_parts.append(f"• Context ban đầu: {len(initial_chunks)} đoạn")
    summary_parts.append(f"• Context sau rerank: {len(reranked_chunks)} đoạn")
    summary_parts.append(f"• Context được chọn: {len(selected_contexts)} đoạn")
    summary_parts.append("")
    
    if selected_contexts:
        summary_parts.append("**CONTEXT ĐƯỢC SỬ DỤNG:**")
        for i, ctx in enumerate(selected_contexts):
            score = ctx.get('rerank_score', 0)
            content_preview = ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
            summary_parts.append(f"**Context {i+1}** (Rerank Score: {score:.4f})")
            summary_parts.append(f"{content_preview}")
            summary_parts.append("")
    
    available_images = output.get("available_images", [])
    if available_images:
        summary_parts.append("**THÔNG TIN ẢNH CÓ SẴN:**")
        summary_parts.append(f"• Số lượng ảnh tìm thấy: {len(available_images)}")
        for i, img in enumerate(available_images[:3]):
            summary_parts.append(f"• Ảnh {i+1}: Score {img.get('score', 0):.4f} - {img.get('caption', 'N/A')[:60]}...")
        summary_parts.append("")
    
    return "\n".join(summary_parts)

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="🏥 Medical RAG Chatbot", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🏥 Medical RAG Chatbot với GPT-4o + AI Image Generation
        
        **Hệ thống hỏi đáp y tế thông minh** sử dụng:
        - 🔍 **BGE-M3** cho tìm kiếm context 
        - 🇻🇳 **Vietnamese Reranker** cho sắp xếp lại
        - 🧠 **GPT-4o** cho lựa chọn context và trả lời
        - 🖼️ **BGE-M3-Image** cho tìm kiếm ảnh minh họa
        - 🎨 **Azure OpenAI Image Generation** cho tạo ảnh mới
        
        Đặt câu hỏi về y tế, thuốc, bệnh, thảo dược, và nhận câu trả lời chi tiết kèm hình ảnh!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="💬 Câu hỏi y tế",
                    placeholder="VD: Thuốc Paracetamol có tác dụng phụ gì?",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Hỏi", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Xóa", scale=1)
                
                gr.Examples(
                    examples=[
                        "Bách thảo sương dùng để chữa chảy máu kẽ răng như thế nào?",
                        "Thuốc Paracetamol có tác dụng gì và liều dùng ra sao?",
                        "Calci D Hasan có thể gây ảnh hưởng gì đến bệnh nhân mắc bệnh lý nền?",
                        "Cây lá lốt có công dụng gì trong y học cổ truyền?",
                        "Vitamin D thiếu hụt có triệu chứng gì?",
                    ],
                    inputs=question_input
                )
            
            with gr.Column(scale=1):
                final_image_output = gr.Image(
                    label="🖼️ Ảnh được chọn",
                    height=300
                )
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="💡 Câu trả lời",
                    lines=8,
                    max_lines=15
                )
            
            with gr.Column():
                steps_output = gr.Textbox(
                    label="⚙️ Các bước thực hiện",
                    lines=8,
                    max_lines=15
                )
        
        with gr.Row(visible=False) as image_selection_row:
            with gr.Column():
                gr.Markdown("## 🖼️ Chọn ảnh minh họa")
                gr.Markdown("**Chọn 1 trong 5 ảnh bên dưới hoặc tạo ảnh mới:**")
                
                available_images_gallery = gr.Gallery(
                    label="📷 Ảnh có sẵn (click để chọn)",
                    show_label=True,
                    elem_id="gallery",
                    columns=5,
                    rows=1,
                    height=200,
                    allow_preview=True
                )
                
                with gr.Row():
                    generate_new_btn = gr.Button("🎨 Tạo ảnh mới với AI", variant="secondary")
        
        with gr.Accordion("📊 Chi tiết Context & Thông tin", open=False):
            context_output = gr.Textbox(
                label="Thông tin chi tiết",
                lines=10,
                max_lines=20
            )
        
        image_selection_visible = gr.State(False)
        
        def submit_question(question):
            if not question.strip():
                return "Vui lòng nhập câu hỏi.", "", [], "", None, gr.update(visible=False)
            
            answer, steps, images, context, final_img_ignored, show_selection = process_medical_question(question)
            
            return (
                answer, 
                steps, 
                images, 
                context, 
                None,
                gr.update(visible=show_selection)
            )
        
        def clear_all():
            global current_state
            current_state = None
            return "", "", "", [], "", None, gr.update(visible=False)
        
        submit_btn.click(
            fn=submit_question,
            inputs=[question_input],
            outputs=[answer_output, steps_output, available_images_gallery, context_output, final_image_output, image_selection_row]
        )
        
        question_input.submit(
            fn=submit_question,
            inputs=[question_input],
            outputs=[answer_output, steps_output, available_images_gallery, context_output, final_image_output, image_selection_row]
        )
        
        available_images_gallery.select(
            fn=select_existing_image,
            outputs=[answer_output, final_image_output]
        )
        
        generate_new_btn.click(
            fn=generate_new_image,
            outputs=[answer_output, final_image_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[question_input, answer_output, steps_output, available_images_gallery, context_output, final_image_output, image_selection_row]
        )
        
        gr.Markdown("""
        ---
        **🔧 Technical Stack:** BGE-M3 • Vietnamese Reranker • GPT-4o • Qdrant • LangGraph • Azure OpenAI Image Generation
        
        **⚠️ Lưu ý:** Thông tin chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến bác sĩ cho chẩn đoán và điều trị chính xác.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="localhost",
        server_port=7860
    )
