import gradio as gr
from pipeline import create_enhanced_rag_graph
import os
from PIL import Image
import base64
from io import BytesIO

# Initialize pipeline globally
app = create_enhanced_rag_graph()

def process_medical_question(message, progress=gr.Progress()):
    """Process medical question with detailed step-by-step output"""
    if not message.strip():
        return "Vui lòng nhập câu hỏi.", "", None, ""
    
    try:
        inputs = {"question": message}
        
        # Initialize outputs
        step_details = []
        final_answer = ""
        image_output = None
        context_info = ""
        
        # Track progress through pipeline
        progress(0, desc="Khởi tạo pipeline...")
        
        step_count = 0
        total_steps = 6
        
        for output in app.stream(inputs, stream_mode="values"):
            step_count += 1
            progress(step_count / total_steps, desc=f"Bước {step_count}/{total_steps}")
            
            # Step 1: Initial Context Search
            if output.get("initial_chunks") and not any("BƯỚC 1" in detail for detail in step_details):
                initial_chunks = output["initial_chunks"]
                step_details.append(f"✅ **BƯỚC 1: TÌM KIẾM CONTEXT BAN ĐẦU**")
                step_details.append(f"   📄 Tìm thấy {len(initial_chunks)} đoạn context từ vector store")
                step_details.append(f"   🔍 Model embedding: bge-m3-v3")
                step_details.append("")
            
            # Step 2: Reranking
            if output.get("reranked_chunks") and not any("BƯỚC 2" in detail for detail in step_details):
                reranked_chunks = output["reranked_chunks"]
                step_details.append(f"✅ **BƯỚC 2: RERANK CONTEXT VỚI VIETNAMESE RERANKER**")
                step_details.append(f"   🔄 Đã rerank và chọn top {len(reranked_chunks)} context")
                step_details.append(f"   🤖 Model: AITeamVN/Vietnamese_Reranker")
                
                # Show top rerank scores
                for i, chunk in enumerate(reranked_chunks[:3]):
                    score = chunk.get('rerank_score', 0)
                    preview = chunk['content'][:80] + "..." if len(chunk['content']) > 80 else chunk['content']
                    step_details.append(f"      {i+1}. Score: {score:.4f} - {preview}")
                step_details.append("")
            
            # Step 3: Context Selection
            if output.get("selected_contexts") and not any("BƯỚC 3" in detail for detail in step_details):
                selected_contexts = output["selected_contexts"]
                step_details.append(f"✅ **BƯỚC 3: GPT-4O LỰA CHỌN CONTEXT TỐT NHẤT**")
                step_details.append(f"   🎯 Đã chọn {len(selected_contexts)} context từ {len(output.get('reranked_chunks', []))} context")
                step_details.append(f"   🧠 AI phân tích và chọn context phù hợp nhất")
                
                # Show selected context preview
                for i, ctx in enumerate(selected_contexts):
                    preview = ctx['content'][:100] + "..." if len(ctx['content']) > 100 else ctx['content']
                    rerank_score = ctx.get('rerank_score', 0)
                    step_details.append(f"      Context {i+1} (Score: {rerank_score:.4f}): {preview}")
                step_details.append("")
            
            # Step 4: Answer Generation
            if output.get("answer") and not any("BƯỚC 4" in detail for detail in step_details):
                answer = output["answer"]
                word_count = len(answer.split())
                step_details.append(f"✅ **BƯỚC 4: GPT-4O SINH CÂU TRẢ LỜI**")
                step_details.append(f"   💬 Đã sinh câu trả lời ({word_count} từ)")
                step_details.append(f"   🎯 Dựa trên {len(output.get('selected_contexts', []))} context được chọn")
                step_details.append("")
            
            # Step 5: Image Search
            if output.get("image_info") and not any("BƯỚC 5" in detail for detail in step_details):
                image_info = output["image_info"]
                step_details.append(f"✅ **BƯỚC 5: TÌM KIẾM ẢNH MINH HỌA**")
                step_details.append(f"   🖼️ Tìm thấy ảnh liên quan với score: {image_info.get('score', 0):.4f}")
                step_details.append(f"   📸 Model embedding: bge-m3-image")
                step_details.append(f"   📝 Caption: {image_info.get('caption', 'N/A')[:100]}...")
                if image_info.get('image_name'):
                    step_details.append(f"   📁 Tên ảnh: {image_info['image_name']}")
                step_details.append("")
            
            # Step 6: Final Answer
            if output.get("final_answer") and not any("BƯỚC 6" in detail for detail in step_details):
                step_details.append(f"✅ **BƯỚC 6: HOÀN THIỆN CÂU TRẢ LỜI**")
                step_details.append(f"   ✨ Kết hợp câu trả lời + thông tin ảnh")
                step_details.append(f"   🏁 Pipeline hoàn tất!")
                step_details.append("")
        
        # Get final outputs
        final_output = output
        
        if final_output and final_output.get("final_answer"):
            final_answer = final_output["final_answer"]
            
            # Process image if available
            image_info = final_output.get("image_info")
            if image_info:
                image_output = process_image_output(image_info)
            
            # Create context information
            context_info = create_context_summary(final_output)
            
        else:
            final_answer = "Xin lỗi, tôi không thể trả lời câu hỏi này."
        
        # Combine step details
        step_display = "\n".join(step_details)
        
        return final_answer, step_display, image_output, context_info
            
    except Exception as e:
        error_msg = f"Đã xảy ra lỗi: {str(e)}"
        return error_msg, f"❌ Lỗi trong quá trình xử lý: {str(e)}", None, ""

def process_image_output(image_info):
    """Process image information to display in Gradio"""
    if not image_info:
        return None
    
    try:
        # Try to load image from various sources
        image_path = image_info.get('image_path')
        image_url = image_info.get('image_url')
        image_name = image_info.get('image_name')
        
        # Try local path first
        if image_path and os.path.exists(image_path):
            return Image.open(image_path)
        
        # Try alternative paths
        possible_paths = [
            f"images/{image_name}" if image_name else None,
            f"datasets/images/{image_name}" if image_name else None,
            f"./images/{image_name}" if image_name else None,
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return Image.open(path)
        
        # If no image found, return placeholder info
        return None
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def create_context_summary(output):
    """Create summary of context and processing information"""
    try:
        summary_parts = []
        
        # Question info
        question = output.get("question", "N/A")
        summary_parts.append(f"**CÂU HỎI:** {question}")
        summary_parts.append("")
        
        # Context statistics
        initial_chunks = output.get("initial_chunks", [])
        reranked_chunks = output.get("reranked_chunks", [])
        selected_contexts = output.get("selected_contexts", [])
        
        summary_parts.append("**THỐNG KÊ CONTEXT:**")
        summary_parts.append(f"• Context ban đầu: {len(initial_chunks)} đoạn")
        summary_parts.append(f"• Context sau rerank: {len(reranked_chunks)} đoạn")
        summary_parts.append(f"• Context được chọn: {len(selected_contexts)} đoạn")
        summary_parts.append("")
        
        # Selected contexts details
        if selected_contexts:
            summary_parts.append("**CONTEXT ĐƯỢC SỬ DỤNG:**")
            for i, ctx in enumerate(selected_contexts):
                score = ctx.get('rerank_score', 0)
                content_preview = ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
                summary_parts.append(f"**Context {i+1}** (Rerank Score: {score:.4f})")
                summary_parts.append(f"{content_preview}")
                summary_parts.append("")
        
        # Image info
        image_info = output.get("image_info")
        if image_info:
            summary_parts.append("**THÔNG TIN ẢNH:**")
            summary_parts.append(f"• Caption: {image_info.get('caption', 'N/A')}")
            summary_parts.append(f"• Score: {image_info.get('score', 0):.4f}")
            if image_info.get('source'):
                summary_parts.append(f"• Nguồn: {image_info['source']}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Lỗi tạo tóm tắt: {str(e)}"

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="🏥 Medical RAG Chatbot", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🏥 Medical RAG Chatbot với GPT-4o
        
        **Hệ thống hỏi đáp y tế thông minh** sử dụng:
        - 🔍 **BGE-M3** cho tìm kiếm context 
        - 🇻🇳 **Vietnamese Reranker** cho sắp xếp lại
        - 🧠 **GPT-4o** cho lựa chọn context và trả lời
        - 🖼️ **BGE-M3-Image** cho tìm kiếm ảnh minh họa
        
        Đặt câu hỏi về y tế, thuốc, bệnh, thảo dược, và nhận câu trả lời chi tiết kèm hình ảnh!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                question_input = gr.Textbox(
                    label="💬 Câu hỏi y tế",
                    placeholder="VD: Thuốc Paracetamol có tác dụng phụ gì?",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Hỏi", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Xóa", scale=1)
                
                # Examples
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
                # Image output
                image_output = gr.Image(
                    label="🖼️ Ảnh minh họa",
                    height=300
                )
        
        with gr.Row():
            with gr.Column():
                # Main answer
                answer_output = gr.Textbox(
                    label="💡 Câu trả lời",
                    lines=8,
                    max_lines=15
                )
            
            with gr.Column():
                # Step-by-step process
                steps_output = gr.Textbox(
                    label="⚙️ Các bước thực hiện",
                    lines=8,
                    max_lines=15
                )
        
        # Context details (collapsible)
        with gr.Accordion("📊 Chi tiết Context & Thông tin", open=False):
            context_output = gr.Textbox(
                label="Thông tin chi tiết",
                lines=10,
                max_lines=20
            )
        
        # Event handlers
        def submit_question(question):
            if not question.strip():
                return "Vui lòng nhập câu hỏi.", "", None, ""
            return process_medical_question(question)
        
        def clear_all():
            return "", "", None, "", ""
        
        submit_btn.click(
            fn=submit_question,
            inputs=[question_input],
            outputs=[answer_output, steps_output, image_output, context_output]
        )
        
        question_input.submit(
            fn=submit_question,
            inputs=[question_input],
            outputs=[answer_output, steps_output, image_output, context_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[question_input, answer_output, steps_output, image_output, context_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **🔧 Technical Stack:** BGE-M3 • Vietnamese Reranker • GPT-4o • Qdrant • LangGraph
        
        **⚠️ Lưu ý:** Thông tin chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến bác sĩ cho chẩn đoán và điều trị chính xác.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
