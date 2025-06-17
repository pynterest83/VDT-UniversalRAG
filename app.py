import gradio as gr
from pipeline import create_enhanced_rag_graph, run_image_generation
import os
from PIL import Image
from langchain_core.runnables.graph import MermaidDrawMethod

# Initialize pipeline globally
app = create_enhanced_rag_graph()

# Global state to store current pipeline result
current_state = None

def process_medical_question(message):
    """
    Process medical question, yielding streamed updates for the chatbot.
    """
    global current_state
    
    if not message.strip():
        yield "Vui lòng nhập câu hỏi.", [], gr.update(visible=False), "", ""
        return

    inputs = {"question": message}
    
    step_details = []
    full_answer = ""
    processed_steps = set()
    streaming_message = ""

    # Stream the pipeline execution
    for output in app.stream(inputs, stream_mode="values"):
        new_step_message = None
        if "initial_chunks" in output and "initial" not in processed_steps:
            new_step_message = "✅ **Bước 1:** Tìm kiếm context ban đầu...\n"
            processed_steps.add("initial")

        if "reranked_chunks" in output and "reranked" not in processed_steps:
            new_step_message = "✅ **Bước 2:** Rerank context...\n"
            processed_steps.add("reranked")

        if "selected_contexts" in output and "selected" not in processed_steps:
            new_step_message = "✅ **Bước 3:** AI lựa chọn context tốt nhất...\n"
            processed_steps.add("selected")
        
        if "available_images" in output and "image_search" not in processed_steps:
            new_step_message = "✅ **Bước 4:** Tìm kiếm ảnh minh họa...\n"
            processed_steps.add("image_search")

        if "answer" in output and "answer_generation" not in processed_steps:
            new_step_message = "✅ **Bước 5:** AI sinh câu trả lời...\n"
            processed_steps.add("answer_generation")
        
        if new_step_message:
            streaming_message += new_step_message
            yield streaming_message, [], gr.update(visible=False), "", ""

        if "answer" in output and output["answer"] != full_answer:
            full_answer = output.get("answer", "")
            yield streaming_message + full_answer, [], gr.update(visible=False), "", ""

    # Final processing after stream
    final_output = output
    current_state = final_output
    
    # --- Build step-by-step details ---
    if final_output.get("initial_chunks"):
        initial_chunks = final_output["initial_chunks"]
        step_details.append("✅ **BƯỚC 1: TÌM KIẾM CONTEXT BAN ĐẦU**")
        step_details.append(f"   📄 Tìm thấy {len(initial_chunks)} đoạn context từ vector store")
        step_details.append("   🔍 Model embedding: bge-m3-v3\n")

    if final_output.get("reranked_chunks"):
        reranked_chunks = final_output["reranked_chunks"]
        step_details.append("✅ **BƯỚC 2: RERANK CONTEXT VỚI VIETNAMESE RERANKER**")
        step_details.append(f"   🔄 Đã rerank và chọn top {len(reranked_chunks)} context")
        step_details.append("   🤖 Model: AITeamVN/Vietnamese_Reranker")
        for i, chunk in enumerate(reranked_chunks[:3]):
            score = chunk.get('rerank_score', 0)
            preview = chunk['content'][:80] + "..."
            step_details.append(f"      {i+1}. Score: {score:.4f} - {preview}")
        step_details.append("")

    if final_output.get("selected_contexts"):
        selected_contexts = final_output["selected_contexts"]
        step_details.append("✅ **BƯỚC 3: GPT-4O LỰA CHỌN CONTEXT TỐT NHẤT**")
        step_details.append(f"   🎯 Đã chọn {len(selected_contexts)} context từ {len(final_output.get('reranked_chunks', []))} context")
        step_details.append("   🧠 AI phân tích và chọn context phù hợp nhất\n")

    if final_output.get("available_images"):
        available_images_data = final_output["available_images"]
        step_details.append("✅ **BƯỚC 4: TÌM KIẾM 5 ẢNH MINH HỌA**")
        step_details.append(f"   🖼️ Tìm thấy {len(available_images_data)} ảnh liên quan")
        step_details.append("   📸 Model embedding: bge-m3-image")
        for i, img in enumerate(available_images_data[:3]):
            step_details.append(f"      {i+1}. Score: {img.get('score', 0):.4f} - {img.get('caption', 'N/A')[:60]}...")
        step_details.append("")

    if final_output.get("answer"):
        word_count = len(final_output['answer'].split())
        step_details.append("✅ **BƯỚC 5: GPT-4O SINH CÂU TRẢ LỜI**")
        step_details.append(f"   💬 Đã sinh câu trả lời ({word_count} từ)")
        step_details.append(f"   🎯 Dựa trên {len(final_output.get('selected_contexts', []))} context được chọn\n")

    step_display = "\n".join(step_details)
    context_info = create_context_summary(final_output)
    
    available_images = process_available_images(final_output.get("available_images", []))
    show_image_selection = len(available_images) > 0
    
    yield streaming_message + full_answer, available_images, gr.update(visible=show_image_selection), step_display, context_info

def process_available_images(available_images_data):
    """Process image information to display in Gradio"""
    images = []
    for img_info in available_images_data:
        image_path = img_info.get('image_path')
        image_name = img_info.get('image_name')
        
        actual_path = None
        if image_path and os.path.isfile(image_path):
            actual_path = image_path
        else:
            possible_paths = [ f"images/{image_name}", f"datasets/images/{image_name}", f"./images/{image_name}", f"imgs/{image_name}"]
            for path in [p for p in possible_paths if p is not None]:
                if os.path.isfile(path):
                    actual_path = path
                    break
        
        if actual_path:
            caption = f"Score: {img_info.get('score', 0):.4f}\n{img_info.get('caption', 'Ảnh liên quan')}"
            images.append((actual_path, caption))
    return images

def select_existing_image(history, evt: gr.SelectData):
    """Handle selection of existing image from gallery, append to chat."""
    global current_state
    if not current_state or not current_state.get("available_images"):
        return history, None
    
    selected_index = evt.index
    available_images = current_state["available_images"]
    
    if 0 <= selected_index < len(available_images):
        selected_image_info = available_images[selected_index]
        selected_img_path = load_single_image(selected_image_info)

        if selected_img_path:
            current_state["selected_image"] = selected_image_info
            current_state["user_choice"] = "select_existing"
            
            final_result = finalize_with_selected_image(current_state)
            
            # Update last bot message with final answer and add image
            history[-1][1] = final_result["final_answer"]
            history.append([None, (selected_img_path, "Ảnh được chọn")])

    return history, gr.update(visible=False)


def generate_new_image(history):
    """Generate new image and append to chat."""
    global current_state
    if not current_state:
        return history
    
    result = run_image_generation(current_state)
    generated_image_info = result.get("generated_image")
    
    if generated_image_info:
        generated_img_path = load_single_image(generated_image_info)
        if generated_img_path:
            current_state["generated_image"] = generated_image_info
            current_state["user_choice"] = "generate_new"
            final_result = finalize_with_selected_image(current_state)
            
            history[-1][1] = final_result["final_answer"]
            history.append([None, (generated_img_path, "Ảnh do AI tạo")])
            
    return history, gr.update(visible=False)


def finalize_with_selected_image(state):
    from pipeline import finalize_answer_node
    return finalize_answer_node(state)


def load_single_image(image_info):
    if not image_info: return None
    image_path = image_info.get('image_path')
    image_name = image_info.get('image_name')
    if image_path and os.path.isfile(image_path):
        return image_path
    possible_paths = [f"images/{image_name}", f"datasets/images/{image_name}", f"./images/{image_name}", f"imgs/{image_name}"]
    for path in [p for p in possible_paths if p is not None]:
        if os.path.isfile(path):
            return path
    return None


def create_context_summary(output):
    """Create summary of context and processing information"""
    if not output: return ""
    summary_parts = []
    question = output.get("question", "N/A")
    summary_parts.append(f"**CÂU HỎI:** {question}\n")
    
    summary_parts.append("**THỐNG KÊ CONTEXT:**")
    summary_parts.append(f"• Context ban đầu: {len(output.get('initial_chunks', []))} đoạn")
    summary_parts.append(f"• Context sau rerank: {len(output.get('reranked_chunks', []))} đoạn")
    summary_parts.append(f"• Context được chọn: {len(output.get('selected_contexts', []))} đoạn\n")
    
    if output.get('selected_contexts'):
        summary_parts.append("**CONTEXT ĐƯỢC SỬ DỤNG:**")
        for i, ctx in enumerate(output['selected_contexts']):
            score = ctx.get('rerank_score', 0)
            content_preview = ctx['content'][:200] + "..."
            summary_parts.append(f"**Context {i+1}** (Rerank Score: {score:.4f})\n{content_preview}\n")
    
    if output.get('available_images'):
        summary_parts.append("**THÔNG TIN ẢNH CÓ SẴN:**")
        summary_parts.append(f"• Số lượng ảnh tìm thấy: {len(output['available_images'])}")
        for i, img in enumerate(output['available_images'][:3]):
            summary_parts.append(f"• Ảnh {i+1}: Score {img.get('score', 0):.4f} - {img.get('caption', 'N/A')[:60]}...")
    
    return "\n".join(summary_parts)


def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="🏥 Medical RAG Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 Medical RAG Chatbot với GPT-4o + AI Image Generation
        **Hệ thống hỏi đáp y tế thông minh**: Đặt câu hỏi về y tế, thuốc, bệnh, và nhận câu trả lời chi tiết kèm hình ảnh!
        """)
        
        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
        )
        
        with gr.Row():
            question_input = gr.Textbox(
                show_label=False,
                placeholder="VD: Thuốc Paracetamol có tác dụng phụ gì?",
                scale=4,
                container=False
            )
            submit_btn = gr.Button("Gửi", variant="primary", scale=1, min_width=150)

        with gr.Row():
            gr.Examples(
                [
                    "Bách thảo sương dùng để chữa chảy máu kẽ răng như thế nào?",
                    "Thuốc Paracetamol có tác dụng gì và liều dùng ra sao?",
                    "Cây lá lốt có công dụng gì trong y học cổ truyền?",
                ],
                inputs=question_input,
                label="Câu hỏi ví dụ"
            )

        with gr.Row(visible=False) as image_selection_row:
            with gr.Column():
                gr.Markdown("### 🖼️ Chọn ảnh minh họa")
                gr.Markdown("Chọn 1 trong 5 ảnh bên dưới hoặc yêu cầu AI tạo ảnh mới.")
                available_images_gallery = gr.Gallery(
                    label="Ảnh có sẵn (click để chọn)",
                    show_label=False,
                    elem_id="gallery",
                    columns=5,
                    rows=1,
                    height=200
                )
                generate_new_btn = gr.Button("🎨 Tạo ảnh mới với AI", variant="secondary")

        with gr.Accordion("⚙️ Chi tiết xử lý", open=False):
            steps_output = gr.Markdown(label="Các bước thực hiện")
            context_output = gr.Markdown(label="Thông tin context")

        # --- Event Handlers ---
        
        def add_user_message(history, user_message):
            if not user_message.strip():
                return history, ""
            return history + [[user_message, None]], ""

        def bot_response(history):
            if not history or not history[-1][0]:
                return history, gr.update(visible=False), None, None

            user_message = history[-1][0]
            history[-1][1] = ""
            
            available_images_update = []
            image_row_update = gr.update(visible=False)
            steps_update = ""
            context_update = ""

            stream = process_medical_question(user_message)
            for response_chunk, available_images, image_row_viz, steps, context in stream:
                history[-1][1] = response_chunk
                available_images_update = available_images
                image_row_update = image_row_viz
                steps_update = steps
                context_update = context
                yield history, available_images_update, image_row_update, steps_update, context_update

        # Chain of events for submitting a question
        question_input.submit(
            add_user_message,
            [chatbot, question_input],
            [chatbot, question_input]
        ).then(
            bot_response,
            [chatbot],
            [chatbot, available_images_gallery, image_selection_row, steps_output, context_output]
        )
        
        submit_btn.click(
            add_user_message,
            [chatbot, question_input],
            [chatbot, question_input],
            queue=False
        ).then(
            bot_response,
            [chatbot],
            [chatbot, available_images_gallery, image_selection_row, steps_output, context_output]
        )

        # Event for selecting an image from the gallery
        available_images_gallery.select(
            select_existing_image,
            [chatbot],
            [chatbot, image_selection_row],
        )

        # Event for generating a new image
        generate_new_btn.click(
            generate_new_image,
            [chatbot],
            [chatbot, image_selection_row]
        )
        
        gr.Markdown("""
        ---
        **🔧 Technical Stack:** BGE-M3 • Vietnamese Reranker • GPT-4o • Qdrant • LangGraph • Azure OpenAI Image Generation
        **⚠️ Lưu ý:** Thông tin chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến bác sĩ cho chẩn đoán và điều trị chính xác.
        """)

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(
        share=True,
        server_name="localhost",
        server_port=7860
    )
