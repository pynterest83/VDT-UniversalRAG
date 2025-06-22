import streamlit as st
import time
from typing import List, Dict, Any
from pipeline import (
    create_app_rag_graph, 
    generate_medical_image
)
import os
import base64
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Hệ thống truy xuất và hỏi đáp bổ sung hình ảnh bằng tiếng Việt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .user-message {
        background-color: #2d2d2d;
        padding: 15px 20px;
        border-radius: 20px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #3d3d3d;
        padding: 15px 20px;
        border-radius: 20px;
        margin: 10px 0;
        margin-right: 20%;
    }
    
    .process-step {
        background-color: #2a2a2a;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 3px solid #4CAF50;
        animation: pulse 1.5s infinite;
        transition: all 0.3s ease;
    }
    
    .process-step.completed {
        background-color: #1a4a1a;
        border-left: 3px solid #4CAF50;
        animation: none;
        opacity: 1;
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .image-item {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 10px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .image-item:hover {
        transform: scale(1.02);
        background-color: #3a3a3a;
    }
    
    .selected-image {
        border: 2px solid #4CAF50;
    }
    
    .generate-button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        font-size: 16px;
        margin: 20px 0;
    }
    
    .final-answer {
        background-color: #2d4a2d;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .processing-indicator {
        text-align: center;
        padding: 20px;
    }
    
    .spinner {
        border: 3px solid #333;
        border-top: 3px solid #4CAF50;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .example-button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin: 3px 0 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        font-size: 12px !important;
    }
    
    .example-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15) !important;
        background: linear-gradient(135deg, #45a049 0%, #4CAF50 100%) !important;
    }
    
    .example-section {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "message_results" not in st.session_state:
    st.session_state.message_results = {}

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 for display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def display_process_step(step_name: str, status: str = "running"):
    """Display processing step with animation"""
    icon = "⏳" if status == "running" else "✅"
    st.markdown(f"""
    <div class="process-step">
        {icon} {step_name}
    </div>
    """, unsafe_allow_html=True)

def process_question_with_steps(question: str):
    """Process question using LangGraph pipeline with real-time state tracking"""
    import time
    
    # Create placeholder for processing steps
    process_container = st.container()
    
    with process_container:
        st.markdown("### 🔄 Đang xử lý câu hỏi...")
        
        # Initialize step placeholders
        step_placeholders = {
            "search_initial_context": st.empty(),
            "rerank_context": st.empty(), 
            "select_context": st.empty(),
            "search_images": st.empty(),
            "generate_answer": st.empty()
        }
        
        step_names = {
            "search_initial_context": "Tìm kiếm context ban đầu",
            "rerank_context": "Sắp xếp lại context theo độ liên quan", 
            "select_context": "Lựa chọn context tối ưu",
            "search_images": "Tìm kiếm ảnh liên quan",
            "generate_answer": "Tạo câu trả lời"
        }
        
        try:
            # Create the LangGraph pipeline
            graph = create_app_rag_graph()
            
            # Initialize state for pipeline
            initial_state = {"question": question}
            
            # Use streaming to track real-time progress
            final_result = None
            
            # Stream through the pipeline and update UI in real-time
            for chunk in graph.stream(initial_state):
                # chunk contains: {node_name: updated_state}
                for node_name, state in chunk.items():
                    if node_name in step_placeholders:
                        # Show loading for current step
                        step_placeholders[node_name].markdown(
                            f'<div class="process-step">⏳ {step_names[node_name]}...</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # Small delay to show the loading state
                        time.sleep(0.3)
                        
                        # Show completion
                        step_placeholders[node_name].markdown(
                            f'<div class="process-step completed">✅ {step_names[node_name]} hoàn thành</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # Store the final state
                        final_result = state
                        
                        # Check for errors
                        if state.get("error"):
                            st.error(f"❌ Lỗi ở bước {step_names[node_name]}: {state['error']}")
                            return None
            
            if not final_result:
                st.error("❌ Không nhận được kết quả từ pipeline")
                return None
            
            # Convert pipeline result to app format
            result = {
                "question": final_result.get("question"),
                "initial_chunks": final_result.get("initial_chunks", []),
                "reranked_chunks": final_result.get("reranked_chunks", []),
                "selected_contexts": final_result.get("selected_contexts", []),
                "answer": final_result.get("answer"),
                "available_images": final_result.get("available_images", []),
                "error": final_result.get("error")
            }
            
            return result
            
        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {str(e)}")
            return None

def display_response_parts(result: Dict, message_key: str):
    """Display the three parts of response"""
    if not result:
        return
    
    # Check if user has made a choice for this message
    choice_key = f"choice_{message_key}"
    selected_image_key = f"selected_image_{message_key}"
    generated_image_key = f"generated_image_{message_key}"
    
    # If final choice is made, show final answer
    if choice_key in st.session_state:
        if st.session_state[choice_key] == "select_existing" and selected_image_key in st.session_state:
            final_state = {
                **result,
                "user_choice": "select_existing",
                "selected_image": st.session_state[selected_image_key]
            }
            display_final_answer(final_state)
            return
        elif st.session_state[choice_key] == "generate_new" and generated_image_key in st.session_state:
            final_state = {
                **result,
                "user_choice": "generate_new", 
                "generated_image": st.session_state[generated_image_key]
            }
            display_final_answer(final_state)
            return
    
    # Otherwise show the selection interface
    
    # Part 1: Text Response
    st.markdown("### 💬 Câu trả lời:")
    answer = result.get("answer", "Không có câu trả lời.")
    st.markdown(f"""
    <div class="assistant-message">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Part 2: Available Images
    available_images = result.get("available_images", [])
    if available_images:
        st.markdown("### 🖼️ Ảnh liên quan:")
        st.info("👆 Chọn một ảnh phù hợp hoặc tạo ảnh mới bên dưới")
        
        # Remove all spacing between elements
        st.markdown("""
        <style>
        .element-container {
            margin-bottom: 0px !important;
        }
        .stForm {
            margin-bottom: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display each image in its own row with completely integrated layout
        for i, img_data in enumerate(available_images):
            caption = img_data.get("caption", "Không có mô tả")
            
            # Get image data with larger size
            if img_data.get("image_path") and os.path.exists(img_data["image_path"]):
                image = Image.open(img_data["image_path"])
                image = image.resize((140, 140), Image.Resampling.LANCZOS)
                
                # Convert to base64 for inline display
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:140px;height:140px;border-radius:10px;object-fit:cover;">'
            else:
                img_html = '<div style="width:140px;height:140px;background-color:#2a2a2a;display:flex;align-items:center;justify-content:center;border-radius:10px;border:2px solid #444;"><span style="color:#666;font-size:14px;">❌ Ảnh lỗi</span></div>'
            
            # Create the visual container
            st.markdown(f"""
            <div style="width: 100%; margin: 0; background-color: #333; border-radius: 15px; padding: 15px; display: flex; align-items: center; gap: 15px; height: 140px;">
                <div style="flex-shrink: 0;">
                    {img_html}
                </div>
                <div style="flex-grow: 1; padding: 0 10px;">
                    <p style="margin: 0; font-size: 18px; line-height: 1.5; color: #f5f5f5; font-weight: 400;">{caption}</p>
                </div>
                <div style="flex-shrink: 0; min-width: 80px; text-align: center;">
                    <div id="button_placeholder_{message_key}_{i}" style="height: 40px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add the actual button right after, positioned to overlay
            st.markdown('<div style="margin-top: -55px; text-align: right; padding-right: 30px;">', unsafe_allow_html=True)
            if st.button(f"📸 Chọn", key=f"select_img_{message_key}_{i}", type="secondary"):
                st.session_state[choice_key] = "select_existing"
                st.session_state[selected_image_key] = img_data
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Không tìm thấy ảnh liên quan trong cơ sở dữ liệu.")
    
    # Part 3: Generate Image Button
    st.markdown("### 🎨 Hoặc tạo ảnh mới:")
    if st.button("🖼️ Tạo ảnh minh họa cho câu trả lời", key=f"generate_new_image_{message_key}"):
        # Create process container for image generation steps
        image_process_container = st.container()
        
        with image_process_container:
            st.markdown("#### 🎨 Đang tạo ảnh...")
            
            # Initialize empty placeholders for image generation steps
            img_step1 = st.empty()
            img_step2 = st.empty()
            
            try:
                # Step 1: Prepare context
                img_step1.markdown('<div class="process-step">⏳ Chuẩn bị context cho sinh ảnh...</div>', unsafe_allow_html=True)
                time.sleep(0.8)
                
                question = result.get("question", "")
                selected_contexts = result.get("selected_contexts", [])
                answer = result.get("answer", "")
                
                if not answer:
                    img_step1.markdown('<div class="process-step">❌ Không có câu trả lời để tạo ảnh</div>', unsafe_allow_html=True)
                    return
                
                main_context = selected_contexts[0]['content'] if selected_contexts else ""
                
                img_step1.markdown('<div class="process-step completed">✅ Chuẩn bị context hoàn thành</div>', unsafe_allow_html=True)
                time.sleep(0.5)
                
                # Step 2: Generate image with AI (using pipeline function)
                img_step2.markdown('<div class="process-step">⏳ Đang sinh ảnh với AI (có thể mất 30-90s)...</div>', unsafe_allow_html=True)
                
                # Use the generate_medical_image function from pipeline.py
                generated_image = generate_medical_image(question, main_context, answer)
                
                img_step2.markdown('<div class="process-step completed">✅ Đã tạo ảnh thành công!</div>', unsafe_allow_html=True)
                time.sleep(0.3)
                
                st.success("🎉 Đã tạo ảnh thành công!")
                
                # Save generated image to session state
                st.session_state[choice_key] = "generate_new"
                st.session_state[generated_image_key] = generated_image
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Lỗi tạo ảnh: {str(e)}")
                # Clear any loading steps on error
                img_step1.empty()
                img_step2.empty()

def display_final_answer(final_state: Dict):
    """Display final answer with selected/generated image"""
    st.markdown("### 🎯 Câu trả lời cuối cùng:")
    
    answer = final_state.get("answer", "")
    user_choice = final_state.get("user_choice", "")
    
    # Display text answer
    st.markdown(f"""
    <div class="final-answer">
        <strong>📝 Câu trả lời:</strong><br>
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Display selected or generated image
    if user_choice == "generate_new":
        generated_image = final_state.get("generated_image")
        if generated_image and os.path.exists(generated_image["image_path"]):
            st.markdown("**🖼️ Ảnh được tạo:**")
            image = Image.open(generated_image["image_path"])
            st.image(image, caption=generated_image.get("caption", ""), width=400)
    
    elif user_choice == "select_existing":
        selected_image = final_state.get("selected_image")
        if selected_image and os.path.exists(selected_image["image_path"]):
            st.markdown("**🖼️ Ảnh đã chọn:**")
            image = Image.open(selected_image["image_path"])
            st.image(image, caption=selected_image.get("caption", ""), width=400)

# Main app
def main():
    st.title("Hệ thống truy xuất và hỏi đáp bổ sung hình ảnh bằng tiếng Việt")
    st.markdown("*Hệ thống tìm kiếm và tạo câu trả lời thông minh với hình ảnh*")
    
    # Display chat history and responses
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                👤 {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # For assistant messages, check if we have a result to display
            message_key = f"msg_{i}"
            if message_key in st.session_state.message_results:
                # Display the interactive response parts
                display_response_parts(st.session_state.message_results[message_key], message_key)
            else:
                # Display simple text message
                st.markdown(f"""
                <div class="assistant-message">
                    🤖 {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Example questions - always visible above chat input
    st.markdown("""
    <div class="example-section">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: #4CAF50;">💡 Câu hỏi mẫu</h4>
        <p style="margin-bottom: 10px; color: #ccc; font-size: 13px;">Click để tự động gửi câu hỏi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define example questions from the test file
    example_questions = [
        "Có bắt buộc phải sử dụng thuốc Carbogast trong khi mang thai và cho con bú không?",
        "Ngoài giảm đau, Acepron 250 mg còn có thêm tác dụng nào khác?",
        "Uống thuốc Comiaryl đúng cách như thế nào?"
    ]
    
    # Create columns for the example questions  
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(
            "🤰 Thuốc Carbogast\nvà thai kỳ", 
            key="example_1",
            help="Click để tự động gửi câu hỏi",
            use_container_width=True
        ):
            if not st.session_state.processing:
                st.session_state.messages.append({"role": "user", "content": example_questions[0]})
                st.session_state.processing = True
                st.rerun()
    
    with col2:
        if st.button(
            "💊 Tác dụng\nAcepron 250mg", 
            key="example_2",
            help="Click để tự động gửi câu hỏi",
            use_container_width=True
        ):
            if not st.session_state.processing:
                st.session_state.messages.append({"role": "user", "content": example_questions[1]})
                st.session_state.processing = True
                st.rerun()
    
    with col3:
        if st.button(
            "🥤 Cách uống\nComiaryl", 
            key="example_3",
            help="Click để tự động gửi câu hỏi",
            use_container_width=True
        ):
            if not st.session_state.processing:
                st.session_state.messages.append({"role": "user", "content": example_questions[2]})
                st.session_state.processing = True
                st.rerun()
    
    # Chat input
    question = st.chat_input("Nhập câu hỏi của bạn...")
    
    if question and not st.session_state.processing:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.processing = True
        st.rerun()  # Rerun to show user message immediately
    
    # Process the latest question if we're not already processing
    if st.session_state.processing and len(st.session_state.messages) > 0:
        latest_message = st.session_state.messages[-1]
        if latest_message["role"] == "user":
            # Process question with steps
            result = process_question_with_steps(latest_message["content"])
            
            if result:
                # Store result for the response message
                message_key = f"msg_{len(st.session_state.messages)}"
                st.session_state.message_results[message_key] = result
                
                # Add assistant message to history
                answer = result.get("answer", "Không có câu trả lời.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Add error message
                st.session_state.messages.append({"role": "assistant", "content": "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi."})
            
            st.session_state.processing = False
            st.rerun()  # Rerun to show the response

if __name__ == "__main__":
    main()
