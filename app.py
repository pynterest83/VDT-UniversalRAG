import gradio as gr
from pipeline import create_enhanced_rag_graph

def create_simple_chat():
    """Create simple chat interface"""
    
    # Initialize pipeline
    app = create_enhanced_rag_graph()
    
    def chat_fn(message, history):
        """Simple chat function"""
        try:
            inputs = {"question": message}
            
            # Get final result from pipeline
            final_output = None
            for output in app.stream(inputs, stream_mode="values"):
                final_output = output
            
            if final_output and final_output.get("final_answer"):
                return final_output["final_answer"]
            else:
                return "Xin lỗi, tôi không thể trả lời câu hỏi này."
                
        except Exception as e:
            return f"Đã xảy ra lỗi: {str(e)}"
    
    # Create interface
    interface = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        title="🏥 Medical RAG Chatbot",
        description="Chatbot AI y tế sử dụng RAG và GPT-4o",
        examples=[
            "Bách thảo sương dùng để chữa chảy máu kẽ răng như thế nào?",
            "Thuốc Paracetamol có tác dụng gì?",
        ]
    )
    
    return interface

if __name__ == "__main__":
    demo = create_simple_chat()
    demo.launch()
