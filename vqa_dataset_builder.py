import os
import openai
from datasets import load_dataset
import json
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DETAILED_SYSTEM_PROMPT = """
Bạn là một chuyên gia phân loại mô tả hình ảnh. Nhiệm vụ của bạn là đọc kỹ từng mô tả hình ảnh được cung cấp (trong vai trò user message) và quyết định xem nên 'GIỮ LẠI' hay 'LOẠI BỎ' mô tả đó. Mục tiêu là chỉ giữ lại các mô tả tập trung vào một thực thể cụ thể, riêng biệt, có khả năng được sử dụng để truy xuất thông tin chi tiết cho các câu hỏi về thực thể đó.

**QUY TRÌNH PHÂN TÍCH BẮT BUỘC (Hãy suy nghĩ nội bộ theo các bước này trước khi đưa ra quyết định cuối cùng):**

1.  **Liệt kê Thực thể Tiềm năng:** Xác định và liệt kê tất cả các đối tượng, người, địa điểm, sản phẩm, hoặc sự kiện cụ thể, được đặt tên hoặc mô tả chi tiết trong mô tả được cung cấp.
2.  **Đánh giá Mức độ Chi tiết:** Với mỗi thực thể tiềm năng đã liệt kê:
    * Nó có tên riêng (ví dụ: tên thương hiệu, tên địa danh) không?
    * Nó có được mô tả với các đặc điểm nhận dạng riêng biệt không?
3.  **Phân tích Mối quan hệ và Vai trò (Nếu có nhiều hơn một đối tượng/sản phẩm chi tiết):**
    * Có **DUY NHẤT MỘT** thực thể nào rõ ràng là **trung tâm chính** của toàn bộ mô tả không?
    * Nếu có nhiều đối tượng/sản phẩm được mô tả chi tiết:
        * Chúng có phải là bộ phận của một thực thể lớn hơn không (ví dụ: biển hiệu là một phần của cửa hàng)?
        * **Quan trọng:** Chúng có cùng thuộc một thương hiệu và được trình bày như một **dòng sản phẩm, bộ sưu tập, hoặc trong cùng một quảng cáo cho thương hiệu đó không?** Nếu vậy, "dòng sản phẩm của Thương hiệu X" hoặc "quảng cáo cho Thương hiệu X" có thể được xem là thực thể chính.
        * Hay chúng là các thực thể riêng biệt, được trình bày với mức độ quan trọng và chi tiết tương đương nhau, tạo thành nhiều tiêu điểm không liên quan chặt chẽ dưới một thực thể bao trùm duy nhất?
4.  **Kiểm tra Yếu tố Nguồn/Ghi công:** Có tên website, công ty nào trong mô tả chỉ mang tính chất là nguồn gốc của bức ảnh (watermark), hoặc một ghi công rất nhỏ cho người tạo ra sản phẩm/biển hiệu/quảng cáo không? (Ví dụ: "Ảnh bởi X", "Biển hiệu làm bởi Y ở góc nhỏ", "Poster thiết kế bởi Z"). Nếu có, các yếu tố này **không phải là thực thể chính của *nội dung được mô tả trong ảnh*** và cần được bỏ qua khi xác định thực thể chính của cảnh.

**SAU KHI THỰC HIỆN QUY TRÌNH PHÂN TÍCH TRÊN, HÃY ÁP DỤNG CÁC QUY TẮC SAU ĐỂ ĐƯA RA QUYẾT ĐỊNH:**

1.  **TIÊU CHÍ CHÍNH ĐỂ GIỮ LẠI:**
    * Mô tả phải tập trung vào **DUY NHẤT MỘT thực thể chính, cụ thể và có thể nhận dạng** (được xác định từ quy trình phân tích ở trên).
        * **Lưu ý quan trọng:** Một "thực thể chính duy nhất" cũng có thể là **"dòng sản phẩm của Thương hiệu X"** hoặc **"một quảng cáo cho Thương hiệu X"** nếu mô tả trình bày nhiều sản phẩm riêng lẻ nhưng chúng rõ ràng thuộc cùng một thương hiệu và được giới thiệu chung trong một ngữ cảnh (ví dụ: ảnh chụp sản phẩm, poster quảng cáo).
    * Tất cả các câu và chi tiết trong mô tả phải dùng để mô tả hoặc bổ sung thông tin cho **thực thể chính duy nhất đó**.
    * Hình ảnh được mô tả phải có khả năng trả lời các câu hỏi cụ thể về thực thể đó.

2.  **CÁC TRƯỜNG HỢP CẦN LOẠI BỎ:**
    * **Nội dung quá chung chung:** Chụp một con phố bất kỳ không tên, một đám đông không xác định, một cảnh sinh hoạt thường ngày không có điểm nhấn đặc biệt, một biển chỉ dẫn thông thường, một quảng cáo chung chung cho một loại hình dịch vụ/sản phẩm mà không có một thực thể cụ thể làm trung tâm.
    * **Nhiều thực thể chính hoặc nhiều tiêu điểm ngang bằng:** Nếu quy trình phân tích ở trên cho thấy mô tả đề cập đến **nhiều hơn một thực thể được đặt tên và mô tả chi tiết**, và chúng được trình bày với **mức độ quan trọng, chi tiết tương đương nhau, tạo thành nhiều tiêu điểm cho cảnh**, mà không có một thực thể nào là trung tâm rõ ràng vượt trội **(và chúng không cùng thuộc một dòng sản phẩm/quảng cáo của một thương hiệu duy nhất như đã nêu ở mục GIỮ LẠI)**.
    * **Không rõ thực thể chính:** Mô tả lan man, không làm nổi bật được một thực thể cụ thể nào sau khi đã phân tích.

3.  **XỬ LÝ CÁC YẾU TỐ PHỤ (QUAN TRỌNG):** (Quy tắc này đã tốt, nhắc lại để áp dụng sau khi xác định thực thể chính của cảnh)
    * **Nguồn ảnh/Watermark/Ghi công nhỏ:** Nếu mô tả chứa tên website, tên công ty, hoặc một dòng chữ nhỏ mà rõ ràng chỉ là **nguồn gốc của bức ảnh (watermark)**, hoặc một **ghi công rất nhỏ** cho người tạo ra sản phẩm trong ảnh (ví dụ: tên người làm biển hiệu ở góc nhỏ, tên nhà xuất bản trên bìa sách), hãy **XEM ĐÓ LÀ THÔNG TIN PHỤ TRỢ hoặc siêu dữ liệu của bức ảnh**.
    * **KHÔNG loại bỏ mô tả CHỈ VÌ sự có mặt của các yếu tố phụ này NẾU phần còn lại của mô tả (nội dung chính của cảnh được chụp) vẫn tập trung vào một thực thể chính duy nhất theo tiêu chí ở mục 1.**

4.  **VÍ DỤ CỤ THỂ ĐỂ THAM KHẢO (Áp dụng sau quy trình phân tích CoT):**
    * **GIỮ LẠI:**
        * Mô tả tảng đá "Sở chỉ huy chiến dịch Điện Biên Phủ".
        * Mô tả cửa hàng "NĂM SÁNH 79".
        * Mô tả siêu thị "K - MARKET".
        * Mô tả "Bánh đậu xanh đặc sản Hội An" thương hiệu "Nghĩa Anh", ngay cả khi có tên một website (được xem là nguồn ảnh/watermark, không phải thực thể chính của món bánh).
        * Biển hiệu "BON BON PARK", ngay cả khi có chữ "TRUNG CAO" nhỏ ở góc (ghi công nhỏ).
        * **Mô tả ba gói kẹo thanh khác nhau (ví dụ: 'PRO BAR bite', 'PRO BAR Meal', 'GREEN SUPERFOOD') được đặt cùng nhau, tất cả đều thuộc cùng một thương hiệu chính (ví dụ: PROBAR). Mô tả có thể bao gồm tên của một đơn vị đóng gói hoặc thiết kế quảng cáo (ví dụ: 'LOUIS PACKAGING') được xem là thông tin phụ/ghi công.** (Lý do sau CoT: Các gói kẹo là các sản phẩm trong cùng một dòng/quảng cáo của thương hiệu PROBAR. Thực thể chính là "quảng cáo/dòng sản phẩm của PROBAR". Louis Packaging là ghi công phụ.)
    * **LOẠI BỎ:**
        * Mô tả một con phố chung chung có nhiều tòa nhà, xe cộ, ngay cả khi có biển số xe hay SĐT trên một xe tải (thực thể chính là "con phố chung chung", xe tải chỉ là một chi tiết).
        * Mô tả một tấm biển hiệu mà trên đó quảng cáo cho hai công ty khác nhau với đầy đủ thông tin cho cả hai (ví dụ: "Quảng Cáo Đại Phát" và "RANCHO RELAXO" trên cùng một biển mà không rõ cái nào chính, cái nào phụ).
        * Mô tả chung chung về lời khuyên mở SPA với hình ảnh một người phụ nữ không tên.
        * **Mô tả một chiếc xích lô (có chi tiết) đang đi qua một quán ăn 'Mỹ Hải' (cũng có chi tiết về món ăn trên biển hiệu).** (Lý do sau CoT: Cả xích lô và quán Mỹ Hải đều là các thực thể được mô tả với chi tiết riêng, tạo thành hai tiêu điểm trong cảnh, không có một thực thể chính duy nhất vượt trội.)

5.  **ĐỊNH DẠNG TRẢ LỜI (QUAN TRỌNG):**
    * Dòng đầu tiên: Chỉ ghi "GIỮ LẠI" hoặc "LOẠI BỎ".
    * Dòng thứ hai: Ghi "Lý do: " theo sau là một giải thích ngắn gọn, rõ ràng cho quyết định của bạn, dựa trên các quy tắc và kết quả phân tích.
"""

openai_client = None
try:
    openai_client = openai.OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure that the OPENAI_API_KEY environment variable is set.")

def call_llm_for_classification(description_to_classify):
    if not openai_client:
        tqdm.write("CLIENT ERROR: OpenAI client was not initialized correctly (possibly missing or invalid API key).")
        return "CLIENT ERROR\nReason: OpenAI client was not initialized correctly (possibly missing or invalid API key)."
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DETAILED_SYSTEM_PROMPT},
                {"role": "user", "content": description_to_classify}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except openai.APIConnectionError as e:
        error_message = f"OpenAI API connection error: {e.__cause__}" if e.__cause__ else f"OpenAI API connection error: {e}"
        tqdm.write(error_message)
        return f"CONNECTION ERROR\nReason: {error_message}"
    except openai.RateLimitError as e:
        error_message = f"OpenAI API rate limit error: {e.body.get('message', str(e)) if e.body else str(e)}"
        tqdm.write(error_message)
        return f"RATE LIMIT ERROR\nReason: {error_message}"
    except openai.APIStatusError as e:
        error_message = f"OpenAI API status error: {e.status_code} - {e.message}"
        tqdm.write(error_message)
        return f"API ERROR ({e.status_code})\nReason: {e.message}"
    except Exception as e:
        tqdm.write(f"Unknown error when calling LLM API: {e}")
        return "UNKNOWN ERROR\nReason: An unexpected error occurred while calling the API."

if __name__ == "__main__":
    if not openai_client:
        print("Cannot continue because OpenAI client is not initialized. Please check your API key.")
    else:
        print("Loading dataset '5CD-AI/Viet-ViTextVQA-gemini-VQA'...")
        kept_samples = []
        output_json_file = "kept_samples_dataset.json"

        try:
            ds = load_dataset("5CD-AI/Viet-ViTextVQA-gemini-VQA")
            print("Dataset loaded successfully!")
            actual_samples_to_test = len(ds["train"])
            
            print(f"\n--- STARTING CLASSIFICATION OF {actual_samples_to_test} DESCRIPTIONS (USING GPT-4O-MINI) ---")

            for i in tqdm(range(actual_samples_to_test), total=actual_samples_to_test, desc="Processing classifications"):
                original_row_data = ds["train"][i]
                description_to_filter = original_row_data.get("description", "")
                
                llm_response = call_llm_for_classification(description_to_filter)

                if "ERROR" in llm_response.splitlines()[0]:
                    tqdm.write(f"Error processing sample index {i}: {llm_response.splitlines()[0]}")
                
                response_parts = llm_response.strip().split("\nLý do: ", 1)
                decision = response_parts[0].strip()
                
                reason = ""
                if len(response_parts) > 1:
                    reason = response_parts[1].strip()
                elif decision not in ["CLIENT ERROR", "CONNECTION ERROR", "RATE LIMIT ERROR", "API ERROR", "UNKNOWN ERROR", "ERROR PROCESSING"]:
                    reason = "No reason provided by LLM or incorrect response format."
                else:
                    if "\nReason: " in llm_response:
                        reason = llm_response.split("\nReason: ",1)[1]
                    else:
                        reason = decision

                if decision == "GIỮ LẠI":
                    output_sample = {}
                    original_image_id_str = original_row_data.get("id")
                    if original_image_id_str is not None:
                        try:
                            output_sample["id"] = int(original_image_id_str)
                        except ValueError:
                            tqdm.write(f"Warning: id '{original_image_id_str}' is not a pure number. Keeping string form for 'id' for sample index {i}.")
                            output_sample["id"] = original_image_id_str 
                    else:
                        tqdm.write(f"Warning: Missing id for sample at index {i}. Using index as temporary 'id'.")
                        output_sample["id"] = i 

                    output_sample["description"] = description_to_filter
                    output_sample["llm_reason"] = reason

                    kept_samples.append(output_sample) 
            
            print(f"\nSaving {len(kept_samples)} kept samples to file '{output_json_file}'...")
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(kept_samples, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved! There are a total of {len(kept_samples)} samples in the file '{output_json_file}'.")

        except Exception as e:
            print(f"An error occurred while loading or processing the dataset: {e}")
            print("Please check the dataset name, network connection, or other issues.")