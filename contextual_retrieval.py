import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ContextualRetrieval:
    def __init__(self, max_workers: int = 5):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.model_name = os.getenv("AZURE_OPENAI_MODEL")
        self.max_workers = max_workers
        self.stats_lock = threading.Lock()
    
    def load_chunks_dataset(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load chunks dataset from JSONL file"""
        chunks_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks_data.append(json.loads(line))
        return chunks_data
    
    def enhance_content_with_context(self, chunk: Dict[str, Any]) -> str:
        """Use LLM to enhance content with contextual information from chunk metadata"""
        
        original_content = chunk.get('content', '')
        section_path = chunk.get('section_path', [])
        keyword = chunk.get('metadata', {}).get('keyword', '')
        
        system_prompt = """Bạn là chuyên gia phân tích nội dung y tế/dược phẩm. Nhiệm vụ: phân tích nội dung và quyết định có cần bổ sung thông tin ngữ cảnh hay không.

CÁCH PHÂN TÍCH:
1. Đọc kỹ nội dung gốc
2. Xem xét thông tin ngữ cảnh có sẵn (keyword + section)
3. Quyết định: nội dung có cần bổ sung thông tin để rõ ràng hơn không?

NGUYÊN TẮC:
- CHỈ bổ sung khi thực sự cần thiết (nội dung mơ hồ, thiếu ngữ cảnh)
- BỔ SUNG tự nhiên, viết thành câu văn hoàn chỉnh và mạch lạc
- GIỮ NGUYÊN ý nghĩa của nội dung gốc
- TÍCH HỢP thông tin từ keyword/section một cách tự nhiên

VÍ DỤ:
- "thuốc này" → "thuốc [Tên thuốc]"
- "bệnh này" → "bệnh [Tên bệnh]" 
- Thêm ngữ cảnh: "Về [chủ đề], [nội dung gốc]"
- Hoặc: "[Chủ đề]: [nội dung gốc cải thiện]"

QUAN TRỌNG: Chỉ trả về JSON format với 2 trường:
- "modify": "yes" nếu cần sửa, "no" nếu không cần
- "new_content": nội dung sau khi cải thiện (rỗng nếu modify = "no")"""

        # Build context info
        section_str = ' > '.join(section_path) if section_path else ''
        
        user_prompt = f"""PHÂN TÍCH NỘI DUNG:
"{original_content}"

THÔNG TIN NGỮ CẢNH:
- Keyword: {keyword if keyword else 'Không có'}
- Section: {section_str if section_str else 'Không có'}

NHIỆM VỤ:
1. Nội dung có rõ ràng và đầy đủ ngữ cảnh không?
2. Có cần bổ sung thông tin từ keyword/section không?
3. Nếu cần: viết lại thành câu văn hoàn chỉnh, mạch lạc với thông tin bổ sung

Trả về JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Clean response if needed
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()
                
                result = json.loads(response_text)
                
                # Validate JSON structure
                if 'modify' not in result or 'new_content' not in result:
                    return original_content
                
                modify = result.get('modify', '').lower()
                new_content = result.get('new_content', '').strip()
                
                if modify == 'yes' and new_content:
                    return new_content
                else:
                    return original_content
                    
            except json.JSONDecodeError as e:
                # JSON parsing failed, return original content
                return original_content
                
        except Exception as e:
            # Handle Azure OpenAI content filter errors and other API errors silently
            # Just return original content for any API errors (including content filter)
            return original_content
    
    def process_single_chunk(self, chunk_with_index: tuple) -> Dict[str, Any]:
        """Process a single chunk with enhancement"""
        index, chunk = chunk_with_index
        
        try:
            original_content = chunk.get('content', '')
            
            if not original_content:
                return {
                    'index': index,
                    'chunk': chunk,
                    'enhanced': False,
                    'error': None
                }
            
            # Enhance content directly
            enhanced_content = self.enhance_content_with_context(chunk)
            
            # Create enhanced chunk
            enhanced_chunk = chunk.copy()
            enhanced = False
            
            # Update if content changed
            if enhanced_content != original_content:
                enhanced_chunk['content'] = enhanced_content
                enhanced_chunk['char_count'] = len(enhanced_content)
                enhanced_chunk['token_count'] = len(enhanced_content.split())
                
                # Add metadata for tracking
                if 'metadata' not in enhanced_chunk:
                    enhanced_chunk['metadata'] = {}
                enhanced_chunk['metadata']['original_content'] = original_content
                enhanced_chunk['metadata']['enhancement_applied'] = True
                enhanced = True
            else:
                enhanced_chunk['metadata'] = enhanced_chunk.get('metadata', {})
                enhanced_chunk['metadata']['enhancement_applied'] = False
            
            return {
                'index': index,
                'chunk': enhanced_chunk,
                'enhanced': enhanced,
                'original_length': len(original_content),
                'enhanced_length': len(enhanced_content),
                'error': None
            }
            
        except Exception as e:
            return {
                'index': index,
                'chunk': chunk,
                'enhanced': False,
                'error': str(e)
            }

    
    def enhance_chunks_dataset(self, input_jsonl_path: str, output_jsonl_path: str, 
                              sample_limit: int = None, max_workers: int = None) -> Dict[str, Any]:
        """Enhance content in chunks dataset with parallel processing"""
        
        if max_workers is None:
            max_workers = self.max_workers
        
        print(f"Đang tải chunks dataset từ {input_jsonl_path}...")
        chunks_dataset = self.load_chunks_dataset(input_jsonl_path)
        
        if sample_limit:
            chunks_dataset = chunks_dataset[:sample_limit]
            print(f"Giới hạn xử lý {sample_limit} chunks")
        
        print(f"Sẽ xử lý {len(chunks_dataset)} chunks với {max_workers} workers song song")
        
        # Prepare chunks with indices for parallel processing
        chunks_with_indices = [(i, chunk) for i, chunk in enumerate(chunks_dataset)]
        
        enhanced_chunks = [None] * len(chunks_dataset)  # Pre-allocate to maintain order
        enhancement_stats = {
            'total_processed': 0,
            'successfully_enhanced': 0,
            'skipped_already_good': 0,
            'content_filtered': 0,  # New stat for content filter cases
            'failed_enhancements': 0,
            'api_errors': 0,
            'avg_length_increase': 0
        }
        
        total_length_increase = 0
        content_filtered_count = 0
        
        print(f"\nĐang xử lý chunks song song với {max_workers} workers...")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.process_single_chunk, chunk_with_index): chunk_with_index[0]
                for chunk_with_index in chunks_with_indices
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(chunks_dataset), desc="Đang xử lý") as pbar:
                for future in as_completed(future_to_chunk):
                    try:
                        result = future.result()
                        index = result['index']
                        enhanced_chunks[index] = result['chunk']
                        
                        # Update stats with thread safety
                        with self.stats_lock:
                            enhancement_stats['total_processed'] += 1
                            
                            if result['error']:
                                if "content_filter" in str(result['error']):
                                    enhancement_stats['content_filtered'] += 1
                                else:
                                    enhancement_stats['api_errors'] += 1
                            elif result['enhanced']:
                                enhancement_stats['successfully_enhanced'] += 1
                                total_length_increase += result['enhanced_length'] - result['original_length']
                            else:
                                enhancement_stats['skipped_already_good'] += 1
                        
                        pbar.update(1)
                        
                        # Update progress info every 50 items
                        if enhancement_stats['total_processed'] % 50 == 0:
                            success_rate = enhancement_stats['successfully_enhanced'] / enhancement_stats['total_processed'] * 100
                            error_rate = enhancement_stats['api_errors'] / enhancement_stats['total_processed'] * 100
                            filter_rate = enhancement_stats['content_filtered'] / enhancement_stats['total_processed'] * 100
                            pbar.set_postfix({
                                'Cải thiện': f"{success_rate:.1f}%",
                                'Lỗi': f"{error_rate:.1f}%",
                                'Filtered': f"{filter_rate:.1f}%"
                            })
                        
                    except Exception as e:
                        print(f"Lỗi khi xử lý task: {e}")
                        enhancement_stats['failed_enhancements'] += 1
                        pbar.update(1)
        
        # Calculate average length increase
        if enhancement_stats['successfully_enhanced'] > 0:
            enhancement_stats['avg_length_increase'] = total_length_increase / enhancement_stats['successfully_enhanced']
        
        # Save enhanced dataset
        print(f"\nĐang lưu chunks dataset đã cải thiện vào {output_jsonl_path}...")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in enhanced_chunks:
                if chunk is not None:  # Safety check
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # Print statistics
        print(f"\n=== KẾT QUẢ CẢI THIỆN CHUNKS (PARALLEL) ===")
        print(f"Tổng số chunks: {len(chunks_dataset):,}")
        print(f"Đã xử lý: {enhancement_stats['total_processed']:,}")
        print(f"Đã cải thiện: {enhancement_stats['successfully_enhanced']:,}")
        print(f"Giữ nguyên (đã tốt): {enhancement_stats['skipped_already_good']:,}")
        print(f"Content filtered: {enhancement_stats['content_filtered']:,}")
        print(f"Lỗi API: {enhancement_stats['api_errors']:,}")
        print(f"Lỗi khác: {enhancement_stats['failed_enhancements']:,}")
        print(f"Tỷ lệ cải thiện: {enhancement_stats['successfully_enhanced']/enhancement_stats['total_processed']*100:.2f}%")
        print(f"Tỷ lệ content filter: {enhancement_stats['content_filtered']/enhancement_stats['total_processed']*100:.2f}%")
        print(f"Tỷ lệ lỗi: {enhancement_stats['api_errors']/enhancement_stats['total_processed']*100:.2f}%")
        if enhancement_stats['successfully_enhanced'] > 0:
            print(f"Độ dài tăng trung bình: {enhancement_stats['avg_length_increase']:.1f} ký tự")
        print(f"Số workers sử dụng: {max_workers}")
        
        # Show some examples
        self.show_enhancement_examples(enhanced_chunks[:10])
        
        return enhancement_stats
    
    def show_enhancement_examples(self, chunks: List[Dict[str, Any]], num_examples: int = 5):
        """Show examples of content enhancement"""
        print(f"\n=== VÍ DỤ CẢI THIỆN CONTENT ===")
        
        enhanced_count = 0
        for i, chunk in enumerate(chunks):
            if chunk is None:  # Skip None chunks
                continue
                
            if enhanced_count >= num_examples:
                break
                
            if chunk.get('metadata', {}).get('enhancement_applied', False):
                original = chunk['metadata']['original_content']
                enhanced = chunk['content']
                section_path = chunk.get('section_path', [])
                keyword = chunk.get('metadata', {}).get('keyword', '')
                
                enhanced_count += 1
                print(f"\nVí dụ {enhanced_count}:")
                print(f"Section: {' > '.join(section_path)}")
                print(f"Keyword: {keyword}")
                print(f"Gốc: {original}")
                print(f"Cải thiện: {enhanced}")
                print(f"Tăng {len(enhanced) - len(original)} ký tự")
                print("-" * 80)
        
        if enhanced_count == 0:
            print("Không có ví dụ nào được cải thiện trong batch này")


def main():
    # Initialize contextual retrieval with parallel processing
    contextual_retrieval = ContextualRetrieval(max_workers=8)
    
    # Enhance the chunks dataset with parallel processing
    results = contextual_retrieval.enhance_chunks_dataset(
        input_jsonl_path="datasets/chunks/context_corpus_clean.jsonl",
        output_jsonl_path="datasets/chunks/context_corpus_enhanced.jsonl",
        sample_limit=None,
        max_workers=10
    )
    
    print(f"\nHoàn thành cải thiện chunks dataset!")
    print(f"Dataset đã cải thiện được lưu tại: datasets/chunks/context_corpus_enhanced.jsonl")


if __name__ == "__main__":
    main()
