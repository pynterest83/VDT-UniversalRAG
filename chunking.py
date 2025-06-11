import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import uuid
from transformers import AutoTokenizer

# Configuration
INPUT_DIR = 'corpus'
OUTPUT_DIR = 'chunks'
OUTPUT_FILE = 'chunked_corpus.jsonl'

# Chunking parameters for BGE-M3 optimization
CHUNK_SIZE = 512  # tokens
OVERLAP_SIZE = 128  # tokens
CONTEXT_PREFIX_SIZE = 100  # Reserve space for Vietnamese context prefix

# Initialize tokenizer for BGE-M3
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for Vietnamese text
    """
    # Vietnamese sentence ending patterns
    sentence_endings = r'[.!?…]+'
    
    # Split by sentence endings but keep the delimiter
    sentences = re.split(f'({sentence_endings})', text)
    
    # Combine sentences with their endings
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            ending = sentences[i + 1].strip()
            sentence = sentence + ending
        
        if sentence:
            combined_sentences.append(sentence)
    
    # Handle last sentence if no ending punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        combined_sentences.append(sentences[-1].strip())
    
    return combined_sentences

def count_tokens(text: str) -> int:
    """
    Count tokens using BGE-M3 tokenizer
    """
    return len(tokenizer.encode(text, add_special_tokens=False))

def create_vietnamese_context(document_metadata: Dict, section_path: List[str]) -> str:
    """
    Create Vietnamese context prefix
    """
    context_parts = []
    
    if document_metadata.get('heading'):
        context_parts.append(f"Tài liệu: {document_metadata['heading']}")
    
    if document_metadata.get('keyword'):
        context_parts.append(f"Chủ đề: {document_metadata['keyword']}")
    
    if section_path:
        if len(section_path) == 1:
            context_parts.append(f"Mục: {section_path[0]}")
        else:
            context_parts.append(f"Mục: {' > '.join(section_path)}")
    
    if context_parts:
        return f"Nội dung này thuộc {', '.join(context_parts)}. "
    else:
        return ""

def create_fixed_size_chunks(
    text: str,
    document_id: Any,
    document_metadata: Dict,
    section_path: List[str],
    chunk_type: str = "content"
) -> List[Dict]:
    """
    Create fixed-size chunks with overlap and sentence boundary preservation
    """
    if not text:
        return []
    
    # Create context prefix
    context_prefix = create_vietnamese_context(document_metadata, section_path)
    context_tokens = count_tokens(context_prefix)
    
    # Available space for actual content
    available_tokens = CHUNK_SIZE - context_tokens - 10  # 10 tokens buffer
    
    if available_tokens <= 0:
        print(f"Warning: Context prefix too long for document {document_id}")
        available_tokens = CHUNK_SIZE // 2
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence is too long, split it by words
        if sentence_tokens > available_tokens:
            # Handle oversized sentence
            words = sentence.split()
            word_chunk = []
            word_chunk_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(' '.join(word_chunk + [word]))
                if word_tokens <= available_tokens:
                    word_chunk.append(word)
                    word_chunk_tokens = word_tokens
                else:
                    if word_chunk:
                        # Create chunk from accumulated words
                        chunk_text = ' '.join(word_chunk)
                        chunks.append(create_chunk_object(
                            document_id, chunk_text, context_prefix,
                            document_metadata, section_path, chunk_type, len(chunks)
                        ))
                        
                        # Start overlap
                        overlap_words = word_chunk[-OVERLAP_SIZE//4:] if len(word_chunk) > OVERLAP_SIZE//4 else word_chunk
                        word_chunk = overlap_words + [word]
                        word_chunk_tokens = count_tokens(' '.join(word_chunk))
                    else:
                        # Single word too long, force add
                        word_chunk = [word]
                        word_chunk_tokens = count_tokens(word)
            
            # Add remaining words
            if word_chunk:
                chunk_text = ' '.join(word_chunk)
                chunks.append(create_chunk_object(
                    document_id, chunk_text, context_prefix,
                    document_metadata, section_path, chunk_type, len(chunks)
                ))
            
            i += 1
            continue
        
        # Check if adding current sentence exceeds limit
        potential_tokens = current_chunk_tokens + sentence_tokens
        
        if potential_tokens <= available_tokens:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens = potential_tokens
            i += 1
        else:
            # Current chunk is full, create it
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(create_chunk_object(
                    document_id, chunk_text, context_prefix,
                    document_metadata, section_path, chunk_type, len(chunks)
                ))
                
                # Calculate overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                # Take sentences from the end for overlap
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sent_tokens = count_tokens(current_chunk_sentences[j])
                    if overlap_tokens + sent_tokens <= OVERLAP_SIZE:
                        overlap_sentences.insert(0, current_chunk_sentences[j])
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_tokens
            else:
                # No current sentences, start fresh
                current_chunk_sentences = []
                current_chunk_tokens = 0
    
    # Add final chunk if there are remaining sentences
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(create_chunk_object(
            document_id, chunk_text, context_prefix,
            document_metadata, section_path, chunk_type, len(chunks)
        ))
    
    return chunks

def create_chunk_object(
    document_id: Any,
    chunk_text: str,
    context_prefix: str,
    document_metadata: Dict,
    section_path: List[str],
    chunk_type: str,
    chunk_index: int
) -> Dict:
    """
    Create a chunk object with contextual content
    """
    contextual_content = context_prefix + clean_text(chunk_text)
    
    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": document_id,
        "content": contextual_content,
        "chunk_type": chunk_type,
        "section_path": section_path,
        "char_count": len(contextual_content),
        "token_count": count_tokens(contextual_content),
        "metadata": {
            **document_metadata,
            "original_content": clean_text(chunk_text),
            "context_added": len(context_prefix),
            "chunk_index": chunk_index,
            "section_level": len(section_path)
        }
    }
    return chunk

def process_section(
    document_id: Any,
    section: Dict,
    document_metadata: Dict,
    section_path: List[str] = None
) -> List[Dict]:
    """
    Process a section using fixed-size chunking with Vietnamese context
    """
    if section_path is None:
        section_path = []
    
    chunks = []
    section_title = section.get('title', '')
    
    # Add current section to path
    current_path = section_path + [section_title] if section_title else section_path
    
    # Combine all content items into one text for consistent chunking
    content_list = section.get('content', [])
    combined_content = ' '.join([item for item in content_list if item])
    
    if combined_content:
        # Create fixed-size chunks
        section_chunks = create_fixed_size_chunks(
            combined_content,
            document_id,
            document_metadata,
            current_path,
            "content"
        )
        chunks.extend(section_chunks)
    
    # Process subsections recursively
    subsections = section.get('subsections', [])
    for subsection in subsections:
        subsection_chunks = process_section(
            document_id=document_id,
            section=subsection,
            document_metadata=document_metadata,
            section_path=current_path
        )
        chunks.extend(subsection_chunks)
    
    return chunks

def process_document(doc_data: Dict) -> List[Dict]:
    """
    Process a complete document using fixed-size Vietnamese Contextual Retrieval
    """
    chunks = []
    
    # Extract document metadata
    document_metadata = {
        "document_id": doc_data.get('document_id'),
        "keyword": doc_data.get('keyword'),
        "url": doc_data.get('url'),
        "heading": doc_data.get('heading'),
        "abstracts": doc_data.get('abstracts')
    }
    
    document_id = doc_data.get('document_id')
    
    # Process abstracts if available
    abstracts = doc_data.get('abstracts')
    if abstracts:
        abstract_chunks = create_fixed_size_chunks(
            abstracts,
            document_id,
            document_metadata,
            ["Tóm tắt"],
            "abstract"
        )
        chunks.extend(abstract_chunks)
    
    # Process all sections
    sections = doc_data.get('sections', [])
    for section in sections:
        section_chunks = process_section(
            document_id=document_id,
            section=section,
            document_metadata=document_metadata
        )
        chunks.extend(section_chunks)
    
    return chunks

def process_corpus_directory(input_dir: str) -> List[Dict]:
    """
    Process all JSON files in the corpus directory
    """
    all_chunks = []
    
    # Get all JSON files
    json_files = list(Path(input_dir).glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Process the document
            document_chunks = process_document(doc_data)
            all_chunks.extend(document_chunks)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(json_files)} files...")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def save_chunks(chunks: List[Dict], output_file: str):
    """
    Save chunks to JSONL format
    """
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(chunks)} chunks to {output_path}")

def print_statistics(chunks: List[Dict]):
    """
    Print detailed statistics about the chunks in Vietnamese
    """
    if not chunks:
        print("Không có chunk nào để phân tích")
        return
    
    chunk_types = {}
    total_chars = 0
    total_tokens = 0
    doc_count = len(set(chunk['document_id'] for chunk in chunks))
    
    token_sizes = []
    char_sizes = []
    
    for chunk in chunks:
        chunk_type = chunk['chunk_type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        total_chars += chunk['char_count']
        total_tokens += chunk.get('token_count', 0)
        
        token_sizes.append(chunk.get('token_count', 0))
        char_sizes.append(chunk['char_count'])
    
    print("\n" + "="*60)
    print("THỐNG KÊ CHUNKING CỐ ĐỊNH (BGE-M3 OPTIMIZED)")
    print("="*60)
    print(f"Tổng số tài liệu đã xử lý: {doc_count}")
    print(f"Tổng số chunk được tạo: {len(chunks)}")
    print(f"Tổng số ký tự: {total_chars:,}")
    print(f"Tổng số token: {total_tokens:,}")
    print(f"Độ dài chunk trung bình: {total_chars/len(chunks):.1f} ký tự")
    print(f"Số token trung bình: {total_tokens/len(chunks):.1f} token")
    print(f"Kích thước token tối đa: {max(token_sizes)} token")
    print(f"Kích thước token tối thiểu: {min(token_sizes)} token")
    print(f"Mục tiêu: {CHUNK_SIZE} token với overlap {OVERLAP_SIZE} token")
    
    print("\nPhân bố loại chunk:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"  {chunk_type}: {count}")
    
    # Token distribution analysis
    optimal_chunks = len([t for t in token_sizes if 400 <= t <= 600])
    print(f"\nChunk trong khoảng tối ưu (400-600 token): {optimal_chunks}/{len(chunks)} ({optimal_chunks/len(chunks)*100:.1f}%)")
    
    # Sample chunks
    print("\nMẫu chunk:")
    sample = chunks[0] if chunks else None
    if sample:
        print(f"\nMẫu đầu tiên:")
        print(f"  Token count: {sample.get('token_count', 'N/A')}")
        print(f"  Char count: {sample['char_count']}")
        print(f"  Section path: {' > '.join(sample['section_path']) if sample['section_path'] else 'N/A'}")
        print(f"  Content: {sample['content'][:200]}...")

def main():
    print("Bắt đầu quá trình chunking cố định với BGE-M3...")
    print(f"Cấu hình: {CHUNK_SIZE} token/chunk, overlap {OVERLAP_SIZE} token")
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Lỗi: Không tìm thấy thư mục đầu vào '{INPUT_DIR}'!")
        return
    
    # Process all documents
    chunks = process_corpus_directory(INPUT_DIR)
    
    if chunks:
        # Save chunks
        save_chunks(chunks, OUTPUT_FILE)
        
        # Print statistics
        print_statistics(chunks)
        
        print(f"\n✅ Chunking cố định hoàn thành!")
        print(f"📁 Kết quả được lưu tại: {OUTPUT_DIR}/{OUTPUT_FILE}")
    else:
        print("❌ Không có chunk nào được tạo!")

if __name__ == "__main__":
    main()
