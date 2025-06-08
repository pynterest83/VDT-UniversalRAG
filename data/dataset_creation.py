import requests
from bs4 import BeautifulSoup
import os
import json
import hashlib
import concurrent.futures
from tqdm import tqdm
import time
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime

OUTPUT_CONFIG = {
    'processed_corpus': 'datasets/processed_medical_corpus.jsonl',
    'structured_paragraphs': 'datasets/structured_paragraphs.jsonl',
    'max_workers': 5,
    'delay_between_requests': 1
}

CORPUS_FILE = 'datasets/text_corpus_youmed_filtered.jsonl'

def load_corpus_data():
    documents = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            documents.append(data)
    return documents

def extract_youmed_content(soup):
    main_content = soup.select_one('.prose.max-w-none')
    
    if not main_content:
        return {"title": "", "sections": [], "raw_text": ""}
    
    title_elem = soup.select_one('h1')
    title = title_elem.get_text(strip=True) if title_elem else ""
    
    sections = []
    current_section = None
    
    for element in main_content.find_all(['h2', 'h3', 'h4', 'p']):
        if element.name in ['h2', 'h3', 'h4']:
            if current_section and current_section['content'].strip():
                sections.append(current_section)
            
            header_text = element.get_text(strip=True)
            if header_text and not header_text.lower().startswith('nội dung'):
                current_section = {
                    'header': header_text,
                    'level': int(element.name[1]),
                    'content': ""
                }
        elif element.name == 'p' and current_section is not None:
            text = element.get_text(strip=True)
            if text and len(text) > 20:
                current_section['content'] += text + "\n"
    
    if current_section and current_section['content'].strip():
        sections.append(current_section)
    
    raw_text = main_content.get_text(separator='\n', strip=True)
    
    return {
        "title": title,
        "sections": sections,
        "raw_text": raw_text
    }

def clean_medical_text(text):
    patterns_to_remove = [
        r'Bài viết.*?kiểm duyệt.*?\n',
        r'Nguồn:.*?\n',
        r'Tác giả:.*?\n',
        r'Cập nhật:.*?\n',
        r'YouMed.*?\n',
        r'>>.*?\n',
        r'Tìm hiểu thêm.*?\n',
        r'\d+\.\d+\.\s*',
        r'^\d+\.\s*',
        r'Nội dung bài viết.*?\n'
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_focused_paragraphs(sections, title, url):
    paragraphs = []
    
    for i, section in enumerate(sections):
        header = section['header']
        header = re.sub(r'^\d+[\.\)]\s*', '', header)
        
        content = clean_medical_text(section['content'])
        if not content or len(content.split()) < 10:
            continue
        
        sentences = re.split(r'[.!?](?=\s+[A-ZÀ-Ỹ])', content)
        
        current_chunk = ""
        current_words = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > 80:
                if current_chunk.strip():
                    paragraph_data = {
                        'url': url,
                        'title': title,
                        'section_header': header,
                        'section_level': section['level'],
                        'content': current_chunk.strip(),
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk),
                        'created_at': datetime.now().isoformat()
                    }
                    paragraphs.append(paragraph_data)
                
                current_chunk = sentence + ". "
                current_words = sentence_words
            else:
                current_chunk += sentence + ". "
                current_words += sentence_words
        
        if current_chunk.strip() and current_words >= 20:
            paragraph_data = {
                'url': url,
                'title': title,
                'section_header': header,
                'section_level': section['level'],
                'content': current_chunk.strip(),
                'word_count': len(current_chunk.split()),
                'char_count': len(current_chunk),
                'created_at': datetime.now().isoformat()
            }
            paragraphs.append(paragraph_data)
    
    return paragraphs

def process_single_document(doc_data):
    url = doc_data['url']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    structured = extract_youmed_content(soup)
    
    processed_text = clean_medical_text(structured['raw_text'])
    
    paragraphs = create_focused_paragraphs(
        structured['sections'], 
        structured['title'], 
        url
    )
    
    processed_doc = {
        'url': url,
        'title': structured['title'],
        'original_title': doc_data.get('title', ''),
        'processed_content': processed_text,
        'sections': structured['sections'],
        'paragraph_count': len(paragraphs),
        'word_count': len(processed_text.split()),
        'char_count': len(processed_text),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    return processed_doc, paragraphs

def save_processed_data(documents, paragraphs):
    with open(OUTPUT_CONFIG['processed_corpus'], 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    with open(OUTPUT_CONFIG['structured_paragraphs'], 'w', encoding='utf-8') as f:
        for para in paragraphs:
            f.write(json.dumps(para, ensure_ascii=False) + '\n')

def create_processing_summary(documents, paragraphs):
    total_words = sum(doc['word_count'] for doc in documents if doc)
    total_chars = sum(doc['char_count'] for doc in documents if doc)
    
    summary = {
        'total_documents': len(documents),
        'total_paragraphs': len(paragraphs),
        'total_words': total_words,
        'total_chars': total_chars,
        'avg_words_per_doc': total_words / len(documents) if documents else 0,
        'avg_paragraphs_per_doc': len(paragraphs) / len(documents) if documents else 0,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    with open('datasets/processing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary

def main():
    corpus_data = load_corpus_data()
    
    os.makedirs('datasets', exist_ok=True)
    
    processed_documents = []
    all_paragraphs = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=OUTPUT_CONFIG['max_workers']) as executor:
        future_to_doc = {executor.submit(process_single_document, doc): doc for doc in corpus_data}
        
        with tqdm(total=len(corpus_data), desc="Processing medical content") as pbar:
            for future in concurrent.futures.as_completed(future_to_doc):
                processed_doc, paragraphs = future.result()
                if processed_doc:
                    processed_documents.append(processed_doc)
                    all_paragraphs.extend(paragraphs)
                
                pbar.update(1)
                time.sleep(OUTPUT_CONFIG['delay_between_requests'])
    
    save_processed_data(processed_documents, all_paragraphs)
    summary = create_processing_summary(processed_documents, all_paragraphs)

if __name__ == "__main__":
    main()
