import json
import re
import requests
from bs4 import BeautifulSoup
import os
import hashlib
import concurrent.futures
from tqdm import tqdm
import time
from urllib.parse import urljoin, urlparse
from datetime import datetime
from typing import List, Dict, Any

class MedicalDocumentChunker:
    def __init__(self, 
                 images_dir: str = 'datasets/images',
                 max_workers: int = 5):
        self.images_dir = images_dir
        self.max_workers = max_workers
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.MAX_IMAGE_SIZE = 10 * 1024 * 1024
        self.MIN_IMAGE_SIZE = 1024
    
    def load_documents(self, input_path: str) -> List[Dict[str, Any]]:
        documents = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
        return documents
    
    def extract_chunks_from_markdown(self, markdown_text: str, title: str, url: str) -> List[Dict[str, Any]]:
        chunks = []
        main_sections = re.split(r'\n## ', markdown_text)
        
        if main_sections[0].startswith('#'):
            intro_content = re.sub(r'^#[^#\n]*\n+', '', main_sections[0]).strip()
            if intro_content and len(intro_content.split()) >= 10:
                chunk_id = self._generate_chunk_id(url, 0, 0)
                chunks.append({
                    'content': intro_content,
                    'metadata': {
                        'url': url,
                        'title': title,
                        'section_title': 'Giới thiệu',
                        'section_index': 0,
                        'chunk_index': 0,
                        'word_count': len(intro_content.split()),
                        'char_count': len(intro_content),
                        'created_at': datetime.now().isoformat(),
                        'has_images': False,
                        'image_count': 0
                    },
                    'chunk_id': chunk_id
                })
            main_sections = main_sections[1:]
        
        for section_idx, section in enumerate(main_sections, 1):
            if not section.strip():
                continue
                
            section = '## ' + section
            lines = section.split('\n')
            section_title = lines[0].replace('##', '').strip()
            section_content = '\n'.join(lines[1:]).strip()
            
            subsection_splits = re.split(r'\n(\d+\.\d+(?:\.\d+)?)\.\s+([^\n]+)', section_content)
            
            if len(subsection_splits) > 1:
                chunk_idx = 0
                i = 1
                while i < len(subsection_splits):
                    if i + 2 < len(subsection_splits):
                        subsection_number = subsection_splits[i]
                        subsection_title = subsection_splits[i + 1]
                        raw_content = subsection_splits[i + 2]
                        
                        lines = raw_content.split('\n')
                        content_lines = []
                        
                        for line in lines:
                            line = line.strip()
                            if line and line != subsection_title and not line.startswith('>>'):
                                content_lines.append(line)
                        
                        clean_content = ' '.join(content_lines).strip()
                        
                        if clean_content and len(clean_content.split()) >= 5:
                            chunk_id = self._generate_chunk_id(url, section_idx, chunk_idx)
                            chunks.append({
                                'content': clean_content,
                                'metadata': {
                                    'url': url,
                                    'title': title,
                                    'section_title': section_title,
                                    'subsection_number': subsection_number,
                                    'subsection_title': subsection_title,
                                    'section_index': section_idx,
                                    'chunk_index': chunk_idx,
                                    'word_count': len(clean_content.split()),
                                    'char_count': len(clean_content),
                                    'created_at': datetime.now().isoformat(),
                                    'has_images': False,
                                    'image_count': 0
                                },
                                'chunk_id': chunk_id
                            })
                            chunk_idx += 1
                        
                        i += 3
                    else:
                        break
            else:
                lines = section_content.split('\n')
                content_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('>>'):
                        content_lines.append(line)
                
                clean_content = ' '.join(content_lines).strip()
                
                if clean_content and len(clean_content.split()) >= 10:
                    chunk_id = self._generate_chunk_id(url, section_idx, 0)
                    chunks.append({
                        'content': clean_content,
                        'metadata': {
                            'url': url,
                            'title': title,
                            'section_title': section_title,
                            'section_index': section_idx,
                            'chunk_index': 0,
                            'word_count': len(clean_content.split()),
                            'char_count': len(clean_content),
                            'created_at': datetime.now().isoformat(),
                            'has_images': False,
                            'image_count': 0
                        },
                        'chunk_id': chunk_id
                    })
        
        return chunks

    def _generate_chunk_id(self, url: str, section_idx: int, chunk_idx: int) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(datetime.now().timestamp() * 1000) % 10000
        return f"{url_hash}_{section_idx}_{chunk_idx}_{timestamp}"
    
    def extract_images_from_url(self, url: str) -> List[Dict[str, Any]]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.select_one('.prose.max-w-none')
            
            if not main_content:
                return []
        
            images = []
            figures = main_content.find_all("figure")
            
            for figure in figures:
                img_element = figure.find("img")
                if img_element:
                    img_src = img_element.get('src') or img_element.get('data-src')
                    if img_src and "data:image" not in img_src:
                        if not img_src.startswith('http'):
                            img_src = urljoin(url, img_src)
                        
                        caption = ""
                        figcaption = figure.find("figcaption")
                        if figcaption:
                            caption = figcaption.get_text(strip=True)
                        
                        image_data = self.download_image(
                            img_src, url,
                            img_element.get('alt', ''),
                            img_element.get('title', ''),
                            caption
                        )
                        
                        if image_data:
                            images.append(image_data)
            
            return images
        except:
            return []
        
    def download_image(self, img_src: str, source_url: str, 
                      alt_text: str, title: str, caption: str) -> Dict[str, Any]:
        try:
            filename = self.sanitize_filename(source_url, img_src)
            filepath = os.path.join(self.images_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    content = f.read()
                image_hash = hashlib.md5(content).hexdigest()
            else:
                response = requests.get(img_src, timeout=15)
                response.raise_for_status()
                
                if len(response.content) < self.MIN_IMAGE_SIZE or len(response.content) > self.MAX_IMAGE_SIZE:
                    return None
                
                with open(filepath, "wb") as file:
                    file.write(response.content)
                
                image_hash = hashlib.md5(response.content).hexdigest()
            
            return {
                'filename': filename,
                'filepath': filepath,
                'source_url': source_url,
                'image_url': img_src,
                'alt_text': alt_text,
                'title': title,
                'caption': caption,
                'file_size': os.path.getsize(filepath),
                'image_hash': image_hash
            }
        except:
            return None
    
    def sanitize_filename(self, url: str, img_src: str) -> str:
        domain = urlparse(url).netloc.replace('.', '_')
        filename = img_src.split('/')[-1]
        if not filename or '.' not in filename:
            filename = hashlib.md5(img_src.encode()).hexdigest()[:12] + '.jpg'
        return f"{domain}_{filename}"
    
    def associate_images_with_chunks(self, chunks: List[Dict[str, Any]], 
                                   images: List[Dict[str, Any]]) -> None:
        if not images:
            return
        
        sections = {}
        for chunk in chunks:
            section_idx = chunk['metadata']['section_index']
            if section_idx not in sections:
                sections[section_idx] = []
            sections[section_idx].append(chunk)
        
        if not sections:
            return
            
        images_per_section = len(images) // len(sections)
        remaining_images = len(images) % len(sections)
        
        image_idx = 0
        for section_idx in sorted(sections.keys()):
            section_chunks = sections[section_idx]
            images_for_section = images_per_section
            if remaining_images > 0:
                images_for_section += 1
                remaining_images -= 1
            
            if section_chunks and image_idx < len(images):
                target_chunk = section_chunks[0]
                chunk_id = target_chunk['chunk_id']
                
                for i in range(min(images_for_section, len(images) - image_idx)):
                    if image_idx + i < len(images):
                        images[image_idx + i]['chunk_id'] = chunk_id
                
                target_chunk['metadata']['has_images'] = True
                target_chunk['metadata']['image_count'] = min(images_for_section, len(images) - image_idx)
                
                image_idx += images_for_section
    
    def process_document(self, document: Dict[str, Any]) -> tuple:
        url = document['url']
        title = document['title']
        markdown = document['markdown']
        
        chunks = self.extract_chunks_from_markdown(markdown, title, url)
        images = self.extract_images_from_url(url)
        self.associate_images_with_chunks(chunks, images)
        
        return chunks, images
    
    def process_documents(self, input_path: str, 
                         chunks_output: str = 'datasets/chunked_medical_paragraphs.jsonl',
                         images_output: str = 'datasets/image_metadata.jsonl') -> tuple:
        documents = self.load_documents(input_path)
        
        all_chunks = []
        all_images = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self.process_document, doc): doc for doc in documents}
            
            with tqdm(total=len(documents), desc="Processing medical documents") as pbar:
                for future in concurrent.futures.as_completed(future_to_doc):
                    try:
                        chunks, images = future.result()
                        all_chunks.extend(chunks)
                        all_images.extend(images)
                    except:
                        pass
                    
                    pbar.update(1)
                    time.sleep(0.1)
        
        with open(chunks_output, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        with open(images_output, 'w', encoding='utf-8') as f:
            for img in all_images:
                f.write(json.dumps(img, ensure_ascii=False) + '\n')
        
        chunks_with_images = sum(1 for chunk in all_chunks if chunk['metadata']['has_images'])
        chunks_without_images = len(all_chunks) - chunks_with_images
        
        summary = {
            'total_documents': len(documents),
            'total_chunks': len(all_chunks),
            'chunks_with_images': chunks_with_images,
            'chunks_without_images': chunks_without_images,
            'total_images': len(all_images),
            'unique_images': len(set(img['image_hash'] for img in all_images)),
            'avg_chunks_per_doc': len(all_chunks) / len(documents) if documents else 0,
            'avg_words_per_chunk': sum(chunk['metadata']['word_count'] for chunk in all_chunks) / len(all_chunks) if all_chunks else 0,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open('datasets/chunking_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return all_chunks, all_images


def main():
    chunker = MedicalDocumentChunker(
        images_dir='datasets/images',
        max_workers=3
    )
    
    chunks, images = chunker.process_documents(
        input_path='datasets/text_corpus_youmed_filtered.jsonl',
        chunks_output='datasets/chunked_medical_paragraphs.jsonl',
        images_output='datasets/image_metadata.jsonl'
    )
    
    return chunks, images


if __name__ == "__main__":
    chunks, images = main()
