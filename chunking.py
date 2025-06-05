import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
import os
from tqdm import tqdm
import hashlib


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass


class MedicalParagraphChunker(ChunkingStrategy):
    def __init__(self, min_paragraph_words: int = 50, max_paragraph_words: int = 800):
        self.min_paragraph_words = min_paragraph_words
        self.max_paragraph_words = max_paragraph_words
    
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not content.strip():
            return []
        
        content = self._preprocess_medical_content(content)
        paragraphs = self._extract_paragraphs(content)
        chunks = self._process_paragraphs(paragraphs)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'chunk_type': 'medical_paragraph'
            })
            
            chunk_objects.append({
                'content': chunk_text.strip(),
                'metadata': chunk_metadata
            })
        
        return chunk_objects
    
    def _preprocess_medical_content(self, content: str) -> str:
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove URLs and links
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        content = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        # Remove non-medical promotional content patterns
        content = re.sub(r'(subscribe|follow us|like us|share|comment below|click here)', '', content, flags=re.IGNORECASE)
        
        # Clean excessive whitespace but preserve paragraph structure
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        return content.strip()
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        # Split by double newlines to respect article structure
        paragraphs = content.split('\n\n')
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para.split()) >= self.min_paragraph_words:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _process_paragraphs(self, paragraphs: List[str]) -> List[str]:
        chunks = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count <= self.max_paragraph_words:
                # Paragraph fits as single chunk
                chunks.append(para)
            else:
                # Split large paragraphs at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if len(potential_chunk.split()) <= self.max_paragraph_words:
                        current_chunk = potential_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks


class SimpleTextChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 512, overlap_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not content.strip():
            return []
        
        content = self._clean_content(content)
        sentences = self._split_into_sentences(content)
        chunks = self._group_sentences_into_chunks(sentences)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text)
            })
            
            chunk_objects.append({
                'content': chunk_text.strip(),
                'metadata': chunk_metadata
            })
        
        return chunk_objects
    
    def _clean_content(self, content: str) -> str:
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        return content.strip()
    
    def _split_into_sentences(self, content: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            
            if current_chunk and current_word_count + sentence_words > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                overlap_chunk = self._get_overlap_text(current_chunk)
                current_chunk = overlap_chunk
                current_word_count = sum(len(s.split()) for s in current_chunk)
                continue
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
                i += 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, sentences: List[str]) -> List[str]:
        if not sentences or self.overlap_size <= 0:
            return []
        
        overlap_sentences = []
        word_count = 0
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= self.overlap_size:
                overlap_sentences.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        return overlap_sentences


class NewlineChunker(ChunkingStrategy):
    def __init__(self, min_words: int = 5):
        self.min_words = min_words
    
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not content.strip():
            return []
        
        # Split by single newlines to get paragraphs
        paragraphs = content.split('\n')
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            # Filter out very short paragraphs
            if para and len(para.split()) >= self.min_words:
                chunks.append(para)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'chunk_type': 'newline_paragraph'
            })
            
            chunk_objects.append({
                'content': chunk_text.strip(),
                'metadata': chunk_metadata
            })
        
        return chunk_objects


class DocumentChunker:
    def __init__(self):
        self.strategies = {
            'medical_paragraph': MedicalParagraphChunker(),
            'medical_paragraph_large': MedicalParagraphChunker(min_paragraph_words=30, max_paragraph_words=1200),
            'simple': SimpleTextChunker(),
            'simple_large': SimpleTextChunker(chunk_size=1024, overlap_size=100),
            'simple_small': SimpleTextChunker(chunk_size=256, overlap_size=25),
            'newline': NewlineChunker(),
            'newline_no_filter': NewlineChunker(min_words=0)
        }
    
    def add_strategy(self, name: str, strategy: ChunkingStrategy):
        self.strategies[name] = strategy
    
    def chunk_document(self, document: Dict[str, Any], strategy_name: str = 'medical_paragraph', 
                      document_id: str = None) -> List[Dict[str, Any]]:
        strategy = self.strategies[strategy_name]
        
        base_metadata = {
            'source_url': document.get('url', ''),
            'document_title': document.get('title', ''),
            'chunking_strategy': strategy_name,
            'chunked_at': datetime.now().isoformat()
        }
        
        content = document.get('markdown', '') or document.get('content', '') or document.get('text', '')
        chunks = strategy.chunk(content, base_metadata)
        
        # Generate unique chunk IDs using document_id
        for i, chunk in enumerate(chunks):
            if document_id:
                chunk['chunk_id'] = f"{document_id}_{strategy_name}_{i:04d}"
            else:
                # Fallback using source URL hash if no document_id provided
                url_hash = hashlib.md5(document.get('url', '').encode()).hexdigest()[:8]
                chunk['chunk_id'] = f"{url_hash}_{strategy_name}_{i:04d}"
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str, format: str = 'jsonl'):
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        elif format.lower() == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def load_documents_from_jsonl(input_path: str) -> List[Dict[str, Any]]:
    documents = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def chunk_documents_from_jsonl(input_path: str = 'text_corpus_youmed_filtered.jsonl',
                              output_path: str = 'chunked_documents.jsonl',
                              strategy: str = 'medical_paragraph',
                              format: str = 'jsonl') -> List[Dict[str, Any]]:
    documents = load_documents_from_jsonl(input_path)
    chunker = DocumentChunker()
    all_chunks = []
    
    for i, document in enumerate(tqdm(documents, desc="Chunking documents")):
        chunks = chunker.chunk_document(document, strategy, document_id=f"doc_{i:04d}")
        for chunk in chunks:
            chunk['metadata']['document_index'] = i
        all_chunks.extend(chunks)
    
    chunker.save_chunks(all_chunks, output_path, format)
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_documents_from_jsonl(
        input_path='datasets/text_corpus_youmed_filtered.jsonl',
        output_path='datasets/chunked_medical_paragraphs.jsonl',
        strategy='newline_no_filter',
        format='jsonl'
    )
