import requests
from bs4 import BeautifulSoup
import os
import json
import hashlib
import concurrent.futures
from tqdm import tqdm
import time
from urllib.parse import urljoin, urlparse

OUTPUT_CONFIG = {
    'images_dir': 'datasets/images',
    'metadata_file': 'datasets/image_metadata.jsonl',
    'max_workers': 5,
    'delay_between_requests': 1
}

CORPUS_FILE = 'datasets/text_corpus_youmed_filtered.jsonl'

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_IMAGE_SIZE = 1024  # 1KB

def load_filtered_corpus_urls():
    urls = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            urls.append(data['url'])
    return urls

def sanitize_filename(url, img_src):
    domain = urlparse(url).netloc.replace('.', '_')
    filename = img_src.split('/')[-1]
    if not filename or '.' not in filename:
        filename = hashlib.md5(img_src.encode()).hexdigest()[:12] + '.jpg'
    return f"{domain}_{filename}"

def get_image_hash(image_content):
    return hashlib.md5(image_content).hexdigest()

def extract_and_download_images(url_data):
    url, images_dir = url_data
    downloaded_metadata = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        page = requests.get(url, headers=headers, timeout=10)
        page.raise_for_status()
        
        soup = BeautifulSoup(page.content, 'html.parser')
        
        # Add debug output
        print(f"Processing URL: {url}")
        
        # Target the main content area specifically for youmed.vn
        main_content = soup.select_one('.prose.max-w-none')
        
        if not main_content:
            # Fallback selectors for youmed.vn content
            main_content = soup.select_one('article') or soup.select_one('.post-content')
            
        if not main_content:
            print(f"No main content found for {url}")
            return downloaded_metadata
            
        print(f"Found main content area")
        
        # Find images specifically in figure tags within main content
        figures = main_content.find_all("figure")
        print(f"Found {len(figures)} figure elements")
        
        for figure in figures:
            images = figure.find_all("img")
            print(f"Found {len(images)} images in figure")
            
            for image in images:
                try:
                    img_src = image.get('src') or image.get('data-src')
                    if not img_src:
                        print("No src found for image")
                        continue
                        
                    if "data:image" in img_src:
                        print("Skipping data:image")
                        continue
                    
                    print(f"Processing image: {img_src}")
                    
                    if not img_src.startswith('http'):
                        img_src = urljoin(url, img_src)
                    
                    filename = sanitize_filename(url, img_src)
                    filepath = os.path.join(images_dir, filename)
                    
                    if os.path.exists(filepath):
                        print(f"File already exists: {filename}")
                        continue
                    
                    print(f"Downloading: {img_src}")
                    response = requests.get(img_src, timeout=15)
                    response.raise_for_status()
                    
                    if len(response.content) < MIN_IMAGE_SIZE or len(response.content) > MAX_IMAGE_SIZE:
                        print(f"Image size out of range: {len(response.content)} bytes")
                        continue
                    
                    with open(filepath, "wb") as file:
                        file.write(response.content)
                    
                    # Get caption text from figcaption
                    caption = ""
                    figcaption = figure.find("figcaption")
                    if figcaption:
                        caption = figcaption.get_text(strip=True)
                    
                    metadata = {
                        'filename': filename,
                        'filepath': filepath,
                        'source_url': url,
                        'image_url': img_src,
                        'alt_text': image.get('alt', ''),
                        'title': image.get('title', ''),
                        'caption': caption,
                        'file_size': len(response.content),
                        'image_hash': get_image_hash(response.content)
                    }
                    
                    downloaded_metadata.append(metadata)
                    print(f"Successfully downloaded: {filename}")
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error downloading image {img_src}: {e}")
                    continue
                
    except Exception as e:
        print(f"Error processing {url}: {e}")
    
    print(f"Downloaded {len(downloaded_metadata)} images from {url}")
    return downloaded_metadata

def save_metadata(all_metadata, metadata_file):
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

def create_summary_stats(all_metadata, total_urls):
    unique_hashes = set(item['image_hash'] for item in all_metadata)
    total_size = sum(item['file_size'] for item in all_metadata)
    
    summary = {
        'total_urls_processed': total_urls,
        'total_images_downloaded': len(all_metadata),
        'unique_images': len(unique_hashes),
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2)
    }
    
    with open('datasets/crawl_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary

def print_statistics(summary):
    print(f"Crawling completed!")
    print(f"URLs processed: {summary['total_urls_processed']}")
    print(f"Images downloaded: {summary['total_images_downloaded']}")
    print(f"Unique images: {summary['unique_images']}")
    print(f"Total size: {summary['total_size_mb']} MB")

def main():
    urls = load_filtered_corpus_urls()
    
    # Process all URLs
    all_urls = urls
    print(f"Total URLs to process: {len(all_urls)}")
    
    os.makedirs(OUTPUT_CONFIG['images_dir'], exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    print(f"Starting image crawling for all URLs from {CORPUS_FILE}...")
    
    url_data = [(url, OUTPUT_CONFIG['images_dir']) for url in all_urls]
    all_metadata = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=OUTPUT_CONFIG['max_workers']) as executor:
        future_to_url = {executor.submit(extract_and_download_images, data): data[0] for data in url_data}
        
        with tqdm(total=len(all_urls), desc="Crawling images") as pbar:
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    metadata_list = future.result()
                    all_metadata.extend(metadata_list)
                    pbar.update(1)
                    time.sleep(OUTPUT_CONFIG['delay_between_requests'])
                except Exception as e:
                    url = future_to_url[future]
                    print(f"Error processing {url}: {e}")
                    pbar.update(1)
    
    save_metadata(all_metadata, OUTPUT_CONFIG['metadata_file'])
    summary = create_summary_stats(all_metadata, len(all_urls))
    print_statistics(summary)

if __name__ == "__main__":
    main()
