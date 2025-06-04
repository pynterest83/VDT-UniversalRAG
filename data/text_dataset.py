from huggingface_hub import login
from datasets import load_dataset
import os
import json
import re

OUTPUT_FILES = {
    'train': 'q_a_train_filtered.jsonl',
    'test': 'q_a_test_filtered.jsonl',
    'validation': 'q_a_validation_filtered.jsonl',
    'corpus': 'text_corpus_youmed_filtered.jsonl'
}

DOMAIN_FILTER = "youmed.vn"

def authenticate_huggingface():
    login(token=os.getenv("HF_TOKEN"))

def load_datasets():
    qa_dataset = load_dataset("tmnam20/ViMedAQA", "all")
    corpus_dataset = load_dataset("codin-research/medical-website-pretrain")
    return qa_dataset, corpus_dataset

def filter_corpus_by_domain(corpus_dataset, domain):
    return corpus_dataset['train'].filter(lambda sample: domain in sample['url'])

def extract_corpus_urls(filtered_corpus):
    return set(sample['url'] for sample in filtered_corpus)

def extract_qa_urls(qa_dataset):
    qa_urls = set()
    for split in ['train', 'test', 'validation']:
        qa_urls.update(sample['article_url'] for sample in qa_dataset[split])
    return qa_urls

def find_missing_urls(qa_urls, corpus_urls):
    return qa_urls - corpus_urls

def remove_samples_with_missing_urls(qa_dataset, missing_urls):
    return {
        split: qa_dataset[split].filter(lambda sample: sample['article_url'] not in missing_urls)
        for split in ['train', 'test', 'validation']
    }

def calculate_dataset_sizes(qa_dataset):
    return {split: len(qa_dataset[split]) for split in ['train', 'test', 'validation']}

def print_statistics(original_sizes, filtered_sizes, corpus_size, missing_urls_count):
    print(f"Number of text_corpus samples with {DOMAIN_FILTER} URLs: {corpus_size}")
    print(f"Number of Q&A URLs not found in corpus: {missing_urls_count}")
    print(f"Original q_a_dataset sizes:")
    for split, size in original_sizes.items():
        print(f"  {split.capitalize()}: {size}")
    print(f"Filtered q_a_dataset sizes:")
    for split, size in filtered_sizes.items():
        print(f"  {split.capitalize()}: {size}")

def save_datasets(filtered_qa_dataset, filtered_corpus):
    for split in ['train', 'test', 'validation']:
        filtered_qa_dataset[split].to_json(OUTPUT_FILES[split], lines=True, force_ascii=False)
    
    filtered_corpus.to_json(OUTPUT_FILES['corpus'], lines=True, force_ascii=False)

def print_save_summary(filtered_qa_dataset, filtered_corpus):
    print(f"Saved filtered datasets:")
    for split in ['train', 'test', 'validation']:
        print(f"  {OUTPUT_FILES[split]}: {len(filtered_qa_dataset[split])} samples")
    print(f"  {OUTPUT_FILES['corpus']}: {len(filtered_corpus)} samples")

def main():
    authenticate_huggingface()
    
    qa_dataset, corpus_dataset = load_datasets()
    filtered_corpus = filter_corpus_by_domain(corpus_dataset, DOMAIN_FILTER)
    corpus_urls = extract_corpus_urls(filtered_corpus)
    
    qa_urls = extract_qa_urls(qa_dataset)
    missing_urls = find_missing_urls(qa_urls, corpus_urls)
    print(missing_urls)
    
    original_sizes = calculate_dataset_sizes(qa_dataset)
    filtered_qa_dataset = remove_samples_with_missing_urls(qa_dataset, missing_urls)
    filtered_sizes = calculate_dataset_sizes(filtered_qa_dataset)
    
    print_statistics(original_sizes, filtered_sizes, len(filtered_corpus), len(missing_urls))
    save_datasets(filtered_qa_dataset, filtered_corpus)
    print_save_summary(filtered_qa_dataset, filtered_corpus)

if __name__ == "__main__":
    main()
