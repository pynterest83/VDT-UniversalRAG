import json
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
import sys

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from ragas import SingleTurnSample
from ragas.metrics import AnswerAccuracy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Thread-local storage for metric instances
thread_local = threading.local()

# Global flag for graceful shutdown
shutdown_requested = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n\nReceived interrupt signal ({signum}). Initiating graceful shutdown...")
    print("Please wait for current evaluations to complete...")
    shutdown_requested.set()

def get_thread_metric():
    """Get or create a thread-local metric instance"""
    if not hasattr(thread_local, 'metric'):
        # Create a new LLM and metric instance for this thread
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60,
            max_retries=2,
        )
        wrapped_llm = LangchainLLMWrapper(llm)
        thread_local.metric = AnswerAccuracy(llm=wrapped_llm)
    return thread_local.metric

def evaluate_sample(data):
    """Evaluate a single sample using thread-local metric"""
    # Check if shutdown was requested
    if shutdown_requested.is_set():
        return float('nan')
    
    try:
        sample = SingleTurnSample(**data)
        metric = get_thread_metric()
        result = metric.single_turn_score(sample)
        return result
    except Exception as e:
        print(f'Exception in thread {threading.current_thread().ident}: {e}')
        return float('nan')

# Set up global metric for sequential processing
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60,
    max_retries=2,
)
wrapped_llm = LangchainLLMWrapper(llm)
global_metric = AnswerAccuracy(llm=wrapped_llm)

def evaluate_sample_sequential(data):
    """Evaluate a single sample using global metric (for sequential processing)"""
    # Check if shutdown was requested
    if shutdown_requested.is_set():
        return float('nan')
    
    try:
        sample = SingleTurnSample(**data)
        result = global_metric.single_turn_score(sample)
        return result
    except Exception as e:
        print(f'Exception: {e}')
        return float('nan')


def calculate_threshold_percentages(scores, thresholds):
    """Calculate percentage of scores meeting each threshold"""
    valid_scores = [s for s in scores if not (isinstance(s, float) and s != s)]
    if not valid_scores:
        return {threshold: 0.0 for threshold in thresholds}, 0
    
    total_count = len(valid_scores)
    percentages = {}
    
    for threshold in thresholds:
        count_above_threshold = sum(1 for score in valid_scores if score >= threshold)
        percentage = (count_above_threshold / total_count) * 100
        percentages[threshold] = percentage
    
    return percentages, total_count

def calculate_avg_score_sequential(file_path, max_samples=None):
    """Calculate threshold percentages sequentially (no threading)"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line.strip())
            samples.append(data)
    
    print(f"Processing {len(samples)} samples from {file_path}")
    scores = []
    
    try:
        for sample in tqdm(samples, desc=f"Evaluating {file_path}"):
            if shutdown_requested.is_set():
                print(f"\nShutdown requested. Processed {len(scores)}/{len(samples)} samples.")
                break
            score = evaluate_sample_sequential(sample)
            scores.append(score)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {len(scores)}/{len(samples)} samples.")
    
    # Calculate both average and threshold percentages
    valid_scores = [s for s in scores if not (isinstance(s, float) and s != s)]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    threshold_percentages, total_count = calculate_threshold_percentages(scores, SCORE_THRESHOLDS)
    
    return avg_score, threshold_percentages, total_count

def calculate_avg_score_parallel(file_path, max_samples=None, max_workers=3):
    """Calculate threshold percentages using parallel processing with ThreadPoolExecutor"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line.strip())
            samples.append(data)
    
    print(f"Processing {len(samples)} samples from {file_path} with {max_workers} workers")
    scores = [None] * len(samples)
    error_count = 0
    completed_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit all tasks
            future_to_index = {
                executor.submit(evaluate_sample, sample): i 
                for i, sample in enumerate(samples)
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(samples), desc=f"Evaluating {file_path}") as pbar:
                for future in as_completed(future_to_index):
                    if shutdown_requested.is_set():
                        print(f"\nShutdown requested. Cancelling remaining tasks...")
                        # Cancel remaining futures
                        for remaining_future in future_to_index:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    
                    index = future_to_index[future]
                    try:
                        score = future.result()
                        scores[index] = score
                        completed_count += 1
                        if isinstance(score, float) and score != score:  # Check for NaN
                            error_count += 1
                    except Exception as e:
                        print(f"Error processing sample {index}: {e}")
                        scores[index] = float('nan')
                        error_count += 1
                        completed_count += 1
                    pbar.update(1)
                    
        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt received. Shutting down executor...")
            shutdown_requested.set()
            # The executor will be properly shutdown by the context manager
    
    # Calculate both average and threshold percentages
    valid_scores = [s for s in scores if s is not None and not (isinstance(s, float) and s != s)]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    threshold_percentages, total_count = calculate_threshold_percentages(scores, SCORE_THRESHOLDS)
    
    print(f"Completed {completed_count}/{len(samples)} samples with {error_count} errors")
    
    if shutdown_requested.is_set():
        print("Evaluation was interrupted. Results are partial.")
    
    return avg_score, threshold_percentages, total_count

# Configuration
MAX_WORKERS = 3  # Adjust based on your system and API rate limits
USE_PARALLEL = True  # Set to False to use sequential processing
SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]  # Thresholds to calculate percentage of samples meeting each threshold

files = [
    "results/benchmark_results_top_5.jsonl",
]

# Set up signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

try:
    if USE_PARALLEL:
        print(f"PARALLEL MODE - Using {MAX_WORKERS} workers for threading")
        print("Press Ctrl+C for graceful shutdown...")
        for file_path in files:
            if shutdown_requested.is_set():
                print("Shutdown requested. Skipping remaining files.")
                break
            print(f"\n--- Starting parallel evaluation of {file_path} ---")
            start_time = time.time()
            avg_score, threshold_percentages, total_count = calculate_avg_score_parallel(file_path, max_workers=MAX_WORKERS)
            end_time = time.time()
            
            print(f"\n=== RESULTS FOR {file_path} ===")
            print(f"Average Score: {avg_score:.4f}")
            print(f"Evaluation Time: {end_time - start_time:.2f}s")
            print(f"Total Valid Samples: {total_count}")
            print("\n--- Threshold Analysis ---")
            for threshold in sorted(threshold_percentages.keys()):
                percentage = threshold_percentages[threshold]
                sample_count = int(percentage * total_count / 100)
                print(f"Score >= {threshold}: {percentage:.2f}% ({sample_count}/{total_count} samples)")
            print("=" * 50)
    else:
        print("SEQUENTIAL MODE - Testing without threading")
        print("Press Ctrl+C for graceful shutdown...")
        for file_path in files:
            if shutdown_requested.is_set():
                print("Shutdown requested. Skipping remaining files.")
                break
            print(f"\n--- Starting sequential evaluation of {file_path} ---")
            start_time = time.time()
            avg_score, threshold_percentages, total_count = calculate_avg_score_sequential(file_path)
            end_time = time.time()
            
            print(f"\n=== RESULTS FOR {file_path} ===")
            print(f"Average Score: {avg_score:.4f}")
            print(f"Evaluation Time: {end_time - start_time:.2f}s")
            print(f"Total Valid Samples: {total_count}")
            print("\n--- Threshold Analysis ---")
            for threshold in sorted(threshold_percentages.keys()):
                percentage = threshold_percentages[threshold]
                sample_count = int(percentage * total_count / 100)
                print(f"Score >= {threshold}: {percentage:.2f}% ({sample_count}/{total_count} samples)")
            print("=" * 50)

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if shutdown_requested.is_set():
        print("\nGraceful shutdown completed.")
    else:
        print("\nProgram completed normally.")

