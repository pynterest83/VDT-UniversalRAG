from pathlib import Path
import modal
import os

VOL_MOUNT_PATH = Path("/vol")
BASE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Modal image following multimodal_embeddings.py pattern
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "wget", 
        "curl",
        "build-essential",
        "cmake",
        "ninja-build",
        "python3-dev",
        "python3-pip",
        "software-properties-common",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/gcc",
        "TORCH_CUDA_ARCH_LIST": "6.0;6.1;7.0;7.5;8.0;8.6;9.0",
        "FORCE_CUDA": "1",
        "MAX_JOBS": "4",
    })
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2", 
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers>=4.30.0",
        "huggingface_hub",
        "accelerate",
        "datasets",
        "peft",
        "deepspeed",
        "triton",
        "packaging",
        "wheel",
        "ninja",
    )
    .pip_install("FlagEmbedding[finetune]")
    .run_commands("python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")'")
)

app = modal.App(name="bge-reranker-m3-training", image=image)

# Volume for persistent storage
volume = modal.Volume.from_name("reranker-training-vol", create_if_missing=True)

def create_deepspeed_config():
    """Create DeepSpeed configuration for FP16 training"""
    config = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        }
    }
    return config

def install_flash_attn():
    """Install flash-attention in runtime if needed"""
    import subprocess
    import sys
    
    try:
        import flash_attn
        print("flash-attn already installed")
        return True
    except ImportError:
        print("Installing flash-attn...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "flash-attn==2.5.6", "--no-build-isolation"
            ])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install flash-attn: {e}")
            return False

@app.function(
    gpu=modal.gpu.A100(count=1),
    volumes={str(VOL_MOUNT_PATH): volume},
    timeout=3600 * 4,  # 4 hours
    cpu=8.0,
    memory=32768,
)
def train_reranker_with_prebuilt_data(
    num_train_epochs: int = 2,
    learning_rate: float = 6e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    train_group_size: int = 8,
    query_max_len: int = 512,
    passage_max_len: int = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    logging_steps: int = 1,
    save_steps: int = 1000,
):
    import subprocess
    import json
    import os
    import shutil
    from pathlib import Path
    
    print("Setting up training environment...")
    
    # Install flash-attn if needed
    install_flash_attn()
    
    # Create working directory
    work_dir = VOL_MOUNT_PATH / "reranker_training"
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    # Create cache directories
    cache_dir = work_dir / "cache"
    model_cache_dir = cache_dir / "model"
    data_cache_dir = cache_dir / "data"
    cache_dir.mkdir(exist_ok=True)
    model_cache_dir.mkdir(exist_ok=True)
    data_cache_dir.mkdir(exist_ok=True)
    
    # Create example data directory and sample file
    example_data_dir = work_dir / "example_data" / "normal"
    example_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample training data if it doesn't exist
    sample_data_file = example_data_dir / "examples.jsonl"
    if not sample_data_file.exists():
        print("Creating sample training data...")
        sample_data = [
            {
                "query": "What is machine learning?",
                "pos": ["Machine learning is a subset of artificial intelligence that focuses on algorithms."],
                "neg": ["Cooking is the art of preparing food.", "Sports are physical activities."]
            },
            {
                "query": "How does deep learning work?",
                "pos": ["Deep learning uses neural networks with multiple layers to learn patterns."],
                "neg": ["Music is an art form.", "Photography captures moments."]
            }
        ]
        
        with open(sample_data_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        print(f"Sample data created at {sample_data_file}")
    
    # Create DeepSpeed config
    print("Creating DeepSpeed configuration...")
    ds_config = create_deepspeed_config()
    ds_config_file = work_dir / "ds_stage0.json"
    with open(ds_config_file, 'w') as f:
        json.dump(ds_config, f, indent=2)
    print(f"DeepSpeed config created at {ds_config_file}")
    
    # Create output directory
    output_dir = work_dir / f"trained_reranker_m3_epochs_{num_train_epochs}"
    
    # Prepare training command
    cmd = [
        "torchrun", "--nproc_per_node", "1",
        "-m", "FlagEmbedding.finetune.reranker.encoder_only.base",
        "--model_name_or_path", BASE_RERANKER_MODEL,
        "--cache_dir", str(model_cache_dir),
        "--train_data", str(sample_data_file),
        "--cache_path", str(data_cache_dir),
        "--train_group_size", str(train_group_size),
        "--query_max_len", str(query_max_len),
        "--passage_max_len", str(passage_max_len),
        "--pad_to_multiple_of", "8",
        "--knowledge_distillation", "False",
        "--output_dir", str(output_dir),
        "--overwrite_output_dir",
        "--learning_rate", str(learning_rate),
        "--fp16",
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--dataloader_drop_last", "True",
        "--warmup_ratio", str(warmup_ratio),
        "--gradient_checkpointing",
        "--weight_decay", str(weight_decay),
        "--deepspeed", str(ds_config_file),
        "--logging_steps", str(logging_steps),
        "--save_steps", str(save_steps),
    ]
    
    print("Starting training with command:")
    print(" ".join(cmd))
    print("-" * 80)
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
    })
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            capture_output=False,
            text=True,
            check=True
        )
        
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        
        # List output files
        if output_dir.exists():
            print("\nOutput files:")
            for file in output_dir.iterdir():
                print(f"  {file.name}")
        
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "message": "Training completed successfully"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Training failed"
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Unexpected error occurred"
        }

@app.function(
    gpu=modal.gpu.A100(count=1),
    volumes={str(VOL_MOUNT_PATH): volume},
    timeout=1800,
)
def test_trained_reranker(model_path: str = None):
    """Test the trained reranker model"""
    import subprocess
    import json
    from pathlib import Path
    
    work_dir = VOL_MOUNT_PATH / "reranker_training"
    
    if not model_path:
        # Find the most recent training output
        output_dirs = list(work_dir.glob("trained_reranker_m3_*"))
        if not output_dirs:
            return {"error": "No trained models found"}
        model_path = str(max(output_dirs, key=os.path.getctime))
    
    print(f"Testing model from: {model_path}")
    
    try:
        # Test the model
        from FlagEmbedding import FlagReranker
        
        reranker = FlagReranker(model_path, use_fp16=True)
        
        # Test queries
        test_cases = [
            {
                "query": "What is artificial intelligence?",
                "passages": [
                    "Artificial intelligence is a branch of computer science.",
                    "Cooking involves preparing food with various ingredients.",
                    "AI systems can learn and make decisions like humans."
                ]
            }
        ]
        
        results = []
        for case in test_cases:
            query = case["query"]
            passages = case["passages"]
            
            # Create pairs for reranking
            pairs = [[query, passage] for passage in passages]
            scores = reranker.compute_score(pairs)
            
            # Sort by scores
            passage_scores = list(zip(passages, scores))
            passage_scores.sort(key=lambda x: x[1], reverse=True)
            
            results.append({
                "query": query,
                "ranked_passages": passage_scores
            })
            
            print(f"Query: {query}")
            for i, (passage, score) in enumerate(passage_scores):
                print(f"  {i+1}. Score: {score:.4f} - {passage}")
            print()
        
        return {
            "status": "success",
            "model_path": model_path,
            "test_results": results
        }
        
    except Exception as e:
        print(f"Testing failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("Starting reranker training...")
    result = train_reranker_with_prebuilt_data.remote()
    print("Training result:", result)
    
    if result.get("status") == "success":
        print("Testing trained model...")
        test_result = test_trained_reranker.remote(result.get("output_dir"))
        print("Test result:", test_result)

if __name__ == "__main__":
    main()
