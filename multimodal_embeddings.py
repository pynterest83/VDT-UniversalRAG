# Finetuning Jina CLIP v2 Multimodal Embedding Model for Portuguese COCO Captions

# This example demonstrates how to finetune the Jina CLIP v2 multimodal embedding model 
# using Modal cloud infrastructure with A100 GPU acceleration. The model is trained on 
# Portuguese COCO captions using Matryoshka representation learning.

from pathlib import Path
import modal

VOL_MOUNT_PATH = Path("/vol")

# Jina CLIP v2 base model for multimodal embedding finetuning
BASE_MODEL = "jinaai/jina-clip-v2"

# Use CUDA-enabled base image with WORKING CUDA version
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
        "CUDA_ROOT": "/usr/local/cuda", 
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .run_commands(
        # Check CUDA version first
        "nvcc --version"
    )
    .run_commands(
        # Upgrade pip, setuptools, wheel first
        "pip install --upgrade pip setuptools wheel"
    )
    .pip_install(
        # Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.9)
        "torch>=2.5.0", "torchvision", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        # Verify PyTorch installation - simplified to avoid quote issues
        "python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"
    )
    .pip_install(
        # Essential build dependencies for flash-attention
        "packaging", 
        "ninja",
        "cmake"
    )
    .pip_install(
        # Core ML libraries with compatible versions - FIXED for Jina CLIP v2 compatibility
        "sentence-transformers>=3.0.0,<4.0.0",  # Use compatible version with older transformers
        "transformers>=4.37.0,<4.41.0",         # Fixed version for EVAVisionTransformer compatibility
        "accelerate>=1.7.0",
        "datasets>=3.6.0",
        "huggingface-hub>=0.31.0",
        "numpy>=1.26.0",
        "scipy",
        "scikit-learn",
        "pillow",
        "tqdm",
        "tensorboard",
        "flash-attn>=2.7.0",
        "triton",
        "pyyaml",
        "regex",
        "safetensors",
        "tokenizers",
        "einops",
        "timm",
    )
)

app = modal.App(
    "jina-clip-multimodal-embeddings",
    image=image,
)

# Create volumes for persistent storage
model_volume = modal.Volume.from_name("vdt-models", create_if_missing=True)
data_volume = modal.Volume.from_name("vdt-datasets", create_if_missing=True) 
output_vol = modal.Volume.from_name("multimodal-embeddings-vol", create_if_missing=True)

MODELS_DIR = "/models"
DATA_DIR = "/data"

def install_flash_attn_runtime():
    """Install flash-attention at runtime using Method 1 only (Dao-AILab)"""
    import subprocess
    import torch
    import os
    
    print("🔍 Checking flash-attention installation...")
    
    # First check if it's already installed
    try:
        import flash_attn
        print(f"✅ Flash-attention already installed: {flash_attn.__version__}")
        return True
    except ImportError:
        pass
    
    print("🚀 Installing flash-attention from Dao-AILab/flash-attention...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        
        # Check CUDA version
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("NVCC version info:")
                print(result.stdout)
        except Exception:
            print("⚠️ Could not check NVCC version")
    
    try:
        # Clean up any existing clone
        subprocess.run(["rm", "-rf", "/tmp/flash-attention"], capture_output=True)
        
        # Clone the repository
        print("Cloning Dao-AILab/flash-attention...")
        clone_result = subprocess.run([
            "git", "clone", "https://github.com/Dao-AILab/flash-attention.git",
            "/tmp/flash-attention"
        ], capture_output=True, text=True, timeout=300)
        
        if clone_result.returncode != 0:
            print(f"❌ Git clone failed: {clone_result.stderr}")
            raise Exception("Git clone failed")
        
        # Change to the directory
        original_dir = os.getcwd()
        os.chdir("/tmp/flash-attention")
        
        try:
            # Install build dependencies
            print("Installing build dependencies...")
            subprocess.run(["pip", "install", "packaging", "ninja"], check=True, timeout=120)
            
            # Set environment variables for compilation
            env = os.environ.copy()
            env.update({
                'MAX_JOBS': '2',  # Limit parallel jobs to prevent memory issues
                'NVCC_THREADS': '1',
                'TORCH_CUDA_ARCH_LIST': '8.0',  # A100 compute capability
                'CUDA_HOME': '/usr/local/cuda',
                'CUDA_ROOT': '/usr/local/cuda',
            })
            
            # Install flash-attention with no build isolation
            print("Installing flash-attention (this may take 10-15 minutes)...")
            install_result = subprocess.run([
                "pip", "install", "flash-attn", "--no-build-isolation", "--verbose"
            ], capture_output=True, text=True, timeout=1200, env=env)  # 20 minutes timeout
            
            if install_result.returncode == 0:
                print("✅ Flash-attention installed successfully!")
                try:
                    import flash_attn
                    print(f"✅ Flash Attention version: {flash_attn.__version__}")
                    return True
                except ImportError as e:
                    print(f"⚠️ Installation succeeded but import failed: {e}")
                    return False
            else:
                print(f"❌ Installation failed: {install_result.stderr[-1000:]}")
                return False
                
        finally:
            os.chdir(original_dir)
            
    except Exception as e:
        print(f"❌ Flash-attention installation failed: {e}")
        print("⚠️ Continuing without flash-attention (training will be slower)")
        return False

@app.function(
    gpu="A100",
    volumes={
        MODELS_DIR: model_volume,
        DATA_DIR: data_volume,
        VOL_MOUNT_PATH: output_vol,
    },
    timeout=1800,  # 30 minutes
)
def verify_environment():
    """Verify that all dependencies and GPU are working correctly"""
    print("=== Environment Verification ===")
    
    # Test PyTorch and CUDA
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Install and test flash attention
    flash_attn_available = install_flash_attn_runtime()
    
    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence-transformers import failed: {e}")
    
    # Test datasets library
    try:
        from datasets import load_dataset
        print("✅ Datasets library available")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
    
    # Test model loading
    try:
        print("\n=== Testing Model Loading ===")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(BASE_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=True, truncate_dim=64)
        model.to(torch.float16)  # Use float16 for Jina CLIP v2 flash attention compatibility
        print(f"✅ Model loaded successfully: {BASE_MODEL}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test model with sample input
        print("\n🧪 Testing model with sample input...")
        sample_texts = ["A beautiful sunset over the ocean", "A cat sitting on a table"]
        text_emb = model.encode(sample_texts)
        print(f"✅ Text encoding works: {text_emb.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    print("\n=== Directory Setup ===")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {VOL_MOUNT_PATH}")
    
    return "Environment verification complete"

@app.function(
    gpu="A100",
    cpu=4,
    memory=32768,  # 32GB RAM
    volumes={
        MODELS_DIR: model_volume,
        DATA_DIR: data_volume,
        VOL_MOUNT_PATH: output_vol,
    },
    timeout=7200,  # 2 hours
)
def run_training(
    max_train_samples: int = 5000,
    max_steps: int = 1000,
    batch_size: int = 1,
    eval_samples: int = 500,
):
    """Complete COCO training function with evaluation"""
    import torch
    import os
    import json
    from datetime import datetime
    from random import randint
    from tqdm import tqdm
    from datasets import load_dataset, Features, Value
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    import logging
    
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    print("🚀 Starting multimodal embeddings training with COCO dataset...")
    
    # Verify GPU setup
    print("=== GPU Setup ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Install flash-attention
    flash_attn_available = install_flash_attn_runtime()
    
    # Set optimal environment for A100
    os.environ.update({
        "CUDA_LAUNCH_BLOCKING": "0",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TOKENIZERS_PARALLELISM": "false",
    })
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VOL_MOUNT_PATH, exist_ok=True)
    
    # Load and process COCO dataset
    print("\n=== Loading COCO Dataset ===")
    print("Loading Portuguese COCO captions dataset...")
    
    train_dataset = load_dataset("laicsiifes/coco-captions-pt-br", split="train", streaming=True)
    eval_dataset = load_dataset("laicsiifes/coco-captions-pt-br", split="validation", streaming=True)
    
    def process_example(example):
        """Process COCO example to extract single caption per image"""
        captions = example['caption']
        caption_ids = example['sentids']
        
        # Choose random caption for this image
        idx = randint(0, len(captions) - 1)
        caption = captions[idx]
        
        return {
            'url': example['url'],
            'caption': caption
        }
    
    # Process datasets
    print("Processing datasets...")
    drop_columns = ['filepath', 'filename', 'sentids', 'split', 'cocoid', 'imgid', 'image']
    features = Features({
        'url': Value(dtype='string', id=None),
        'caption': Value(dtype='string', id=None),
    })
    
    train_dataset_processed = train_dataset.map(process_example, remove_columns=drop_columns).cast(features)
    eval_dataset_processed = eval_dataset.map(process_example, remove_columns=drop_columns).cast(features)
    
    # Limit training samples if specified
    if max_train_samples > 0:
        train_dataset_processed = train_dataset_processed.take(max_train_samples)
    
    # Load model
    print(f"\n=== Loading Model: {BASE_MODEL} ===")
    try:
        model = SentenceTransformer(BASE_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=True, truncate_dim=64)
        model.to(torch.float16)
        print(f"✅ Model loaded successfully: {BASE_MODEL}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # ROBUST APPROACH: Unfreeze by parameter position (last layers)
        print("\n🔧 Applying Robust Layer Unfreezing...")
        
        # Get all parameters
        all_params = list(model.named_parameters())
        total_params = sum(p.numel() for _, p in all_params)
        
        # First freeze everything
        for _, param in all_params:
            param.requires_grad = False
        
        # Unfreeze last 5-10% of parameters (by position in the model)
        num_to_unfreeze = 1  # Unfreeze 5% of parameter groups
        
        trainable_params = 0
        for i in range(num_to_unfreeze):
            name, param = all_params[-(i+1)]  # Count from the end
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"  ✅ Unfrozen (position-based): {name} ({param.numel():,} params)")
        
        print(f"\n📊 Position-based Parameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Percentage trainable: {(trainable_params/total_params)*100:.2f}%")
        
        # Test model with sample input
        print("\n🧪 Testing model with sample input...")
        sample_texts = ["A beautiful sunset over the ocean", "A cat sitting on a table"]
        text_emb = model.encode(sample_texts)
        print(f"✅ Text encoding works: {text_emb.shape}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e
    
    # Create evaluation functions
    def create_evaluator_dataset(dataset, max_samples=500):
        """Create evaluation dataset for Information Retrieval evaluation"""
        corpus = {}
        queries = {}
        relevant_docs = {}
        
        print("Creating evaluation dataset...")
        
        # Convert streaming dataset to list for easier processing
        dataset_list = []
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            dataset_list.append(example)
        
        for i, example in enumerate(tqdm(dataset_list, desc="Processing evaluation data")):
            img_url = example['url']
            caption = example['caption']
            
            # Use index as unique ID
            query_id = f"query_{i}"
            doc_id = f"doc_{i}"
            
            # Queries: images (we want to find relevant captions for images)
            queries[query_id] = img_url
            
            # Corpus: captions (the documents to be retrieved)
            corpus[doc_id] = caption
            
            # Relevant docs: each query (image) should retrieve its corresponding caption
            relevant_docs[query_id] = {doc_id}
        
        return corpus, queries, relevant_docs
    
    def evaluate_model_performance(model, eval_dataset, name_suffix=""):
        """Evaluate model performance using Information Retrieval metrics"""
        print(f"Creating evaluation dataset{name_suffix}...")
        corpus, queries, relevant_docs = create_evaluator_dataset(eval_dataset, max_samples=eval_samples)
        
        print(f"Creating evaluator{name_suffix}...")
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"Image-Caption-Retrieval-Evaluator{name_suffix}",
            show_progress_bar=True,
            batch_size=32,  # Reduced batch size for memory efficiency
        )
        
        print(f"Running evaluation{name_suffix}...")
        results = ir_evaluator(model)
        
        # Print detailed results
        print(f"\n=== Evaluation Results{name_suffix} ===")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
        
        return results, ir_evaluator
    
    # Evaluate model before training
    print("\n" + "=" * 50)
    print("EVALUATING MODEL BEFORE FINE-TUNING")
    print("=" * 50)
    
    pre_training_results, pre_training_evaluator = evaluate_model_performance(
        model, eval_dataset_processed, name_suffix=" (Before Fine-tuning)"
    )
    
    # Store the original evaluation results
    original_performance = {
        'map': pre_training_results.get('Image-Caption-Retrieval-Evaluator (Before Fine-tuning)_cosine_map@100', 0),
        'ndcg': pre_training_results.get('Image-Caption-Retrieval-Evaluator (Before Fine-tuning)_cosine_ndcg@10', 0),
        'recall': pre_training_results.get('Image-Caption-Retrieval-Evaluator (Before Fine-tuning)_cosine_recall@10', 0),
        'precision': pre_training_results.get('Image-Caption-Retrieval-Evaluator (Before Fine-tuning)_cosine_precision@10', 0),
        'accuracy@1': pre_training_results.get('Image-Caption-Retrieval-Evaluator (Before Fine-tuning)_cosine_accuracy@1', 0),
    }
    
    print(f"\nOriginal Performance Summary:")
    for metric, score in original_performance.items():
        print(f"{metric}: {score:.4f}")
    
    # Setup training
    print("\n=== Setting Up Training ===")
    
    # Create output directory
    output_dir = VOL_MOUNT_PATH / f"training_jina_clip_v2_cocoptbr-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define loss functions - FIXED: Remove MatryoshkaLoss for Jina CLIP v2 compatibility  
    train_loss = MultipleNegativesRankingLoss(model=model)
    # Note: MatryoshkaLoss is not compatible with Jina CLIP v2 architecture
    # The model already has truncate_dim=64 built-in for dimension reduction
    
    # Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        fp16=False,  # Use float16 through model.to() instead
        bf16=False,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        max_steps=max_steps,
        logging_first_step=True,
        logging_steps=50,
        report_to=["none"],
        gradient_accumulation_steps=2,
        accelerator_config={'dispatch_batches': False}
    )
    
    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        loss=train_loss,
    )
    
    print(f"Training configuration:")
    print(f"- Output directory: {output_dir}")
    print(f"- Batch size: {batch_size}")
    print(f"- Max steps: {max_steps}")
    print(f"- Training samples: {max_train_samples}")
    print(f"- Evaluation samples: {eval_samples}")
    print(f"- Flash Attention: {'Enabled' if flash_attn_available else 'Disabled'}")
    
    # Start training
    print("\n=== Starting Training ===")
    print("🚀 Beginning fine-tuning...")
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise e
    
    # Evaluate model after training
    print("\n" + "=" * 50)
    print("EVALUATING MODEL AFTER FINE-TUNING")
    print("=" * 50)
    
    post_training_results, post_training_evaluator = evaluate_model_performance(
        model, eval_dataset_processed, name_suffix=" (After Fine-tuning)"
    )
    
    # Store the fine-tuned evaluation results
    fine_tuned_performance = {
        'map': post_training_results.get('Image-Caption-Retrieval-Evaluator (After Fine-tuning)_cosine_map@100', 0),
        'ndcg': post_training_results.get('Image-Caption-Retrieval-Evaluator (After Fine-tuning)_cosine_ndcg@10', 0),
        'recall': post_training_results.get('Image-Caption-Retrieval-Evaluator (After Fine-tuning)_cosine_recall@10', 0),
        'precision': post_training_results.get('Image-Caption-Retrieval-Evaluator (After Fine-tuning)_cosine_precision@10', 0),
        'accuracy@1': post_training_results.get('Image-Caption-Retrieval-Evaluator (After Fine-tuning)_cosine_accuracy@1', 0),
    }
    
    print(f"\nFine-tuned Performance Summary:")
    for metric, score in fine_tuned_performance.items():
        print(f"{metric}: {score:.4f}")
    
    # Compare performance
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    
    print(f"{'Metric':<15} {'Before':<10} {'After':<10} {'Improvement':<12} {'% Change':<10}")
    print("-" * 65)
    
    for metric in original_performance.keys():
        before = original_performance[metric]
        after = fine_tuned_performance[metric]
        improvement = after - before
        if before > 0:
            percent_change = (improvement / before) * 100
        else:
            percent_change = float('inf') if improvement > 0 else 0
        
        print(f"{metric:<15} {before:<10.4f} {after:<10.4f} {improvement:<12.4f} {percent_change:<10.2f}%")
    
    # Save model
    print("\n=== Saving Results ===")
    model_save_path = VOL_MOUNT_PATH / "trained_model"
    model.save(str(model_save_path))
    print(f"✅ Model saved to {model_save_path}")
    
    # Save evaluation results and training info
    results_summary = {
        'original_performance': original_performance,
        'fine_tuned_performance': fine_tuned_performance,
        'training_config': {
            'model_name': BASE_MODEL,
            'batch_size': batch_size,
            'max_steps': max_steps,
            'max_train_samples': max_train_samples,
            'eval_samples': eval_samples,
            'truncate_dim': 64,
            'flash_attn_available': flash_attn_available,
            'output_dir': str(output_dir)
        },
        'training_completed': True,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save detailed results
    results_path = VOL_MOUNT_PATH / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save training info for monitoring
    info_path = VOL_MOUNT_PATH / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Commit volume changes
    output_vol.commit()
    
    print(f"✅ Results saved to: {results_path}")
    print("✅ Training completed successfully!")
    
    return results_summary

@app.function(
    volumes={MODELS_DIR: model_volume},
)
def list_saved_models():
    """List all saved models in the models directory"""
    models_path = Path(MODELS_DIR)
    if models_path.exists():
        models = list(models_path.iterdir())
        print("Saved models:")
        for model in models:
            print(f"  - {model.name}")
        return [str(m) for m in models]
    else:
        print("No models directory found")
        return []

@app.function(
    gpu="A100",
    volumes={MODELS_DIR: model_volume},
    timeout=600,
)
def test_saved_model(model_path: str):
    """Test a saved model"""
    import torch
    from sentence_transformers import SentenceTransformer
    
    full_path = Path(MODELS_DIR) / model_path
    if not full_path.exists():
        print(f"Model not found: {full_path}")
        return
    
    try:
        model = SentenceTransformer(str(full_path), device='cuda', trust_remote_code=True, truncate_dim=64)
        model.to(torch.float16)  # Use float16 for Jina CLIP v2 flash attention compatibility
        test_texts = ["A person riding a bicycle", "Uma pessoa andando de bicicleta"]
        embeddings = model.encode(test_texts)
        print(f"✅ Model test successful!")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        
        # Test similarity
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(embeddings[0]), 
            torch.tensor(embeddings[1]), 
            dim=0
        )
        print(f"Text similarity: {similarity:.4f}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

# Web monitoring endpoint  
@app.function(volumes={VOL_MOUNT_PATH: output_vol})
@modal.wsgi_app()
def monitor():
    """Web interface to monitor training progress"""
    from flask import Flask, jsonify
    import json
    import os
    
    web_app = Flask(__name__)
    
    @web_app.route("/")
    def home():
        return """
        <h1>🎯 Multimodal Embeddings Training Monitor</h1>
        <p><a href="/status">📊 Check Status</a></p>
        <p><a href="/results">📈 View Results</a></p>
        """
    
    @web_app.route("/status")
    def status():
        try:
            output_dir = VOL_MOUNT_PATH / "output"
            model_dir = VOL_MOUNT_PATH / "trained_model"
            
            status_info = {
                "output_exists": output_dir.exists(),
                "model_exists": model_dir.exists(),
                "files": [str(f) for f in VOL_MOUNT_PATH.glob("*")] if VOL_MOUNT_PATH.exists() else []
            }
            return jsonify(status_info)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    @web_app.route("/results")
    def results():
        try:
            results_path = VOL_MOUNT_PATH / "training_info.json"
            if results_path.exists():
                with open(results_path) as f:
                    return jsonify(json.load(f))
            else:
                return jsonify({"error": "No results file found"})
        except Exception as e:
            return jsonify({"error": str(e)})
    
    return web_app

@app.local_entrypoint()
def main(
    verify: bool = True,
    max_train_samples: int = 5000,
    max_steps: int = 1000,
    batch_size: int = 1,
    eval_samples: int = 500,
):
    """
    Start multimodal embeddings training with COCO dataset
    
    Args:
        verify: Whether to run environment verification first
        max_train_samples: Number of training samples (0 for all)
        max_steps: Maximum training steps
        batch_size: Training batch size
        eval_samples: Number of evaluation samples
    """
    
    if verify:
        print("🔍 Verifying environment...")
        verify_environment.remote()
    
    print("🚀 Starting COCO training...")
    result = run_training.remote(
        max_train_samples=max_train_samples,
        max_steps=max_steps,
        batch_size=batch_size,
        eval_samples=eval_samples,
    )
    
    print("✅ Training completed!")
    print("📊 Results:", result)
    print(f"🌐 Monitor at: https://ise703--jina-clip-multimodal-embeddings-monitor-dev.modal.run")
