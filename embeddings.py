from pathlib import Path
import json
import modal
import hashlib

VOL_MOUNT_PATH = Path("/vol")

# BAAI/bge-m3 base model for embedding finetuning (1024 dimensions)
BASE_MODEL = "BAAI/bge-m3"

# Define the Modal image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "tensorboard", 
    "sentence-transformers",
    "datasets",
    "transformers",
    "huggingface_hub",
    "accelerate",
    "scikit-learn",
    "scipy",
    "tqdm",
).add_local_dir("datasets", "/datasets", ignore=["images/*", "corpus/*"])  # Add datasets directory to image

app = modal.App(name="bge-m3-medical-embeddings", image=image)
output_vol = modal.Volume.from_name("embeddings-volume", create_if_missing=True)

# ### Handling preemption for long-running training jobs

restart_tracker_dict = modal.Dict.from_name(
    "embeddings-restart-tracker", create_if_missing=True
)

def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count

# ## Data loading and preparation functions

def load_jsonl_file(file_path: str):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def prepare_data_for_training(qa_data):
    """Convert QA data to format expected by sentence transformers - KHÔNG deduplicate"""
    prepared_data = []
    
    for item in qa_data:
        prepared_item = {
            "id": item["question_idx"],
            "anchor": item["question"],
            "positive": item["context"],
            "answer": item["answer"],
            "title": item.get("title", ""),
            "keyword": item.get("keyword", ""),
        }
        prepared_data.append(prepared_item)
    
    print(f"📊 Training data prepared:")
    print(f"   Total samples: {len(prepared_data)}")
    
    return prepared_data

def deduplicate_corpus_for_evaluation(corpus_data):
    """Remove duplicate content from corpus used for evaluation"""
    content_to_first_id = {}
    unique_corpus = {}
    
    for item_id, content in corpus_data.items():
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash not in content_to_first_id:
            content_to_first_id[content_hash] = item_id
            unique_corpus[item_id] = content
        # Skip duplicates
    
    print(f"📊 Corpus deduplication:")
    print(f"   Original entries: {len(corpus_data)}")
    print(f"   Unique entries: {len(unique_corpus)}")
    
    return unique_corpus

# ## Finetuning BAAI/bge-m3 on Vietnamese Medical QA dataset

@app.function(
    gpu="A100",
    volumes={VOL_MOUNT_PATH: output_vol},
    timeout=43200,
)
def finetune_embeddings(
    num_train_epochs: int = 4,
):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.util import cos_sim
    
    restarts = track_restarts(restart_tracker_dict)
    
    print("Loading BAAI/bge-m3 model...")
    # Load model with multilingual support (1024 dimensions)
    model = SentenceTransformer(
        BASE_MODEL,
        model_card_data=SentenceTransformerModelCardData(
            language="vi",
            license="apache-2.0",
            model_name="BGE-m3-medical",
        ),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Loading FULL training data from JSONL files...")
    
    # Load ALL data from JSONL files (no sample limiting)
    train_data = load_jsonl_file("/datasets/q_a_train_filtered.jsonl")
    test_data = load_jsonl_file("/datasets/q_a_test_filtered.jsonl")
    validation_data = load_jsonl_file("/datasets/q_a_validation_filtered.jsonl")
    
    print(f"Loaded {len(train_data)} training samples from JSONL (FULL DATASET)")
    print(f"Loaded {len(test_data)} test samples from JSONL (FULL DATASET)")
    print(f"Loaded {len(validation_data)} validation samples from JSONL (FULL DATASET)")
    
    # Prepare data for training - GIỮ NGUYÊN tất cả samples
    prepared_train = prepare_data_for_training(train_data)
    prepared_test = prepare_data_for_training(test_data)
    prepared_validation = prepare_data_for_training(validation_data)
    
    # Convert to datasets - FULL training data
    train_dataset = Dataset.from_list(prepared_train)
    test_dataset = Dataset.from_list(prepared_test)
    validation_dataset = Dataset.from_list(prepared_validation)
    
    print(f"Prepared {len(train_dataset)} training samples (FULL DATASET)")
    print(f"Prepared {len(test_dataset)} test samples (FULL DATASET)")
    print(f"Prepared {len(validation_dataset)} validation samples (FULL DATASET)")
    
    # Sample data structure for verification
    if train_dataset:
        sample = train_dataset[0]
        print(f"\nSample training data:")
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['anchor'][:100]}...")
        print(f"Context: {sample['positive'][:100]}...")
    
    # CHỈ deduplicate cho evaluation corpus
    all_data = prepared_train + prepared_validation + prepared_test
    corpus_dataset = Dataset.from_list(all_data)
    
    # Tạo corpus RAW (có duplicates) 
    corpus_raw = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
    
    # CHỈ deduplicate cho evaluation
    corpus = deduplicate_corpus_for_evaluation(corpus_raw)
    
    # Helper function to build relevant_docs mapping
    def build_relevant_docs(dataset, corpus_raw, corpus):
        """Build relevant docs mapping handling deduplication correctly"""
        queries = dict(zip(dataset["id"], dataset["anchor"]))
        relevant_docs = {}
        
        for q_id in queries:
            if q_id in corpus:  # Nếu question này có trong deduplicated corpus
                relevant_docs[q_id] = [q_id]  # Chỉ point đến chính nó
            else:
                # Nếu bị deduplicate, tìm ID đại diện có cùng content
                original_content = corpus_raw[q_id]
                for corpus_id, corpus_content in corpus.items():
                    if corpus_content == original_content:
                        relevant_docs[q_id] = [corpus_id]
                        break
                else:
                    relevant_docs[q_id] = []
        
        return queries, relevant_docs
    
    # Create SEPARATE evaluators for validation and test to avoid data leakage
    
    # 1. Validation evaluator (for training monitoring)
    val_queries, val_relevant_docs = build_relevant_docs(validation_dataset, corpus_raw, corpus)
    
    val_ir_evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=corpus,
        relevant_docs=val_relevant_docs,
        name="validation",
        score_functions={"cosine": cos_sim},
        show_progress_bar=True,
        batch_size=32,
    )
    
    # 2. Test evaluator (for baseline and final evaluation only)
    test_queries, test_relevant_docs = build_relevant_docs(test_dataset, corpus_raw, corpus)
    
    test_ir_evaluator = InformationRetrievalEvaluator(
        queries=test_queries,
        corpus=corpus,
        relevant_docs=test_relevant_docs,
        name="test",
        score_functions={"cosine": cos_sim},
        show_progress_bar=True,
        batch_size=32,
    )
    
    print(f"\nEvaluation setup:")
    print(f"Number of validation queries: {len(val_queries)}")
    print(f"Number of test queries: {len(test_queries)}")
    print(f"Number of corpus documents: {len(corpus)}")
    print(f"Number of validation relevant docs mappings: {len(val_relevant_docs)}")
    print(f"Number of test relevant docs mappings: {len(test_relevant_docs)}")
    
    # Run baseline evaluation on TEST SET before training
    print("\n" + "="*50)
    print("BASELINE EVALUATION (Before Training) - TEST SET")
    print("="*50)
    
    baseline_results = test_ir_evaluator(model)
    
    print("Baseline Results (Test Set):")
    print(f"cosine_ndcg@10: {baseline_results['test_cosine_ndcg@10']:.4f}")
    
    # Clear GPU cache after baseline evaluation
    torch.cuda.empty_cache()
    print("GPU memory cleared after baseline evaluation")

    val_baseline_results = val_ir_evaluator(model)
    print("Validation Baseline Results:")
    print(f"cosine_ndcg@10: {val_baseline_results['validation_cosine_ndcg@10']:.4f}")
    
    # Setup standard contrastive loss
    train_loss = MultipleNegativesRankingLoss(model)
    
    # Define training arguments - with validation evaluation
    args = SentenceTransformerTrainingArguments(
        output_dir=str(VOL_MOUNT_PATH / "bge-m3-medical"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,             
        gradient_accumulation_steps=32,             
        per_device_eval_batch_size=32,              
        warmup_ratio=0.1,                           
        learning_rate=2e-5,                         
        lr_scheduler_type="cosine",                 
        optim="adamw_torch_fused",                  
        tf32=True,                                  
        bf16=True,                                  
        gradient_checkpointing=True,                
        batch_sampler=BatchSamplers.NO_DUPLICATES,  
        eval_strategy="epoch",                                               
        save_strategy="epoch",                      
        logging_steps=50,                           
        save_total_limit=3,                         
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_cosine_ndcg@10"
    )

    # Create trainer with validation dataset and VALIDATION evaluator (NO TEST DATA LEAKAGE)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["anchor", "positive"]),
        loss=train_loss,
        evaluator=val_ir_evaluator,
    )
    
    try:
        # Better checkpoint detection and validation
        checkpoint_dir = VOL_MOUNT_PATH / "bge-m3-medical"
        resume_from_checkpoint = None
        
        if checkpoint_dir.exists():
            # Find all valid checkpoint directories
            checkpoint_subdirs = [d for d in checkpoint_dir.glob("checkpoint-*") if d.is_dir()]
            
            if checkpoint_subdirs and restarts > 0:
                # Sort by step number and find the latest
                valid_checkpoints = []
                for ckpt_dir in checkpoint_subdirs:
                    # Check if checkpoint has required files
                    required_files = [
                        ckpt_dir / "trainer_state.json",
                        ckpt_dir / "training_args.bin",
                    ]
                    # Check for either pytorch_model.bin or model.safetensors
                    model_files = [
                        ckpt_dir / "pytorch_model.bin",
                        ckpt_dir / "model.safetensors",
                    ]
                    
                    if all(f.exists() for f in required_files) and any(f.exists() for f in model_files):
                        try:
                            step_num = int(ckpt_dir.name.split("-")[1])
                            valid_checkpoints.append((step_num, ckpt_dir))
                        except (IndexError, ValueError):
                            continue
                
                if valid_checkpoints:
                    # Get the latest valid checkpoint
                    latest_step, latest_checkpoint = max(valid_checkpoints, key=lambda x: x[0])
                    resume_from_checkpoint = str(latest_checkpoint)
                    print(f"Resuming from checkpoint: {resume_from_checkpoint} (step {latest_step}, restart count: {restarts})")
                else:
                    print(f"No valid checkpoints found. Starting fresh training (restart count: {restarts})")
            else:
                print(f"No checkpoints to resume from (restart count: {restarts})")
        else:
            print(f"Starting fresh training (restart count: {restarts})")
        
        print("Starting training on FULL DATASET...")
        print("📊 Training will be monitored using VALIDATION SET (no test data leakage)")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Clear GPU cache after training completes
        torch.cuda.empty_cache()
        print("GPU memory cleared after training completion")
        
    except KeyboardInterrupt:
        print("Received interrupt; saving state and model")
        trainer.save_state()
        trainer.save_model()
        # Clear memory on interrupt
        torch.cuda.empty_cache()
        raise
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save current state even on failure
        try:
            trainer.save_state()
            trainer.save_model()
        except:
            pass
        torch.cuda.empty_cache()
        raise

    # Save the best model with explicit path checking
    print("Saving final model...")
    try:
        trainer.save_model()
        
        # Verify the model was saved correctly
        final_model_path = VOL_MOUNT_PATH / "bge-m3-medical"
        model_files = list(final_model_path.glob("*.bin")) + list(final_model_path.glob("*.safetensors"))
        
        if not model_files:
            print("Warning: No model files found after saving. Attempting manual save...")
            # Manual save as fallback
            model.save(str(final_model_path))
        
        print(f"Model saved successfully to {final_model_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        # Fallback save
        fallback_path = VOL_MOUNT_PATH / "bge-m3-medical-fallback"
        model.save(str(fallback_path))
        print(f"Model saved to fallback path: {fallback_path}")
    
    # Evaluate fine-tuned model after training ON TEST SET
    print("\n" + "="*50)
    print("FINAL EVALUATION (After Training) - TEST SET")
    print("="*50)
    
    # Load the fine-tuned model
    fine_tuned_model = SentenceTransformer(
        str(VOL_MOUNT_PATH / "bge-m3-medical"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Run final evaluation on TEST SET
    final_results = test_ir_evaluator(fine_tuned_model)
    print("Fine-tuned Results (Test Set):")
    print(f"cosine_ndcg@10: {final_results['test_cosine_ndcg@10']:.4f}")
    
    # Clear GPU cache after final evaluation
    torch.cuda.empty_cache()
    print("GPU memory cleared after final evaluation")
    
    # Show improvement comparison
    print("\n" + "="*50)
    print("IMPROVEMENT SUMMARY (TEST SET)")
    print("="*50)
    baseline_score = baseline_results['test_cosine_ndcg@10']
    final_score = final_results['test_cosine_ndcg@10']
    improvement = final_score - baseline_score
    if baseline_score > 0:
        improvement_pct = (improvement / baseline_score) * 100
        print(f"NDCG@10: {baseline_score:.4f} → {final_score:.4f} (+{improvement:.4f}, +{improvement_pct:.1f}%)")
    else:
        print(f"NDCG@10: baseline skipped → {final_score:.4f}")
    print("="*50)
    
    # Reset restart counter on successful completion
    restart_tracker_dict["count"] = 0
    print("Reset restart counter after successful training")
    
    # Commit changes to volume
    output_vol.commit()
    print("✅ Training completed successfully on FULL DATASET!")
    
    return {
        "baseline_results": baseline_results,
        "final_results": final_results,
        "training_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "validation_samples": len(validation_dataset),
        "status": "completed"
    }

# ## Monitoring with Tensorboard

@app.function(volumes={VOL_MOUNT_PATH: output_vol})
@modal.wsgi_app()
def monitor():
    import tensorboard
    
    board = tensorboard.program.TensorBoard()
    board.configure(logdir=f"{VOL_MOUNT_PATH}/bge-m3-medical/logs")
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app

# ## Model Inference Class

@app.cls(volumes={VOL_MOUNT_PATH: output_vol})
class EmbeddingModel:
    @modal.enter()
    def load_model(self):
        import torch
        from sentence_transformers import SentenceTransformer
        
        # Load the finetuned model
        model_path = VOL_MOUNT_PATH / "bge-m3-medical"
        self.model = SentenceTransformer(
            str(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully!")
    
    @modal.method()
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to 1024-dimensional embeddings"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    @modal.method()
    def similarity(self, query: str, documents: list[str]) -> list[float]:
        """Compute similarity between query and documents"""
        from sentence_transformers.util import cos_sim
        
        query_embedding = self.model.encode([query])
        doc_embeddings = self.model.encode(documents)
        
        similarities = cos_sim(query_embedding, doc_embeddings)[0]
        return similarities.tolist()

@app.function(gpu="A100", timeout=43200)
def run_complete_workflow(num_train_epochs: int = 4):
    """Run the complete training workflow with built-in evaluation"""
    
    print("🚀 Starting complete BGE-M3 finetuning workflow...")
    
    # Just run training - it includes all evaluation now
    training_results = finetune_embeddings.remote(num_train_epochs=num_train_epochs)
    
    print("\n✅ Complete workflow finished!")
    return training_results

@app.function(
    gpu="A10G", 
    volumes={VOL_MOUNT_PATH: output_vol}, 
    timeout=21600  # 6 hours timeout
)
def embed_chunked_corpus():
    import json
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    
    model = SentenceTransformer(
        str(VOL_MOUNT_PATH / "bge-m3-medical" ),
        device="cuda"
    )
    
    input_file = "/datasets/chunks/context_corpus_clean.jsonl"
    output_file = str(VOL_MOUNT_PATH / "context_corpus_embedded_clean_2.jsonl")
    
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        batch_size = 32
        batch_lines = []
        batch_contents = []
        
        with tqdm(total=total_lines, desc="Embedding chunks") as pbar:
            for line in infile:
                data = json.loads(line.strip())
                batch_lines.append(data)
                batch_contents.append(data["content"])
                
                if len(batch_contents) == batch_size:
                    embeddings = model.encode(batch_contents)
                    
                    for i, embedding in enumerate(embeddings):
                        batch_lines[i]["embedding"] = embedding.tolist()
                        outfile.write(json.dumps(batch_lines[i], ensure_ascii=False) + '\n')
                    
                    pbar.update(len(batch_lines))
                    batch_lines = []
                    batch_contents = []
            
            if batch_contents:
                embeddings = model.encode(batch_contents)
                
                for i, embedding in enumerate(embeddings):
                    batch_lines[i]["embedding"] = embedding.tolist()
                    outfile.write(json.dumps(batch_lines[i], ensure_ascii=False) + '\n')
                
                pbar.update(len(batch_lines))
    
    output_vol.commit()
    return output_file

# ## CLI Usage Examples

# To start finetuning:
# modal run --detach embeddings.py::finetune_embeddings --num-train-epochs=4

# To monitor training with tensorboard:
# Visit: https://ise703--bge-m3-medical-embeddings-monitor-dev.modal.run

# To test inference with real data:
# modal run embeddings.py