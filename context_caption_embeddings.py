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

app = modal.App(name="bge-m3-context-caption-embeddings", image=image)
output_vol = modal.Volume.from_name("context-caption-embeddings-volume", create_if_missing=True)

# ### Handling preemption for long-running training jobs

restart_tracker_dict = modal.Dict.from_name(
    "context-caption-restart-tracker", create_if_missing=True
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

def prepare_context_caption_data(image_data, dataset_prefix=""):
    """Convert image question mappings data to context-caption pairs for training"""
    prepared_data = []
    
    for idx, item in enumerate(image_data):
        # Create unique ID with dataset prefix and index
        unique_id = f"{dataset_prefix}_{idx}" if dataset_prefix else f"img_{idx}"
        
        prepared_item = {
            "id": unique_id,
            "anchor": item["context"],
            "positive": item["caption"],
            "question": item["question"],
            "answer": item["answer"],
            "image_filename": item.get("image_filename", ""),
            "support_type": item.get("support_type", ""),
        }
        prepared_data.append(prepared_item)
    
    print(f"📊 Context-Caption training data prepared:")
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

# ## Finetuning BAAI/bge-m3 on Context-Caption pairs

@app.function(
    gpu="A100",
    volumes={VOL_MOUNT_PATH: output_vol},
    timeout=43200,
)
def finetune_context_caption_embeddings(
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
    
    print("Loading BAAI/bge-m3 model for context-caption training...")
    # Load model with multilingual support (1024 dimensions)
    model = SentenceTransformer(
        BASE_MODEL,
        model_card_data=SentenceTransformerModelCardData(
            language="vi",
            license="apache-2.0",
            model_name="BGE-m3-context-caption",
        ),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Loading context-caption data from cleaned JSONL files...")
    
    # Load cleaned data from image question mappings
    train_data = load_jsonl_file("/datasets/image_question_mappings_train_clean.jsonl")
    test_data = load_jsonl_file("/datasets/image_question_mappings_test_clean.jsonl")
    validation_data = load_jsonl_file("/datasets/image_question_mappings_val_clean.jsonl")
    
    print(f"Loaded {len(train_data)} training samples from cleaned JSONL")
    print(f"Loaded {len(test_data)} test samples from cleaned JSONL")
    print(f"Loaded {len(validation_data)} validation samples from cleaned JSONL")
    
    # Prepare data for context-caption training with unique IDs
    prepared_train = prepare_context_caption_data(train_data, "train")
    prepared_test = prepare_context_caption_data(test_data, "test") 
    prepared_validation = prepare_context_caption_data(validation_data, "val")
    
    # Convert to datasets
    train_dataset = Dataset.from_list(prepared_train)
    test_dataset = Dataset.from_list(prepared_test)
    validation_dataset = Dataset.from_list(prepared_validation)
    
    print(f"Prepared {len(train_dataset)} training samples for context-caption")
    print(f"Prepared {len(test_dataset)} test samples for context-caption")
    print(f"Prepared {len(validation_dataset)} validation samples for context-caption")
    
    # Sample data structure for verification
    if train_dataset:
        sample = train_dataset[0]
        print(f"\nSample context-caption training data:")
        print(f"ID: {sample['id']}")
        print(f"Context (anchor): {sample['anchor'][:100]}...")
        print(f"Caption (positive): {sample['positive'][:100]}...")
        print(f"Support type: {sample.get('support_type', 'N/A')}")
    
    # Helper function to build relevant_docs mapping for context-caption
    def build_context_caption_relevant_docs(dataset, corpus_raw, corpus):
        """Build relevant docs mapping for context-caption pairs"""
        queries = dict(zip(dataset["id"], dataset["anchor"]))  # CONTEXTS as queries
        relevant_docs = {}
        
        for q_id in queries:
            # Each context should match its corresponding caption
            if q_id in corpus:  # If this caption exists in deduplicated corpus
                relevant_docs[q_id] = [q_id]  # Context should match its caption
            else:
                # Find the caption that matches this context
                original_caption = corpus_raw[q_id]  # Get the original caption
                for corpus_id, corpus_caption in corpus.items():
                    if corpus_caption == original_caption:
                        relevant_docs[q_id] = [corpus_id]
                        break
                else:
                    relevant_docs[q_id] = []
        
        return queries, relevant_docs
    
    # Create corpus for evaluation (captions as documents, contexts as queries)
    all_data = prepared_train + prepared_validation + prepared_test
    corpus_dataset = Dataset.from_list(all_data)
    
    # Build corpus: caption ID -> caption content (CAPTIONS are the corpus)
    corpus_raw = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))  # captions
    
    # Deduplicate corpus for evaluation
    corpus = deduplicate_corpus_for_evaluation(corpus_raw)
    
    # Create evaluators for validation and test
    val_queries, val_relevant_docs = build_context_caption_relevant_docs(validation_dataset, corpus_raw, corpus)
    
    val_ir_evaluator = InformationRetrievalEvaluator(
        queries=val_queries,  # contexts as queries
        corpus=corpus,        # captions as corpus
        relevant_docs=val_relevant_docs,
        name="validation_context_caption",
        score_functions={"cosine": cos_sim},
        show_progress_bar=True,
        batch_size=32,
    )
    
    test_queries, test_relevant_docs = build_context_caption_relevant_docs(test_dataset, corpus_raw, corpus)
    
    test_ir_evaluator = InformationRetrievalEvaluator(
        queries=test_queries,  # contexts as queries  
        corpus=corpus,         # captions as corpus
        relevant_docs=test_relevant_docs,
        name="test_context_caption",
        score_functions={"cosine": cos_sim},
        show_progress_bar=True,
        batch_size=32,
    )
    
    print(f"\nContext-Caption evaluation setup:")
    print(f"Number of validation queries (contexts): {len(val_queries)}")
    print(f"Number of test queries (contexts): {len(test_queries)}")
    print(f"Number of corpus documents (captions): {len(corpus)}")
    
    # Run baseline evaluation on TEST SET before training
    print("\n" + "="*50)
    print("BASELINE EVALUATION - CONTEXT-CAPTION (Before Training)")
    print("="*50)
    
    baseline_results = test_ir_evaluator(model)
    print("Baseline Results (Context-Caption Test Set):")
    print(f"cosine_ndcg@10: {baseline_results['test_context_caption_cosine_ndcg@10']:.4f}")
    
    torch.cuda.empty_cache()
    
    val_baseline_results = val_ir_evaluator(model)
    print("Validation Baseline Results (Context-Caption):")
    print(f"cosine_ndcg@10: {val_baseline_results['validation_context_caption_cosine_ndcg@10']:.4f}")
    
    # Setup contrastive loss for context-caption pairs
    train_loss = MultipleNegativesRankingLoss(model)
    
    # Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(VOL_MOUNT_PATH / "bge-m3-context-caption"),
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
        metric_for_best_model="eval_validation_context_caption_cosine_ndcg@10"
    )

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["anchor", "positive"]),  # context-caption pairs
        loss=train_loss,
        evaluator=val_ir_evaluator,
    )
    
    try:
        # Checkpoint detection for context-caption training
        checkpoint_dir = VOL_MOUNT_PATH / "bge-m3-context-caption"
        resume_from_checkpoint = None
        
        if checkpoint_dir.exists():
            checkpoint_subdirs = [d for d in checkpoint_dir.glob("checkpoint-*") if d.is_dir()]
            
            if checkpoint_subdirs and restarts > 0:
                valid_checkpoints = []
                for ckpt_dir in checkpoint_subdirs:
                    required_files = [
                        ckpt_dir / "trainer_state.json",
                        ckpt_dir / "training_args.bin",
                    ]
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
                    latest_step, latest_checkpoint = max(valid_checkpoints, key=lambda x: x[0])
                    resume_from_checkpoint = str(latest_checkpoint)
                    print(f"Resuming context-caption training from: {resume_from_checkpoint}")
        
        print("Starting context-caption training...")
        print("📊 Training contexts to match captions using validation set monitoring")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print("Received interrupt; saving context-caption model state")
        trainer.save_state()
        trainer.save_model()
        torch.cuda.empty_cache()
        raise
    except Exception as e:
        print(f"Context-caption training failed: {e}")
        try:
            trainer.save_state()
            trainer.save_model()
        except:
            pass
        torch.cuda.empty_cache()
        raise

    # Save the context-caption model
    print("Saving final context-caption model...")
    try:
        trainer.save_model()
        
        final_model_path = VOL_MOUNT_PATH / "bge-m3-context-caption"
        model_files = list(final_model_path.glob("*.bin")) + list(final_model_path.glob("*.safetensors"))
        
        if not model_files:
            print("Warning: No model files found. Attempting manual save...")
            model.save(str(final_model_path))
        
        print(f"Context-caption model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Error saving context-caption model: {e}")
        fallback_path = VOL_MOUNT_PATH / "bge-m3-context-caption-fallback"
        model.save(str(fallback_path))
        print(f"Model saved to fallback path: {fallback_path}")
    
    # Final evaluation on TEST SET
    print("\n" + "="*50)
    print("FINAL EVALUATION - CONTEXT-CAPTION (After Training)")
    print("="*50)
    
    fine_tuned_model = SentenceTransformer(
        str(VOL_MOUNT_PATH / "bge-m3-context-caption"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    final_results = test_ir_evaluator(fine_tuned_model)
    print("Fine-tuned Results (Context-Caption Test Set):")
    print(f"cosine_ndcg@10: {final_results['test_context_caption_cosine_ndcg@10']:.4f}")
    
    torch.cuda.empty_cache()
    
    # Show improvement
    print("\n" + "="*50)
    print("CONTEXT-CAPTION IMPROVEMENT SUMMARY")
    print("="*50)
    baseline_score = baseline_results['test_context_caption_cosine_ndcg@10']
    final_score = final_results['test_context_caption_cosine_ndcg@10']
    improvement = final_score - baseline_score
    if baseline_score > 0:
        improvement_pct = (improvement / baseline_score) * 100
        print(f"Context-Caption NDCG@10: {baseline_score:.4f} → {final_score:.4f} (+{improvement:.4f}, +{improvement_pct:.1f}%)")
    else:
        print(f"Context-Caption NDCG@10: baseline skipped → {final_score:.4f}")
    print("="*50)
    
    # Reset restart counter
    restart_tracker_dict["count"] = 0
    
    # Commit changes
    output_vol.commit()
    print("✅ Context-caption training completed successfully!")
    
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
    board.configure(logdir=f"{VOL_MOUNT_PATH}/bge-m3-context-caption/logs")
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
class ContextCaptionEmbeddingModel:
    @modal.enter()
    def load_model(self):
        import torch
        from sentence_transformers import SentenceTransformer
        
        # Load the finetuned context-caption model
        model_path = VOL_MOUNT_PATH / "bge-m3-context-caption"
        self.model = SentenceTransformer(
            str(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Context-Caption model loaded successfully!")
    
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
    
    @modal.method()
    def find_best_captions_for_context(self, context: str, captions: list[str], top_k: int = 5) -> list[tuple]:
        """Find best matching captions for a given context"""
        from sentence_transformers.util import cos_sim
        
        context_embedding = self.model.encode([context])
        caption_embeddings = self.model.encode(captions)
        
        similarities = cos_sim(context_embedding, caption_embeddings)[0]
        
        # Get top-k results
        top_indices = similarities.argsort(descending=True)[:top_k]
        results = [(captions[i], float(similarities[i])) for i in top_indices]
        
        return results
    
    @modal.method()
    def find_best_contexts_for_caption(self, caption: str, contexts: list[str], top_k: int = 5) -> list[tuple]:
        """Find best matching contexts for a given caption"""
        from sentence_transformers.util import cos_sim
        
        caption_embedding = self.model.encode([caption])
        context_embeddings = self.model.encode(contexts)
        
        similarities = cos_sim(caption_embedding, context_embeddings)[0]
        
        # Get top-k results
        top_indices = similarities.argsort(descending=True)[:top_k]
        results = [(contexts[i], float(similarities[i])) for i in top_indices]
        
        return results

@app.function(gpu="A100", timeout=43200)
def run_context_caption_workflow(num_train_epochs: int = 4):
    """Run the complete context-caption training workflow"""
    
    print("🚀 Starting BGE-M3 context-caption finetuning workflow...")
    
    training_results = finetune_context_caption_embeddings.remote(num_train_epochs=num_train_epochs)
    
    print("\n✅ Context-caption workflow finished!")
    return training_results

@app.function(
    gpu="A10G", 
    volumes={VOL_MOUNT_PATH: output_vol}, 
    timeout=21600  # 6 hours timeout
)
def embed_context_caption_pairs():
    """Embed all context-caption pairs for fast retrieval"""
    import json
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    
    model = SentenceTransformer(
        str(VOL_MOUNT_PATH / "bge-m3-context-caption"),
        device="cuda"
    )
    
    # Process all cleaned files
    files_to_process = [
        "/datasets/image_question_mappings_train_clean.jsonl",
        "/datasets/image_question_mappings_test_clean.jsonl", 
        "/datasets/image_question_mappings_val_clean.jsonl"
    ]
    
    output_file = str(VOL_MOUNT_PATH / "context_caption_embeddings.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in files_to_process:
            print(f"Processing {input_file}...")
            
            # Count total lines for progress bar
            total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
            
            with open(input_file, 'r', encoding='utf-8') as infile:
                batch_size = 32
                batch_contexts = []
                batch_captions = []
                batch_data = []
                
                with tqdm(total=total_lines, desc=f"Embedding {input_file}") as pbar:
                    for line in infile:
                        data = json.loads(line.strip())
                        
                        batch_contexts.append(data["context"])
                        batch_captions.append(data["caption"])
                        batch_data.append(data)
                        
                        if len(batch_contexts) == batch_size:
                            # Embed contexts and captions
                            context_embeddings = model.encode(batch_contexts)
                            caption_embeddings = model.encode(batch_captions)
                            
                            # Save with embeddings
                            for i, original_data in enumerate(batch_data):
                                embedded_data = original_data.copy()
                                embedded_data["context_embedding"] = context_embeddings[i].tolist()
                                embedded_data["caption_embedding"] = caption_embeddings[i].tolist()
                                outfile.write(json.dumps(embedded_data, ensure_ascii=False) + '\n')
                            
                            pbar.update(len(batch_data))
                            batch_contexts = []
                            batch_captions = []
                            batch_data = []
                    
                    # Process remaining items
                    if batch_contexts:
                        context_embeddings = model.encode(batch_contexts)
                        caption_embeddings = model.encode(batch_captions)
                        
                        for i, original_data in enumerate(batch_data):
                            embedded_data = original_data.copy()
                            embedded_data["context_embedding"] = context_embeddings[i].tolist()
                            embedded_data["caption_embedding"] = caption_embeddings[i].tolist()
                            outfile.write(json.dumps(embedded_data, ensure_ascii=False) + '\n')
                        
                        pbar.update(len(batch_data))
    
    output_vol.commit()
    print(f"✅ Context-caption embeddings saved to {output_file}")
    return output_file

# ## CLI Usage Examples

# To start context-caption finetuning:
# modal run --detach context_caption_embeddings.py::finetune_context_caption_embeddings --num-train-epochs=4

# To run complete workflow:
# modal run --detach context_caption_embeddings.py::run_context_caption_workflow --num-train-epochs=4

# To monitor training with tensorboard:
# Visit: https://ise703--bge-m3-context-caption-embeddings-monitor-dev.modal.run

# To embed all context-caption pairs:
# modal run --detach context_caption_embeddings.py::embed_context_caption_pairs
