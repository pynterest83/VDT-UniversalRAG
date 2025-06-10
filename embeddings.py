# # Finetuning BGE-M3 Embedding Model for Vietnamese Medical QA

# This example demonstrates how to finetune the BGE-M3 embedding model using Modal cloud
# infrastructure with GPU acceleration. The model is trained on Vietnamese medical question-answer
# data using Matryoshka representation learning for efficient embeddings at multiple dimensions.

from pathlib import Path
import json
import modal

VOL_MOUNT_PATH = Path("/vol")

# BGE-M3 base model for embedding finetuning
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
).add_local_dir("datasets", "/datasets")  # Add datasets directory to image

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
    """Convert QA data to the format expected by sentence transformers"""
    prepared_data = []
    for item in qa_data:
        prepared_item = {
            "id": item["question_idx"],
            "anchor": item["question"],  # Query
            "positive": item["context"],  # Relevant document
            "answer": item["answer"],
            "title": item.get("title", ""),
            "keyword": item.get("keyword", ""),
        }
        prepared_data.append(prepared_item)
    return prepared_data

# ## Finetuning BGE-M3 on Vietnamese Medical QA dataset

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
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
    from sentence_transformers.util import cos_sim
    
    restarts = track_restarts(restart_tracker_dict)
    
    # Matryoshka dimensions (large to small)
    matryoshka_dimensions = [1024, 768, 512, 256, 128, 64]
    
    print("Loading BGE-M3 model...")
    # Load model with Flash Attention 2 for efficiency
    model = SentenceTransformer(
        BASE_MODEL,
        model_kwargs={"attn_implementation": "sdpa"},
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
    
    # Prepare data for training
    prepared_train = prepare_data_for_training(train_data)
    prepared_test = prepare_data_for_training(test_data)
    prepared_validation = prepare_data_for_training(validation_data)
    
    # Convert to datasets
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
    
    # Create evaluation setup - use full corpus since dataset is smaller now
    all_data = prepared_train + prepared_validation + prepared_test
    corpus_dataset = Dataset.from_list(all_data)
    
    # Use full corpus (no shrinking needed for 10k dataset)
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
    
    # Create relevant docs mapping
    relevant_docs = {}
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    
    print(f"\nEvaluation setup:")
    print(f"Number of queries: {len(queries)}")
    print(f"Number of corpus documents: {len(corpus)}")
    print(f"Number of relevant docs mappings: {len(relevant_docs)}")
    
    # SKIP baseline evaluation to save memory for training
    print("\n" + "="*50)
    print("SKIPPING BASELINE EVALUATION - SAVING MEMORY FOR TRAINING")
    print("="*50)
    
    # Create dummy baseline results for comparison later
    baseline_results = {f"dim_{dim}_cosine_ndcg@10": 0.0 for dim in matryoshka_dimensions}
    
    # Clear any existing GPU cache
    torch.cuda.empty_cache()
    print("GPU memory cleared before training")
    
    # Setup Matryoshka loss
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
    )
    
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
        eval_strategy="steps",                      # Re-enable validation evaluation
        eval_steps=500,                             # Evaluate every 500 steps
        save_strategy="steps",                      
        save_steps=500,                             
        logging_steps=50,                           
        save_total_limit=3,                         
        load_best_model_at_end=True,                # Load best model based on validation
        metric_for_best_model="eval_loss",          # Use validation loss (memory efficient)
        dataloader_num_workers=4,                   
        dataloader_pin_memory=False,                
    )

    # Create trainer with validation dataset
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["anchor", "positive"]),
        eval_dataset=validation_dataset.select_columns(["anchor", "positive"]),
        loss=train_loss,
        evaluator=None,
    )
    
    try:
        # Check if there's actually a valid checkpoint directory before resuming
        checkpoint_dir = VOL_MOUNT_PATH / "bge-m3-medical"
        checkpoint_subdirs = list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
        
        if restarts > 0 and checkpoint_subdirs:
            # Find the latest checkpoint
            latest_checkpoint = max(checkpoint_subdirs, key=lambda x: int(x.name.split("-")[1]))
            resume_from_checkpoint = str(latest_checkpoint)
            print(f"Resuming from checkpoint: {resume_from_checkpoint} (restart count: {restarts})")
        else:
            resume_from_checkpoint = None
            print(f"Starting fresh training (restart count: {restarts})")
        
        print("Starting training on FULL DATASET...")
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
    
    # Save the best model
    print("Saving final model...")
    trainer.save_model()
    
    # Clear GPU cache and delete trainer to free memory
    del trainer
    torch.cuda.empty_cache()
    print("GPU memory cleared after saving model")
    
    # Evaluate fine-tuned model after training
    print("\n" + "="*50)
    print("FINAL EVALUATION (After Training)")
    print("="*50)
    
    # Load the fine-tuned model
    fine_tuned_model = SentenceTransformer(
        str(VOL_MOUNT_PATH / "bge-m3-medical"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Recreate evaluator for final evaluation only
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
            show_progress_bar=True,
            batch_size=64,                          # Smaller batch size
        )
        matryoshka_evaluators.append(ir_evaluator)
    
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    
    final_results = evaluator(fine_tuned_model)
    print("Fine-tuned Results:")
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {final_results[key]:.4f}")
    
    # Clear GPU cache after final evaluation
    torch.cuda.empty_cache()
    print("GPU memory cleared after final evaluation")
    
    # Show improvement comparison
    print("\n" + "="*50)
    print("IMPROVEMENT SUMMARY")
    print("="*50)
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        baseline = baseline_results[key]
        final = final_results[key]
        improvement = final - baseline
        if baseline > 0:
            improvement_pct = (improvement / baseline) * 100
            print(f"dim_{dim}: {baseline:.4f} → {final:.4f} (+{improvement:.4f}, +{improvement_pct:.1f}%)")
        else:
            print(f"dim_{dim}: baseline skipped → {final:.4f}")
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
    def encode(self, texts: list[str], dimension: int = 768) -> list[list[float]]:
        """Encode texts to embeddings with specified dimension"""
        embeddings = self.model.encode(texts)
        
        # Truncate to specified dimension for Matryoshka representation
        if dimension < embeddings.shape[1]:
            embeddings = embeddings[:, :dimension]
        
        return embeddings.tolist()
    
    @modal.method()
    def similarity(self, query: str, documents: list[str], dimension: int = 768) -> list[float]:
        """Compute similarity between query and documents"""
        from sentence_transformers.util import cos_sim
        
        query_embedding = self.model.encode([query])
        doc_embeddings = self.model.encode(documents)
        
        # Truncate embeddings if needed
        if dimension < query_embedding.shape[1]:
            query_embedding = query_embedding[:, :dimension]
            doc_embeddings = doc_embeddings[:, :dimension]
        
        similarities = cos_sim(query_embedding, doc_embeddings)[0]
        return similarities.tolist()

# ## Evaluation function

@app.function(gpu="A100", volumes={VOL_MOUNT_PATH: output_vol}, timeout=43200)
def evaluate_model():
    """Evaluate the finetuned model using FULL test dataset"""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
    from sentence_transformers.util import cos_sim
    from datasets import Dataset
    
    # Load the finetuned model
    model_path = VOL_MOUNT_PATH / "bge-m3-medical"
    model = SentenceTransformer(
        str(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Loading FULL test data for evaluation...")
    # Load ALL test data from datasets directory
    test_data = load_jsonl_file("/datasets/q_a_test_filtered.jsonl")  # No limiting
    prepared_test = prepare_data_for_training(test_data)
    
    test_dataset = Dataset.from_list(prepared_test)
    
    # Use full corpus for evaluation
    all_data = load_jsonl_file("/datasets/q_a_train_filtered_10k.jsonl") + \
               load_jsonl_file("/datasets/q_a_validation_filtered.jsonl") + \
               test_data
    all_prepared = prepare_data_for_training(all_data)
    all_corpus_dataset = Dataset.from_list(all_prepared)
    
    # Use full corpus (no shrinking)
    corpus = dict(zip(all_corpus_dataset["id"], all_corpus_dataset["positive"]))
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
    relevant_docs = {q_id: [q_id] for q_id in queries}
    
    print(f"Evaluating on {len(queries)} test queries (FULL DATASET)")
    
    # Create evaluators for different dimensions
    matryoshka_dimensions = [1024, 768, 512, 256, 128, 64]
    matryoshka_evaluators = []
    
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
            show_progress_bar=True,
            batch_size=256,                         # Maximized for A100 40GB
        )
        matryoshka_evaluators.append(ir_evaluator)
    
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    
    # Run evaluation
    results = evaluator(model)
    
    # Clear GPU cache after evaluation
    torch.cuda.empty_cache()
    print("GPU memory cleared after evaluation")
    
    # Print results
    print("\nEvaluation Results (FULL DATASET):")
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {results[key]:.4f}")
    
    return results

# ## Local entrypoint

@app.local_entrypoint()
def main():
    """Test the finetuned embedding model with real data"""
    
    # For local entrypoint, we need to access local files directly
    import os
    
    if os.path.exists("datasets/q_a_test_filtered.jsonl"):
        test_data = load_jsonl_file("datasets/q_a_test_filtered.jsonl")[:3]
    else:
        print("No test data found locally. Please ensure datasets/q_a_test_filtered.jsonl exists.")
        return
    
    # Example usage with real data
    embedding_model = EmbeddingModel()
    
    for i, sample in enumerate(test_data):
        query = sample["question"]
        # Use the context as one of the documents along with some variations
        documents = [
            sample["context"],
            f"Thông tin y tế khác không liên quan đến {sample.get('keyword', 'chủ đề')}",
            f"Tài liệu y tế về {sample.get('keyword', 'thuốc')} với nội dung khác"
        ]
        
        # Test similarity computation
        similarities = embedding_model.similarity.remote(query, documents, dimension=512)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Query: {query}")
        print(f"Answer: {sample['answer']}")
        print("\nDocument similarities:")
        for j, (doc, sim) in enumerate(zip(documents, similarities)):
            print(f"{j+1}. Similarity: {sim:.4f}")
            print(f"   Document: {doc[:100]}{'...' if len(doc) > 100 else ''}")
        print()

# ## CLI Usage Examples

# To start finetuning:
# modal run --detach embeddings.py::finetune_embeddings --num-train-epochs=4

# To monitor training with tensorboard:
# Visit: https://ise703--bge-m3-medical-embeddings-monitor-dev.modal.run

# To evaluate the model:
# modal run embeddings.py::evaluate_model

# To test inference with real data:
# modal run embeddings.py

# ## Complete training and evaluation workflow

@app.function(gpu="A100", timeout=43200)
def run_complete_workflow(num_train_epochs: int = 4):
    """Run the complete workflow on FULL dataset: train model and then evaluate it separately"""
    
    print("🚀 Starting complete BGE-M3 finetuning workflow on FULL DATASET...")
    
    # Step 1: Run training on full dataset
    print("\n📊 Step 1: Training the model on FULL DATASET...")
    training_results = finetune_embeddings.remote(num_train_epochs=num_train_epochs)
    
    # Step 2: Run separate evaluation on full dataset
    print("\n📈 Step 2: Running separate evaluation on FULL DATASET...")
    evaluation_results = evaluate_model.remote()
    
    print("\n✅ Complete workflow finished on FULL DATASET!")
    return {
        "training": training_results,
        "evaluation": evaluation_results
    }