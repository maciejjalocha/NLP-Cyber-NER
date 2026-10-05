"""
BERT-NER: Baseline BERT model for Named Entity Recognition

This implementation serves as a baseline model using pre-trained BERT NER
for comparison with LST-NER approach.
"""

import os
import sys
import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support

# Import span F1 evaluation functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_cyber_ner.span_f1 import toSpans, span_f1

# Import deduplication function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_cyber_ner.dataset import remove_leakage


def main():
    try:
        print("Starting BERT-NER baseline script...")
        
        # Set random seed for reproducibility
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        set_seed(args.seed)

        # Configuration
        config = {
            "base_model_name": "roberta-base",  # Pre-trained NER model
            "max_length": 128,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "epochs": 3,
            "output_dir": ("bert_ner_baseline_output_roberta_attackner" if args.seed == 42 else f"bert_ner_baseline_output_roberta_attackner_seed{args.seed}"),
            "target_train_data": "data/interim/attackner/train.txt",
            "target_dev_data": "data/interim/attackner/dev.txt"
        }
        
        print("Configuration set successfully")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA current device:", torch.cuda.current_device())
            print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Load pre-trained NER model
        print("Loading pre-trained BERT NER model (CoNLL-2003)...")
        pretrained_model = AutoModelForTokenClassification.from_pretrained(config["base_model_name"])
        pretrained_model.to(device)
        
        # Get source labels from the pre-trained model
        source_config = pretrained_model.config
        num_source_labels = source_config.num_labels
        source_labels = source_config.id2label if hasattr(source_config, 'id2label') else None
        
        print(f"Pre-trained NER model loaded: {config['base_model_name']}")
        print(f"Number of source labels: {num_source_labels}")
        if source_labels:
            print(f"Source labels: {list(source_labels.values())}")
        
        # Load attackner dataset
        print("Loading attackner datasets...")
        # add_prefix_space=True is required for RoBERTa-family fast tokenizers (e.g. SecureBERT)
        # when used with pretokenized input (is_split_into_words=True); harmless no-op for BERT tokenizers.
        tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], add_prefix_space=True)
        
        # Helper function to convert BIOES to BIO format
        def convert_bioes_to_bio(tag):
            if tag == 'O' or tag == '':
                return tag
            
            # Split into prefix and entity type
            parts = tag.split('-', 1)
            if len(parts) != 2:
                return tag  # Return as is if not in expected format
            
            prefix, entity_type = parts
            
            # Convert BIOES to BIO
            if prefix == 'S':  # Single token entity
                return f"B-{entity_type}"
            elif prefix == 'E':  # End of entity
                return f"I-{entity_type}"
            else:  # B- and I- stay the same
                return tag
        
        # Create NER Dataset class
        class NERDataset(Dataset):
            def __init__(self, file_path, tokenizer, labels_to_id, max_length=128):
                self.texts = []
                self.tags = []
                
                # Read CoNLL-like format file
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_words = []
                    current_tags = []
                    
                    for line in f:
                        line = line.strip()
                        if line == '':
                            # End of sentence
                            if current_words:
                                self.texts.append(current_words)
                                self.tags.append(current_tags)
                                current_words = []
                                current_tags = []
                        else:
                            # Token and tag
                            parts = line.split(' ')
                            if len(parts) >= 2:
                                token, tag = parts[0], parts[1]
                                # Convert BIOES to BIO format
                                bio_tag = convert_bioes_to_bio(tag)
                                current_words.append(token)
                                current_tags.append(bio_tag)
                    
                    # Add the last sentence if not empty
                    if current_words:
                        self.texts.append(current_words)
                        self.tags.append(current_tags)
                
                self.tokenizer = tokenizer
                self.labels_to_id = labels_to_id
                self.max_length = max_length
                print(f"Loaded {len(self.texts)} sentences from {file_path}")
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                words = self.texts[idx]
                tags = self.tags[idx]
                
                # Tokenize the words and align the tags
                encoded_inputs = self.tokenizer(words,
                                              is_split_into_words=True,
                                              max_length=self.max_length,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt')
                
                # Extract features
                input_ids = encoded_inputs['input_ids'].squeeze(0)
                attention_mask = encoded_inputs['attention_mask'].squeeze(0)
                
                # Convert tags to label ids
                labels = torch.ones(self.max_length, dtype=torch.long) * -100 # -100 is ignored in loss
                
                # Align tags with tokenized words
                word_ids = encoded_inputs.word_ids()
                previous_word_idx = None
                label_idx = 0
                
                for i, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx == previous_word_idx:
                        # Skip special tokens and continuation tokens
                        continue
                    
                    if label_idx < len(tags):
                        tag = tags[label_idx]
                        labels[i] = self.labels_to_id.get(tag, 0)  # Default to 'O' (0) if unknown
                        label_idx += 1
                    
                    previous_word_idx = word_idx
                
                # Convert lists to tensors or strings for proper batching
                # Store original text and tags as strings to avoid variable length lists
                text_str = ' '.join(words)
                tags_str = ' '.join(tags)
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'text': text_str,  # String instead of list
                    'tags': tags_str   # String instead of list
                }
        
        # Extract all labels from the attackner dataset and convert to BIO
        print("Extracting target labels from attackner dataset and converting to BIO format...")
        all_labels = set()
        
        # Read training file to extract labels
        with open(config["target_train_data"], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split(' ')) >= 2:
                    _, label = line.split(' ')[:2]
                    # Convert to BIO format for label collection
                    bio_label = convert_bioes_to_bio(label)
                    all_labels.add(bio_label)
        
        # Read dev file to extract additional labels (if any)
        with open(config["target_dev_data"], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split(' ')) >= 2:
                    _, label = line.split(' ')[:2]
                    # Convert to BIO format for label collection
                    bio_label = convert_bioes_to_bio(label)
                    all_labels.add(bio_label)
        
        # Sort labels to ensure consistent ordering, with 'O' as the first label
        target_labels = ['O'] if 'O' in all_labels else []
        sorted_labels = sorted(all_labels - {'O'} if 'O' in all_labels else all_labels)
        target_labels.extend(sorted_labels)
        
        print(f"Extracted {len(target_labels)} unique labels from the dataset")
        print(f"Labels: {target_labels}")
        
        # Create label mappings
        target_id2label = {str(i): label for i, label in enumerate(target_labels)}
        labels_to_id = {label: i for i, label in enumerate(target_labels)}  # Use this name for NERDataset
        
        # Store in config for later use
        config["id2label"] = target_id2label
        
        # Load raw training data
        print("Loading raw data for deduplication...")
        raw_train_data = []

        # Read training data
        with open(config["target_train_data"], 'r', encoding='utf-8') as f:
            current_words = []
            current_tags = []
            for line in f:
                line = line.strip()
                if line == '':
                    if current_words:
                        raw_train_data.append((current_words, current_tags))
                        current_words = []
                        current_tags = []
                else:
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        token, tag = parts[0], parts[1]
                        current_words.append(token)
                        current_tags.append(tag)
            if current_words:
                raw_train_data.append((current_words, current_tags))

        # Union-dedup: drop any train sentence whose tokens appear in ANY of the four
        # datasets' eval sets (not just this dataset's own dev set). Matches the
        # leakage-removal approach in nlp_cyber_ner/modeling/train_hf_ner.py (naim-version
        # branch), so one model per dataset stays comparable across all four eval columns.
        EVAL_PATHS = {
            "attackner": "data/interim/attackner/dev.txt",
            "APTNer": "data/interim/APTNer/APTNERdev.cleaned",
            "DNRTI": "data/interim/DNRTI/valid.cleaned",
            "CYNER": "data/interim/CYNER/valid.txt",
        }

        def load_eval_token_union():
            union = set()
            for name, path in EVAL_PATHS.items():
                if not os.path.exists(path):
                    print(f"Warning: eval path for {name} not found at {path}, skipping in union")
                    continue
                current_words = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line == '':
                            if current_words:
                                union.add(tuple(current_words))
                                current_words = []
                        else:
                            parts = line.split(' ')
                            if parts and parts[0]:
                                current_words.append(parts[0])
                    if current_words:
                        union.add(tuple(current_words))
                print(f"  {name}: eval file loaded from {path}")
            return union

        # Apply deduplication to remove overlapping examples
        print(f"Before deduplication - Train: {len(raw_train_data)} examples")
        eval_union_tokens = load_eval_token_union()
        print(f"Union of eval-set token sequences across all 4 datasets: {len(eval_union_tokens)}")
        union_pseudo_eval = [(list(toks), []) for toks in eval_union_tokens]
        raw_train_data, removed_train_examples = remove_leakage(raw_train_data, union_pseudo_eval)
        print(f"After deduplication - Train: {len(raw_train_data)} examples")
        print(f"Removed {len(removed_train_examples)} overlapping examples from train dataset (union-dedup across 4 eval sets)")

        # Convert the deduplicated data back to temporary files
        deduplicated_train_path = config["target_train_data"] + ".deduplicated"
        with open(deduplicated_train_path, 'w', encoding='utf-8') as f:
            for words, tags in raw_train_data:
                for word, tag in zip(words, tags):
                    bio_tag = convert_bioes_to_bio(tag)
                    f.write(f"{word} {bio_tag}\n")
                f.write("\n")
        
        # Create datasets using deduplicated data
        train_dataset = NERDataset(deduplicated_train_path, tokenizer, labels_to_id, config["max_length"])
        dev_dataset = NERDataset(config["target_dev_data"], tokenizer, labels_to_id, config["max_length"])
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config["batch_size"], shuffle=False)
        print("Dataloaders created")
        
        # Create a new BERT NER model for fine-tuning on target domain
        print("Creating BERT NER model for fine-tuning...")
        
        # Build the classification model directly from the configured backbone (rather
        # than loading a bert-base-cased skeleton and swapping in a RoBERTa backbone
        # afterward). The skeleton-swap approach left the model's weights RoBERTa-shaped
        # (514-row position embeddings) while its *saved config* still said bert-base-cased
        # (512-row) - harmless during a single training run, but it made from_pretrained()
        # crash with an Embedding size mismatch when reloading the checkpoint later. Building
        # directly from base_model_name keeps the saved config and weights consistent.
        model = AutoModelForTokenClassification.from_pretrained(
            config["base_model_name"],
            num_labels=len(target_labels)
        )
        model.to(device)
        
        print(f"Model initialized with {len(target_labels)} target labels")
        
        # Perform a test forward pass
        print("Testing forward pass...")
        batch = next(iter(train_dataloader))
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**batch)
        print(f"Forward pass successful, output shapes: {[k + ': ' + str(v.shape) if isinstance(v, torch.Tensor) else k + ': N/A' for k, v in outputs.items()]}")
        
        # Full training setup
        print("\nStarting full training...")
        
        # Prepare model for training
        optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate total training steps for scheduler
        num_training_steps = len(train_dataloader) * config["epochs"]
        num_warmup_steps = int(num_training_steps * 0.1)  # 10% of total steps for warmup
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training tracking variables
        best_f1 = 0.0
        steps_no_improve = 0
        early_stop_patience = 3  # Stop if no improvement for 3 epochs
        epochs_completed = 0
        
        # Create directory for saving models
        os.makedirs(config["output_dir"], exist_ok=True)
        
        print(f"\nTraining for {config['epochs']} epochs with {len(train_dataloader)} batches per epoch")
        print(f"Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        print(f"Batch size: {config['batch_size']}, Learning rate: {config['learning_rate']}")
        
        # Training loop
        for epoch in range(config["epochs"]):
            print(f"\n{'='*80}\nStarting Epoch {epoch+1}/{config['epochs']}\n{'='*80}")
            model.train()
            
            # Training metrics
            epoch_loss = 0
            num_batches = 0
            
            # Process each batch
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(**batch)
                
                # Get loss
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                current_loss = loss.item()
                epoch_loss += current_loss
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Evaluate on dev set
            print("\nEvaluating on dev set...")
            # Use the id2label mapping from config
            eval_results = evaluate_model(model, dev_dataloader, device, config["id2label"])
            print(f"Dev set results: F1={eval_results['span_f1']:.4f}, Precision={eval_results['span_precision']:.4f}, Recall={eval_results['span_recall']:.4f}")
            
            # Save model if it's the best so far
            current_f1 = eval_results['span_f1']  # Use span-based F1
            if current_f1 > best_f1:
                print(f"New best F1: {current_f1:.4f} (previous: {best_f1:.4f}). Saving model...")
                best_f1 = current_f1
                steps_no_improve = 0
                
                # Save the best model
                best_model_path = os.path.join(config["output_dir"], "best_model")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                
                # Save training info
                with open(os.path.join(config["output_dir"], "training_info.json"), "w") as f:
                    json.dump({
                        "best_f1": best_f1,
                        "best_epoch": epoch + 1,
                        "span_precision": eval_results['span_precision'],
                        "span_recall": eval_results['span_recall'],
                        "token_precision": eval_results['token_precision'],
                        "token_recall": eval_results['token_recall'],
                        "token_accuracy": eval_results['token_accuracy'],
                        "config": config
                    }, f, indent=2)
            else:
                steps_no_improve += 1
                print(f"No improvement for {steps_no_improve} epochs. Best F1: {best_f1:.4f}")
            
            # Early stopping check
            if steps_no_improve >= early_stop_patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
            
            epochs_completed = epoch + 1
        
        print(f"\nTraining completed after {epochs_completed} epochs.")
        print(f"Best F1 score: {best_f1:.4f}")
        
        # Load the best model for final evaluation. Use from_pretrained rather than
        # hunting for a specific weights filename (pytorch_model.bin vs model.safetensors)
        # so this works regardless of which format save_pretrained used.
        print("\nLoading best model for final evaluation...")
        best_model_path = os.path.join(config["output_dir"], "best_model")
        if os.path.isdir(best_model_path) and os.listdir(best_model_path):
            model = AutoModelForTokenClassification.from_pretrained(best_model_path)
            model.to(device)

            # Final evaluation
            print("\nFinal evaluation on dev set:")
            final_eval_results = evaluate_model(model, dev_dataloader, device, target_id2label, save_predictions_path=os.path.join(config["output_dir"], "predictions_dev.json"))
            print(f"Final evaluation results: Span F1={final_eval_results['span_f1']:.4f}, Precision={final_eval_results['span_precision']:.4f}, Recall={final_eval_results['span_recall']:.4f}")
        else:
            print("Best model not found. Using last model for final evaluation.")
            final_eval_results = eval_results
            
        print("\nTraining and evaluation complete!")
        
        print("\nScript execution complete")
        
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nScript execution failed with error")


# Set random seed for reproducibility
def set_seed(seed=42):
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(model, eval_dataloader, device, id2label=None, save_predictions_path=None):
    """
    Evaluate the model on a validation set using span-based F1 metrics
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        id2label: Dictionary mapping from label ID (as string) to label name
    """
    # Create a fallback id2label mapping if none is provided
    if id2label is None:
        print("Warning: No id2label dictionary provided, using default mapping")
        id2label = {
            "0": "O", 
            "1": "B-Entity", "2": "I-Entity",
            "3": "B-Other", "4": "I-Other"
        }
    print("Evaluating model...")
    model.eval()
    
    # For token-level metrics
    all_token_preds = []
    all_token_labels = []
    
    # For span-based metrics
    all_pred_tags = []
    all_gold_tags = []
    current_pred_tags = []
    current_gold_tags = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Get predictions
            preds = torch.argmax(logits, dim=2)
            
            # Collect predictions and labels, excluding padded tokens (-100)
            labels = batch['labels']
            
            for i in range(labels.shape[0]):  # Iterate over batch
                current_pred_tags = []
                current_gold_tags = []
                
                for j in range(labels.shape[1]):  # Iterate over sequence
                    if labels[i, j] != -100:  # Exclude padding tokens
                        all_token_preds.append(preds[i, j].item())
                        all_token_labels.append(labels[i, j].item())
                        
                        # Convert numeric labels to BIO tags for span evaluation
                        try:
                            if id2label:
                                pred_tag = id2label[str(preds[i, j].item())]
                                gold_tag = id2label[str(labels[i, j].item())]
                            else:
                                # Default conversion if id2label not provided
                                if preds[i, j].item() == 0:
                                    pred_tag = 'O'
                                else:
                                    tag_type = 'B' if preds[i, j].item() % 2 == 1 else 'I'
                                    tag_class = f"-{(preds[i, j].item() + 1) // 2}"
                                    pred_tag = tag_type + tag_class
                                    
                                if labels[i, j].item() == 0:
                                    gold_tag = 'O'
                                else:
                                    tag_type = 'B' if labels[i, j].item() % 2 == 1 else 'I'
                                    tag_class = f"-{(labels[i, j].item() + 1) // 2}"
                                    gold_tag = tag_type + tag_class
                        except Exception as e:
                            print(f"Error converting labels to tags: {e}")
                            # Fallback to simple O tag if there's an error
                            pred_tag = 'O'
                            gold_tag = 'O'
                        
                        current_pred_tags.append(pred_tag)
                        current_gold_tags.append(gold_tag)
                    else:
                        # End of sentence
                        if current_pred_tags and current_gold_tags:
                            break
                            
                if current_pred_tags and current_gold_tags:
                    all_pred_tags.append(current_pred_tags)
                    all_gold_tags.append(current_gold_tags)
    
    # Calculate token-level metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_token_labels, all_token_preds, average='weighted', zero_division=1
    )
    
    # Calculate accuracy
    accuracy = (np.array(all_token_preds) == np.array(all_token_labels)).mean()
    
    # Calculate span-based metrics
    span_metrics = span_f1(all_gold_tags, all_pred_tags)

    # Optionally dump per-example gold/pred tags for later significance testing
    # (paired bootstrap comparison against another model's predictions on the same dev set)
    if save_predictions_path:
        with open(save_predictions_path, 'w', encoding='utf-8') as f_pred:
            json.dump({'gold': all_gold_tags, 'pred': all_pred_tags}, f_pred)
        print(f"Saved {len(all_gold_tags)} per-example predictions to {save_predictions_path}")
    
    # Print both metrics for comparison
    print(f"Token-level metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}")
    print(f"Span-based metrics: F1={span_metrics['slot-f1']:.4f}, Precision={span_metrics['precision']:.4f}, Recall={span_metrics['recall']:.4f}")
    
    return {
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1,
        'token_accuracy': accuracy,
        'span_precision': span_metrics['precision'],
        'span_recall': span_metrics['recall'],
        'span_f1': span_metrics['slot-f1'],
        'f1': span_metrics['slot-f1']  # Use span F1 as the primary metric
    }


if __name__ == "__main__":
    main()
