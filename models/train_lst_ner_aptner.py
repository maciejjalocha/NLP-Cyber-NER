"""
LST-NER: Label Structure Transfer for Named Entity Recognition

A simplified implementation based on the paper:
"Cross-domain Named Entity Recognition via Graph Matching"
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
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
# Import span F1 evaluation functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_cyber_ner.span_f1 import toSpans, span_f1
# Import POT for Gromov-Wasserstein calculations
try:
    import ot
except ImportError:
    print("Warning: POT (Python Optimal Transport) package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "POT"])
    import ot

# Import deduplication function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp_cyber_ner.dataset import remove_leakage

def main():
    try:
        print("Starting LST-NER script...")
        
        # Set random seed for reproducibility
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        set_seed(args.seed)

        # Configuration
        config = {
            "base_model_name": "roberta-base",
            "max_length": 128,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "epochs": 3,
            "output_dir": ("lst_ner_output_roberta_aptner" if args.seed == 42 else f"lst_ner_output_roberta_aptner_seed{args.seed}"),
            "temp": 4.0,  # Temperature for distribution smoothing
            "edge_threshold": 1.5,  # Threshold for adding edges in the label graph
            "gwd_lambda": 0.01,  # Weight for GWD loss
            "target_train_data": "data/interim/APTNer/APTNERtrain.cleaned",
            "target_dev_data": "data/interim/APTNer/APTNERdev.cleaned"
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
        
        # Load real datasets and pre-trained NER model
        print("Loading datasets and pre-trained NER model...")
        
        # Load pre-trained NER model from Hugging Face (CoNLL-2003 English)
        print("Loading pre-trained NER model...")
        source_model_name = "dslim/bert-base-NER"  # Trained on CoNLL-2003 (English)
        source_model = AutoModelForTokenClassification.from_pretrained(source_model_name)
        source_model.to(device)
        
        # Get source labels from the config
        source_config = source_model.config
        num_source_labels = source_config.num_labels
        source_labels = source_config.id2label if hasattr(source_config, 'id2label') else None
        
        print(f"Pre-trained NER model loaded: {source_model_name}")
        print(f"Number of source labels: {num_source_labels}")
        if source_labels:
            print(f"Source labels: {list(source_labels.values())}")
        
        # Load real APTNer dataset
        print("Loading APTNer datasets...")
        
        # add_prefix_space=True is required for RoBERTa-family fast tokenizers (e.g. SecureBERT)
        # when used with pretokenized input (is_split_into_words=True); harmless no-op for BERT tokenizers.
        tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], add_prefix_space=True)

        # Check tokenizer compatibility with original BERT
        print("Checking tokenizer compatibility...")
        tokenizer_original = AutoTokenizer.from_pretrained("bert-base-cased")
        print("Vocab sizes match:", len(tokenizer_original.vocab) == len(tokenizer.vocab))
        print("Special tokens match:", tokenizer_original.special_tokens_map == tokenizer.special_tokens_map)
        print("Tokenizer check complete.\n")
        
        
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
                                # APTNer keeps its native BIOES tags (no BIO collapse)
                                current_words.append(token)
                                current_tags.append(tag)
                    
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
        
        # Extract all labels from the APTNer dataset. APTNer's raw tags are natively
        # BIOES (B-/I-/E-/S-); keep them as-is instead of collapsing to BIO, since this
        # dataset trains its own separate classifier (no shared label space with the
        # other three datasets) and comparability to the original APTNER benchmark
        # requires the native BIOES scheme.
        print("Extracting target labels from APTNer dataset (native BIOES scheme)...")
        all_labels = set()

        # Read training file to extract labels
        with open(config["target_train_data"], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split(' ')) >= 2:
                    _, label = line.split(' ')[:2]
                    all_labels.add(label)

        # Read dev file to extract additional labels (if any)
        with open(config["target_dev_data"], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split(' ')) >= 2:
                    _, label = line.split(' ')[:2]
                    all_labels.add(label)
        
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

        # Convert the deduplicated data back to temporary files. APTNer keeps its native
        # BIOES tags here (no convert_bioes_to_bio call) to match the label vocab above.
        deduplicated_train_path = config["target_train_data"] + ".deduplicated"
        with open(deduplicated_train_path, 'w', encoding='utf-8') as f:
            for words, tags in raw_train_data:
                for word, tag in zip(words, tags):
                    f.write(f"{word} {tag}\n")
                f.write("\n")
        
        # Create datasets using deduplicated data
        train_dataset = NERDataset(deduplicated_train_path, tokenizer, labels_to_id, config["max_length"])
        dev_dataset = NERDataset(config["target_dev_data"], tokenizer, labels_to_id, config["max_length"])
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config["batch_size"], shuffle=False)
        print("Dataloaders created")
        
        # Estimate label distributions
        print("Estimating label distributions...")
        num_target_labels = len(target_labels)
        label_ids = list(range(num_target_labels))
        # The source model (dslim/bert-base-NER) has its own vocabulary, which may not
        # match the target backbone's tokenizer (e.g. SecureBERT/RoBERTa). Re-tokenize the
        # training data with the source model's own tokenizer for this pass, so input_ids
        # stay within its embedding table's range instead of the target tokenizer's vocab space.
        source_tokenizer = AutoTokenizer.from_pretrained(source_model_name)
        source_train_dataset = NERDataset(deduplicated_train_path, source_tokenizer, labels_to_id, config["max_length"])

        prob_distributions = estimate_label_distributions(
            source_model,
            source_train_dataset,
            label_ids,
            num_source_labels,
            device,
            temp=config["temp"]
        )
        print(f"Created label distributions with shape: {prob_distributions.shape}")
        
        # Construct source graph
        print("Constructing source graph...")
        source_graph = construct_label_graph(
            prob_distributions,
            edge_threshold=config["edge_threshold"]
        )
        print(f"Source graph constructed with shape: {source_graph.shape}")
        
        # Create LST-NER model with GWD
        print("Creating LST-NER model with GWD...")
        lst_ner_model = LSTNER(
            base_model_name=config["base_model_name"],
            target_labels=target_labels,
            temp=config["temp"],
            edge_threshold=config["edge_threshold"],
            gwd_lambda=config["gwd_lambda"]
        )
        
        # Set source graph
        print("Setting source graph...")
        lst_ner_model.set_source_graph(source_graph)
        
        # Perform a test forward pass
        print("Testing forward pass...")
        lst_ner_model.to(device)
        batch = next(iter(train_dataloader))
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = lst_ner_model(**batch)
        print(f"Forward pass successful, output shapes: {[k + ': ' + str(v.shape) if isinstance(v, torch.Tensor) else k + ': N/A' for k, v in outputs.items()]}")
        
        # Full training setup
        print("\nStarting full training...")
        
        # Prepare model for training
        optimizer = AdamW(
            lst_ner_model.parameters(),
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
        print(f"GWD lambda: {config['gwd_lambda']}, Edge threshold: {config['edge_threshold']}")
        
        # Training loop
        for epoch in range(config["epochs"]):
            print(f"\n{'='*80}\nStarting Epoch {epoch+1}/{config['epochs']}\n{'='*80}")
            lst_ner_model.train()
            
            # Training metrics
            epoch_loss = 0
            epoch_gwd_loss = 0
            num_batches = 0
            
            # Process each batch
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = lst_ner_model(**batch)
                
                # Get loss components
                loss = outputs['loss']
                gwd_loss = outputs['gwd_loss']
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(lst_ner_model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                current_loss = loss.item()
                epoch_loss += current_loss
                num_batches += 1
                
                if gwd_loss is not None:
                    current_gwd = gwd_loss.item()
                    epoch_gwd_loss += current_gwd
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}", 
                        'gwd_loss': f"{current_gwd:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
                else:
                    # Update progress bar without GWD loss
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            avg_gwd_loss = epoch_gwd_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}, Average GWD loss: {avg_gwd_loss:.4f}")
            
            # Evaluate on dev set
            print("\nEvaluating on dev set...")
            # Use the id2label mapping from config
            eval_results = evaluate_model(lst_ner_model, dev_dataloader, device, config["id2label"])
            print(f"Dev set results: F1={eval_results['span_f1']:.4f}, Precision={eval_results['span_precision']:.4f}, Recall={eval_results['span_recall']:.4f}")
            
            # Save model if it's the best so far
            current_f1 = eval_results['span_f1']  # Use span-based F1
            if current_f1 > best_f1:
                print(f"New best F1: {current_f1:.4f} (previous: {best_f1:.4f}). Saving model...")
                best_f1 = current_f1
                steps_no_improve = 0
                
                # Save the best model
                best_model_path = os.path.join(config["output_dir"], "best_model")
                lst_ner_model.save_pretrained(best_model_path)
                
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
        
        # Load the best model for final evaluation
        print("\nLoading best model for final evaluation...")
        best_model_path = os.path.join(config["output_dir"], "best_model")
        if os.path.exists(os.path.join(best_model_path, "pytorch_model.bin")):
            # Load the best model state dict
            best_state_dict = torch.load(os.path.join(best_model_path, "pytorch_model.bin"))
            lst_ner_model.load_state_dict(best_state_dict)
            
            # Final evaluation
            print("\nFinal evaluation on dev set:")
            final_eval_results = evaluate_model(lst_ner_model, dev_dataloader, device, target_id2label, save_predictions_path=os.path.join(config["output_dir"], "predictions_dev.json"))
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

# Function to estimate label distributions using source model
def estimate_label_distributions(source_model, dataset, label_ids, num_labels, device, temp=4.0):
    """
    Estimate label probability distributions using a source model
    Returns distribution matrix of shape [num_target_labels, num_source_labels]
    """
    print("Estimating label distributions from source model...")
    source_model.eval()
    
    # Initialize probability distributions
    prob_distributions = torch.zeros(len(label_ids), num_labels)
    sample_counts = torch.zeros(len(label_ids))
    
    # Create a label_id to index mapping
    label_id_to_idx = {label_id: idx for idx, label_id in enumerate(label_ids)}
    
    with torch.no_grad():
        # Process each example in dataset
        for i, sample in enumerate(dataset):
            # Get input tensors
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            labels = sample['labels']
            
            # Get model outputs
            outputs = source_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0).cpu()  # [seq_len, num_labels]
            
            # Apply temperature scaling
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)  # [seq_len, num_labels]
            
            # For each token with a valid label, accumulate distribution
            for j, label in enumerate(labels):
                if label.item() != -100 and label.item() in label_id_to_idx:
                    idx = label_id_to_idx[label.item()]
                    prob_distributions[idx] += probs[j]
                    sample_counts[idx] += 1
    
    # Average the probability distributions
    for i in range(len(label_ids)):
        if sample_counts[i] > 0:
            prob_distributions[i] = prob_distributions[i] / sample_counts[i]
        else:
            # For labels with no samples, use uniform distribution
            prob_distributions[i] = torch.ones(num_labels) / num_labels
    
    print(f"Label distribution estimation complete for {len(label_ids)} labels")
    return prob_distributions

# Function to construct label graph from probability distributions
def construct_label_graph(prob_distributions, edge_threshold=0.5):
    """
    Construct a label graph using probability distributions
    Returns adjacency matrix of shape [num_nodes, num_nodes]
    """
    print(f"Constructing label graph with threshold {edge_threshold}...")
    num_nodes = prob_distributions.shape[0]
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    # Normalize node representations
    norm_factor = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = F.pairwise_distance(
                    prob_distributions[i].unsqueeze(0), 
                    prob_distributions[j].unsqueeze(0), 
                    p=2
                ).item()
                norm_factor += distance
    
    if norm_factor > 0:
        norm_factor = norm_factor / (num_nodes * (num_nodes - 1))
    else:
        norm_factor = 1.0
    
    # Calculate similarities and construct graph
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                # Self-loop
                adj_matrix[i, j] = 1.0
            else:
                # Calculate normalized distance
                distance = F.pairwise_distance(
                    prob_distributions[i].unsqueeze(0), 
                    prob_distributions[j].unsqueeze(0), 
                    p=2
                ).item() / norm_factor
                
                # Add edge if distance is below threshold
                if distance < edge_threshold:
                    similarity = 1.0 - distance / edge_threshold
                    adj_matrix[i, j] = similarity
    
    print(f"Label graph constructed with shape: {adj_matrix.shape}")
    return adj_matrix

# Graph Convolutional Network layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        print(f"Initializing GraphConvolution layer with in_features={in_features}, out_features={out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
        """
        input: node features [batch_size, num_nodes, in_features]
        adj: adjacency matrix [num_nodes, num_nodes]
        """
        # Normalize adjacency matrix
        adj = self._normalize_adj(adj)
        
        # Message passing
        support = torch.matmul(input, self.weight)  # [batch_size, num_nodes, out_features]
        output = torch.matmul(adj, support)  # [batch_size, num_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _normalize_adj(self, adj):
        """
        Symmetrically normalize adjacency matrix
        """
        # Replace inf with 0
        adj = torch.where(adj == float('inf'), torch.zeros_like(adj), adj)
        
        # Add self-loops
        identity = torch.eye(adj.size(0), device=adj.device)
        adj = adj + identity
        
        # Calculate degree matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Normalize
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

# LST-NER model with target graph and GWD
class LSTNER(nn.Module):
    def __init__(self, base_model_name, target_labels, temp=4.0, edge_threshold=0.5, gwd_lambda=0.01):
        super(LSTNER, self).__init__()
        print(f"Initializing LSTNER with {len(target_labels)} labels and edge_threshold={edge_threshold}")
        # BERT backbone (swappable via base_model_name, e.g. SecureBERT/RoBERTa are RoBERTa-based)
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.dp_dim = self.hidden_dim  # Projection dimension
        
        # Target labels
        self.target_labels = target_labels
        self.num_target_labels = len(target_labels)
        
        # Label semantic fusion layers
        self.Wp = nn.Linear(self.hidden_dim, self.dp_dim)
        self.bp = nn.Parameter(torch.zeros(self.dp_dim))
        
        # Label representation (randomly initialized)
        self.label_embeddings = nn.Parameter(torch.randn(self.num_target_labels, self.dp_dim))
        
        # GCN for label graph structure
        self.gcn = GraphConvolution(self.dp_dim, self.dp_dim)
        
        # Projection layers for label fusion
        self.W_prime = nn.Linear(self.dp_dim, self.hidden_dim)
        self.b_prime = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Task-specific classifiers
        self.classifier = nn.Linear(self.hidden_dim, self.num_target_labels)
        self.aux_classifier = nn.Linear(self.hidden_dim, self.num_target_labels) 
        
        # Source and target graphs
        self.source_graph = None
        self.target_graph = None
        self.target_distributions = None
        
        # Tracking target distribution accumulation
        self.target_dist_accum = torch.zeros(self.num_target_labels, self.num_target_labels)
        self.target_dist_count = torch.zeros(self.num_target_labels)
        
        # Hyperparameters
        self.temp = temp
        self.edge_threshold = edge_threshold
        self.gwd_lambda = gwd_lambda  # Weight for GWD loss
        
    def set_source_graph(self, source_graph):
        """
        Set the source graph for label structure transfer
        """
        self.source_graph = source_graph
    
    def update_target_distributions(self, logits, labels, update_graph=False):
        """
        Update target label distributions based on model predictions
        
        Args:
            logits: Model predictions [batch_size, seq_len, num_target_labels]
            labels: Ground truth labels [batch_size, seq_len]
            update_graph: Whether to update the target graph after updating distributions
        """
        batch_size, seq_len, _ = logits.shape
        device = logits.device
        
        # Apply temperature scaling to logits
        scaled_logits = logits / self.temp
        probs = F.softmax(scaled_logits, dim=-1)  # [batch_size, seq_len, num_target_labels]
        
        # Accumulate probability distributions for each label
        for b in range(batch_size):
            for s in range(seq_len):
                label = labels[b, s].item()
                if label != -100 and 0 <= label < self.num_target_labels:
                    self.target_dist_accum[label] += probs[b, s].detach().cpu()
                    self.target_dist_count[label] += 1
        
        # If we have enough samples and update_graph is True, update target graph
        if update_graph and torch.any(self.target_dist_count > 0):
            self.compute_target_graph()
    
    def compute_target_graph(self):
        """
        Compute target graph from accumulated distributions
        """
        # Average the accumulated distributions
        target_distributions = torch.zeros_like(self.target_dist_accum)
        for i in range(self.num_target_labels):
            if self.target_dist_count[i] > 0:
                target_distributions[i] = self.target_dist_accum[i] / self.target_dist_count[i]
            else:
                # For labels with no samples, use uniform distribution
                target_distributions[i] = torch.ones(self.num_target_labels) / self.num_target_labels
        
        # Store the distributions
        self.target_distributions = target_distributions
        
        # Construct the target graph
        print("Constructing target graph...")
        self.target_graph = self.construct_graph_from_distributions(target_distributions)
        print(f"Target graph constructed with shape: {self.target_graph.shape}")
    
    def construct_graph_from_distributions(self, distributions):
        """
        Construct a graph from label distributions - similar to construct_label_graph function
        but implemented as a method for internal use
        """
        num_nodes = distributions.shape[0]
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        
        # Normalize node representations
        norm_factor = 0.0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distance = F.pairwise_distance(
                        distributions[i].unsqueeze(0), 
                        distributions[j].unsqueeze(0), 
                        p=2
                    ).item()
                    norm_factor += distance
        
        if norm_factor > 0:
            norm_factor = norm_factor / (num_nodes * (num_nodes - 1))
        else:
            norm_factor = 1.0
        
        # Calculate similarities and construct graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    # Self-loop
                    adj_matrix[i, j] = 1.0
                else:
                    # Calculate normalized distance
                    distance = F.pairwise_distance(
                        distributions[i].unsqueeze(0), 
                        distributions[j].unsqueeze(0), 
                        p=2
                    ).item() / norm_factor
                    
                    # Add edge if distance is below threshold
                    if distance < self.edge_threshold:
                        similarity = 1.0 - distance / self.edge_threshold
                        adj_matrix[i, j] = similarity
        
        return adj_matrix
    
    def compute_gwd(self, source_graph, target_graph):
        """
        Compute the Gromov-Wasserstein Distance between source and target graphs
        
        Returns:
            gwd_loss: The GWD loss value
        """
        # Convert PyTorch tensors to NumPy for POT library
        source_adj = source_graph.cpu().numpy()
        target_adj = target_graph.cpu().numpy()
        
        # Create uniform distributions for graph nodes
        p = np.ones(source_adj.shape[0]) / source_adj.shape[0]
        q = np.ones(target_adj.shape[0]) / target_adj.shape[0]
        
        # Compute the Gromov-Wasserstein distance
        # Note: We use the squared Euclidean loss for structure matching
        gwd = ot.gromov.gromov_wasserstein2(
            source_adj, target_adj, p, q, 'square_loss', verbose=False
        )
        
        return torch.tensor(gwd, device=source_graph.device)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass with label-guided attention and GCN
        """
        # Get BERT contextual embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        batch_size, seq_len, _ = token_embeds.shape
        
        # Check if source graph exists
        if self.source_graph is None:
            print("Warning: Source graph not set, using identity matrix")
            self.source_graph = torch.eye(self.num_target_labels)
        
        # Ensure source graph is on the correct device
        source_graph = self.source_graph.to(input_ids.device)
        
        # Label-guided attention for extracting label-specific components
        q = self.Wp(token_embeds) + self.bp  # [batch_size, seq_len, dp_dim]
        
        # Calculate label-specific components for each sentence
        label_specific_comps = torch.zeros(batch_size, self.num_target_labels, self.dp_dim, device=input_ids.device)
        
        for i in range(batch_size):
            # Calculate attention weights between tokens and label embeddings
            alpha = F.softmax(torch.matmul(q[i], self.label_embeddings.transpose(0, 1)), dim=0)  # [seq_len, num_target_labels]
            
            # Extract label-specific components
            for l in range(self.num_target_labels):
                label_specific_comps[i, l] = torch.matmul(alpha[:, l], q[i])  # [dp_dim]
        
        # Apply GCN to enhance label-specific components with graph structure
        enhanced_label_comps = self.gcn(label_specific_comps, source_graph)
        
        # Token-guided attention to fuse label-specific components into token embeddings
        enhanced_token_embeds = token_embeds.clone()
        
        for i in range(batch_size):
            for j in range(seq_len):
                # Calculate attention weights
                beta = F.softmax(torch.matmul(q[i, j], enhanced_label_comps[i].transpose(0, 1)), dim=0)  # [num_target_labels]
                # Fuse label-specific components
                weighted_sum = torch.matmul(beta, enhanced_label_comps[i])  # [dp_dim]
                enhanced_token_embeds[i, j] += self.W_prime(weighted_sum) + self.b_prime
        
        # Classification for NER
        logits = self.classifier(enhanced_token_embeds)  # [batch_size, seq_len, num_target_labels]
        
        # Auxiliary task for entity type detection (sentence level)
        # Use average pooling for sequence representation
        avg_token_embeds = torch.mean(enhanced_token_embeds, dim=1)  # [batch_size, hidden_dim]
        aux_logits = self.aux_classifier(avg_token_embeds)  # [batch_size, num_target_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # NER token classification loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            token_loss = loss_fct(logits.view(-1, self.num_target_labels), labels.view(-1))
            
            # Auxiliary loss (entity type detection)
            # Since we don't have sentence-level entity type labels in this implementation,
            # we'll create approximate labels: a label is present if any token has that label
            aux_labels = torch.zeros(batch_size, self.num_target_labels, device=labels.device)
            for i in range(batch_size):
                for j in range(seq_len):
                    if labels[i, j] >= 0 and labels[i, j] < self.num_target_labels:
                        aux_labels[i, labels[i, j]] = 1.0
            
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_labels)
            
            # Update target distributions based on current predictions
            self.update_target_distributions(logits, labels, update_graph=True)
            
            # GWD loss (if both graphs are available)
            gwd_loss = torch.tensor(0.0, device=labels.device)
            if self.source_graph is not None and self.target_graph is not None:
                # Move target graph to current device
                target_graph = self.target_graph.to(labels.device)
                gwd_loss = self.compute_gwd(source_graph, target_graph)
                print(f"GWD Loss: {gwd_loss.item():.4f}")
            
            # Combine losses
            loss = token_loss + 0.1 * aux_loss  # Base loss
            
            # Add GWD loss if available
            if gwd_loss > 0:
                loss = loss + self.gwd_lambda * gwd_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'aux_logits': aux_logits,
            'enhanced_embeds': enhanced_token_embeds,
            'gwd_loss': gwd_loss if 'gwd_loss' in locals() else None
        }
    
    def save_pretrained(self, output_path):
        """
        Save model to disk
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save config
        config = {
            "hidden_dim": self.hidden_dim,
            "dp_dim": self.dp_dim,
            "edge_threshold": self.edge_threshold,
            "temp": self.temp,
            "gwd_lambda": self.gwd_lambda,
            "num_target_labels": self.num_target_labels,
            "target_labels": self.target_labels
        }
        
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

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
            logits = outputs['logits']
            
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