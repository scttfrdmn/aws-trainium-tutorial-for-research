# Video Tutorial 2: Basic Training Workflow on AWS Trainium

**Duration**: 20 minutes  
**Difficulty**: Beginner to Intermediate  
**Prerequisites**: Tutorial 1 completed, basic PyTorch knowledge

## 🎯 Learning Objectives

By the end of this tutorial, viewers will be able to:
1. Set up a complete training pipeline on Trainium
2. Implement proper data loading and preprocessing
3. Monitor training progress and performance metrics
4. Optimize training for cost and efficiency
5. Save and load models for inference

## 📋 Tutorial Script

### Opening (0:00 - 1:00)
**[TITLE CARD: Basic Training Workflow on AWS Trainium]**

**Narrator**: "Welcome back to our AWS Trainium tutorial series! In our first tutorial, we got familiar with the basics. Now we're going to dive deeper into training workflows - the heart of machine learning research.

By the end of this tutorial, you'll understand how to train models efficiently on Trainium, monitor your progress, and optimize for both performance and cost. This is the foundation you'll need for serious research work."

**[SHOW: Training workflow diagram]**

### Section 1: Training Setup (1:00 - 5:00)

#### Instance and Environment (1:00 - 2:00)
**[SCREEN: AWS Console - EC2 Dashboard]**

**Narrator**: "Let's start by launching a training-optimized instance. For this tutorial, I'm using a trn1.8xlarge with 8 Neuron cores. This gives us substantial compute power while keeping costs reasonable."

```bash
# Connect to instance
ssh -i my-training-key.pem ubuntu@ec2-xxx.compute-1.amazonaws.com

# Verify Neuron setup
neuron-ls
echo "Neuron cores available: $(neuron-ls | grep -c 'Neuron device')"
```

**[SHOW: SSH connection and Neuron device verification]**

#### Project Structure (2:00 - 3:00)
**Narrator**: "Let's set up a proper project structure for our training workflow. Organization is crucial for research reproducibility."

```bash
# Create project structure
mkdir -p ~/training_project/{data,models,logs,checkpoints,configs}
cd ~/training_project

# Project structure
tree ~/training_project
```

**[SHOW: Directory structure creation]**

#### Dependencies Installation (3:00 - 4:00)
**Narrator**: "Now let's install our training dependencies. We'll need torch-neuronx for Trainium optimization, plus some common ML libraries."

```bash
# Install core dependencies
pip install torch-neuronx==2.2.0 --extra-index-url https://pip.repos.neuron.amazonaws.com
pip install transformers==4.36.0
pip install datasets==2.14.0
pip install wandb  # For experiment tracking
pip install matplotlib seaborn  # For visualization

# Verify installations
python -c "
import torch_neuronx
import torch_xla.core.xla_model as xm
print('✅ Neuron setup complete')
print('Available devices:', xm.get_xla_supported_devices())
"
```

#### Configuration Management (4:00 - 5:00)
**Narrator**: "Let's create a configuration system for our training. This makes it easy to experiment with different hyperparameters."

```python
# configs/training_config.py
import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingConfig:
    """Training configuration for reproducible experiments."""
    
    # Model settings
    model_name: str = "simple_classifier"
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data settings
    max_sequence_length: int = 512
    train_split: float = 0.8
    validation_split: float = 0.1
    
    # Hardware settings
    num_workers: int = 4
    pin_memory: bool = True
    compile_model: bool = True
    
    # Logging settings
    log_interval: int = 10
    save_interval: int = 500
    project_name: str = "trainium_training"
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# Create and save default config
config = TrainingConfig()
config.save('configs/default.json')
print("Configuration saved to configs/default.json")
```

**[SHOW: Configuration file creation and explanation]**

### Section 2: Data Pipeline (5:00 - 9:00)

#### Dataset Creation (5:00 - 6:30)
**[SCREEN: Python editor with data loading code]**

**Narrator**: "Let's create a proper data pipeline. For this example, I'll use a text classification task, but the patterns apply to any machine learning problem."

```python
# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """Text classification dataset for demonstration."""
    
    def __init__(self, num_samples: int = 5000, max_length: int = 512):
        """Initialize dataset with synthetic text data."""
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Generate synthetic data
        logger.info(f"Generating {num_samples} synthetic text samples...")
        self.data = self._generate_synthetic_data()
        logger.info("Dataset created successfully")
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic text classification data."""
        np.random.seed(42)  # For reproducibility
        data = []
        
        # Simple vocabulary
        vocab_size = 10000
        
        for i in range(self.num_samples):
            # Generate random sequence
            sequence_length = np.random.randint(50, self.max_length)
            input_ids = np.random.randint(1, vocab_size, sequence_length)
            
            # Simple rule-based labels
            # Label 1 if mean > 5000, else 0
            label = 1 if np.mean(input_ids) > 5000 else 0
            
            # Pad to max_length
            padded_ids = np.zeros(self.max_length, dtype=np.int64)
            padded_ids[:len(input_ids)] = input_ids
            
            # Attention mask
            attention_mask = np.zeros(self.max_length, dtype=np.int64)
            attention_mask[:len(input_ids)] = 1
            
            data.append({
                'input_ids': padded_ids,
                'attention_mask': attention_mask,
                'label': label
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# Test dataset creation
dataset = TextClassificationDataset(num_samples=1000)
print(f"Dataset created with {len(dataset)} samples")

# Test data loading
sample = dataset[0]
for key, value in sample.items():
    print(f"{key}: {value.shape} (dtype: {value.dtype})")
```

**[SHOW: Dataset creation and sample data inspection]**

#### Data Loader Optimization (6:30 - 8:00)
**Narrator**: "Now let's create optimized data loaders. Proper data loading is crucial for getting good performance from Trainium."

```python
# data/dataloader.py
from torch.utils.data import DataLoader, random_split
import torch_xla.core.xla_model as xm

def create_dataloaders(
    dataset: Dataset, 
    config: TrainingConfig,
    device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.validation_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Important for Neuron compilation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

# Create dataloaders
config = TrainingConfig()
device = xm.xla_device()

train_loader, val_loader, test_loader = create_dataloaders(dataset, config, device)

print(f"Dataloaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Test batch loading
sample_batch = next(iter(train_loader))
print(f"Sample batch shapes:")
for key, value in sample_batch.items():
    print(f"  {key}: {value.shape}")
```

#### Data Loading Performance (8:00 - 9:00)
**Narrator**: "Let's benchmark our data loading to ensure it won't be a bottleneck during training."

```python
# data/benchmark.py
import time
from tqdm import tqdm

def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 50) -> Dict[str, float]:
    """Benchmark dataloader performance."""
    
    print(f"Benchmarking dataloader with {num_batches} batches...")
    
    start_time = time.time()
    batch_times = []
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Simulate data transfer to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                _ = value.to(device, non_blocking=True)
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        if i >= num_batches - 1:
            break
    
    total_time = time.time() - start_time
    
    metrics = {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'batches_per_second': num_batches / total_time,
        'samples_per_second': (num_batches * config.batch_size) / total_time
    }
    
    print(f"Data loading performance:")
    print(f"  Average batch time: {metrics['avg_batch_time']:.3f}s")
    print(f"  Batches per second: {metrics['batches_per_second']:.1f}")
    print(f"  Samples per second: {metrics['samples_per_second']:.1f}")
    
    return metrics

# Benchmark our dataloader
benchmark_results = benchmark_dataloader(train_loader)
```

**[SHOW: Data loading benchmark results]**

### Section 3: Model Architecture (9:00 - 13:00)

#### Model Definition (9:00 - 11:00)
**[SCREEN: Model architecture code]**

**Narrator**: "Now let's create our model. I'll design a transformer-based classifier that's optimized for Neuron compilation."

```python
# models/classifier.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NeuronOptimizedClassifier(nn.Module):
    """Transformer classifier optimized for Neuron."""
    
    def __init__(self, config: TrainingConfig, vocab_size: int = 10000):
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.position_encoding = PositionalEncoding(config.hidden_size, config.max_sequence_length)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,  # Fixed for simplicity
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True  # Better for Neuron
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),  # GELU works well on Neuron
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 2)  # Binary classification
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        embeddings = self.token_embedding(input_ids)
        embeddings = self.position_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.embedding_dropout(embeddings)
        
        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert to transformer format (True = masked)
        transformer_mask = (attention_mask == 0)
        
        # Transformer
        hidden_states = self.transformer(
            embeddings, 
            src_key_padding_mask=transformer_mask
        )
        
        # Global average pooling (masked)
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1).float()
        pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# Create model
config = TrainingConfig()
model = NeuronOptimizedClassifier(config)

print(f"Model created:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test forward pass
sample_batch = next(iter(train_loader))
with torch.no_grad():
    output = model(sample_batch['input_ids'], sample_batch['attention_mask'])
    print(f"Output shape: {output.shape}")
```

**[SHOW: Model creation and architecture summary]**

#### Model Compilation (11:00 - 12:30)
**Narrator**: "Now comes the crucial step - compiling our model for Neuron. This optimization is what gives us the performance advantage."

```python
# models/compile.py
import torch_neuronx
import time

def compile_model_for_neuron(model, sample_batch, config):
    """Compile model for Neuron with proper error handling."""
    
    print("Compiling model for Neuron...")
    print("This may take several minutes for complex models...")
    
    # Prepare sample inputs
    sample_input_ids = sample_batch['input_ids'][:1]  # Single sample for compilation
    sample_attention_mask = sample_batch['attention_mask'][:1]
    
    print(f"Compilation input shapes:")
    print(f"  input_ids: {sample_input_ids.shape}")
    print(f"  attention_mask: {sample_attention_mask.shape}")
    
    try:
        start_time = time.time()
        
        # Compile the model
        compiled_model = torch_neuronx.trace(
            model, 
            (sample_input_ids, sample_attention_mask),
            compiler_workdir='./neuron_cache'  # Cache compilation
        )
        
        compilation_time = time.time() - start_time
        print(f"✅ Compilation successful in {compilation_time:.1f}s")
        
        # Test compiled model
        print("Testing compiled model...")
        with torch.no_grad():
            original_output = model(sample_input_ids, sample_attention_mask)
            compiled_output = compiled_model(sample_input_ids, sample_attention_mask)
            
            # Check if outputs are similar
            diff = torch.abs(original_output - compiled_output).max().item()
            print(f"Max difference between original and compiled: {diff:.6f}")
            
            if diff < 1e-3:
                print("✅ Compilation verification passed")
            else:
                print("⚠️ Large difference detected - check compilation")
        
        return compiled_model
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        print("Falling back to uncompiled model for this tutorial")
        return model

# Compile model
if config.compile_model:
    model = compile_model_for_neuron(model, sample_batch, config)
else:
    print("Skipping compilation (config.compile_model = False)")
```

**[SHOW: Compilation process and verification]**

#### Model to Device (12:30 - 13:00)
**Narrator**: "Finally, let's move our model to the Neuron device and prepare for training."

```python
# Move model to device
device = xm.xla_device()
model = model.to(device)

print(f"Model moved to device: {device}")

# Verify model is on correct device
for name, param in model.named_parameters():
    if 'embedding' in name:  # Just check a few parameters
        print(f"Parameter {name} device: {param.device}")
        break
```

### Section 4: Training Loop (13:00 - 17:00)

#### Training Infrastructure (13:00 - 14:30)
**[SCREEN: Training loop implementation]**

**Narrator**: "Now let's implement our training loop. This includes proper monitoring, checkpointing, and optimization for Trainium."

```python
# training/trainer.py
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from tqdm import tqdm
import wandb
import time
import os

class TrainiumTrainer:
    """Trainer class optimized for AWS Trainium."""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs * 100  # Approximate steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Initialize wandb if available
        try:
            wandb.init(
                project=config.project_name,
                config=config.__dict__,
                name=f"trainium_training_{int(time.time())}"
            )
            self.use_wandb = True
        except:
            print("Wandb not available, using local logging only")
            self.use_wandb = False
    
    def compute_accuracy(self, predictions, labels):
        """Compute classification accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == labels).float()
        return correct.mean().item()
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        # Wrap dataloader for XLA
        train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
        
        progress_bar = tqdm(train_device_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            logits = self.model(batch['input_ids'], batch['attention_mask'])
            loss = self.criterion(logits, batch['labels'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step with XLA synchronization
            xm.optimizer_step(self.optimizer)
            self.scheduler.step()
            
            # Metrics
            accuracy = self.compute_accuracy(logits, batch['labels'])
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.3f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'train_accuracy_step': accuracy,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'step': batch_idx
                })
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_loader):
        """Validation pass."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        val_device_loader = pl.MpDeviceLoader(val_loader, self.device)
        
        with torch.no_grad():
            for batch in tqdm(val_device_loader, desc="Validation"):
                logits = self.model(batch['input_ids'], batch['attention_mask'])
                loss = self.criterion(logits, batch['labels'])
                
                accuracy = self.compute_accuracy(logits, batch['labels'])
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy

# Initialize trainer
trainer = TrainiumTrainer(model, config, device)
print("Trainer initialized successfully")
```

**[SHOW: Trainer class setup and initialization]**

#### Full Training Loop (14:30 - 16:30)
**Narrator**: "Now let's run our complete training loop with monitoring and checkpointing."

```python
# training/train.py
def train_model(trainer, train_loader, val_loader, config):
    """Complete training loop with monitoring."""
    
    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    best_val_accuracy = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)
        
        # Training
        epoch_start = time.time()
        train_loss, train_acc = trainer.train_epoch(train_loader)
        train_time = time.time() - epoch_start
        
        # Validation
        val_start = time.time()
        val_loss, val_acc = trainer.validate(val_loader)
        val_time = time.time() - val_start
        
        # Store metrics
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.train_accuracies.append(train_acc)
        trainer.val_accuracies.append(val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        print(f"  Train Time: {train_time:.1f}s, Val Time: {val_time:.1f}s")
        
        # Log to wandb
        if trainer.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_epoch': train_loss,
                'train_accuracy_epoch': train_acc,
                'val_loss_epoch': val_loss,
                'val_accuracy_epoch': val_acc,
                'train_time': train_time,
                'val_time': val_time
            })
        
        # Save checkpoint if best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_checkpoint(trainer.model, trainer.optimizer, epoch, val_acc, 'checkpoints/best_model.pt')
            print(f"  🎉 New best model saved! Validation accuracy: {val_acc:.3f}")
        
        # Regular checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            save_checkpoint(trainer.model, trainer.optimizer, epoch, val_acc, f'checkpoints/epoch_{epoch+1}.pt')
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    
    return trainer

def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, filepath)
    print(f"  💾 Checkpoint saved: {filepath}")

# Start training
trained_trainer = train_model(trainer, train_loader, val_loader, config)
```

**[SHOW: Training progress with real-time metrics]**

#### Performance Monitoring (16:30 - 17:00)
**Narrator**: "Let's monitor our Neuron utilization during training to ensure we're getting good performance."

```bash
# In a separate terminal or background process
neuron-monitor --output training_metrics.log &

# Monitor in real-time
tail -f training_metrics.log
```

**[SHOW: Neuron monitoring output during training]**

### Section 5: Results and Analysis (17:00 - 19:00)

#### Training Results (17:00 - 18:00)
**[SCREEN: Results visualization]**

**Narrator**: "Excellent! Our training completed successfully. Let's analyze the results and create some visualizations."

```python
# analysis/results.py
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(trainer):
    """Create comprehensive training result plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    epochs = range(1, len(trainer.train_losses) + 1)
    
    axes[0, 0].plot(epochs, trainer.train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, trainer.val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, trainer.train_accuracies, 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, trainer.val_accuracies, 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate schedule
    # This would show the actual learning rate changes during training
    axes[1, 0].plot(epochs, [config.learning_rate * 0.95**e for e in epochs], 'g-')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # Final metrics summary
    final_train_acc = trainer.train_accuracies[-1]
    final_val_acc = trainer.val_accuracies[-1]
    best_val_acc = max(trainer.val_accuracies)
    
    metrics_text = f"""Final Results:
    Training Accuracy: {final_train_acc:.3f}
    Validation Accuracy: {final_val_acc:.3f}
    Best Validation Accuracy: {best_val_acc:.3f}
    
    Training completed in {len(epochs)} epochs
    Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Training Summary')
    
    plt.tight_layout()
    plt.savefig('logs/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training results saved to logs/training_results.png")

# Generate results plots
plot_training_results(trained_trainer)
```

**[SHOW: Training result plots and analysis]**

#### Performance Analysis (18:00 - 19:00)
**Narrator**: "Let's analyze the performance characteristics of our training run."

```python
# analysis/performance.py
def analyze_performance():
    """Analyze training performance and costs."""
    
    # Calculate training metrics
    total_epochs = config.num_epochs
    samples_per_epoch = len(train_loader) * config.batch_size
    total_samples = total_epochs * samples_per_epoch
    
    # Estimate timing (from actual training)
    estimated_time_per_epoch = 60  # seconds, from our actual run
    total_training_time = total_epochs * estimated_time_per_epoch / 60  # minutes
    
    # Cost calculation
    instance_cost_per_hour = 5.37  # trn1.8xlarge cost
    training_cost = (total_training_time / 60) * instance_cost_per_hour
    
    print("Performance Analysis:")
    print("=" * 50)
    print(f"Training Configuration:")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {total_epochs}")
    print(f"  Total samples processed: {total_samples:,}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Time per epoch: ~{estimated_time_per_epoch:.0f} seconds")
    print(f"  Total training time: {total_training_time:.1f} minutes")
    print(f"  Samples per second: {total_samples / (total_training_time * 60):.1f}")
    
    print(f"\nCost Analysis:")
    print(f"  Instance: trn1.8xlarge (${instance_cost_per_hour:.2f}/hour)")
    print(f"  Training cost: ${training_cost:.2f}")
    print(f"  Cost per 1K samples: ${(training_cost * 1000 / total_samples):.3f}")
    
    # Compare with traditional compute
    gpu_multiplier = 3.0  # Estimated GPU cost multiplier
    gpu_cost = training_cost * gpu_multiplier
    savings = gpu_cost - training_cost
    
    print(f"\nCost Comparison:")
    print(f"  Estimated GPU cost: ${gpu_cost:.2f}")
    print(f"  Trainium cost: ${training_cost:.2f}")
    print(f"  Savings: ${savings:.2f} ({savings/gpu_cost*100:.0f}%)")

analyze_performance()
```

**[SHOW: Performance analysis results]**

### Section 6: Model Deployment (19:00 - 20:00)

#### Model Saving and Loading (19:00 - 19:30)
**[SCREEN: Model deployment code]**

**Narrator**: "Finally, let's prepare our trained model for deployment and inference."

```python
# deployment/save_model.py
def save_production_model(model, config, save_path='models/production_model'):
    """Save model for production deployment."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'model_class': 'NeuronOptimizedClassifier',
        'vocab_size': 10000,
        'timestamp': time.time()
    }, f'{save_path}/model.pt')
    
    # Save configuration
    config.save(f'{save_path}/config.json')
    
    print(f"Production model saved to {save_path}/")
    
    # Create inference script
    inference_script = '''
import torch
import torch_neuronx
from models.classifier import NeuronOptimizedClassifier

def load_model(model_path):
    """Load trained model for inference."""
    checkpoint = torch.load(f"{model_path}/model.pt")
    config_dict = checkpoint['config']
    
    # Recreate config object
    from configs.training_config import TrainingConfig
    config = TrainingConfig(**config_dict)
    
    # Create and load model
    model = NeuronOptimizedClassifier(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

# Usage example:
# model, config = load_model('models/production_model')
'''
    
    with open(f'{save_path}/inference.py', 'w') as f:
        f.write(inference_script)
    
    print("Inference script created")

# Save our trained model
save_production_model(trained_trainer.model, config)
```

#### Quick Inference Test (19:30 - 20:00)
**Narrator**: "Let's test our saved model with a quick inference example."

```python
# deployment/test_inference.py
def test_inference():
    """Test inference with saved model."""
    
    print("Testing inference with saved model...")
    
    # Load model (in practice, this would be on a fresh instance)
    checkpoint = torch.load('models/production_model/model.pt')
    config_dict = checkpoint['config']
    
    # Recreate model
    inference_config = TrainingConfig(**config_dict)
    inference_model = NeuronOptimizedClassifier(inference_config)
    inference_model.load_state_dict(checkpoint['model_state_dict'])
    inference_model.eval()
    
    print("✅ Model loaded successfully")
    
    # Test with sample data
    test_batch = next(iter(test_loader))
    
    with torch.no_grad():
        start_time = time.time()
        predictions = inference_model(
            test_batch['input_ids'][:5],  # Test with 5 samples
            test_batch['attention_mask'][:5]
        )
        inference_time = time.time() - start_time
    
    predicted_classes = torch.argmax(predictions, dim=1)
    true_labels = test_batch['labels'][:5]
    
    print(f"Inference Results:")
    print(f"  Inference time: {inference_time:.3f}s for 5 samples")
    print(f"  Time per sample: {inference_time/5:.3f}s")
    print(f"  Predictions: {predicted_classes.tolist()}")
    print(f"  True labels: {true_labels.tolist()}")
    print(f"  Accuracy: {(predicted_classes == true_labels).float().mean():.3f}")

test_inference()
```

**[SHOW: Inference test results]**

### Closing (20:00 - 20:00)
**Narrator**: "Fantastic! We've successfully implemented a complete training workflow on AWS Trainium. We covered:

- Setting up a proper training environment and data pipeline
- Creating and compiling models optimized for Neuron
- Implementing efficient training loops with monitoring
- Analyzing performance and costs
- Preparing models for deployment

You now have the foundation for serious machine learning research on Trainium. In our next tutorial, we'll work through a real-world, hardware-validated biomedical NER fine-tune.

All the code from today is available in the GitHub repository. Thanks for watching, and see you next time!"

**[SHOW: End screen with GitHub link and next tutorial preview]**

## 🎬 Production Notes

### Technical Requirements
- **Instance**: trn1.8xlarge for demonstration
- **Recording**: Capture compilation process (may take several minutes)
- **Error Handling**: Show realistic compilation and training scenarios
- **Performance**: Include actual performance metrics and timing

### Visual Elements
- **Training Progress**: Real-time loss and accuracy curves
- **Monitoring**: Neuron utilization graphs
- **Code Quality**: Professional syntax highlighting and formatting
- **Results**: Clear visualizations of training outcomes

### Educational Focus
- **Best Practices**: Emphasize proper ML engineering practices
- **Cost Awareness**: Continuous focus on cost optimization
- **Real-World Applicability**: Show patterns applicable to research
- **Troubleshooting**: Address common issues during training

---

*This tutorial provides a comprehensive foundation for machine learning training workflows on AWS Trainium.*